"""Bar 1 hero render — full Audi at GGX + HDRI environment ambient + cusp sub-pixel.

Output: renders/crypsorender_v01/SHOWCASE_AUDI_BAR1.png

Pipeline:
  1. Load v40 .3dphox (geometry + SH + opacity)
  2. Load v31 normals + edges, v33 material_hints, v40 kappa + cusp_norm
  3. Project to camera, decode SH for view-dependent base color
  4. Compute kNN soft shadows + graph AO (cached if available, otherwise on the fly)
  5. Compute v32b curvature visibility from MLS
  6. Apply Bar 1 lighting (GGX + HDRI ambient + cusp sub-pixel proper integration)
  7. Rasterize at 2048x2048, Lanczos to 1024x1024
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path('/sessions/ecstatic-sleepy-curie/mnt/Crypsoid')
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np
from PIL import Image

from crypsorender.io.phox_loader import load_3dphox, load_aux_from_3dphox
from crypsorender.io.splat_buffer import SplatBuffer
from crypsorender.pipeline.camera import Camera, CameraParams
from crypsorender.pipeline.project import project_splats
from crypsorender.pipeline.rasterize_numba import rasterize_splats_numba
from crypsorender.math.sh import eval_sh_color
from crypsorender.math.shadows_knn import knn_shadow_factor, knn_graph_ao
from crypsorender.math.bar1_lighting import apply_bar1_lighting


SRC = ROOT / 'outputs' / 'v40_audi_full_mipfilled.3dphox'
OUT = ROOT / 'renders' / 'crypsorender_v01' / 'SHOWCASE_AUDI_BAR1.png'

CAMERA = CameraParams(yaw_deg=35, pitch_deg=18, distance=2.4, fov_deg=42, size=2048)
OUT_SIZE = 1024

# Same sun direction we used for the prior MAX hero
SUN_DIR  = np.array([0.4, -0.7, 0.6], dtype=np.float32)
SUN_RGB  = np.array([1.0, 0.96, 0.85], dtype=np.float32) * 1.55


def main():
    t0 = time.perf_counter()
    print(f"[1/7] Loading v40 .3dphox base ...")
    sb = load_3dphox(SRC)
    print(f"      base N = {sb.n:,}")

    print(f"[2/7] Loading aux data (normals + materials + kappa + cusp) ...")
    aux = load_aux_from_3dphox(SRC)
    normals_full = aux.get('normals')
    material_hint_full = aux.get('material_hint')
    view_dep_full      = aux.get('material_view_dep')
    cusp_full          = aux.get('cusp_norm')
    edges_full         = aux.get('edges')
    k_full             = aux.get('k', 4)
    print(f"      normals={normals_full is not None}, material={material_hint_full is not None}, "
          f"cusp={cusp_full is not None}, edges={edges_full is not None}")

    print(f"[3/7] Building camera ...")
    cam = Camera(sb.xyz, CAMERA)
    print(f"      eye={cam.eye}, focal={cam.focal:.0f}")

    print(f"[4/7] Projecting splats ...")
    t1 = time.perf_counter()
    centers_2d, cov_2d_inv, radii, depths, keep, idx = project_splats(sb, cam)
    print(f"      {len(centers_2d):,} splats survive ({time.perf_counter()-t1:.2f}s)")

    print(f"[5/7] Decoding view-dependent SH color ...")
    view_dirs = sb.xyz[idx] - cam.eye[None, :]
    view_dirs_norm = view_dirs / (np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-9)
    if sb.sh_rest is not None:
        albedo = eval_sh_color(sb.sh_dc[idx], sb.sh_rest[idx], view_dirs_norm)
    else:
        albedo = sb.sh_dc[idx] * 0.28209479177387814 + 0.5
    albedo = np.clip(albedo, 0.0, 1.0).astype(np.float32)

    # Subsetted aux to visible splats
    normals_v       = normals_full[idx]      if normals_full is not None       else None
    mh_v            = material_hint_full[idx] if material_hint_full is not None else None
    vd_v            = view_dep_full[idx]      if view_dep_full is not None      else None
    cusp_v          = cusp_full[idx]          if cusp_full is not None          else None
    xyz_v           = sb.xyz[idx]
    opa_v           = sb.opacities[idx].astype(np.float32)

    if normals_v is None:
        raise SystemExit("Bar 1 lighting requires v31 normals; aux dict had none.")

    # Fast scale recovery for shadows + max pixel size
    scales_v   = np.exp(sb.scales[idx]).astype(np.float32)
    max_sigma  = scales_v.max(axis=1)
    pix_at_d   = cam.focal * max_sigma / np.maximum(np.abs(depths), 0.05)

    print(f"[6/7] kNN shadows + graph AO ({sb.n:,} full splats, k={k_full}) ...")
    t2 = time.perf_counter()
    if edges_full is not None:
        # Compute on the FULL set (neighbor indices index into full xyz),
        # then subset down to visible splats afterwards.
        nbr_opa_full = sb.opacities[edges_full].astype(np.float32)
        nbr_log_scales_full = sb.scales[edges_full]
        shadow_full = knn_shadow_factor(
            xyz=sb.xyz, neighbors=edges_full,
            neighbor_scales=nbr_log_scales_full,
            neighbor_opacities=nbr_opa_full,
            light_dir=SUN_DIR, strength=0.85,
        )
        ao_full = knn_graph_ao(
            xyz=sb.xyz, neighbors=edges_full,
            normals=normals_full.astype(np.float32),
            neighbor_opacities=nbr_opa_full,
            ao_radius=0.05, gamma=0.7,
        )
        shadow = shadow_full[idx]
        ao = ao_full[idx]
    else:
        # Fallback: no shadows / no AO
        shadow = np.ones(len(idx), dtype=np.float32)
        ao     = np.ones(len(idx), dtype=np.float32)
    print(f"      shadows+AO {time.perf_counter()-t2:.2f}s   "
          f"shadow mean={shadow.mean():.3f}, AO mean={ao.mean():.3f}")

    # v32b curvature visibility (inline; uses cusp + N.L)
    NdotL = np.maximum(0.0, normals_v @ -(SUN_DIR/np.linalg.norm(SUN_DIR)))
    if cusp_v is not None:
        # high cusp → visibility softens at grazing
        kappa_eff = cusp_v.astype(np.float32)
        beta = 0.5
        curv_vis = NdotL * (1.0 - beta * kappa_eff * (1.0 - NdotL))
        curv_vis = np.clip(curv_vis, 0.0, 1.0).astype(np.float32)
    else:
        curv_vis = NdotL.astype(np.float32)

    print(f"[7/7] Bar 1 lighting compose (GGX + HDRI + cusp sub-pixel) ...")
    shaded = apply_bar1_lighting(
        albedo=albedo, normals=normals_v.astype(np.float32),
        xyz=xyz_v, eye=cam.eye.astype(np.float32),
        sun_dir=SUN_DIR, sun_rgb=SUN_RGB,
        shadow_factor=shadow.astype(np.float32),
        ao_factor=ao.astype(np.float32),
        curvature_visibility=curv_vis,
        material_hint=mh_v, view_dep=vd_v, cusp_norm=cusp_v,
        max_pixel_size=pix_at_d.astype(np.float32),
        ambient_intensity=0.65,
        specular_strength=1.4,
        cusp_glint_strength=0.7,
    )

    # Sort back-to-front and rasterize
    print(f"      sorting + rasterizing ({CAMERA.size}x{CAMERA.size}) ...")
    t3 = time.perf_counter()
    order = np.argsort(-depths)
    fb, ab = rasterize_splats_numba(
        centers_2d[order].astype(np.float32),
        cov_2d_inv[order].astype(np.float32),
        radii[order].astype(np.float32),
        opa_v[order], shaded[order],
        CAMERA.size, CAMERA.size,
    )
    print(f"      raster {time.perf_counter()-t3:.2f}s")

    fb = np.clip(fb, 0.0, 1.0)
    # Mild tone curve (gamma 0.92, gentle smoothstep)
    tone = np.power(fb, 0.92)
    tone = tone * tone * (3.0 - 2.0 * tone)
    img = (tone * 255).astype(np.uint8)
    pil = Image.fromarray(img).resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
    pil.save(OUT)
    print()
    print("=" * 70)
    print(f"DONE in {time.perf_counter()-t0:.2f}s   ->  {OUT}")
    print("=" * 70)


if __name__ == '__main__':
    main()
