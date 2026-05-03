"""Bar 2 hero render — full PBR + environment reflections + Mip-Splatting prefilter.

Output: renders/crypsorender_v01/SHOWCASE_AUDI_BAR2.png

Pipeline:
  1. Load v40 .3dphox (geom + SH + opacity) + aux (normals/edges/material/kappa/cusp/mip)
  2. Per-splat PBR decomposition (albedo / metallic / roughness / F0 / kd) from SH
  3. Project to camera + Mip-Splatting prefilter on 2D covariance + opacity
  4. Decode SH → view-dependent base color
  5. kNN soft shadows + graph AO
  6. v32b curvature visibility from cusp_norm
  7. Build procedural environment (sky/ground/sun)
  8. Apply Bar 2 lighting: PBR direct (sun) + env-ambient + env-reflections + cusp glint
  9. Rasterize at 2048x2048, mild tone curve, Lanczos to 1024x1024
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path('/sessions/ecstatic-sleepy-curie/mnt/Crypsoid')
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np
from PIL import Image

from crypsorender.io.phox_loader import load_3dphox, load_aux_from_3dphox
from crypsorender.pipeline.camera import Camera, CameraParams
from crypsorender.pipeline.project import project_splats
from crypsorender.pipeline.rasterize_numba import rasterize_splats_numba
from crypsorender.math.sh import eval_sh_color
from crypsorender.math.shadows_knn import knn_shadow_factor, knn_graph_ao
from crypsorender.math.material_decompose import decompose_pbr, decompose_summary
from crypsorender.math.environment import ProceduralEnvironment
from crypsorender.math.mip_splatting_filter import apply_mip_splatting_filter, per_splat_filter_radius
from crypsorender.math.bar2_lighting import apply_bar2_lighting


SRC = ROOT / 'outputs' / 'v40_audi_full_mipfilled.3dphox'
OUT = ROOT / 'renders' / 'crypsorender_v01' / 'SHOWCASE_AUDI_BAR2.png'

CAMERA = CameraParams(yaw_deg=35, pitch_deg=18, distance=2.4, fov_deg=42, size=2048)
OUT_SIZE = 1024

SUN_DIR = np.array([0.4, -0.7, 0.6], dtype=np.float32)
SUN_RGB = np.array([1.0, 0.96, 0.85], dtype=np.float32) * 1.45


def main():
    t0 = time.perf_counter()
    print(f"[1/9] Loading v40 .3dphox base ...")
    sb = load_3dphox(SRC)
    print(f"      base N = {sb.n:,}")

    print(f"[2/9] Loading aux data ...")
    aux = load_aux_from_3dphox(SRC)
    normals_full   = aux.get('normals')
    material_full  = aux.get('material_hint')
    cusp_full      = aux.get('cusp_norm')
    edges_full     = aux.get('edges')
    mip_full       = aux.get('material_mip')           # mip_zoom byte from v33
    print(f"      normals={normals_full is not None}  material={material_full is not None}  "
          f"cusp={cusp_full is not None}  edges={edges_full is not None}  mip={mip_full is not None}")
    if normals_full is None:
        raise SystemExit("Bar 2 needs v31 normals.")

    print(f"[3/9] PBR material decomposition (per-splat, full set) ...")
    t1 = time.perf_counter()
    pbr_full = decompose_pbr(sb.sh_dc, sb.sh_rest, sb.opacities)
    print(decompose_summary(pbr_full))
    print(f"      decomposed in {time.perf_counter()-t1:.2f}s")

    print(f"[4/9] Building camera ...")
    cam = Camera(sb.xyz, CAMERA)
    print(f"      eye={cam.eye}, focal={cam.focal:.0f}")

    print(f"[5/9] Projecting splats ...")
    t2 = time.perf_counter()
    centers_2d, cov_2d_inv, radii, depths, keep, idx = project_splats(sb, cam)
    print(f"      {len(centers_2d):,} splats survive ({time.perf_counter()-t2:.2f}s)")

    # We need the covariance (not its inverse) to apply the Mip-Splatting filter.
    # Re-invert from cov_2d_inv since the project pipeline doesn't return cov_2d directly.
    # cov_2d = (cov_2d_inv)^-1 elementwise via the analytic 2x2 inverse.
    a = cov_2d_inv[:, 0, 0]; b = cov_2d_inv[:, 0, 1]; d = cov_2d_inv[:, 1, 1]
    det_inv = a * d - b * b
    inv_det = 1.0 / np.where(np.abs(det_inv) < 1e-12, 1e-12, det_inv)
    cov_2d = np.empty_like(cov_2d_inv, dtype=np.float32)
    cov_2d[:, 0, 0] =  d * inv_det
    cov_2d[:, 0, 1] = -b * inv_det
    cov_2d[:, 1, 0] = -b * inv_det
    cov_2d[:, 1, 1] =  a * inv_det

    # Subset all per-splat aux to visible
    normals_v   = normals_full[idx].astype(np.float32)
    cusp_v      = cusp_full[idx]      if cusp_full     is not None else None
    mip_v       = mip_full[idx]       if mip_full      is not None else None
    xyz_v       = sb.xyz[idx]
    opa_v       = sb.opacities[idx].astype(np.float32)
    pbr_v = {k: v[idx] for k, v in pbr_full.items()}

    # ---------- Step 6: Mip-Splatting prefilter (if mip_zoom present) ----------
    if mip_v is not None:
        # Per-splat min filter radius from mip_zoom byte
        per_radius = per_splat_filter_radius(mip_v, depths, cam.focal, min_px=0.5)
        # Use the AVERAGE radius as the global filter (Mip-Splatting paper does
        # this per-splat; our implementation uses one global radius for speed).
        global_radius = float(np.median(per_radius))
        cov_2d_filt, opa_filt, radii_filt = apply_mip_splatting_filter(
            cov_2d, opa_v, radii, min_filter_px=global_radius
        )
        n_attenuated = int((opa_filt < opa_v).sum())
        print(f"      Mip-Splatting prefilter: median radius = {global_radius:.3f}px, "
              f"attenuated {n_attenuated:,} small splats ({100*n_attenuated/len(opa_v):.1f}%)")
        # Re-invert filtered covariance for rasterizer
        a = cov_2d_filt[:, 0, 0]; b = cov_2d_filt[:, 0, 1]; d = cov_2d_filt[:, 1, 1]
        det_post = a * d - b * b
        inv_det = 1.0 / np.where(np.abs(det_post) < 1e-12, 1e-12, det_post)
        cov_2d_inv_filt = np.empty_like(cov_2d_filt, dtype=np.float32)
        cov_2d_inv_filt[:, 0, 0] =  d * inv_det
        cov_2d_inv_filt[:, 0, 1] = -b * inv_det
        cov_2d_inv_filt[:, 1, 0] = -b * inv_det
        cov_2d_inv_filt[:, 1, 1] =  a * inv_det
    else:
        cov_2d_inv_filt = cov_2d_inv
        opa_filt = opa_v
        radii_filt = radii

    print(f"[6/9] Decoding view-dependent SH for albedo recolor ...")
    view_dirs = sb.xyz[idx] - cam.eye[None, :]
    view_dirs_norm = view_dirs / (np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-9)
    if sb.sh_rest is not None:
        view_color = eval_sh_color(sb.sh_dc[idx], sb.sh_rest[idx], view_dirs_norm)
    else:
        view_color = sb.sh_dc[idx] * 0.28209479177387814 + 0.5
    view_color = np.clip(view_color, 0.0, 1.0).astype(np.float32)
    # Use the view-dependent color as the working "albedo" passed to Bar 2 — the
    # decomposition already produced a view-INdependent albedo from DC, but for
    # visual fidelity we use the view-blended one for the diffuse term and let
    # the env reflection layer provide the specular reflection of the surroundings.
    bar2_albedo = view_color

    print(f"[7/9] kNN shadows + graph AO (full set, k={aux.get('k', 4)}) ...")
    t3 = time.perf_counter()
    if edges_full is not None:
        nbr_opa = sb.opacities[edges_full].astype(np.float32)
        shadow_full = knn_shadow_factor(
            xyz=sb.xyz, neighbors=edges_full,
            neighbor_scales=sb.scales[edges_full],
            neighbor_opacities=nbr_opa,
            light_dir=SUN_DIR, strength=0.85,
        )
        ao_full = knn_graph_ao(
            xyz=sb.xyz, neighbors=edges_full,
            normals=normals_full.astype(np.float32),
            neighbor_opacities=nbr_opa,
            ao_radius=0.05, gamma=0.7,
        )
        shadow_v = shadow_full[idx]
        ao_v = ao_full[idx]
    else:
        shadow_v = np.ones(len(idx), dtype=np.float32)
        ao_v     = np.ones(len(idx), dtype=np.float32)
    print(f"      shadows+AO {time.perf_counter()-t3:.2f}s   "
          f"shadow mean={shadow_v.mean():.3f}, AO mean={ao_v.mean():.3f}")

    # v32b curvature visibility from cusp_norm
    NdotL = np.maximum(0.0, normals_v @ -(SUN_DIR/np.linalg.norm(SUN_DIR)))
    if cusp_v is not None:
        kappa_eff = cusp_v.astype(np.float32)
        curv_vis = NdotL * (1.0 - 0.5 * kappa_eff * (1.0 - NdotL))
        curv_vis = np.clip(curv_vis, 0.0, 1.0).astype(np.float32)
    else:
        curv_vis = NdotL.astype(np.float32)

    # Per-splat max projected pixel size (for cusp sub-pixel)
    scales_v_lin = np.exp(sb.scales[idx]).astype(np.float32)
    max_sigma = scales_v_lin.max(axis=1)
    pix_at_d  = cam.focal * max_sigma / np.maximum(np.abs(depths), 0.05)

    print(f"[8/9] Building procedural environment + Bar 2 PBR compose ...")
    t4 = time.perf_counter()
    env = ProceduralEnvironment(sun_dir=SUN_DIR, intensity=1.0)
    shaded = apply_bar2_lighting(
        albedo=bar2_albedo,
        metallic=pbr_v['metallic'],
        roughness=pbr_v['roughness'],
        F0=pbr_v['F0'],
        kd=pbr_v['kd'],
        normals=normals_v,
        xyz=xyz_v,
        eye=cam.eye.astype(np.float32),
        sun_dir=SUN_DIR, sun_rgb=SUN_RGB,
        environment=env,
        shadow_factor=shadow_v.astype(np.float32),
        ao_factor=ao_v.astype(np.float32),
        curvature_visibility=curv_vis,
        cusp_norm=cusp_v,
        max_pixel_size=pix_at_d,
        sun_strength=1.4,
        env_ambient_strength=0.55,
        env_reflection_strength=0.85,
        cusp_glint_strength=0.55,
    )
    print(f"      compose {time.perf_counter()-t4:.2f}s")

    # ---------- Step 9: Sort + rasterize ----------
    print(f"[9/9] Sorting + rasterizing ({CAMERA.size}x{CAMERA.size}) ...")
    t5 = time.perf_counter()
    order = np.argsort(-depths)
    fb, ab = rasterize_splats_numba(
        centers_2d[order].astype(np.float32),
        cov_2d_inv_filt[order].astype(np.float32),
        radii_filt[order].astype(np.float32),
        opa_filt[order].astype(np.float32),
        shaded[order].astype(np.float32),
        CAMERA.size, CAMERA.size,
    )
    print(f"      raster {time.perf_counter()-t5:.2f}s")

    fb = np.clip(fb, 0.0, 1.0)
    # Mild tone curve (gamma 0.92 + smoothstep contrast)
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
