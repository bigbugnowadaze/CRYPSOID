"""Photoreal Audi — studio multi-light + ACES tonemap + presentation backdrop.

Output: renders/crypsorender_v01/SHOWCASE_AUDI_PHOTOREAL.png
        (and a 2K version for hero use)

This is "what would this look like if a 3D artist published it" — three-point
lighting (key + fill + rim), bright photographic backdrop, ACES filmic tonemap,
subtle vignette + color grade.

Same source data as Bar 2 (v40 .3dphox). The renderer hasn't changed; we're
just lighting the scene the way a marketing render would be set up.
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
from crypsorender.math.mip_splatting_filter import apply_mip_splatting_filter, per_splat_filter_radius
from crypsorender.math.photoreal import (
    StudioEnvironment, three_point_directions,
    apply_photoreal_lighting,
    aces_filmic, color_grade, vignette,
)


SRC = ROOT / 'outputs' / 'v40_audi_full_mipfilled.3dphox'
OUT_2K = ROOT / 'renders' / 'crypsorender_v01' / 'SHOWCASE_AUDI_PHOTOREAL_2k.png'
OUT_1K = ROOT / 'renders' / 'crypsorender_v01' / 'SHOWCASE_AUDI_PHOTOREAL.png'

CAMERA = CameraParams(yaw_deg=35, pitch_deg=18, distance=2.4, fov_deg=42, size=2048)


def main():
    t0 = time.perf_counter()
    print(f"[1/9] Loading v40 .3dphox + aux ...")
    sb = load_3dphox(SRC)
    aux = load_aux_from_3dphox(SRC)
    normals_full = aux['normals'].astype(np.float32)
    cusp_full    = aux.get('cusp_norm')
    edges_full   = aux.get('edges')
    mip_full     = aux.get('material_mip')
    print(f"      base N = {sb.n:,}")

    print(f"[2/9] PBR decomposition ...")
    pbr_full = decompose_pbr(sb.sh_dc, sb.sh_rest, sb.opacities)
    print(decompose_summary(pbr_full))

    print(f"[3/9] Camera + projection ...")
    cam = Camera(sb.xyz, CAMERA)
    centers_2d, cov_2d_inv, radii, depths, keep, idx = project_splats(sb, cam)
    print(f"      {len(centers_2d):,} splats survive at {CAMERA.size}px")

    # Recover cov_2d for Mip-Splatting
    a = cov_2d_inv[:, 0, 0]; b = cov_2d_inv[:, 0, 1]; d = cov_2d_inv[:, 1, 1]
    inv_det = 1.0 / np.where(np.abs(a*d - b*b) < 1e-12, 1e-12, a*d - b*b)
    cov_2d = np.empty_like(cov_2d_inv, dtype=np.float32)
    cov_2d[:, 0, 0] =  d * inv_det
    cov_2d[:, 0, 1] = -b * inv_det
    cov_2d[:, 1, 0] = -b * inv_det
    cov_2d[:, 1, 1] =  a * inv_det

    # Subset to visible
    normals_v = normals_full[idx]
    cusp_v    = cusp_full[idx] if cusp_full is not None else None
    mip_v     = mip_full[idx]  if mip_full  is not None else None
    xyz_v     = sb.xyz[idx]
    opa_v     = sb.opacities[idx].astype(np.float32)
    pbr_v     = {k: v[idx] for k, v in pbr_full.items()}

    # Mip-Splatting prefilter
    if mip_v is not None:
        per_radius = per_splat_filter_radius(mip_v, depths, cam.focal, min_px=0.5)
        cov_2d_filt, opa_filt, radii_filt = apply_mip_splatting_filter(
            cov_2d, opa_v, radii, min_filter_px=float(np.median(per_radius))
        )
        a = cov_2d_filt[:, 0, 0]; b = cov_2d_filt[:, 0, 1]; d = cov_2d_filt[:, 1, 1]
        det_post = a*d - b*b
        inv_det = 1.0 / np.where(np.abs(det_post) < 1e-12, 1e-12, det_post)
        cov_2d_inv_filt = np.empty_like(cov_2d_filt, dtype=np.float32)
        cov_2d_inv_filt[:, 0, 0] =  d * inv_det
        cov_2d_inv_filt[:, 0, 1] = -b * inv_det
        cov_2d_inv_filt[:, 1, 0] = -b * inv_det
        cov_2d_inv_filt[:, 1, 1] =  a * inv_det
    else:
        cov_2d_inv_filt = cov_2d_inv; opa_filt = opa_v; radii_filt = radii

    print(f"[4/9] SH decode (visible) ...")
    view_dirs = sb.xyz[idx] - cam.eye[None, :]
    view_dirs_norm = view_dirs / (np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-9)
    if sb.sh_rest is not None:
        view_color = eval_sh_color(sb.sh_dc[idx], sb.sh_rest[idx], view_dirs_norm)
    else:
        view_color = sb.sh_dc[idx] * 0.28209479177387814 + 0.5
    view_color = np.clip(view_color, 0.0, 1.0).astype(np.float32)

    print(f"[5/9] kNN shadows + graph AO ...")
    nbr_opa = sb.opacities[edges_full].astype(np.float32)
    # KEY-light shadow uses the key direction; computed below after lights are known.
    # For Bar 2 within photoreal we compute one shadow pass against the KEY light.
    # This is faithful for the key contribution (the dominant light) and is good
    # enough — fill + rim get unshadowed which matches a real studio setup.

    print(f"[6/9] Three-point lighting rig setup ...")
    lights = three_point_directions(
        camera_eye=cam.eye, scene_center=cam.center,
        key_az=35, key_el=20,
        fill_az=-50, fill_el=12,
        rim_az=170, rim_el=42,
    )
    KEY_RGB  = np.array([1.0, 0.96, 0.88], dtype=np.float32) * 1.55     # warm key
    FILL_RGB = np.array([0.55, 0.65, 0.85], dtype=np.float32) * 1.20    # cool fill
    RIM_RGB  = np.array([0.95, 0.95, 1.00], dtype=np.float32) * 1.30    # bright neutral rim

    # Now compute key-light shadow + AO
    shadow_full = knn_shadow_factor(
        xyz=sb.xyz, neighbors=edges_full,
        neighbor_scales=sb.scales[edges_full],
        neighbor_opacities=nbr_opa,
        light_dir=lights['key_dir'], strength=0.85,
    )
    ao_full = knn_graph_ao(
        xyz=sb.xyz, neighbors=edges_full,
        normals=normals_full,
        neighbor_opacities=nbr_opa,
        ao_radius=0.05, gamma=0.7,
    )
    shadow_v = shadow_full[idx]; ao_v = ao_full[idx]
    print(f"      shadow mean={shadow_v.mean():.3f}  AO mean={ao_v.mean():.3f}")

    # v32b curvature visibility against the KEY light
    NdotL = np.maximum(0.0, normals_v @ -(lights['key_dir']/np.linalg.norm(lights['key_dir'])))
    if cusp_v is not None:
        kappa_eff = cusp_v.astype(np.float32)
        curv_vis = NdotL * (1.0 - 0.5 * kappa_eff * (1.0 - NdotL))
        curv_vis = np.clip(curv_vis, 0.0, 1.0).astype(np.float32)
    else:
        curv_vis = NdotL.astype(np.float32)

    scales_v_lin = np.exp(sb.scales[idx]).astype(np.float32)
    pix_at_d = cam.focal * scales_v_lin.max(axis=1) / np.maximum(np.abs(depths), 0.05)

    print(f"[7/9] Studio environment + photoreal compose (HDR linear) ...")
    t1 = time.perf_counter()
    env = StudioEnvironment(sun_dir=lights['key_dir'], intensity=1.0)
    shaded_hdr = apply_photoreal_lighting(
        albedo=view_color,
        metallic=pbr_v['metallic'], roughness=pbr_v['roughness'],
        F0=pbr_v['F0'], kd=pbr_v['kd'],
        normals=normals_v, xyz=xyz_v, eye=cam.eye.astype(np.float32),
        environment=env,
        key_dir=lights['key_dir'],   key_rgb=KEY_RGB,
        fill_dir=lights['fill_dir'], fill_rgb=FILL_RGB,
        rim_dir=lights['rim_dir'],   rim_rgb=RIM_RGB,
        shadow_factor=shadow_v.astype(np.float32),
        ao_factor=ao_v.astype(np.float32),
        curvature_visibility=curv_vis,
        cusp_norm=cusp_v,
        max_pixel_size=pix_at_d,
        env_ambient_strength=0.50,
        env_reflection_strength=1.0,
        cusp_glint_strength=0.55,
    )
    print(f"      compose {time.perf_counter()-t1:.2f}s   "
          f"shaded HDR range: min={shaded_hdr.min():.3f} max={shaded_hdr.max():.3f}")

    print(f"[8/9] Sort + rasterize at {CAMERA.size}x{CAMERA.size} ...")
    t2 = time.perf_counter()
    order = np.argsort(-depths)
    fb, ab = rasterize_splats_numba(
        centers_2d[order].astype(np.float32),
        cov_2d_inv_filt[order].astype(np.float32),
        radii_filt[order].astype(np.float32),
        opa_filt[order],
        shaded_hdr[order],
        CAMERA.size, CAMERA.size,
    )
    print(f"      raster {time.perf_counter()-t2:.2f}s")

    # Background: composite over a soft photographic grey gradient
    H, W = fb.shape[:2]
    yy = np.arange(H, dtype=np.float32) / max(H - 1, 1)
    grad_top = np.array([0.93, 0.94, 0.96], dtype=np.float32)
    grad_bot = np.array([0.74, 0.75, 0.77], dtype=np.float32)
    bg = grad_top[None, None, :] * (1 - yy[:, None, None]) \
       + grad_bot[None, None, :] *  yy[:, None, None]
    bg = np.broadcast_to(bg, (H, W, 3))
    # Compose: scene where alpha > 0, bg elsewhere (using ab as foreground coverage)
    a = np.clip(ab, 0.0, 1.0)[..., None]
    composited = fb + (1.0 - a) * bg

    print(f"[9/9] ACES tonemap + color grade + vignette + sRGB ...")
    tone = aces_filmic(composited)
    graded = color_grade(tone, exposure_stops=0.05, contrast=1.06,
                         saturation=1.10)
    vig = vignette(graded, strength=0.18, falloff=1.7)
    srgb = linear_to_srgb_safe(vig)
    img8 = (np.clip(srgb, 0, 1) * 255).astype(np.uint8)

    Image.fromarray(img8).save(OUT_2K)
    Image.fromarray(img8).resize((1024, 1024), Image.LANCZOS).save(OUT_1K)
    print()
    print("=" * 70)
    print(f"DONE in {time.perf_counter()-t0:.2f}s")
    print(f"  -> {OUT_2K} (2k)")
    print(f"  -> {OUT_1K} (1k)")
    print("=" * 70)


def linear_to_srgb_safe(x):
    a = 0.055
    return np.where(x <= 0.0031308,
                    12.92 * x,
                    (1.0 + a) * np.power(np.maximum(x, 1e-12), 1.0 / 2.4) - a)


if __name__ == '__main__':
    main()
