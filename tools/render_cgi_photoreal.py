"""F.26 — Render the CGI studio .3dphox through the full photoreal stack.

Same renderer as render_audi_photoreal_v2.py, but:
  - source = outputs/cgi_studio_v1.3dphox  (clean procedural scene)
  - PBR comes from outputs/cgi_studio_v1.pbr.npz (ground-truth, NOT SH-decomposed)
  - camera framed for the studio scene
  - background is a brighter studio sweep (no dark grading)

This closes the AUDI_HERO_attempt.md "input data, not pipeline" finding —
the renderer hasn't changed; only the input has, and the result demonstrates
that the lit stack produces clean Blender-style output when fed clean data.
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
from crypsorender.math.mip_splatting_filter import (
    apply_mip_splatting_filter, per_splat_filter_radius,
)
from crypsorender.math.photoreal import (
    StudioEnvironment, three_point_directions,
    apply_photoreal_lighting,
    aces_filmic, color_grade, vignette,
)
from crypsorender.math.shadows_knn import knn_shadow_factor, knn_graph_ao


SRC     = ROOT / 'outputs' / 'cgi_studio_v1.3dphox'
PBR_SRC = ROOT / 'outputs' / 'cgi_studio_v1.pbr.npz'
OUT_2K  = ROOT / 'renders' / 'crypsorender_v01' / 'SHOWCASE_CGI_STUDIO_2k.png'
OUT_1K  = ROOT / 'renders' / 'crypsorender_v01' / 'SHOWCASE_CGI_STUDIO.png'

# Camera framed on the scene (center ~ (0, 0.30, 0))
CAMERA = CameraParams(yaw_deg=35, pitch_deg=12, distance=1.35, fov_deg=44, size=2048)


def linear_to_srgb_safe(x):
    a = 0.055
    return np.where(x <= 0.0031308,
                    12.92 * x,
                    (1.0 + a) * np.power(np.maximum(x, 1e-12), 1.0 / 2.4) - a)


def main():
    t0 = time.perf_counter()
    print(f'[1/8] Loading CGI .3dphox ...')
    sb = load_3dphox(SRC)
    aux = load_aux_from_3dphox(SRC)
    normals_full = aux['normals'].astype(np.float32)
    cusp_full    = aux.get('cusp_norm')
    edges_full   = aux.get('edges')
    mip_full     = aux.get('material_mip')
    pbr_npz = np.load(PBR_SRC, allow_pickle=True)
    print(f'      N = {sb.n:,} splats')

    print(f'[2/8] Loading ground-truth PBR (no SH decomposition) ...')
    pbr_full = {
        'albedo':    pbr_npz['albedo'].astype(np.float32),
        'metallic':  pbr_npz['metallic'].astype(np.float32),
        'roughness': pbr_npz['roughness'].astype(np.float32),
        'F0':        pbr_npz['F0'].astype(np.float32),
        'kd':        pbr_npz['kd'].astype(np.float32),
    }
    print(f'      metallic min/median/max = '
          f'{pbr_full["metallic"].min():.3f} / '
          f'{np.median(pbr_full["metallic"]):.3f} / '
          f'{pbr_full["metallic"].max():.3f}')
    print(f'      roughness min/median/max = '
          f'{pbr_full["roughness"].min():.3f} / '
          f'{np.median(pbr_full["roughness"]):.3f} / '
          f'{pbr_full["roughness"].max():.3f}')

    print(f'[3/8] Camera + projection ...')
    cam = Camera(sb.xyz, CAMERA)
    centers_2d, cov_2d_inv, radii, depths, keep, idx = project_splats(sb, cam)
    print(f'      {len(centers_2d):,} splats survive at {CAMERA.size}px')

    # Recover cov_2d for Mip-Splatting prefilter
    a = cov_2d_inv[:, 0, 0]; b = cov_2d_inv[:, 0, 1]; d = cov_2d_inv[:, 1, 1]
    inv_det = 1.0 / np.where(np.abs(a*d - b*b) < 1e-12, 1e-12, a*d - b*b)
    cov_2d = np.empty_like(cov_2d_inv, dtype=np.float32)
    cov_2d[:, 0, 0] =  d * inv_det
    cov_2d[:, 0, 1] = -b * inv_det
    cov_2d[:, 1, 0] = -b * inv_det
    cov_2d[:, 1, 1] =  a * inv_det

    normals_v = normals_full[idx]
    cusp_v    = cusp_full[idx] if cusp_full is not None else None
    mip_v     = mip_full[idx]  if mip_full  is not None else None
    xyz_v     = sb.xyz[idx]
    opa_v     = sb.opacities[idx].astype(np.float32)
    pbr_v     = {k: v[idx] for k, v in pbr_full.items()}

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

    print(f'[4/8] Three-point lighting rig ...')
    lights = three_point_directions(
        camera_eye=cam.eye, scene_center=cam.center,
        key_az=40,  key_el=35,
        fill_az=-55, fill_el=18,
        rim_az=160, rim_el=50,
    )
    KEY_RGB  = np.array([1.00, 0.97, 0.90], dtype=np.float32) * 0.95
    FILL_RGB = np.array([0.60, 0.70, 0.90], dtype=np.float32) * 0.42
    RIM_RGB  = np.array([0.96, 0.96, 1.00], dtype=np.float32) * 0.55

    print(f'[5/8] kNN shadows + graph AO (against KEY light) ...')
    nbr_opa = sb.opacities[edges_full].astype(np.float32)
    shadow_full = knn_shadow_factor(
        xyz=sb.xyz, neighbors=edges_full,
        neighbor_scales=sb.scales[edges_full],
        neighbor_opacities=nbr_opa,
        light_dir=lights['key_dir'], strength=0.45,
    )
    ao_full = knn_graph_ao(
        xyz=sb.xyz, neighbors=edges_full,
        normals=normals_full,
        neighbor_opacities=nbr_opa,
        ao_radius=0.03, gamma=0.4,
    )
    shadow_v = shadow_full[idx]; ao_v = ao_full[idx]
    print(f'      shadow mean={shadow_v.mean():.3f}  AO mean={ao_v.mean():.3f}')

    NdotL = np.maximum(0.0, normals_v @ -(lights['key_dir']/np.linalg.norm(lights['key_dir'])))
    if cusp_v is not None:
        kappa_eff = cusp_v.astype(np.float32)
        curv_vis = NdotL * (1.0 - 0.5 * kappa_eff * (1.0 - NdotL))
        curv_vis = np.clip(curv_vis, 0.0, 1.0).astype(np.float32)
    else:
        curv_vis = NdotL.astype(np.float32)

    scales_v_lin = np.exp(sb.scales[idx]).astype(np.float32)
    pix_at_d = cam.focal * scales_v_lin.max(axis=1) / np.maximum(np.abs(depths), 0.05)

    print(f'[6/8] Studio environment + photoreal compose ...')
    t1 = time.perf_counter()
    env = StudioEnvironment(sun_dir=lights['key_dir'], intensity=1.0)
    # albedo IS our ground-truth color for CGI source
    view_color = pbr_full['albedo'][idx].astype(np.float32)
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
        env_ambient_strength=0.30,
        env_reflection_strength=0.55,
        cusp_glint_strength=0.04,
    )
    print(f'      compose {time.perf_counter()-t1:.2f}s   '
          f'shaded HDR range: min={shaded_hdr.min():.3f} max={shaded_hdr.max():.3f}')

    print(f'[7/8] Sort + rasterize at {CAMERA.size}x{CAMERA.size} ...')
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
    print(f'      raster {time.perf_counter()-t2:.2f}s')

    # Bright studio sweep background (light gray vertical gradient)
    H, W = fb.shape[:2]
    yy = np.arange(H, dtype=np.float32) / max(H - 1, 1)
    grad_top = np.array([0.78, 0.80, 0.84], dtype=np.float32)
    grad_bot = np.array([0.30, 0.30, 0.32], dtype=np.float32)
    bg = grad_top[None, None, :] * (1 - yy[:, None, None]) \
       + grad_bot[None, None, :] *  yy[:, None, None]
    bg = np.broadcast_to(bg, (H, W, 3))
    a = np.clip(ab, 0.0, 1.0)[..., None]
    composited = fb + (1.0 - a) * bg

    print(f'[8/8] ACES tonemap + grade + vignette + sRGB ...')
    tone = aces_filmic(composited)
    graded = color_grade(tone, exposure_stops=-0.2, contrast=1.10, saturation=1.06)
    vig = vignette(graded, strength=0.18, falloff=1.6)
    srgb = linear_to_srgb_safe(vig)
    img8 = (np.clip(srgb, 0, 1) * 255).astype(np.uint8)

    OUT_2K.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img8).save(OUT_2K)
    Image.fromarray(img8).resize((1024, 1024), Image.LANCZOS).save(OUT_1K)
    print()
    print('=' * 70)
    print(f'DONE in {time.perf_counter()-t0:.2f}s')
    print(f'  -> {OUT_2K} (2k)')
    print(f'  -> {OUT_1K} (1k)')
    print('=' * 70)


if __name__ == '__main__':
    main()
