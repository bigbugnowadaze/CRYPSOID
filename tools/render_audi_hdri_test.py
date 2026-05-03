"""Smoke test: render Audi using a real HDR file (the synthesized test_smoke_hdr.npy)."""
import sys, time, math, traceback
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
from crypsorender.math.material_decompose import decompose_pbr
from crypsorender.math.mip_splatting_filter import apply_mip_splatting_filter, per_splat_filter_radius
from crypsorender.math.bar2_lighting import apply_bar2_lighting
from crypsorender.math.environment import HDRIEnvironment
from crypsorender.math.photoreal import aces_filmic, color_grade

SRC = ROOT / 'outputs' / 'v40_audi_full_mipfilled.3dphox'
OUT = ROOT / 'renders' / 'crypsorender_v01' / 'SHOWCASE_AUDI_HDRI.png'
HDR = ROOT / 'outputs' / 'test_smoke_hdr.npy'

t0 = time.perf_counter()
sb = load_3dphox(SRC)
aux = load_aux_from_3dphox(SRC)
normals_full = aux['normals'].astype(np.float32)
edges_full   = aux['edges']
cusp_full    = aux.get('cusp_norm')
mip_full     = aux.get('material_mip')

pbr_full = decompose_pbr(sb.sh_dc, sb.sh_rest, sb.opacities)

cam = Camera(sb.xyz, CameraParams(yaw_deg=35, pitch_deg=18, distance=2.4, fov_deg=42, size=2048))
centers_2d, cov_2d_inv, radii, depths, keep, idx = project_splats(sb, cam)

a = cov_2d_inv[:, 0, 0]; b = cov_2d_inv[:, 0, 1]; d = cov_2d_inv[:, 1, 1]
inv_det = 1.0 / np.where(np.abs(a*d - b*b) < 1e-12, 1e-12, a*d - b*b)
cov_2d = np.empty_like(cov_2d_inv, dtype=np.float32)
cov_2d[:, 0, 0] = d * inv_det
cov_2d[:, 0, 1] = -b * inv_det
cov_2d[:, 1, 0] = -b * inv_det
cov_2d[:, 1, 1] = a * inv_det

normals_v = normals_full[idx]
cusp_v    = cusp_full[idx]
mip_v     = mip_full[idx]
xyz_v     = sb.xyz[idx]
opa_v     = sb.opacities[idx].astype(np.float32)
pbr_v     = {k: v[idx] for k, v in pbr_full.items()}

per_radius = per_splat_filter_radius(mip_v, depths, cam.focal, min_px=0.5)
cov_2d_filt, opa_filt, radii_filt = apply_mip_splatting_filter(
    cov_2d, opa_v, radii, min_filter_px=float(np.median(per_radius))
)
a = cov_2d_filt[:, 0, 0]; b = cov_2d_filt[:, 0, 1]; d = cov_2d_filt[:, 1, 1]
inv_det = 1.0 / np.where(np.abs(a*d - b*b) < 1e-12, 1e-12, a*d - b*b)
cov_inv_f = np.empty_like(cov_2d_filt, dtype=np.float32)
cov_inv_f[:, 0, 0] = d * inv_det
cov_inv_f[:, 0, 1] = -b * inv_det
cov_inv_f[:, 1, 0] = -b * inv_det
cov_inv_f[:, 1, 1] = a * inv_det

view_dirs = sb.xyz[idx] - cam.eye[None, :]
view_dirs_norm = view_dirs / (np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-9)
view_color = np.clip(eval_sh_color(sb.sh_dc[idx], sb.sh_rest[idx], view_dirs_norm), 0, 1).astype(np.float32)

sun_lat = math.radians(30); sun_lon = math.radians(60)
sun_dir = -np.array([math.cos(sun_lat) * math.sin(sun_lon),
                     math.sin(sun_lat),
                     -math.cos(sun_lat) * math.cos(sun_lon)], dtype=np.float32)

nbr_opa = sb.opacities[edges_full].astype(np.float32)
shadow = knn_shadow_factor(xyz=sb.xyz, neighbors=edges_full,
                            neighbor_scales=sb.scales[edges_full],
                            neighbor_opacities=nbr_opa,
                            light_dir=sun_dir, strength=0.85)[idx]
ao = knn_graph_ao(xyz=sb.xyz, neighbors=edges_full,
                   normals=normals_full, neighbor_opacities=nbr_opa,
                   ao_radius=0.05, gamma=0.7)[idx]

NdotL = np.maximum(0, normals_v @ -(sun_dir / np.linalg.norm(sun_dir)))
curv_vis = (NdotL * (1 - 0.5 * cusp_v.astype(np.float32) * (1 - NdotL))).astype(np.float32)

scales_v_lin = np.exp(sb.scales[idx]).astype(np.float32)
pix_at_d = cam.focal * scales_v_lin.max(axis=1) / np.maximum(np.abs(depths), 0.05)

env_hdri = HDRIEnvironment(HDR, intensity=1.0)
print(f'env loaded: shape={env_hdri.hdr.shape} max={env_hdri.hdr.max():.2f}')

shaded = apply_bar2_lighting(
    albedo=view_color, metallic=pbr_v['metallic'], roughness=pbr_v['roughness'],
    F0=pbr_v['F0'], kd=pbr_v['kd'], normals=normals_v, xyz=xyz_v,
    eye=cam.eye.astype(np.float32),
    sun_dir=sun_dir, sun_rgb=np.array([1.4, 1.30, 1.10], dtype=np.float32),
    environment=env_hdri,
    shadow_factor=shadow.astype(np.float32),
    ao_factor=ao.astype(np.float32),
    curvature_visibility=curv_vis,
    cusp_norm=cusp_v.astype(np.float32),
    max_pixel_size=pix_at_d.astype(np.float32),
    sun_strength=1.4, env_ambient_strength=0.65,
    env_reflection_strength=1.1, cusp_glint_strength=0.55,
)
print(f'shaded HDR range: min={shaded.min():.3f} max={shaded.max():.3f}')

order = np.argsort(-depths)
fb, ab = rasterize_splats_numba(
    centers_2d[order].astype(np.float32),
    cov_inv_f[order].astype(np.float32),
    radii_filt[order].astype(np.float32),
    opa_filt[order], shaded[order], 2048, 2048
)

H, W = fb.shape[:2]
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
fx = cam.focal; cx = cy = W / 2
ray_x = (xx - cx) / fx; ray_y = -(yy - cy) / fx; ray_z = -np.ones_like(ray_x)
rays_world = np.stack([ray_x, ray_y, ray_z], axis=-1) @ cam.view_rot.T
rays_world = rays_world / (np.linalg.norm(rays_world, axis=-1, keepdims=True) + 1e-9)
bg = env_hdri.sample(rays_world.reshape(-1, 3)).reshape(H, W, 3)

a_msk = np.clip(ab, 0, 1)[..., None]
composited = fb + (1 - a_msk) * bg

tone = aces_filmic(composited)
graded = color_grade(tone, exposure_stops=0.10, contrast=1.05, saturation=1.08)


def to_srgb(x):
    a = 0.055
    return np.where(x <= 0.0031308, 12.92 * x,
                    (1 + a) * np.power(np.maximum(x, 1e-12), 1.0 / 2.4) - a)


img8 = (np.clip(to_srgb(graded), 0, 1) * 255).astype(np.uint8)
Image.fromarray(img8).resize((1024, 1024), Image.LANCZOS).save(OUT)
print(f'wrote {OUT.name} in {time.perf_counter()-t0:.2f}s')
