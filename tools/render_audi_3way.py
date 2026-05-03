"""Render Bug's 3-way Audi comparison.

  1. inputs/audi/Audi A5 Sportback.zip   -> SHOWCASE_AUDI_PLY_color.png
  2. outputs/v40_audi_full_mipfilled.3dphox -> SHOWCASE_AUDI_PHOX_color.png
  3. (already on disk) SHOWCASE_AUDI_MAX.png is the lit version

All renders share the same camera and resolution. 2x supersample then Lanczos.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path('/sessions/ecstatic-sleepy-curie/mnt/Crypsoid')
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np
from PIL import Image

from crypsorender.io.phox_loader import load_3dphox
from crypsorender.io.ply_loader import load_ply
from crypsorender.io.splat_buffer import SplatBuffer
from crypsorender.pipeline.camera import Camera, CameraParams
from crypsorender.pipeline.project import project_splats
from crypsorender.pipeline.rasterize_numba import rasterize_splats_numba
from crypsorender.math.sh import eval_sh_color


CAMERA = CameraParams(yaw_deg=35, pitch_deg=18, distance=2.4, fov_deg=42, size=2048)
OUT_SIZE = 1024


def render_full_color(scene, label, supersample_size=2048):
    t0 = time.perf_counter()
    cam_params = CameraParams(yaw_deg=CAMERA.yaw_deg, pitch_deg=CAMERA.pitch_deg,
                              distance=CAMERA.distance, fov_deg=CAMERA.fov_deg,
                              size=supersample_size)
    cam = Camera(scene.xyz, cam_params)
    print(f"[{label}] camera built. eye={cam.eye}, focal={cam.focal:.0f}", flush=True)

    print(f"[{label}] projecting {scene.n:,} splats ...", flush=True)
    t1 = time.perf_counter()
    centers_2d, cov_2d_inv, radii, depths, keep, idx = project_splats(scene, cam)
    print(f"[{label}]   {len(centers_2d):,} splats survive ({(time.perf_counter()-t1):.2f}s)", flush=True)

    print(f"[{label}] decoding SH color (view-dependent) ...", flush=True)
    view_dirs = scene.xyz[idx] - cam.eye[None, :]
    view_dirs = view_dirs / (np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-9)
    if scene.sh_rest is not None:
        color = eval_sh_color(scene.sh_dc[idx], scene.sh_rest[idx], view_dirs)
    else:
        color = scene.sh_dc[idx] * 0.28209479177387814 + 0.5
    color = np.clip(color, 0.0, 1.0).astype(np.float32)
    opa = scene.opacities[idx].astype(np.float32)

    print(f"[{label}] sorting back-to-front ...", flush=True)
    order = np.argsort(-depths)
    centers_2d = centers_2d[order]
    cov_2d_inv = cov_2d_inv[order]
    radii = radii[order]
    color = color[order]
    opa = opa[order]

    print(f"[{label}] rasterizing at {supersample_size}x{supersample_size} ...", flush=True)
    t2 = time.perf_counter()
    fb, ab = rasterize_splats_numba(
        centers_2d.astype(np.float32),
        cov_2d_inv.astype(np.float32),
        radii.astype(np.float32),
        opa, color, supersample_size, supersample_size,
    )
    print(f"[{label}]   raster {time.perf_counter()-t2:.2f}s", flush=True)

    fb = np.clip(fb, 0.0, 1.0)
    img = (fb * 255).astype(np.uint8)
    pil = Image.fromarray(img).resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
    print(f"[{label}] total wall = {time.perf_counter()-t0:.2f}s", flush=True)
    return np.array(pil)


def main():
    out_dir = ROOT / 'renders' / 'crypsorender_v01'
    out_dir.mkdir(parents=True, exist_ok=True)
    target = sys.argv[1] if len(sys.argv) > 1 else 'both'

    if target in ('ply', 'both'):
        print("=" * 70)
        print("  PANEL 1 - Original Audi PLY (ground truth source)")
        print("=" * 70)
        ply_path = ROOT / 'inputs' / 'audi' / 'Audi A5 Sportback.zip'
        scene = load_ply(ply_path)
        print(f"  loaded {scene.n:,} splats from {ply_path.name} "
              f"({ply_path.stat().st_size/1024/1024:.1f} MB)")
        img1 = render_full_color(scene, "PLY")
        out1 = out_dir / 'SHOWCASE_AUDI_PLY_color.png'
        Image.fromarray(img1).save(out1)
        print(f"  saved {out1.name}")

    if target in ('phox', 'both'):
        print()
        print("=" * 70)
        print("  PANEL 2 - CRYPSOID v40 .3dphox (compressed)")
        print("=" * 70)
        scene = load_3dphox(ROOT / 'outputs' / 'v40_audi_full_mipfilled.3dphox')
        print(f"  loaded {scene.n:,} splats from v40_audi_full_mipfilled.3dphox "
              f"({(ROOT / 'outputs/v40_audi_full_mipfilled.3dphox').stat().st_size/1024/1024:.1f} MB)")
        img2 = render_full_color(scene, "PHOX")
        out2 = out_dir / 'SHOWCASE_AUDI_PHOX_color.png'
        Image.fromarray(img2).save(out2)
        print(f"  saved {out2.name}")


if __name__ == '__main__':
    main()
