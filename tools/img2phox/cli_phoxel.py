"""F.12 driver — Phoxel: CPU Plenoxel-style voxel reconstruction → .3dphox.

The breakout pitch:
  - Replaces the F.10/F.11 splat-from-scratch optimizer (which hit a position-
    frozen ceiling at ~5 dB on Family) with a voxel grid.
  - The voxel grid IS the parameter — there is nothing to "freeze". Density
    and color flow with the photometric residual.
  - At the end we extract phoxoidal blobs from occupied cells, so output is
    still .3dphox compatible.
  - Numba JIT in forward + backward keeps it CPU-only.

Usage:
  python3 -m img2phox.cli_phoxel \\
      --photos inputs/Family --max-photos 8 --max-dim 480 \\
      --grid-res 64 --opt-iters 200 \\
      --out outputs/family_phoxel.3dphox
"""
from __future__ import annotations
import argparse, time
from pathlib import Path

import numpy as np
from PIL import Image

from .load_photos import load_photoset
from .preprocess import preprocess_photoset
from .sfm_global import run_sfm_global
from .phoxel import (PhoxelGrid, fit_phoxel_grid, render_image,
                      extract_blobs_from_grid)
from .encode import encode_blobbundle_to_3dphox


def psnr(a, b, peak=1.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return 10 * np.log10(peak * peak / max(mse, 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--photos',      type=Path, required=True)
    p.add_argument('--max-photos',  type=int,  default=8)
    p.add_argument('--max-dim',     type=int,  default=480)
    p.add_argument('--n-features',  type=int,  default=8000)
    p.add_argument('--ratio',       type=float, default=0.85)
    p.add_argument('--min-matches', type=int,  default=20)
    p.add_argument('--ba-nfev',     type=int,  default=200)
    p.add_argument('--grid-res',    type=int,  default=64)
    p.add_argument('--opt-iters',   type=int,  default=200)
    p.add_argument('--n-samples',   type=int,  default=64,
                    help='Number of samples per ray (volume integration).')
    p.add_argument('--train-scale', type=float, default=0.5,
                    help='Render resolution scale during optimization.')
    p.add_argument('--scene-pad',   type=float, default=1.5,
                    help='AABB padding factor — grid extends scene_pad x ' \
                         'beyond camera bounding box.')
    p.add_argument('--density-thresh', type=float, default=0.5)
    p.add_argument('--max-blobs',   type=int, default=200_000)
    p.add_argument('--out',         type=Path, default=Path('/sessions/ecstatic-sleepy-curie/mnt/Crypsoid/outputs/img2phox_phoxel.3dphox'))
    p.add_argument('--out-render',  type=Path, default=Path('/sessions/ecstatic-sleepy-curie/mnt/Crypsoid/renders/crypsorender_v01/SHOWCASE_PHOXEL_v01.png'))
    args = p.parse_args()

    print("=" * 70)
    print("  Phase F.12 — Phoxel: CPU-only voxel-grid → .3dphox")
    print("=" * 70)

    t0 = time.perf_counter()

    # ---------- Load + preprocess ----------
    print(f"\n[load] photos from {args.photos}, max-dim={args.max_dim} ...")
    photoset = load_photoset(args.photos, max_dim=args.max_dim)
    print(f"  loaded {len(photoset)} photos at {photoset[0].image.shape}")
    if len(photoset) > args.max_photos:
        idxs = np.linspace(0, len(photoset) - 1, args.max_photos).astype(int)
        from .data_classes import PhotoSet
        photoset = PhotoSet(photos=[photoset.photos[i] for i in idxs])
        print(f"  subsampled to {len(photoset)} evenly-spaced views")

    print(f"\n[preprocess] EXIF intrinsics + camera-DB distortion ...")
    photoset, intr = preprocess_photoset(photoset, fov_deg_fallback=50.0,
                                           exposure_method=None, verbose=True)

    # ---------- Global SfM ----------
    print(f"\n[sfm-global] rotation + LUD translation averaging ...")
    rec_cams, sparse_cloud, sfm_stats = run_sfm_global(
        photoset, intr=intr,
        n_features=args.n_features, ratio=args.ratio, min_matches=args.min_matches,
        run_global_ba=True, ba_max_nfev=args.ba_nfev, verbose=True,
    )
    n_reg = sfm_stats['n_cameras_registered']
    print(f"  -> {n_reg}/{len(photoset)} cams registered, "
          f"{len(sparse_cloud)} sparse 3D points")
    if n_reg < 3:
        print(f"FAIL: only {n_reg} cameras registered — pipeline aborts here.")
        return

    # ---------- Scene bounds from camera positions + sparse cloud ----------
    cam_positions = np.array([e.cam_position for e in rec_cams.extrinsics])
    if len(sparse_cloud) > 0:
        all_pts = np.concatenate([cam_positions, sparse_cloud.xyz], axis=0)
    else:
        all_pts = cam_positions
    center = all_pts.mean(axis=0)
    radius = np.linalg.norm(all_pts - center, axis=1).max()
    extent = radius * args.scene_pad
    scene_lo = (center - extent).astype(np.float32)
    scene_hi = (center + extent).astype(np.float32)
    print(f"\n[bounds] center={center}, radius={radius:.3f}, "
          f"AABB extent={(2*extent):.3f}")
    print(f"  grid={args.grid_res}^3 = {args.grid_res**3:,} cells, "
          f"cell_size={(2*extent / args.grid_res):.4f}")

    # ---------- Voxel optimization ----------
    print(f"\n[phoxel] fit voxel grid for {args.opt_iters} iters at "
          f"train scale {args.train_scale:.2f}...")
    grid, history = fit_phoxel_grid(
        photoset, rec_cams,
        scene_lo=scene_lo, scene_hi=scene_hi,
        resolution=args.grid_res,
        n_iters=args.opt_iters,
        n_samples_per_ray=args.n_samples,
        train_resolution_scale=args.train_scale,
        verbose=True,
    )

    # ---------- Extract blobs + encode ----------
    print(f"\n[extract] cells with density > {args.density_thresh} -> blobs ...")
    blobs = extract_blobs_from_grid(grid,
                                       density_threshold=args.density_thresh,
                                       max_blobs=args.max_blobs)
    print(f"  -> {len(blobs):,} blobs extracted")

    if len(blobs) > 0:
        print(f"\n[encode] -> {args.out} ...")
        sz = encode_blobbundle_to_3dphox(blobs, args.out)
        print(f"  wrote {sz:,} bytes ({len(blobs):,} blobs)")
    else:
        print(f"  WARNING: no blobs above density threshold; skipping encode")
        sz = 0

    # ---------- Eval at full resolution ----------
    print(f"\n[eval] render cam 0 at full resolution ...")
    gt = photoset.photos[0].image
    H, W = gt.shape[:2]
    rendered = render_image(grid, rec_cams.intrinsics, rec_cams.extrinsics[0],
                              H=H, W=W, n_samples=args.n_samples)
    eval_psnr = psnr(rendered, gt)
    print(f"  PSNR (cam 0)  = {eval_psnr:.2f} dB")
    print(f"  loss start    = {history[0]:.4f}")
    print(f"  loss final    = {history[-1]:.4f}")

    # Comparison sheet
    panels = [
        ('GT cam 0',                gt),
        ('Phoxel render',           rendered),
        ('|GT - render| x 4',       np.abs(rendered - gt) * 4),
    ]
    args.out_render.parent.mkdir(parents=True, exist_ok=True)
    sep = 8
    sheet_h = H + 60
    sheet_w = len(panels) * W + (len(panels) - 1) * sep
    sheet = np.full((sheet_h, sheet_w, 3), 0.07, dtype=np.float32)
    cursor = 0
    for label, arr in panels:
        sheet[50:50+H, cursor:cursor+W] = arr.clip(0, 1)
        cursor += W + sep
    Image.fromarray((sheet * 255).astype(np.uint8)).save(args.out_render)
    print(f"  comparison -> {args.out_render}")

    print()
    print("=" * 70)
    print(f"  Phase F.12 Phoxel summary:")
    print(f"    photos used                   : {len(photoset)}")
    print(f"    cams registered               : {n_reg}/{len(photoset)}")
    print(f"    grid resolution               : {args.grid_res}^3")
    print(f"    final blobs                   : {len(blobs):,}")
    print(f"    train-view PSNR (cam 0)       : {eval_psnr:.2f} dB")
    if sz:
        print(f"    output .3dphox                : {sz:,} bytes")
    print(f"    total wall                    : {time.perf_counter()-t0:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
