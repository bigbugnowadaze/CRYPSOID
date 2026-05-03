"""F.10 driver — full polished real-photo pipeline.

Uses every polish-round module:
  - F.5 ORB SfM + F.5+ sparse-Jacobian BA
  - F.6 SGBM dense MVS + voxel fusion
  - F.7 + F.7+ EXIF intrinsics + camera-DB distortion + exposure normalize
  - F.8 + F.8+ JIT'd dense optimizer
  - F.8++ 3DGS-paper density schedule

Intended for real photo folders (e.g. Tanks & Temples Family).

Usage:
  python3 -m img2phox.cli_v10 \
      --photos inputs/Family --max-photos 30 \
      --out outputs/family_v10.3dphox \
      --opt-iters 2000
"""
from __future__ import annotations
import argparse, time
from pathlib import Path

import numpy as np
from PIL import Image

from .load_photos import load_photoset
from .preprocess import preprocess_photoset
from .sfm_real import run_sfm_real
from .sfm_global import run_sfm_global
from .mvs import run_dense_mvs
from .optimize import quick_seed_from_pointcloud
from .optimize_jit import render_blobs_to_photo_jit, aggregate_signal_jit
from .density_control import density_step, DensityScheduleConfig, DensityScheduleState
from .encode import encode_blobbundle_to_3dphox


def psnr(a, b, peak=1.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return 10 * np.log10(peak * peak / max(mse, 1e-12))


def jit_optimize_loop(blobs, photoset, cameras,
                       n_iters=1000, lr_color=0.05, lr_opacity=0.04,
                       densify_from=200, densify_until=8000, densify_every=100,
                       prune_every=100, prune_threshold=0.005,
                       max_blobs=200_000, scene_radius=1.0,
                       verbose=True):
    """JIT'd optimization loop with adaptive density control."""
    from .data_classes import BlobBundle
    cur = BlobBundle(
        xyz=blobs.xyz.copy(), scales=blobs.scales.copy(), quats=blobs.quats.copy(),
        opacity=blobs.opacity.copy(), sh_dc=blobs.sh_dc.copy(), sh_rest=None,
        tier=blobs.tier.copy() if blobs.tier is not None else None,
    )
    cfg = DensityScheduleConfig(
        densify_from_iter=densify_from,
        densify_until_iter=densify_until,
        densify_interval=densify_every,
        prune_interval=prune_every,
        prune_opacity_threshold=prune_threshold,
        max_blobs=max_blobs,
    )
    state = DensityScheduleState()
    history = []
    t0 = time.perf_counter()
    for it in range(n_iters):
        # Render every camera + sum L1 loss
        residuals = []
        loss = 0.0
        for ci in range(len(cameras)):
            rendered = render_blobs_to_photo_jit(cur, cameras, ci)
            r = photoset.photos[ci].image - rendered
            loss += float(np.abs(r).mean())
            residuals.append(r)
        history.append(loss / len(cameras))

        signal = aggregate_signal_jit(cur, cameras, residuals)
        seen = signal['coverage_count'] > 0
        if seen.any():
            cnt = signal['coverage_count'][seen][:, None].astype(np.float32)
            cur.sh_dc[seen]   += lr_color   * signal['color_correction'][seen] / cnt
            cur.opacity[seen] += lr_opacity * signal['opacity_gradient'][seen] / cnt[:, 0]
        cur.sh_dc = cur.sh_dc.clip(0, 1)
        cur.opacity = cur.opacity.clip(0.0, 1.0)

        cur, state = density_step(cur, state, signal, scene_radius, it, cfg)

        if verbose and (it < 5 or it == n_iters - 1 or it % max(1, n_iters // 20) == 0):
            print(f"  iter {it:5d}  loss={history[-1]:.4f}  blobs={len(cur):,}  "
                  f"elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    if verbose:
        print(f"  optimize done in {time.perf_counter()-t0:.1f}s "
              f"(L1: {history[0]:.4f} -> {history[-1]:.4f}, blobs {len(blobs)} -> {len(cur)})")
    return cur, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--photos',      type=Path, required=True)
    p.add_argument('--max-photos',  type=int,  default=30,
                    help='If photoset > this, evenly subsample to this many.')
    p.add_argument('--max-dim',     type=int,  default=960,
                    help='Downscale photos so longer side <= this many pixels.')
    p.add_argument('--n-features',  type=int,  default=8000)
    p.add_argument('--ratio',       type=float, default=0.85)
    p.add_argument('--min-matches', type=int,  default=20)
    p.add_argument('--ba-nfev',     type=int,  default=300)
    p.add_argument('--sfm-mode',    type=str,  default='global', choices=['incremental', 'global'])
    p.add_argument('--mvs-pairs',   type=int,  default=8)
    p.add_argument('--voxel-size',  type=float, default=0.02)
    p.add_argument('--opt-iters',   type=int,  default=500)
    p.add_argument('--max-blobs',   type=int,  default=200_000)
    p.add_argument('--holdout',     type=int,  default=2,
                    help='Hold out N cameras as test views; rest used for train.')
    p.add_argument('--out',         type=Path, default=Path('/sessions/ecstatic-sleepy-curie/mnt/Crypsoid/outputs/img2phox_v10.3dphox'))
    p.add_argument('--out-render',  type=Path, default=Path('/sessions/ecstatic-sleepy-curie/mnt/Crypsoid/renders/crypsorender_v01/SHOWCASE_IMG2PHOX_v10.png'))
    args = p.parse_args()

    print("=" * 70)
    print("  Phase F.10 — full polished real-photo pipeline")
    print("=" * 70)

    t0 = time.perf_counter()
    print(f"\n[load] photos from {args.photos}, max-dim={args.max_dim} ...")
    photoset = load_photoset(args.photos, max_dim=args.max_dim)
    print(f"  loaded {len(photoset)} photos at {photoset[0].image.shape}")

    if len(photoset) > args.max_photos:
        idxs = np.linspace(0, len(photoset) - 1, args.max_photos).astype(int)
        from .data_classes import PhotoSet
        photoset = PhotoSet(photos=[photoset.photos[i] for i in idxs])
        print(f"  subsampled to {len(photoset)} evenly-spaced views")

    # Hold out test views
    if args.holdout > 0 and len(photoset) > args.holdout + 4:
        ho_step = len(photoset) // (args.holdout + 1)
        ho_idx = [(i + 1) * ho_step for i in range(args.holdout)]
        from .data_classes import PhotoSet
        train_idx = [i for i in range(len(photoset)) if i not in ho_idx]
        train = PhotoSet(photos=[photoset.photos[i] for i in train_idx])
        test  = PhotoSet(photos=[photoset.photos[i] for i in ho_idx])
        print(f"  holdout split: {len(train)} train + {len(test)} test")
    else:
        train = photoset; test = None

    print(f"\n[F.7+] preprocess: EXIF intrinsics + camera-DB distortion (exposure normalize OFF: it destroys ORB matches) ...")
    train, intr = preprocess_photoset(train, fov_deg_fallback=50.0,
                                        exposure_method=None, verbose=True)
    print(f"  intrinsics: focal={intr.focal_x:.1f}px, principal=({intr.cx:.1f}, {intr.cy:.1f})")

    if args.sfm_mode == 'global':
        print(f"\n[F.11] GLOBAL SfM (rotation + translation averaging) ...")
        rec_cams, sparse_cloud, sfm_stats = run_sfm_global(
            train, intr=intr,
            n_features=args.n_features, ratio=args.ratio, min_matches=args.min_matches,
            run_global_ba=True, ba_max_nfev=args.ba_nfev, verbose=True,
        )
    else:
        print(f"\n[F.5 + F.5+] INCREMENTAL SfM with global sparse BA ...")
        rec_cams, sparse_cloud, sfm_stats = run_sfm_real(
            train, intr=intr,
            n_features=args.n_features, ratio=args.ratio, min_matches=args.min_matches,
            run_global_ba=True, ba_max_nfev=args.ba_nfev, verbose=True,
        )
    n_reg = sfm_stats['n_cameras_registered']
    print(f"  -> {n_reg}/{len(train)} cams registered, {len(sparse_cloud)} sparse 3D points")
    if n_reg < 3:
        print(f"FAIL: only {n_reg} cameras registered — pipeline aborts here.")
        return

    print(f"\n[F.6] dense MVS via SGBM ({args.mvs_pairs} pairs) ...")
    dense_cloud = run_dense_mvs(train, rec_cams, sparse_cloud,
                                  max_pairs=args.mvs_pairs,
                                  voxel_size=args.voxel_size, verbose=True)
    print(f"  -> {len(dense_cloud):,} fused dense points")

    print(f"\n[seed] BlobBundle from dense cloud ...")
    blobs = quick_seed_from_pointcloud(dense_cloud, opacity=0.5, scale_kappa=0.5, k_neighbors=6)
    scene_radius = float(np.linalg.norm(blobs.xyz - blobs.xyz.mean(axis=0), axis=1).mean())
    print(f"  -> {len(blobs):,} seed blobs, scene_radius={scene_radius:.3f}")

    print(f"\n[F.8 + F.8+ + F.8++] JIT'd dense optimizer with 3DGS density schedule ...")
    blobs_opt, history = jit_optimize_loop(
        blobs, train, rec_cams,
        n_iters=args.opt_iters,
        densify_from=max(50, args.opt_iters // 20),
        densify_until=int(args.opt_iters * 0.85),
        densify_every=max(20, args.opt_iters // 30),
        prune_every=max(20, args.opt_iters // 30),
        max_blobs=args.max_blobs,
        scene_radius=scene_radius,
        verbose=True,
    )

    print(f"\n[encode] -> {args.out} ...")
    sz = encode_blobbundle_to_3dphox(blobs_opt, args.out)
    print(f"  wrote {sz:,} bytes ({len(blobs_opt):,} blobs)")

    # Build comparison panel using the FIRST training cam
    train_render = render_blobs_to_photo_jit(blobs_opt, rec_cams, 0)
    gt_train = train.photos[0].image
    panels = [
        ('GT train view',          gt_train),
        ('Reconstructed render',   train_render),
        ('|GT - render| x 4',      np.abs(train_render - gt_train) * 4),
    ]

    # If we have held-out test views, evaluate PSNR on them
    test_psnrs = []
    if test is not None:
        # We need camera poses for the test views. Best-effort: use the closest
        # registered training camera's pose as a proxy. Real eval would need to
        # localize the test views via PnP against the recovered cloud, which is
        # F.10.3-grade work; for F.10.2 we just report training-view PSNR.
        print(f"\n[note] held-out test-view PSNR requires localizing test cams via "
              f"PnP — deferred to F.10.3. Reporting train-view PSNR only.")

    train_psnr = psnr(train_render, gt_train)

    args.out_render.parent.mkdir(parents=True, exist_ok=True)
    H, W = panels[0][1].shape[:2]; sep = 8
    sheet = np.full((H + 60, len(panels) * W + (len(panels) - 1) * sep, 3), 0.07,
                     dtype=np.float32)
    cursor = 0
    for label, arr in panels:
        sheet[50:50+H, cursor:cursor+W] = arr.clip(0, 1)
        cursor += W + sep
    Image.fromarray((sheet * 255).astype(np.uint8)).save(args.out_render)

    print()
    print("=" * 70)
    print(f"  Phase F.10 Family pipeline summary:")
    print(f"    photos used                   : {len(train)}/{args.max_photos}")
    print(f"    cams registered               : {n_reg}/{len(train)}")
    print(f"    sparse 3D points              : {len(sparse_cloud):,}")
    print(f"    dense MVS points (fused)      : {len(dense_cloud):,}")
    print(f"    final blobs                   : {len(blobs_opt):,}")
    print(f"    train-view PSNR (cam 0)       : {train_psnr:.2f} dB")
    print(f"    output .3dphox                : {sz:,} bytes")
    print(f"    total wall                    : {time.perf_counter()-t0:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
