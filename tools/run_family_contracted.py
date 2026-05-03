"""F.23.3 — Run contracted phoxel on the full Family scene + measure PSNR.

Loads the COLMAP cache from F.22 (101 registered cams + sparse cloud) and
fits BOTH a bounded PhoxelGrid (the prior baseline) and a ContractedPhoxelGrid
on the same cameras, with the same iteration budget. Reports per-camera PSNR
mean and the delta.

This is the F.23 acceptance gate that actually matters — the synthetic Gate 2
in F.23.2 was inconclusive because the bounded baseline could "fog-cheat" the
small synthetic bg. On real Family data the bg has too many sharp edges across
too many camera angles for that to work; we expect contracted to win.

Designed to checkpoint every chunk of iterations so we can run across multiple
short bash calls if needed.

Usage:
    # Run N iters of bounded (continues from outputs/_family_bounded_grid.npz)
    python3 tools/run_family_contracted.py --mode bounded --n-iters 10

    # Run N iters of contracted (continues from outputs/_family_contracted_grid.npz)
    python3 tools/run_family_contracted.py --mode contracted --n-iters 10

    # Just evaluate existing grids
    python3 tools/run_family_contracted.py --mode eval-both
"""
from __future__ import annotations
import sys, time, argparse, pickle, os, json
from pathlib import Path

ROOT = Path('/sessions/ecstatic-sleepy-curie/mnt/Crypsoid')
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np
from PIL import Image

from img2phox.phoxel import (PhoxelGrid, render_image, accumulate_grad,
                                 PhoxelOptimizer)
from img2phox.phoxel_contracted import (ContractedPhoxelGrid,
                                            render_image_contracted,
                                            accumulate_grad_contracted)

CACHE   = ROOT / 'outputs' / '_family152_colmap_cache.pkl'
B_GRID  = ROOT / 'outputs' / '_family_bounded_grid.npz'
C_GRID  = ROOT / 'outputs' / '_family_contracted_grid.npz'
RESULTS = ROOT / 'outputs' / '_family_F23_results.json'


def psnr(a, b, peak=1.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse < 1e-12: return 99.0
    return 10.0 * np.log10(peak * peak / mse)


def load_cache():
    with open(CACHE, 'rb') as f:
        return pickle.load(f)


def init_or_load_bounded(cams, res=64, init_density=0.05):
    if B_GRID.exists():
        g = np.load(B_GRID)
        grid = PhoxelGrid(origin=g['origin'], size=g['size'], res=g['res'],
                            density=g['density'].astype(np.float32),
                            color=g['color'].astype(np.float32))
        print(f'  loaded bounded grid {grid.res[0]}^3, '
              f'occupied={(grid.density > 0.5).sum():,}', flush=True)
        return grid
    cam_pos = np.array([e.cam_position for e in cams.extrinsics])
    sp = cams  # for compat
    lo = (cam_pos.min(axis=0) - 1.0).astype(np.float32)
    hi = (cam_pos.max(axis=0) + 1.0).astype(np.float32)
    grid = PhoxelGrid.from_bounds(lo, hi, res=res, init_density=init_density)
    print(f'  init bounded grid {res}^3 from cameras, '
          f'extent={hi-lo}', flush=True)
    return grid


def init_or_load_contracted(cams, res=96, init_density=0.05):
    if C_GRID.exists():
        g = np.load(C_GRID)
        grid = ContractedPhoxelGrid(world_center=g['world_center'],
                                       world_scale=float(g['world_scale']),
                                       res=g['res'],
                                       density=g['density'].astype(np.float32),
                                       color=g['color'].astype(np.float32))
        print(f'  loaded contracted grid {grid.res[0]}^3, '
              f'occupied={(grid.density > 0.5).sum():,}', flush=True)
        return grid
    cam_pos = np.array([e.cam_position for e in cams.extrinsics])
    grid = ContractedPhoxelGrid.from_cameras(cam_pos, res=res,
                                                  init_density=init_density)
    print(f'  init contracted grid {res}^3, world_scale={grid.world_scale:.3f}',
          flush=True)
    return grid


def save_bounded(grid):
    # np.savez auto-appends .npz, so tmp file ends up at <path>.tmp.npz
    tmp_base = str(B_GRID) + '.tmp'
    np.savez(tmp_base, origin=grid.origin, size=grid.size, res=grid.res,
              density=grid.density, color=grid.color)
    os.replace(tmp_base + '.npz', B_GRID)


def save_contracted(grid):
    tmp_base = str(C_GRID) + '.tmp'
    np.savez(tmp_base, world_center=grid.world_center,
              world_scale=np.float32(grid.world_scale),
              res=grid.res, density=grid.density, color=grid.color)
    os.replace(tmp_base + '.npz', C_GRID)


def fit_bounded(photoset, cams, n_iters, train_scale=0.3, n_samples=48,
                lr_density=2.0, lr_color=0.3):
    grid = init_or_load_bounded(cams, res=64)
    opt = PhoxelOptimizer(lr_density=lr_density, lr_color=lr_color)
    Ht = max(8, int(photoset.photos[0].image.shape[0] * train_scale))
    Wt = max(8, int(photoset.photos[0].image.shape[1] * train_scale))
    n_cams = len(cams)
    print(f'  train @ {Ht}x{Wt}, {n_cams} cams', flush=True)

    # Pre-resize GTs once
    gts = []
    for ci in range(n_cams):
        gt = np.asarray(Image.fromarray((photoset.photos[ci].image * 255)
                          .clip(0, 255).astype(np.uint8))
                          .resize((Wt, Ht), Image.LANCZOS),
                          dtype=np.float32) / 255.0
        gts.append(gt)
    t0 = time.perf_counter()
    for it in range(n_iters):
        gd = np.zeros_like(grid.density); gc = np.zeros_like(grid.color)
        loss = 0.0; n_rays = 0
        for ci in range(n_cams):
            r = render_image(grid, cams.intrinsics, cams.extrinsics[ci],
                              H=Ht, W=Wt, n_samples=n_samples)
            loss += float(np.abs(r - gts[ci]).mean())
            accumulate_grad(grid, cams.intrinsics, cams.extrinsics[ci], gts[ci], r,
                              n_samples=n_samples, grad_density=gd, grad_color=gc)
            n_rays += Ht * Wt
        loss /= n_cams
        opt.step(grid, gd, gc, n_rays_seen=n_rays)
        print(f'   bounded iter {it:3d}  L1={loss:.4f}  '
              f'occ={(grid.density > 0.5).sum():,}  '
              f'elapsed={time.perf_counter()-t0:.1f}s', flush=True)
    save_bounded(grid)
    return grid


def fit_contracted(photoset, cams, n_iters, train_scale=0.3, n_samples=64,
                    lr_density=2.0, lr_color=0.3, t_far_norm=30.0):
    grid = init_or_load_contracted(cams, res=96)
    opt = PhoxelOptimizer(lr_density=lr_density, lr_color=lr_color)
    Ht = max(8, int(photoset.photos[0].image.shape[0] * train_scale))
    Wt = max(8, int(photoset.photos[0].image.shape[1] * train_scale))
    n_cams = len(cams)
    print(f'  train @ {Ht}x{Wt}, {n_cams} cams, t_far_norm={t_far_norm}', flush=True)

    gts = []
    for ci in range(n_cams):
        gt = np.asarray(Image.fromarray((photoset.photos[ci].image * 255)
                          .clip(0, 255).astype(np.uint8))
                          .resize((Wt, Ht), Image.LANCZOS),
                          dtype=np.float32) / 255.0
        gts.append(gt)
    t0 = time.perf_counter()
    for it in range(n_iters):
        gd = np.zeros_like(grid.density); gc = np.zeros_like(grid.color)
        loss = 0.0; n_rays = 0
        for ci in range(n_cams):
            r = render_image_contracted(grid, cams.intrinsics, cams.extrinsics[ci],
                                          H=Ht, W=Wt, n_samples=n_samples,
                                          t_far_norm=t_far_norm)
            loss += float(np.abs(r - gts[ci]).mean())
            accumulate_grad_contracted(grid, cams.intrinsics, cams.extrinsics[ci],
                                          gts[ci], r, n_samples=n_samples,
                                          t_far_norm=t_far_norm,
                                          grad_density=gd, grad_color=gc)
            n_rays += Ht * Wt
        loss /= n_cams
        opt.step(grid, gd, gc, n_rays_seen=n_rays)
        print(f'   contracted iter {it:3d}  L1={loss:.4f}  '
              f'occ={(grid.density > 0.5).sum():,}  '
              f'elapsed={time.perf_counter()-t0:.1f}s', flush=True)
    save_contracted(grid)
    return grid


def eval_bounded(photoset, cams, eval_scale=0.5, n_samples=64):
    if not B_GRID.exists():
        print('  no bounded grid yet'); return None
    g = np.load(B_GRID)
    grid = PhoxelGrid(origin=g['origin'], size=g['size'], res=g['res'],
                        density=g['density'], color=g['color'])
    H0, W0 = photoset.photos[0].image.shape[:2]
    H = max(16, int(H0 * eval_scale)); W = max(16, int(W0 * eval_scale))
    psnrs = []
    for ci in range(len(cams)):
        gt = np.asarray(Image.fromarray((photoset.photos[ci].image * 255)
                          .clip(0, 255).astype(np.uint8))
                          .resize((W, H), Image.LANCZOS),
                          dtype=np.float32) / 255.0
        r = render_image(grid, cams.intrinsics, cams.extrinsics[ci],
                          H=H, W=W, n_samples=n_samples)
        psnrs.append(psnr(r, gt))
    return float(np.mean(psnrs)), psnrs


def eval_contracted(photoset, cams, eval_scale=0.5, n_samples=96, t_far_norm=30.0):
    if not C_GRID.exists():
        print('  no contracted grid yet'); return None
    g = np.load(C_GRID)
    grid = ContractedPhoxelGrid(world_center=g['world_center'],
                                   world_scale=float(g['world_scale']),
                                   res=g['res'], density=g['density'],
                                   color=g['color'])
    H0, W0 = photoset.photos[0].image.shape[:2]
    H = max(16, int(H0 * eval_scale)); W = max(16, int(W0 * eval_scale))
    psnrs = []
    for ci in range(len(cams)):
        gt = np.asarray(Image.fromarray((photoset.photos[ci].image * 255)
                          .clip(0, 255).astype(np.uint8))
                          .resize((W, H), Image.LANCZOS),
                          dtype=np.float32) / 255.0
        r = render_image_contracted(grid, cams.intrinsics, cams.extrinsics[ci],
                                      H=H, W=W, n_samples=n_samples,
                                      t_far_norm=t_far_norm)
        psnrs.append(psnr(r, gt))
    return float(np.mean(psnrs)), psnrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['bounded', 'contracted', 'eval-both'],
                     required=True)
    ap.add_argument('--n-iters', type=int, default=10)
    ap.add_argument('--train-scale', type=float, default=0.3)
    ap.add_argument('--n-cams', type=int, default=None,
                     help='Subsample cams (for quick smoke runs)')
    args = ap.parse_args()

    print(f'F.23.3 — Family contracted vs bounded, mode={args.mode}', flush=True)
    print(f'Loading {CACHE.name} ...', flush=True)
    d = load_cache()
    photoset = d['photoset']; cams = d['rec_cams']
    print(f'  {len(cams)} cams, sparse pts={len(d["sparse"].xyz):,}', flush=True)

    # Optional cam subsampling for quick smoke runs
    if args.n_cams is not None and args.n_cams < len(cams):
        idx = np.linspace(0, len(cams)-1, args.n_cams).astype(int)
        from img2phox.data_classes import PhotoSet, CameraBundle
        photoset = PhotoSet(photos=[photoset.photos[i] for i in idx])
        cams     = CameraBundle(intrinsics=cams.intrinsics,
                                  extrinsics=[cams.extrinsics[i] for i in idx])
        print(f'  subsampled to {len(cams)} cams', flush=True)

    if args.mode == 'bounded':
        fit_bounded(photoset, cams, n_iters=args.n_iters,
                     train_scale=args.train_scale)
    elif args.mode == 'contracted':
        fit_contracted(photoset, cams, n_iters=args.n_iters,
                        train_scale=args.train_scale)
    elif args.mode == 'eval-both':
        rb = eval_bounded(photoset, cams)
        rc = eval_contracted(photoset, cams)
        out = {}
        if rb is not None:
            psnr_b, all_b = rb
            print(f'  bounded    PSNR mean = {psnr_b:.2f} dB  '
                  f'(min {min(all_b):.2f}, max {max(all_b):.2f})', flush=True)
            out['bounded_mean'] = psnr_b
            out['bounded_per_cam'] = all_b
        if rc is not None:
            psnr_c, all_c = rc
            print(f'  contracted PSNR mean = {psnr_c:.2f} dB  '
                  f'(min {min(all_c):.2f}, max {max(all_c):.2f})', flush=True)
            out['contracted_mean'] = psnr_c
            out['contracted_per_cam'] = all_c
        if rb is not None and rc is not None:
            out['delta'] = psnr_c - psnr_b
            print(f'  delta = {out["delta"]:+.2f} dB', flush=True)
        with open(RESULTS, 'w') as fh:
            json.dump(out, fh, indent=2)
        print(f'  results -> {RESULTS}', flush=True)


if __name__ == '__main__':
    main()
