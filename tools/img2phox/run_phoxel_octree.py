"""F.12.3 driver — run octree phoxel optimization on cached Family SfM.

Two phases:
  1. Coarse warm-up at 32^3
  2. Subdivide occupied cells (4x), refine

Designed to run in chunks (so each bash call fits the 40s budget) by saving
the octree state to disk between calls.
"""
from __future__ import annotations
import sys, os, pickle, time, argparse
sys.path.insert(0, 'tools')
import numpy as np
from PIL import Image
from img2phox.phoxel_octree import (PhoxelOctree, render_image_oct,
                                       accumulate_grad_oct, OctreeOptimizer)
import img2phox.phoxel_octree as oct_mod


def save_oct(oct_grid: PhoxelOctree, path: str):
    np.savez(path,
              coarse_density=oct_grid.coarse_density,
              coarse_color=oct_grid.coarse_color,
              subdiv=oct_grid.subdiv,
              fine_density=oct_grid.fine_density,
              fine_color=oct_grid.fine_color,
              origin=oct_grid.origin, size=oct_grid.size, res=oct_grid.res,
              fine_factor=np.int32(oct_grid.fine_factor))


def load_oct(path: str) -> PhoxelOctree:
    d = np.load(path)
    return PhoxelOctree(
        origin=d['origin'], size=d['size'], res=d['res'],
        fine_factor=int(d['fine_factor']),
        coarse_density=d['coarse_density'].astype(np.float32),
        coarse_color=d['coarse_color'].astype(np.float32),
        subdiv=d['subdiv'].astype(np.int32),
        fine_density=d['fine_density'].astype(np.float32),
        fine_color=d['fine_color'].astype(np.float32),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['init', 'iter', 'subdivide', 'eval'], required=True)
    p.add_argument('--n-iters', type=int, default=20)
    p.add_argument('--coarse-res', type=int, default=32)
    p.add_argument('--fine-factor', type=int, default=4)
    p.add_argument('--train-scale', type=float, default=0.5)
    p.add_argument('--n-samples', type=int, default=40)
    p.add_argument('--lr-density', type=float, default=2.0)
    p.add_argument('--lr-color', type=float, default=0.3)
    p.add_argument('--subdivide-threshold', type=float, default=0.3)
    p.add_argument('--oct-path', type=str, default='outputs/_phoxel_oct_v01.npz')
    p.add_argument('--sfm-cache', type=str, default='outputs/_phoxel_sfm_cache.pkl')
    args = p.parse_args()

    with open(args.sfm_cache, 'rb') as f: d = pickle.load(f)
    photoset = d['photoset']; rec_cams = d['rec_cams']; sparse = d['sparse']

    if args.mode == 'init':
        cam_pos = np.array([e.cam_position for e in rec_cams.extrinsics])
        all_pts = np.concatenate([cam_pos, sparse.xyz], axis=0)
        center = all_pts.mean(axis=0); radius = np.linalg.norm(all_pts-center, axis=1).max()
        extent = radius * 1.2
        lo = (center - extent).astype(np.float32); hi = (center + extent).astype(np.float32)
        oct = PhoxelOctree.from_bounds(lo, hi, coarse_res=args.coarse_res,
                                          fine_factor=args.fine_factor,
                                          init_density=0.05, init_color=0.5)
        save_oct(oct, args.oct_path)
        print(f'init octree {args.coarse_res}^3 coarse, F={args.fine_factor}, '
              f'effective {args.coarse_res*args.fine_factor}^3 max -> {args.oct_path}')
        return

    if not os.path.exists(args.oct_path):
        print(f'ERR: {args.oct_path} not found, run --mode init first'); return
    oct = load_oct(args.oct_path)
    print(f'loaded oct: {args.coarse_res}^3 coarse, '
          f'{oct.n_subdivided} subdivided, total cells={oct.total_cells:,}')

    if args.mode == 'subdivide':
        # Subdivide all coarse cells with density > threshold AND not already subdivided
        mask = oct.coarse_density > args.subdivide_threshold
        n_sub = oct.subdivide(mask)
        print(f'subdivided {n_sub} new coarse cells (threshold {args.subdivide_threshold})')
        print(f'now: {oct.n_subdivided} subdivided, {oct.fine_density.size:,} fine cells, '
              f'total {oct.total_cells:,} cells')
        save_oct(oct, args.oct_path)
        return

    if args.mode == 'eval':
        def psnr(a, b, p=1.0):
            return 10*np.log10(p*p / max(np.mean((a.astype(np.float64)-b.astype(np.float64))**2), 1e-12))
        psnrs = []
        for ci in range(len(rec_cams)):
            gt = photoset.photos[ci].image; H, W = gt.shape[:2]
            rendered = render_image_oct(oct, rec_cams.intrinsics, rec_cams.extrinsics[ci],
                                          H=H, W=W, n_samples=args.n_samples)
            psnrs.append(psnr(rendered, gt))
        print(f'PSNRs: ' + ' '.join(f'{p:.1f}' for p in psnrs))
        print(f'Mean PSNR: {np.mean(psnrs):.2f} dB')
        return

    # mode == 'iter'
    opt = OctreeOptimizer(lr_density=args.lr_density, lr_color=args.lr_color)
    Ht = max(8, int(photoset.photos[0].image.shape[0] * args.train_scale))
    Wt = max(8, int(photoset.photos[0].image.shape[1] * args.train_scale))
    gts = []
    for ci in range(len(rec_cams)):
        gt = np.asarray(Image.fromarray((photoset.photos[ci].image*255).clip(0,255).astype(np.uint8))
                          .resize((Wt, Ht), Image.LANCZOS), dtype=np.float32) / 255.0
        gts.append(gt)
    t0 = time.perf_counter()
    for it in range(args.n_iters):
        gcd = np.zeros_like(oct.coarse_density); gcc = np.zeros_like(oct.coarse_color)
        gfd = np.zeros_like(oct.fine_density); gfc = np.zeros_like(oct.fine_color)
        loss = 0.0; n_rays = 0
        for ci in range(len(rec_cams)):
            intr = rec_cams.intrinsics; extr = rec_cams.extrinsics[ci]
            rendered = render_image_oct(oct, intr, extr, H=Ht, W=Wt, n_samples=args.n_samples)
            loss += float(np.abs(rendered - gts[ci]).mean())
            accumulate_grad_oct(oct, intr, extr, gts[ci], rendered,
                                  n_samples=args.n_samples,
                                  grad_cd=gcd, grad_cc=gcc, grad_fd=gfd, grad_fc=gfc)
            n_rays += Ht * Wt
        loss /= len(rec_cams)
        opt.step(oct, gcd, gcc, gfd, gfc, n_rays_seen=n_rays)
        if it % 5 == 0 or it == args.n_iters - 1:
            print(f'  + iter {it:3d}  L1={loss:.4f}  '
                  f'cd_max={oct.coarse_density.max():.1f}  '
                  f'fd_max={oct.fine_density.max() if oct.fine_density.size else 0.0:.1f}  '
                  f'elapsed={time.perf_counter()-t0:.1f}s', flush=True)
    save_oct(oct, args.oct_path)


if __name__ == '__main__':
    main()
