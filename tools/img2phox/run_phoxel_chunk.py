"""Helper: run N more iters of phoxel optimization on the cached SfM + grid.

Loads grid from outputs/_phoxel_grid_v03.npz (or initializes fresh if missing).
Runs N iters, saves grid back. Designed to be called repeatedly within one
bash session up to its 40s budget.
"""
from __future__ import annotations
import sys, pickle, time, os, argparse
sys.path.insert(0, 'tools')
import numpy as np
from PIL import Image
from img2phox.phoxel import (PhoxelGrid, render_image, PhoxelOptimizer)
import img2phox.phoxel as phoxel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-iters',     type=int,   default=50)
    p.add_argument('--res',         type=int,   default=96)
    p.add_argument('--train-scale', type=float, default=0.5)
    p.add_argument('--n-samples',   type=int,   default=40)
    p.add_argument('--lr-density',  type=float, default=2.0)
    p.add_argument('--lr-color',    type=float, default=0.3)
    p.add_argument('--grid-path',   type=str,   default='outputs/_phoxel_grid_v03.npz')
    p.add_argument('--sfm-cache',   type=str,   default='outputs/_phoxel_sfm_cache.pkl')
    p.add_argument('--init-density', type=float, default=0.05)
    p.add_argument('--tv-weight-density', type=float, default=0.0,
                    help='TV regularization weight on density. Try 0.001-0.01 to kill floaters.')
    p.add_argument('--tv-weight-color', type=float, default=0.0,
                    help='TV regularization weight on color.')
    args = p.parse_args()

    with open(args.sfm_cache, 'rb') as f: d = pickle.load(f)
    photoset = d['photoset']; rec_cams = d['rec_cams']; sparse = d['sparse']

    # Load or init grid
    if os.path.exists(args.grid_path):
        g = np.load(args.grid_path)
        if int(g['res'][0]) == args.res:
            grid = PhoxelGrid(origin=g['origin'], size=g['size'], res=g['res'],
                                density=g['density'].astype(np.float32),
                                color=g['color'].astype(np.float32))
            print(f'loaded grid {args.res}^3, density max={grid.density.max():.1f}, '
                  f'occupied={(grid.density>0.5).sum()}')
        else:
            print(f'WARN: grid res {int(g["res"][0])} != requested {args.res}, re-init')
            cam_pos = np.array([e.cam_position for e in rec_cams.extrinsics])
            all_pts = np.concatenate([cam_pos, sparse.xyz], axis=0)
            center = all_pts.mean(axis=0); radius = np.linalg.norm(all_pts-center, axis=1).max()
            extent = radius * 1.2
            lo = (center - extent).astype(np.float32); hi = (center + extent).astype(np.float32)
            grid = PhoxelGrid.from_bounds(lo, hi, res=args.res, init_density=args.init_density)
    else:
        cam_pos = np.array([e.cam_position for e in rec_cams.extrinsics])
        all_pts = np.concatenate([cam_pos, sparse.xyz], axis=0)
        center = all_pts.mean(axis=0); radius = np.linalg.norm(all_pts-center, axis=1).max()
        extent = radius * 1.2
        lo = (center - extent).astype(np.float32); hi = (center + extent).astype(np.float32)
        grid = PhoxelGrid.from_bounds(lo, hi, res=args.res, init_density=args.init_density)
        print(f'init grid {args.res}^3 from cameras')

    opt = PhoxelOptimizer(lr_density=args.lr_density, lr_color=args.lr_color)
    t0 = time.perf_counter()
    Ht = max(8, int(photoset.photos[0].image.shape[0] * args.train_scale))
    Wt = max(8, int(photoset.photos[0].image.shape[1] * args.train_scale))

    # Pre-resize all GTs once
    gts = []
    for ci in range(len(rec_cams)):
        gt = np.asarray(Image.fromarray((photoset.photos[ci].image*255).clip(0,255).astype(np.uint8))
                          .resize((Wt, Ht), Image.LANCZOS), dtype=np.float32) / 255.0
        gts.append(gt)

    for it in range(args.n_iters):
        gd = np.zeros_like(grid.density)
        gc = np.zeros_like(grid.color)
        loss = 0.0; n_rays = 0
        for ci in range(len(rec_cams)):
            intr = rec_cams.intrinsics; extr = rec_cams.extrinsics[ci]
            rendered = render_image(grid, intr, extr, H=Ht, W=Wt, n_samples=args.n_samples)
            loss += float(np.abs(rendered - gts[ci]).mean())
            phoxel.accumulate_grad(grid, intr, extr, gts[ci], rendered,
                                     n_samples=args.n_samples,
                                     grad_density=gd, grad_color=gc)
            n_rays += Ht * Wt
        loss /= len(rec_cams)
        # Add TV gradient (smooths neighbors, kills floaters)
        if args.tv_weight_density > 0:
            gd += args.tv_weight_density * phoxel.tv_gradient(grid.density) * n_rays
        if args.tv_weight_color > 0:
            gc += args.tv_weight_color * phoxel.tv_gradient(grid.color) * n_rays
        opt.step(grid, gd, gc, n_rays_seen=n_rays)
        if it % 5 == 0 or it == args.n_iters - 1:
            print(f'  + iter {it:3d}  L1={loss:.4f}  density max={grid.density.max():.1f}  '
                  f'occ={(grid.density>0.5).sum()}  elapsed={time.perf_counter()-t0:.1f}s',
                  flush=True)

    np.savez(args.grid_path,
              density=grid.density, color=grid.color,
              origin=grid.origin, size=grid.size, res=grid.res)

    # Quick PSNR
    def psnr(a, b, peak=1.0):
        mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
        return 10 * np.log10(peak * peak / max(mse, 1e-12))

    psnrs = []
    for ci in range(len(rec_cams)):
        gt = photoset.photos[ci].image; H, W = gt.shape[:2]
        rendered = render_image(grid, rec_cams.intrinsics, rec_cams.extrinsics[ci],
                                  H=H, W=W, n_samples=args.n_samples)
        psnrs.append(psnr(rendered, gt))
    print(f'  PSNR mean: {np.mean(psnrs):.2f} dB  ({" ".join(f"{p:.1f}" for p in psnrs)})')


if __name__ == '__main__':
    main()
