"""F.23.2 — Acceptance Gate 1.

Verify that ContractedPhoxelGrid converges on a bounded synthetic scene to
within 5% of the bounded PhoxelGrid PSNR. This proves the contraction +
disparity sampling don't degrade the bounded case.

Scene: a single colored sphere at the origin, viewed from 12 cameras
arranged on a hemisphere of radius 2.5. All geometry fits inside the
unit ball, so contract is identity for every sample point and the test
ISOLATES the disparity-sampling change from the contraction transform.

Pass criterion:
    contracted_PSNR >= 0.95 * bounded_PSNR
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path('/sessions/ecstatic-sleepy-curie/mnt/Crypsoid')
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np

from img2phox.data_classes import (PhotoSet, Photo, CameraIntrinsics,
                                       CameraExtrinsics, CameraBundle)
from img2phox.phoxel import (PhoxelGrid, render_image, accumulate_grad,
                                 PhoxelOptimizer)
from img2phox.phoxel_contracted import (ContractedPhoxelGrid,
                                            render_image_contracted,
                                            accumulate_grad_contracted)


def make_synthetic_sphere_scene(n_cams=12, H=64, W=64,
                                 sphere_center=(0, 0, 0),
                                 sphere_r=0.7,
                                 sphere_color=(0.85, 0.30, 0.20),
                                 cam_distance=2.5,
                                 focal=80.0):
    """Render a textured sphere from `n_cams` camera positions analytically.

    Cheap analytic ray-vs-sphere; pixels that miss the sphere get gray bg.
    """
    rng = np.random.default_rng(0)
    sphere_center = np.asarray(sphere_center, dtype=np.float32)
    sphere_color  = np.asarray(sphere_color,  dtype=np.float32)
    bg = np.array([0.5, 0.5, 0.55], dtype=np.float32)

    intr = CameraIntrinsics(focal_x=focal, focal_y=focal,
                              cx=W / 2, cy=H / 2, width=W, height=H)
    extrinsics = []
    photos = []
    # Spread cameras on a Fibonacci hemisphere
    i = np.arange(n_cams) + 0.5
    phi = np.arccos(1 - i / n_cams)              # 0 .. pi/2 (hemisphere)
    theta = np.pi * (1 + 5 ** 0.5) * i
    cam_pos = cam_distance * np.stack([
        np.cos(theta) * np.sin(phi),
        np.cos(phi),
        np.sin(theta) * np.sin(phi),
    ], axis=1).astype(np.float32)

    for ci in range(n_cams):
        eye = cam_pos[ci]
        # look at origin
        z = (sphere_center - eye); z /= np.linalg.norm(z) + 1e-9
        up = np.array([0, 1, 0], dtype=np.float32)
        if abs(z @ up) > 0.95:
            up = np.array([1, 0, 0], dtype=np.float32)
        x = np.cross(z, up); x /= np.linalg.norm(x) + 1e-9
        y = np.cross(z, x)
        R = np.stack([x, y, z], axis=0)         # world->cam
        t = -R @ eye

        # Render: per-pixel ray-sphere
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        rd_cam = np.stack([
            (xs - intr.cx) / intr.focal_x,
            (ys - intr.cy) / intr.focal_y,
            np.ones_like(xs, dtype=np.float32),
        ], axis=-1).reshape(-1, 3)
        rd_cam /= np.linalg.norm(rd_cam, axis=1, keepdims=True)
        rd_w = (R.T @ rd_cam.T).T
        ro_w = np.broadcast_to(eye[None, :], rd_w.shape)
        # Solve quadratic for ray-sphere intersection
        oc = ro_w - sphere_center[None, :]
        b = (oc * rd_w).sum(axis=1)
        c = (oc * oc).sum(axis=1) - sphere_r * sphere_r
        disc = b * b - c
        hit = disc >= 0
        # Sphere normal-based shading for some texture
        img = np.broadcast_to(bg[None, :], (H * W, 3)).copy()
        if hit.any():
            t_hit = -b[hit] - np.sqrt(disc[hit])
            front = t_hit > 0
            keep = np.where(hit)[0][front]
            if keep.size > 0:
                p = ro_w[keep] + t_hit[front, None] * rd_w[keep]
                n = (p - sphere_center[None, :]) / sphere_r
                # Lambert-ish from a fixed light direction
                light = np.array([0.5, 1.0, 0.3]); light /= np.linalg.norm(light)
                shade = np.maximum(0.15, n @ light).astype(np.float32)[:, None]
                img[keep] = sphere_color[None, :] * shade
        img = img.reshape(H, W, 3).astype(np.float32)
        photos.append(Photo(image=img, path=Path(f'sphere_{ci:02d}.png')))
        extrinsics.append(CameraExtrinsics(R=R.astype(np.float32),
                                              t=t.astype(np.float32)))
    cams = CameraBundle(intrinsics=intr, extrinsics=extrinsics)
    return PhotoSet(photos=photos), cams


def psnr(a, b, peak=1.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse < 1e-12:
        return 99.0
    return 10.0 * np.log10(peak * peak / mse)


def fit_bounded(photoset, cams, n_iters=80, res=48):
    cam_pos = np.array([e.cam_position for e in cams.extrinsics])
    lo = (cam_pos.min(axis=0) - 0.5).astype(np.float32)
    hi = (cam_pos.max(axis=0) + 0.5).astype(np.float32)
    grid = PhoxelGrid.from_bounds(lo, hi, res=res, init_density=0.05)
    opt = PhoxelOptimizer(lr_density=2.0, lr_color=0.3)
    for it in range(n_iters):
        gd = np.zeros_like(grid.density)
        gc = np.zeros_like(grid.color)
        n_rays = 0
        for ci in range(len(cams)):
            gt = photoset.photos[ci].image
            r = render_image(grid, cams.intrinsics, cams.extrinsics[ci],
                              H=gt.shape[0], W=gt.shape[1], n_samples=48)
            accumulate_grad(grid, cams.intrinsics, cams.extrinsics[ci], gt, r,
                              n_samples=48, grad_density=gd, grad_color=gc)
            n_rays += gt.shape[0] * gt.shape[1]
        opt.step(grid, gd, gc, n_rays_seen=n_rays)
    return grid


def fit_contracted(photoset, cams, n_iters=80, res=48):
    cam_pos = np.array([e.cam_position for e in cams.extrinsics])
    grid = ContractedPhoxelGrid.from_cameras(cam_pos, res=res, init_density=0.05)
    opt = PhoxelOptimizer(lr_density=2.0, lr_color=0.3)
    for it in range(n_iters):
        gd = np.zeros_like(grid.density)
        gc = np.zeros_like(grid.color)
        n_rays = 0
        for ci in range(len(cams)):
            gt = photoset.photos[ci].image
            r = render_image_contracted(grid, cams.intrinsics, cams.extrinsics[ci],
                                          H=gt.shape[0], W=gt.shape[1], n_samples=64)
            accumulate_grad_contracted(grid, cams.intrinsics, cams.extrinsics[ci],
                                          gt, r, n_samples=64,
                                          grad_density=gd, grad_color=gc)
            n_rays += gt.shape[0] * gt.shape[1]
        # Reuse the bounded optimizer; it just walks density+color arrays
        opt.step(grid, gd, gc, n_rays_seen=n_rays)
    return grid


def eval_psnr_bounded(grid, photoset, cams):
    psnrs = []
    for ci in range(len(cams)):
        gt = photoset.photos[ci].image
        r = render_image(grid, cams.intrinsics, cams.extrinsics[ci],
                          H=gt.shape[0], W=gt.shape[1], n_samples=64)
        psnrs.append(psnr(r, gt))
    return float(np.mean(psnrs))


def eval_psnr_contracted(grid, photoset, cams):
    psnrs = []
    for ci in range(len(cams)):
        gt = photoset.photos[ci].image
        r = render_image_contracted(grid, cams.intrinsics, cams.extrinsics[ci],
                                      H=gt.shape[0], W=gt.shape[1], n_samples=96)
        psnrs.append(psnr(r, gt))
    return float(np.mean(psnrs))


def main():
    print('=' * 70)
    print('F.23.2 — GATE 1: contracted-vs-bounded on bounded scene')
    print('=' * 70)
    t0 = time.perf_counter()
    photoset, cams = make_synthetic_sphere_scene(n_cams=12, H=64, W=64)
    print(f'  synth scene: {len(cams)} cams @ {photoset.photos[0].image.shape}')

    N_ITERS = 30
    RES = 32

    print(f'\n[bounded] fitting ({N_ITERS} iters @ {RES}^3) ...')
    t1 = time.perf_counter()
    g_b = fit_bounded(photoset, cams, n_iters=N_ITERS, res=RES)
    psnr_b = eval_psnr_bounded(g_b, photoset, cams)
    print(f'  bounded fit done in {time.perf_counter()-t1:.1f}s, PSNR = {psnr_b:.2f} dB')

    print(f'\n[contracted] fitting ({N_ITERS} iters @ {RES}^3) ...')
    t1 = time.perf_counter()
    g_c = fit_contracted(photoset, cams, n_iters=N_ITERS, res=RES)
    psnr_c = eval_psnr_contracted(g_c, photoset, cams)
    print(f'  contracted fit done in {time.perf_counter()-t1:.1f}s, PSNR = {psnr_c:.2f} dB')

    ratio = psnr_c / psnr_b if psnr_b > 0 else 0.0
    print()
    print('=' * 70)
    print(f'PSNR ratio  contracted / bounded = {ratio:.3f}')
    PASS = ratio >= 0.95
    print(f'GATE 1: {"PASS" if PASS else "FAIL"}  '
          f'(criterion: ratio >= 0.95)')
    print(f'  total {time.perf_counter()-t0:.1f}s')
    print('=' * 70)
    return 0 if PASS else 1


if __name__ == '__main__':
    raise SystemExit(main())
 1


if __name__ == '__main__':
    raise SystemExit(main())
