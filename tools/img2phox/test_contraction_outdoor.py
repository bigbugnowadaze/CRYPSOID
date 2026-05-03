"""F.23.2 — Acceptance Gate 2.

The actual point of scene contraction. Build a scene with a small
foreground subject AND a large textured backdrop sphere far behind.
Cameras orbit close to the foreground but their pixels still capture
the backdrop.

Bounded grid sized to the cameras leaves the backdrop OUTSIDE the AABB,
so background pixels collapse into bg_color and the optimizer puts
spurious foreground density to "explain" them. PSNR stalls.

Contracted grid sees the same backdrop squashed into the [1, 2) shell of
the contracted ball; sky-resolution but present. PSNR climbs.

Pass criterion:
    contracted_PSNR - bounded_PSNR >= 3.0 dB
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


def make_outdoor_scene(n_cams=10, H=64, W=64,
                       fg_r=0.3, fg_color=(0.85, 0.30, 0.20),
                       bg_r=20.0,
                       cam_distance=1.4,
                       focal=70.0):
    """Foreground sphere at origin + huge textured background sphere at r=20.

    Bg colour is a procedural latitude-coloured pattern so it has structure
    (not constant) — that way bounded grids genuinely fail to explain it.
    """
    rng = np.random.default_rng(0)
    fg_color = np.asarray(fg_color, dtype=np.float32)
    intr = CameraIntrinsics(focal_x=focal, focal_y=focal,
                              cx=W / 2, cy=H / 2, width=W, height=H)
    extrinsics = []; photos = []

    i = np.arange(n_cams) + 0.5
    phi = np.full(n_cams, np.pi * 0.45)            # equatorial-ish
    theta = 2 * np.pi * i / n_cams
    cam_pos = cam_distance * np.stack([
        np.cos(theta) * np.sin(phi),
        np.cos(phi),
        np.sin(theta) * np.sin(phi),
    ], axis=1).astype(np.float32)

    def background_color(d):
        """Aggressive backdrop: sharp horizon split + lon checker.

        Designed so bounded "fog inside the AABB" can NOT fake it. The fog
        could match an average colour but not the sharp horizon edge.
        """
        lat = np.arcsin(np.clip(d[:, 1], -1, 1))             # -pi/2..pi/2
        lon = np.arctan2(d[:, 2], d[:, 0])                    # -pi..pi
        # Sky/ground sharp split at lat=0
        sky    = np.array([0.30, 0.55, 0.92], dtype=np.float32)
        ground = np.array([0.45, 0.32, 0.18], dtype=np.float32)
        base = np.where(lat[:, None] > 0, sky[None, :], ground[None, :])
        # Longitude checker (8 bands), strong contrast
        chk = ((np.floor(4 * lon / np.pi) % 2) == 0).astype(np.float32)[:, None]
        tinted = base * (0.6 + 0.4 * chk)
        return np.clip(tinted, 0.0, 1.0).astype(np.float32)

    for ci in range(n_cams):
        eye = cam_pos[ci]
        z = -eye / np.linalg.norm(eye)
        up = np.array([0, 1, 0], dtype=np.float32)
        if abs(z @ up) > 0.95:
            up = np.array([1, 0, 0], dtype=np.float32)
        x = np.cross(z, up); x /= np.linalg.norm(x) + 1e-9
        y = np.cross(z, x)
        R = np.stack([x, y, z], axis=0)
        t = -R @ eye

        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        rd_cam = np.stack([
            (xs - intr.cx) / intr.focal_x,
            (ys - intr.cy) / intr.focal_y,
            np.ones_like(xs, dtype=np.float32),
        ], axis=-1).reshape(-1, 3)
        rd_cam /= np.linalg.norm(rd_cam, axis=1, keepdims=True)
        rd_w = (R.T @ rd_cam.T).T
        ro_w = np.broadcast_to(eye[None, :], rd_w.shape)

        # Foreground sphere intersection
        b = (ro_w * rd_w).sum(axis=1)
        c_fg = (ro_w * ro_w).sum(axis=1) - fg_r * fg_r
        disc_fg = b * b - c_fg
        hit_fg = (disc_fg >= 0) & (-b - np.sqrt(np.maximum(disc_fg, 0)) > 0)

        # Background sphere ALWAYS hit (camera is inside it)
        c_bg = (ro_w * ro_w).sum(axis=1) - bg_r * bg_r
        disc_bg = b * b - c_bg
        t_bg = -b + np.sqrt(np.maximum(disc_bg, 0))
        p_bg = ro_w + t_bg[:, None] * rd_w
        bg_dir = p_bg / (np.linalg.norm(p_bg, axis=1, keepdims=True) + 1e-9)
        img = background_color(bg_dir)

        # Where foreground hit: shade the fg colour
        if hit_fg.any():
            t_hit = -b[hit_fg] - np.sqrt(disc_fg[hit_fg])
            keep = np.where(hit_fg)[0]
            p = ro_w[keep] + t_hit[:, None] * rd_w[keep]
            n = p / fg_r
            light = np.array([0.5, 1.0, 0.3]); light /= np.linalg.norm(light)
            shade = np.maximum(0.20, n @ light).astype(np.float32)[:, None]
            img[keep] = fg_color[None, :] * shade

        img = img.reshape(H, W, 3).astype(np.float32)
        photos.append(Photo(image=img, path=Path(f'outdoor_{ci:02d}.png')))
        extrinsics.append(CameraExtrinsics(R=R.astype(np.float32),
                                              t=t.astype(np.float32)))
    cams = CameraBundle(intrinsics=intr, extrinsics=extrinsics)
    return PhotoSet(photos=photos), cams


def psnr(a, b, peak=1.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse < 1e-12: return 99.0
    return 10.0 * np.log10(peak * peak / mse)


def main():
    print('=' * 70)
    print('F.23.2 — GATE 2: contracted vs bounded on UNBOUNDED outdoor scene')
    print('=' * 70)
    t0 = time.perf_counter()
    photoset, cams = make_outdoor_scene(n_cams=10, H=64, W=64)
    print(f'  synth scene: {len(cams)} cams, fg=tight ball, bg=textured sphere @ r=20')

    N_ITERS = 60
    RES = 48
    T_FAR = 60.0

    print(f'\n[bounded] fitting (cameras-only AABB will MISS background) ...')
    cam_pos = np.array([e.cam_position for e in cams.extrinsics])
    lo = (cam_pos.min(axis=0) - 0.5).astype(np.float32)
    hi = (cam_pos.max(axis=0) + 0.5).astype(np.float32)
    g_b = PhoxelGrid.from_bounds(lo, hi, res=RES, init_density=0.05)
    opt = PhoxelOptimizer(lr_density=2.0, lr_color=0.3)
    t1 = time.perf_counter()
    for it in range(N_ITERS):
        gd = np.zeros_like(g_b.density); gc = np.zeros_like(g_b.color); n_rays = 0
        for ci in range(len(cams)):
            gt = photoset.photos[ci].image
            r = render_image(g_b, cams.intrinsics, cams.extrinsics[ci],
                              H=gt.shape[0], W=gt.shape[1], n_samples=48)
            accumulate_grad(g_b, cams.intrinsics, cams.extrinsics[ci], gt, r,
                              n_samples=48, grad_density=gd, grad_color=gc)
            n_rays += gt.shape[0] * gt.shape[1]
        opt.step(g_b, gd, gc, n_rays_seen=n_rays)
    psnrs_b = [psnr(render_image(g_b, cams.intrinsics, cams.extrinsics[ci],
                                  H=64, W=64, n_samples=64), photoset.photos[ci].image)
                for ci in range(len(cams))]
    psnr_b = float(np.mean(psnrs_b))
    print(f'  bounded fit done in {time.perf_counter()-t1:.1f}s, PSNR = {psnr_b:.2f} dB')

    print(f'\n[contracted] fitting (background squashed into shell) ...')
    g_c = ContractedPhoxelGrid.from_cameras(cam_pos, res=RES, init_density=0.05)
    opt = PhoxelOptimizer(lr_density=2.0, lr_color=0.3)
    t1 = time.perf_counter()
    for it in range(N_ITERS):
        gd = np.zeros_like(g_c.density); gc = np.zeros_like(g_c.color); n_rays = 0
        for ci in range(len(cams)):
            gt = photoset.photos[ci].image
            r = render_image_contracted(g_c, cams.intrinsics, cams.extrinsics[ci],
                                          H=gt.shape[0], W=gt.shape[1], n_samples=64,
                                          t_far_norm=T_FAR)
            accumulate_grad_contracted(g_c, cams.intrinsics, cams.extrinsics[ci],
                                          gt, r, n_samples=64, t_far_norm=T_FAR,
                                          grad_density=gd, grad_color=gc)
            n_rays += gt.shape[0] * gt.shape[1]
        opt.step(g_c, gd, gc, n_rays_seen=n_rays)
    psnrs_c = [psnr(render_image_contracted(g_c, cams.intrinsics, cams.extrinsics[ci],
                                              H=64, W=64, n_samples=96, t_far_norm=T_FAR),
                     photoset.photos[ci].image) for ci in range(len(cams))]
    psnr_c = float(np.mean(psnrs_c))
    print(f'  contracted fit done in {time.perf_counter()-t1:.1f}s, PSNR = {psnr_c:.2f} dB')

    delta = psnr_c - psnr_b
    print()
    print('=' * 70)
    print(f'PSNR delta  contracted - bounded = {delta:+.2f} dB')
    PASS = delta >= 3.0
    print(f'GATE 2: {"PASS" if PASS else "FAIL"}  (criterion: delta >= +3.0 dB)')
    print(f'  total {time.perf_counter()-t0:.1f}s')
    print('=' * 70)
    return 0 if PASS else 1


if __name__ == '__main__':
    raise SystemExit(main())
