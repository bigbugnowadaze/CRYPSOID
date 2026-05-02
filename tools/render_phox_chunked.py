#!/usr/bin/env python3
"""Phoxoidal-aware Audi render driver.

Renders the v28 .3dphox container in TWO modes from the same projected state:
  --mode gaussian   -> all tiers via vanilla Gaussian density (the v0.1 path)
  --mode phoxoidal  -> Tier A / Tier B via phoxoidal_density_screen with a
                       synthetic germ; Tier C falls back to Gaussian.

Per-call resumable state (so it fits in the bash sandbox) is the same shape as
render_audi_chunked.py.

Usage:
  --init   : load scene, project, sort, fit synthetic germs, save state
  --batch N: process next N splats (sorted-rank order)
  --finalize --out path : save PNG
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path('/sessions/ecstatic-sleepy-curie/mnt/Crypsoid')
sys.path.insert(0, str(ROOT / 'tools'))

from crypsorender.io.phox_loader import load_3dphox
from crypsorender.io.ply_loader import load_ply
from crypsorender.io.splat_buffer import SplatBuffer
from crypsorender.pipeline.camera import Camera, CameraParams
from crypsorender.pipeline.project import project_splats
from crypsorender.pipeline.tile import bin_splats_to_tiles
from crypsorender.math.sh import eval_sh_color
from crypsorender.math.germ import fit_synthetic_germs, fit_synthetic_germs_5, phoxoidal_density_screen, phoxoidal_density_germ_full
from crypsorender.pipeline.rasterize import gaussian_density_at_points

DEFAULT_CAM = dict(yaw_deg=35, pitch_deg=18, distance=2.4, fov_deg=42)
# CLI camera overrides applied at init time


def _save(state_dir, **kw):
    np.savez(state_dir / 'state.npz', **kw)


def init_state(scene_path, is_phox, size, max_points, state_dir, use_sh, cam_overrides=None):
    state_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading {scene_path.name} ...", flush=True)
    scene = load_3dphox(scene_path) if is_phox else load_ply(scene_path)
    if max_points and max_points < scene.n:
        rng = np.random.default_rng(2030)
        idx = rng.choice(scene.n, size=max_points, replace=False)
        scene = SplatBuffer(
            n=max_points, xyz=scene.xyz[idx], scales=scene.scales[idx],
            quats=scene.quats[idx], opacities=scene.opacities[idx],
            sh_dc=scene.sh_dc[idx],
            sh_rest=scene.sh_rest[idx] if scene.sh_rest is not None else None,
            tier=scene.tier[idx] if scene.tier is not None else None,
            germ=None, correction=None,
            source=scene.source, scene_format=scene.scene_format,
        )
    print(f"  {scene.n} splats; tier counts: "
          f"{dict(zip(*np.unique(scene.tier, return_counts=True))) if scene.tier is not None else 'none'}", flush=True)

    cam_kw = dict(DEFAULT_CAM)
    if cam_overrides: cam_kw.update(cam_overrides)
    cam = Camera(scene.xyz, CameraParams(size=size, **cam_kw))

    print("projecting ...", flush=True)
    centers_2d, cov_2d_inv, radii, depths, keep, _ = project_splats(scene, cam)
    n_vis = int(centers_2d.shape[0])
    orig = np.where(keep)[0]
    view_dirs = scene.xyz[orig] - cam.eye
    view_dirs = view_dirs / (np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-9)
    sh_rest_vis = scene.sh_rest[orig] if (use_sh and scene.sh_rest is not None) else None
    rgb_vals = eval_sh_color(scene.sh_dc[orig], sh_rest_vis, view_dirs, view_clip=True)
    opacity_vals = scene.opacities[orig].astype(np.float32)
    tier_vis = (scene.tier[orig] if scene.tier is not None else np.full(n_vis, 2, dtype=np.uint8)).astype(np.uint8)

    print("fitting synthetic germs (5-coef Pearcey basis) ...", flush=True)
    t0 = time.perf_counter()
    germ_vis = fit_synthetic_germs_5(scene.xyz[orig], scene.quats[orig], scene.scales[orig],
                                     tier=tier_vis, k=16)
    print(f"  germ fit took {time.perf_counter()-t0:.1f}s; shape {germ_vis.shape}", flush=True)

    # Reconstruct cov_2d (forward) for the faithful evaluator + sigma_n_screen.
    # cov_2d_inv was computed by project_splats; invert it back for cov_2d.
    cov_2d = np.empty_like(cov_2d_inv)
    a = cov_2d_inv[:, 0, 0]; b = cov_2d_inv[:, 0, 1]; d = cov_2d_inv[:, 1, 1]
    det = a * d - b * b
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    cov_2d[:, 0, 0] =  d / det
    cov_2d[:, 0, 1] = -b / det
    cov_2d[:, 1, 0] = -b / det
    cov_2d[:, 1, 1] =  a / det
    # sigma_n_screen: use the smallest screen-space spread as the "normal" sigma.
    # This is a screen-space proxy for the splat's normal-direction sigma.
    eig_min = 0.5 * (a + d - np.sqrt(np.maximum((a - d) ** 2 + 4 * b * b, 0.0)))
    eig_min = np.where(np.abs(eig_min) < 1e-12, 1e-12, eig_min)
    sigma_n_screen = np.sqrt(np.maximum(1.0 / eig_min, 0.0)).astype(np.float32)

    print("tile binning ...", flush=True)
    t0 = time.perf_counter()
    tile_lists, _, n_tiles = bin_splats_to_tiles(centers_2d, radii, cam.size, tile_size=16)
    print(f"  {n_tiles}x{n_tiles} tiles in {time.perf_counter()-t0:.1f}s", flush=True)
    tile_offsets = np.zeros(n_tiles*n_tiles + 1, dtype=np.int64)
    flat_lists = []
    for i in range(n_tiles*n_tiles):
        L = tile_lists[i] if tile_lists[i] is not None else np.array([], dtype=np.int32)
        flat_lists.append(np.asarray(L, dtype=np.int32))
        tile_offsets[i+1] = tile_offsets[i] + len(L)
    flat = np.concatenate(flat_lists) if flat_lists else np.array([], dtype=np.int32)

    sort_order = np.argsort(depths, kind='stable').astype(np.int64)
    framebuffer = np.zeros((size, size, 3), dtype=np.float32)
    alpha_accum = np.zeros((size, size), dtype=np.float32)

    _save(state_dir,
          centers_2d=centers_2d, cov_2d_inv=cov_2d_inv, cov_2d=cov_2d,
          sigma_n_screen=sigma_n_screen, radii=radii,
          depths=depths, rgb_vals=rgb_vals, opacity_vals=opacity_vals,
          tier_vis=tier_vis, germ_vis=germ_vis,
          tile_flat=flat, tile_offsets=tile_offsets,
          sort_order=sort_order, framebuffer=framebuffer, alpha_accum=alpha_accum,
          size=np.int64(size), n_visible=np.int64(n_vis), cursor=np.int64(0))
    meta = dict(scene=str(scene_path), is_phox=is_phox, size=size, n_visible=n_vis,
                use_sh=use_sh,
                tier_counts={int(k): int(v) for k, v in zip(*np.unique(tier_vis, return_counts=True))})
    (state_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f"state ready: {state_dir}/state.npz")


def process_batch(state_dir, batch_size, mode):
    s = np.load(state_dir / 'state.npz', allow_pickle=False)
    centers_2d = s['centers_2d']; cov_2d_inv = s['cov_2d_inv']; radii = s['radii']
    rgb_vals = s['rgb_vals']; opacity_vals = s['opacity_vals']
    tier_vis = s['tier_vis']; germ_vis = s['germ_vis']
    tile_flat = s['tile_flat']; tile_offsets = s['tile_offsets']
    sort_order = s['sort_order']
    framebuffer = s['framebuffer'].copy()
    alpha_accum = s['alpha_accum'].copy()
    size = int(s['size']); n_vis = int(s['n_visible']); cursor = int(s['cursor'])
    # Optional Tier 2 fields (not always present in older state files):
    cov_2d = s['cov_2d'] if 'cov_2d' in s.files else None
    sigma_n_screen = s['sigma_n_screen'] if 'sigma_n_screen' in s.files else None
    if mode == 'faithful' and (cov_2d is None or sigma_n_screen is None):
        print("WARN: 'faithful' mode requires state generated by Tier 2 init; "
              "cov_2d/sigma_n_screen not in state file. Falling back to 'phoxoidal' (Tier 1 screen-space).")
        mode = 'phoxoidal'

    splat_rank = np.empty(n_vis, dtype=np.int64)
    splat_rank[sort_order] = np.arange(n_vis)
    end = min(cursor + batch_size, n_vis)
    if cursor >= n_vis:
        print("already done"); return
    splats_in_batch = sort_order[cursor:end]
    # Boolean mask: True if splat is in this batch (O(K) per-tile check)
    in_batch_mask = np.zeros(n_vis, dtype=bool)
    in_batch_mask[splats_in_batch] = True

    n_tiles_side = int(np.sqrt(len(tile_offsets) - 1))
    tile_size = 16
    dispatch_counts = {0: 0, 1: 0, 2: 0}    # per tier rasterization count

    t0 = time.perf_counter()
    for tile_idx in range(n_tiles_side * n_tiles_side):
        ts = tile_offsets[tile_idx]; te = tile_offsets[tile_idx+1]
        if te == ts: continue
        tile_splats_all = tile_flat[ts:te]
        in_batch = in_batch_mask[tile_splats_all]
        if not in_batch.any(): continue
        tile_splats = tile_splats_all[in_batch]
        order = np.argsort(splat_rank[tile_splats])
        tile_splats = tile_splats[order]

        ty = tile_idx // n_tiles_side; tx = tile_idx % n_tiles_side
        x0 = tx*tile_size; y0 = ty*tile_size
        x1 = min(x0+tile_size, size); y1 = min(y0+tile_size, size)
        tile_h = y1-y0; tile_w = x1-x0
        px_grid, py_grid = np.meshgrid(
            np.arange(x0, x1, dtype=np.float32) + 0.5,
            np.arange(y0, y1, dtype=np.float32) + 0.5)
        px_flat = px_grid.flatten(); py_flat = py_grid.flatten()
        tile_alpha = alpha_accum[y0:y1, x0:x1]
        tile_rgb = framebuffer[y0:y1, x0:x1]

        for s_idx in tile_splats:
            transmittance_flat = (1.0 - tile_alpha).reshape(-1)
            if (transmittance_flat < 1e-4).all(): break

            kind = int(tier_vis[s_idx])
            germ_active = (kind != 2) and np.any(germ_vis[s_idx] != 0)
            if mode == 'faithful' and germ_active:
                density = phoxoidal_density_germ_full(
                    centers_2d, cov_2d, cov_2d_inv, germ_vis, sigma_n_screen,
                    px_flat, py_flat, s_idx,
                )
            elif mode == 'phoxoidal' and germ_active:
                density = phoxoidal_density_screen(centers_2d, cov_2d_inv, germ_vis,
                                                   px_flat, py_flat, s_idx)
            else:
                density = gaussian_density_at_points(centers_2d[s_idx], cov_2d_inv[s_idx],
                                                     px_flat, py_flat)
            if density.max() < 1e-6: continue
            opacity = float(opacity_vals[s_idx])
            alpha_flat = np.minimum(opacity * density, 0.999)
            # Per-pixel alpha threshold (Inria 3DGS convention): below 1/255
            # the splat's contribution to that pixel is invisible, so skip it.
            # This prevents the "wide-tail accumulation" that washes the image
            # to white when each pixel is covered by ~150 splats.
            alpha_flat = np.where(alpha_flat < (1.0 / 255.0), 0.0, alpha_flat)
            if alpha_flat.max() < 1e-6: continue
            alpha_2d = alpha_flat.reshape(tile_h, tile_w)
            transmittance_2d = 1.0 - tile_alpha
            contrib = transmittance_2d * alpha_2d
            color = rgb_vals[s_idx]
            tile_rgb += contrib[:, :, None] * color[None, None, :]
            tile_alpha += contrib
            dispatch_counts[kind] = dispatch_counts.get(kind, 0) + 1

        framebuffer[y0:y1, x0:x1] = tile_rgb
        alpha_accum[y0:y1, x0:x1] = tile_alpha

    elapsed = time.perf_counter() - t0
    print(f"  batch [{cursor},{end}) of {n_vis}: {elapsed:.1f}s   "
          f"per-tier rasterizations: A={dispatch_counts.get(0,0)} B={dispatch_counts.get(1,0)} C={dispatch_counts.get(2,0)}")
    save_kwargs = dict(
        centers_2d=centers_2d, cov_2d_inv=cov_2d_inv, radii=radii,
        depths=s['depths'], rgb_vals=rgb_vals, opacity_vals=opacity_vals,
        tier_vis=tier_vis, germ_vis=germ_vis,
        tile_flat=tile_flat, tile_offsets=tile_offsets,
        sort_order=sort_order, framebuffer=framebuffer, alpha_accum=alpha_accum,
        size=np.int64(size), n_visible=np.int64(n_vis), cursor=np.int64(end),
    )
    if cov_2d is not None: save_kwargs['cov_2d'] = cov_2d
    if sigma_n_screen is not None: save_kwargs['sigma_n_screen'] = sigma_n_screen
    _save(state_dir, **save_kwargs)


def finalize(state_dir, out_png):
    s = np.load(state_dir / 'state.npz', allow_pickle=False)
    fb = s['framebuffer']; alpha = s['alpha_accum']
    img = np.clip(fb * 255, 0, 255).astype(np.uint8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    import imageio
    imageio.imwrite(out_png, img)
    print(f"saved {out_png}; alpha mean={alpha.mean():.3f} max={alpha.max():.3f} fb max={fb.max():.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', type=Path)
    p.add_argument('--is-phox', action='store_true')
    p.add_argument('--size', type=int, default=1024)
    p.add_argument('--max-points', type=int, default=0)
    p.add_argument('--use-sh', action='store_true')
    p.add_argument('--state-dir', type=Path, required=True)
    p.add_argument('--yaw', type=float, default=None)
    p.add_argument('--pitch', type=float, default=None)
    p.add_argument('--distance', type=float, default=None)
    p.add_argument('--fov', type=float, default=None)
    p.add_argument('--init', action='store_true')
    p.add_argument('--batch', type=int, default=0)
    p.add_argument('--mode', choices=['gaussian', 'phoxoidal', 'faithful'], default='phoxoidal',
                   help="gaussian = vanilla; phoxoidal = Tier 1 screen-space approx; "
                        "faithful = Tier 2 5-coef germ in screen-space eigenframe")
    p.add_argument('--finalize', action='store_true')
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    if args.init:
        cam_over = {}
        if args.yaw is not None: cam_over['yaw_deg'] = args.yaw
        if args.pitch is not None: cam_over['pitch_deg'] = args.pitch
        if args.distance is not None: cam_over['distance'] = args.distance
        if args.fov is not None: cam_over['fov_deg'] = args.fov
        init_state(args.scene, args.is_phox, args.size, args.max_points, args.state_dir, args.use_sh, cam_overrides=cam_over)
    if args.batch > 0:
        process_batch(args.state_dir, args.batch, args.mode)
    if args.finalize:
        finalize(args.state_dir, args.out)


if __name__ == '__main__':
    main()
