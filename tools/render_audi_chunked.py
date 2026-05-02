#!/usr/bin/env python3
"""Chunked Audi render driver.

The full 763k-splat render at 1024x1024 doesn't fit in one bash sandbox call.
This driver:
  Step 0: load scene, project + sort all splats, save state to disk
  Step k: process next batch of M splats, update framebuffer on disk
  --finalize: save final PNG + metrics

Resumable: progress lives in state.npz.

Usage:
  python3 tools/render_audi_chunked.py --init
  python3 tools/render_audi_chunked.py --batch 100000   # repeat until --finalize
  python3 tools/render_audi_chunked.py --finalize
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
from crypsorender.pipeline.rasterize import gaussian_density_at_points

OUT = ROOT / 'renders' / 'crypsorender_v01'

DEFAULT_CAMERA = dict(yaw_deg=35, pitch_deg=18, distance=2.4, fov_deg=42)


def init_state(scene_path: Path, is_phox: bool, size: int, max_points: int,
               state_dir: Path, use_sh: bool):
    state_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading {scene_path.name} ...", flush=True)
    t0 = time.perf_counter()
    scene = load_3dphox(scene_path) if is_phox else load_ply(scene_path)
    print(f"  loaded {scene.n} splats in {time.perf_counter()-t0:.1f}s", flush=True)

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
        print(f"  subsampled to {scene.n}", flush=True)

    cam = Camera(scene.xyz, CameraParams(
        yaw_deg=DEFAULT_CAMERA['yaw_deg'], pitch_deg=DEFAULT_CAMERA['pitch_deg'],
        distance=DEFAULT_CAMERA['distance'], fov_deg=DEFAULT_CAMERA['fov_deg'], size=size))

    print("projecting + ewa ...", flush=True)
    t0 = time.perf_counter()
    centers_2d, cov_2d_inv, radii, depths, keep, _ = project_splats(scene, cam)
    print(f"  {keep.sum()}/{scene.n} kept in {time.perf_counter()-t0:.1f}s", flush=True)

    # SH eval (use_sh=False -> just return DC color)
    print("sh eval ...", flush=True)
    t0 = time.perf_counter()
    orig = np.where(keep)[0]
    view_dirs = scene.xyz[orig] - cam.eye
    view_dirs = view_dirs / (np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-9)
    sh_rest_vis = scene.sh_rest[orig] if (use_sh and scene.sh_rest is not None) else None
    rgb_vals = eval_sh_color(scene.sh_dc[orig], sh_rest_vis, view_dirs, view_clip=True)
    opacity_vals = scene.opacities[orig].astype(np.float32)
    print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)

    print("tile binning ...", flush=True)
    t0 = time.perf_counter()
    tile_lists, tile_counts, n_tiles = bin_splats_to_tiles(centers_2d, radii, cam.size, tile_size=16)
    print(f"  {n_tiles}x{n_tiles} tiles in {time.perf_counter()-t0:.1f}s", flush=True)
    # Convert tile_lists to a single concatenated int32 array + offsets (for npz storage)
    tile_offsets = np.zeros(n_tiles*n_tiles + 1, dtype=np.int64)
    flat_lists = []
    for i in range(n_tiles*n_tiles):
        L = tile_lists[i] if tile_lists[i] is not None else np.array([], dtype=np.int32)
        flat_lists.append(np.asarray(L, dtype=np.int32))
        tile_offsets[i+1] = tile_offsets[i] + len(L)
    flat = np.concatenate(flat_lists) if flat_lists else np.array([], dtype=np.int32)

    # Front-to-back order for ALL splats (smallest depth first)
    sort_order = np.argsort(depths, kind='stable').astype(np.int64)

    # Init framebuffer + alpha accumulator
    framebuffer = np.zeros((size, size, 3), dtype=np.float32)
    alpha_accum = np.zeros((size, size), dtype=np.float32)

    # Save state
    np.savez(state_dir / 'state.npz',
             centers_2d=centers_2d, cov_2d_inv=cov_2d_inv, radii=radii,
             depths=depths, rgb_vals=rgb_vals, opacity_vals=opacity_vals,
             tile_flat=flat, tile_offsets=tile_offsets,
             sort_order=sort_order, framebuffer=framebuffer, alpha_accum=alpha_accum,
             size=np.int64(size), n_visible=np.int64(centers_2d.shape[0]),
             cursor=np.int64(0))
    meta = dict(scene=str(scene_path), is_phox=is_phox, size=size, n_visible=int(centers_2d.shape[0]),
                use_sh=use_sh, scene_format=scene.scene_format,
                tier_counts=({int(k): int(v) for k, v in zip(*np.unique(scene.tier, return_counts=True))} if scene.tier is not None else None))
    (state_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f"state saved to {state_dir}/state.npz")


def process_batch(state_dir: Path, batch_size: int):
    s = np.load(state_dir / 'state.npz', allow_pickle=False)
    centers_2d = s['centers_2d']; cov_2d_inv = s['cov_2d_inv']; radii = s['radii']
    rgb_vals = s['rgb_vals']; opacity_vals = s['opacity_vals']
    tile_flat = s['tile_flat']; tile_offsets = s['tile_offsets']
    sort_order = s['sort_order']
    framebuffer = s['framebuffer'].copy()
    alpha_accum = s['alpha_accum'].copy()
    size = int(s['size']); n_vis = int(s['n_visible']); cursor = int(s['cursor'])

    # Build splat_rank lookup once for tile-local sorting
    splat_rank = np.empty(n_vis, dtype=np.int64)
    splat_rank[sort_order] = np.arange(n_vis)

    end = min(cursor + batch_size, n_vis)
    if cursor >= n_vis:
        print(f"all {n_vis} splats already done.")
        return

    print(f"processing splats sorted-rank [{cursor}, {end}) of {n_vis}", flush=True)
    # Splats whose rank is in [cursor, end)
    splats_in_batch = sort_order[cursor:end]
    splat_in_batch_set = set(splats_in_batch.tolist())

    n_tiles_side = int(np.sqrt(len(tile_offsets) - 1))
    tile_size = 16

    t0 = time.perf_counter()
    n_processed = 0
    for tile_idx in range(n_tiles_side * n_tiles_side):
        ts = tile_offsets[tile_idx]; te = tile_offsets[tile_idx + 1]
        if te == ts:
            continue
        # Splats in this tile that fall in this batch
        tile_splats_all = tile_flat[ts:te]
        # Filter to those in this batch
        in_batch_mask = np.isin(tile_splats_all, splats_in_batch, assume_unique=False)
        if not in_batch_mask.any():
            continue
        tile_splats = tile_splats_all[in_batch_mask]
        # Sort by global rank to maintain front-to-back order
        order = np.argsort(splat_rank[tile_splats])
        tile_splats = tile_splats[order]

        ty = tile_idx // n_tiles_side
        tx = tile_idx % n_tiles_side
        x0 = tx * tile_size
        y0 = ty * tile_size
        x1 = min(x0 + tile_size, size); y1 = min(y0 + tile_size, size)
        tile_h = y1 - y0; tile_w = x1 - x0

        px_grid, py_grid = np.meshgrid(
            np.arange(x0, x1, dtype=np.float32) + 0.5,
            np.arange(y0, y1, dtype=np.float32) + 0.5)
        px_flat = px_grid.flatten(); py_flat = py_grid.flatten()

        tile_alpha = alpha_accum[y0:y1, x0:x1]
        tile_rgb = framebuffer[y0:y1, x0:x1]

        for s_idx in tile_splats:
            transmittance_flat = (1.0 - tile_alpha).reshape(-1)
            if (transmittance_flat < 1e-4).all():
                break
            density = gaussian_density_at_points(
                centers_2d[s_idx], cov_2d_inv[s_idx], px_flat, py_flat)
            if density.max() < 1e-6:
                continue
            opacity = float(opacity_vals[s_idx])
            alpha_flat = np.minimum(opacity * density, 0.999)
            alpha_2d = alpha_flat.reshape(tile_h, tile_w)
            transmittance_2d = 1.0 - tile_alpha
            contrib = transmittance_2d * alpha_2d
            color = rgb_vals[s_idx]
            tile_rgb += contrib[:, :, None] * color[None, None, :]
            tile_alpha += contrib
            n_processed += 1

        framebuffer[y0:y1, x0:x1] = tile_rgb
        alpha_accum[y0:y1, x0:x1] = tile_alpha

    elapsed = time.perf_counter() - t0
    print(f"  {n_processed} splat-tile rasterizations in {elapsed:.1f}s")
    # Save state with new cursor
    np.savez(state_dir / 'state.npz',
             centers_2d=centers_2d, cov_2d_inv=cov_2d_inv, radii=radii,
             depths=s['depths'], rgb_vals=rgb_vals, opacity_vals=opacity_vals,
             tile_flat=tile_flat, tile_offsets=tile_offsets,
             sort_order=sort_order, framebuffer=framebuffer, alpha_accum=alpha_accum,
             size=np.int64(size), n_visible=np.int64(n_vis), cursor=np.int64(end))
    print(f"cursor advanced to {end}/{n_vis}")


def finalize(state_dir: Path, out_png: Path):
    s = np.load(state_dir / 'state.npz', allow_pickle=False)
    fb = s['framebuffer']
    alpha = s['alpha_accum']
    img = np.clip(fb * 255, 0, 255).astype(np.uint8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    import imageio
    imageio.imwrite(out_png, img)
    print(f"saved {out_png}")
    print(f"  alpha mean={alpha.mean():.3f} max={alpha.max():.3f}, fb max={fb.max():.4f}")
    return img, alpha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', type=Path, required=False)
    p.add_argument('--is-phox', action='store_true')
    p.add_argument('--size', type=int, default=1024)
    p.add_argument('--max-points', type=int, default=0)
    p.add_argument('--use-sh', action='store_true')
    p.add_argument('--state-dir', type=Path, required=True)
    p.add_argument('--init', action='store_true')
    p.add_argument('--batch', type=int, default=0)
    p.add_argument('--finalize', action='store_true')
    p.add_argument('--out', type=Path)
    args = p.parse_args()

    if args.init:
        init_state(args.scene, args.is_phox, args.size, args.max_points, args.state_dir, args.use_sh)
    if args.batch > 0:
        process_batch(args.state_dir, args.batch)
    if args.finalize:
        if args.out is None:
            raise SystemExit("--finalize needs --out")
        finalize(args.state_dir, args.out)


if __name__ == '__main__':
    main()
