"""Per-tile rasterization with alpha compositing.

Implements the canonical 3DGS-style rasterizer:
- For each tile: walk the depth-sorted splat list front-to-back
- Per splat: compute Mahalanobis density at each pixel in the tile bbox
- Pre-multiplied alpha composite, with early termination on saturated pixels

Self-contained: a single `composite()` function. Tier dispatch happens at the
call site (render.py) by computing rgb_vals/opacity_vals appropriately for the
splats being rendered.
"""

from __future__ import annotations

import numpy as np

from ..io.splat_buffer import SplatBuffer


def gaussian_density_at_points(center_2d, cov_2d_inv, px_flat, py_flat):
    """Evaluate 2D Gaussian density at pixel points.

    α_geom = exp(-0.5 · (p - c)^T · Σ^-1 · (p - c))
    """
    dx = px_flat - center_2d[0]
    dy = py_flat - center_2d[1]
    mahal_sq = (
        cov_2d_inv[0, 0] * dx * dx
        + 2.0 * cov_2d_inv[0, 1] * dx * dy
        + cov_2d_inv[1, 1] * dy * dy
    )
    mahal_sq = np.maximum(mahal_sq, 0)
    power = -0.5 * mahal_sq
    power = np.maximum(power, -20)
    return np.exp(power)


def composite(
    scene: SplatBuffer,
    camera,
    centers_2d: np.ndarray,
    cov_2d_inv: np.ndarray,
    radii: np.ndarray,
    depths: np.ndarray,
    original_indices: np.ndarray,
    tile_lists: list,
    tile_counts: np.ndarray,
    n_tiles: int,
    rgb_vals: np.ndarray,
    opacity_vals: np.ndarray,
    tile_size: int = 16,
):
    """Front-to-back tile-based composite into an RGBA framebuffer.

    Args:
        scene:           SplatBuffer (only `n` is used here for index sanity)
        camera:          Camera object (used for `size`)
        centers_2d:      (m, 2) projected 2D centers (m = visible-after-cull)
        cov_2d_inv:      (m, 2, 2) inverted 2D covariances
        radii:           (m,) splat 3-sigma footprints in pixels
        depths:          (m,) camera-space depths (positive = in front)
        original_indices: (m,) — for debugging only; not used here
        tile_lists:      list of length n_tiles*n_tiles, each entry is a numpy
                         array of splat indices in [0, m) overlapping that tile
        tile_counts:     (n_tiles, n_tiles) splat counts per tile
        n_tiles:         number of tiles per dimension
        rgb_vals:        (m, 3) precomputed linear RGB colors in [0, 1]
        opacity_vals:    (m,) precomputed alpha values in [0, 1]
                         (NOT logits — the loader/sh path already sigmoid-decoded)
        tile_size:       pixels per tile (default 16)

    Returns:
        (framebuffer, alpha_accum):
            framebuffer  (size, size, 3) float32 — pre-multiplied RGB
            alpha_accum  (size, size)    float32 — accumulated coverage in [0,1]
    """
    if rgb_vals is None or opacity_vals is None:
        raise ValueError("rgb_vals and opacity_vals must be non-None")
    if rgb_vals.ndim != 2 or rgb_vals.shape[1] != 3:
        raise ValueError(f"rgb_vals must be (m,3); got {rgb_vals.shape}")

    size = camera.size
    framebuffer = np.zeros((size, size, 3), dtype=np.float32)
    alpha_accum = np.zeros((size, size), dtype=np.float32)

    # Front-to-back: smallest depth first.
    # In the camera convention used by camera.world_to_cam, depths > 0 means
    # in front of the camera and SMALLER depth = CLOSER, so we sort ascending.
    sort_order = np.argsort(depths, kind="stable").astype(np.int64)
    splat_rank = np.empty_like(sort_order)
    splat_rank[sort_order] = np.arange(sort_order.shape[0])

    # Process tiles
    for tile_idx in range(n_tiles * n_tiles):
        ty = tile_idx // n_tiles
        tx = tile_idx % n_tiles
        x0 = tx * tile_size
        y0 = ty * tile_size
        x1 = min(x0 + tile_size, size)
        y1 = min(y0 + tile_size, size)
        tile_h = y1 - y0
        tile_w = x1 - x0
        if tile_h <= 0 or tile_w <= 0:
            continue

        splat_list = tile_lists[tile_idx]
        if splat_list is None or len(splat_list) == 0:
            continue

        # Sort the tile's splat list by global front-to-back rank
        splat_arr = np.asarray(splat_list, dtype=np.int64)
        order = np.argsort(splat_rank[splat_arr])
        splats_in_order = splat_arr[order]

        # Build pixel grid for the tile (use pixel CENTERS at integer+0.5)
        px_grid, py_grid = np.meshgrid(
            np.arange(x0, x1, dtype=np.float32) + 0.5,
            np.arange(y0, y1, dtype=np.float32) + 0.5,
        )
        px_flat = px_grid.flatten()
        py_flat = py_grid.flatten()

        # Tile-local accumulators (one allocation per tile, then write back)
        tile_alpha = alpha_accum[y0:y1, x0:x1].copy()
        tile_rgb = framebuffer[y0:y1, x0:x1].copy()

        for s in splats_in_order:
            # Early termination: if every pixel in tile is already saturated.
            transmittance_flat = (1.0 - tile_alpha).reshape(-1)
            if (transmittance_flat < 1e-4).all():
                break

            density = gaussian_density_at_points(
                centers_2d[s], cov_2d_inv[s], px_flat, py_flat
            )
            if density.max() < 1e-6:
                continue

            opacity = float(opacity_vals[s])
            alpha_flat = opacity * density  # (tile_h*tile_w,)
            # Cap individual alpha to avoid blowing the composite at near singularities
            alpha_flat = np.minimum(alpha_flat, 0.999)
            alpha_2d = alpha_flat.reshape(tile_h, tile_w)

            # Pre-multiplied alpha "over" composite (front-to-back).
            transmittance_2d = 1.0 - tile_alpha  # (tile_h, tile_w)
            contrib = transmittance_2d * alpha_2d  # how much of this splat shows through

            color = rgb_vals[s]  # (3,)
            tile_rgb += contrib[:, :, None] * color[None, None, :]
            tile_alpha += contrib

        framebuffer[y0:y1, x0:x1] = tile_rgb
        alpha_accum[y0:y1, x0:x1] = tile_alpha

    return framebuffer, alpha_accum
