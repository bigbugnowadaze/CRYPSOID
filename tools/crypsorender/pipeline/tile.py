"""16×16 tile binning for efficient rasterization."""

import numpy as np


def bin_splats_to_tiles(
    centers_2d: np.ndarray,
    radii: np.ndarray,
    size: int,
    tile_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin splats into 16×16 tiles.

    Args:
        centers_2d: (m, 2) screen-space centers
        radii: (m,) 3-sigma footprint radii
        size: image dimension (size × size pixels)
        tile_size: tile size in pixels (default 16)

    Returns:
        (tile_grid, tile_lists, tile_counts)
        - tile_grid: (n_tiles, n_tiles, max_splats_per_tile) int32 array of splat indices (-1 padding)
        - tile_lists: list of lists, each containing splat indices for that tile
        - tile_counts: (n_tiles, n_tiles) int32 array of splat counts per tile
    """
    n_tiles = (size + tile_size - 1) // tile_size
    m = centers_2d.shape[0]

    # Allocate output structure
    # Use a dict to collect splats per tile
    tile_dict = {}
    for ty in range(n_tiles):
        for tx in range(n_tiles):
            tile_dict[(ty, tx)] = []

    # For each splat, find overlapping tiles
    for i in range(m):
        cx, cy = centers_2d[i]
        r = radii[i]

        # Bounding box of splat
        x_min = int(np.floor(cx - r))
        x_max = int(np.ceil(cx + r))
        y_min = int(np.floor(cy - r))
        y_max = int(np.ceil(cy + r))

        # Clamp to image bounds
        x_min = max(0, x_min)
        x_max = min(size - 1, x_max)
        y_min = max(0, y_min)
        y_max = min(size - 1, y_max)

        # Tile indices
        tx_min = x_min // tile_size
        tx_max = x_max // tile_size
        ty_min = y_min // tile_size
        ty_max = y_max // tile_size

        # Add splat to all overlapped tiles
        for ty in range(ty_min, ty_max + 1):
            for tx in range(tx_min, tx_max + 1):
                if ty < n_tiles and tx < n_tiles:
                    tile_dict[(ty, tx)].append(i)

    # Convert to arrays for efficient access
    tile_lists = []
    tile_counts = np.zeros((n_tiles, n_tiles), dtype=np.int32)
    for ty in range(n_tiles):
        for tx in range(n_tiles):
            splats = tile_dict[(ty, tx)]
            tile_counts[ty, tx] = len(splats)
            tile_lists.append(np.array(splats, dtype=np.int32))

    return tile_lists, tile_counts, n_tiles
