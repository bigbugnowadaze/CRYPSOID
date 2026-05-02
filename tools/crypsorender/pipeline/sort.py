"""Depth sorting of splats."""

import numpy as np


def sort_splats_by_depth(depths: np.ndarray, splat_indices: np.ndarray) -> np.ndarray:
    """Sort splats by depth (back-to-front).

    Args:
        depths: (m,) depth values
        splat_indices: (m,) indices into original splat array

    Returns:
        (m,) sorted indices into splat_indices (for front-to-back iteration)
    """
    # Sort in descending order (back-to-front)
    order = np.argsort(-depths, kind="stable")
    return order
