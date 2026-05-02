"""Quaternion operations for rotation matrices."""

import numpy as np


def quat_to_rot_matrix(quats: np.ndarray) -> np.ndarray:
    """Convert unit quaternions (wxyz format) to 3x3 rotation matrices.

    Args:
        quats: (n, 4) array of unit quaternions in wxyz order

    Returns:
        (n, 3, 3) array of rotation matrices
    """
    w = quats[:, 0]
    x = quats[:, 1]
    y = quats[:, 2]
    z = quats[:, 3]

    n = quats.shape[0]
    rot = np.zeros((n, 3, 3), dtype=np.float32)

    # Formula from standard quaternion-to-matrix conversion
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - z * w)
    rot[:, 0, 2] = 2 * (x * z + y * w)

    rot[:, 1, 0] = 2 * (x * y + z * w)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - x * w)

    rot[:, 2, 0] = 2 * (x * z - y * w)
    rot[:, 2, 1] = 2 * (y * z + x * w)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return rot
