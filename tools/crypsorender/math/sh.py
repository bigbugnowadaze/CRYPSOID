"""Spherical harmonics (SH) basis evaluation up to degree 3, real basis."""

from __future__ import annotations
import numpy as np


def eval_sh_basis(directions: np.ndarray) -> np.ndarray:
    """Evaluate real-orthonormal SH basis (degrees 0..3 -> 16 coeffs).

    Args:
        directions: (..., 3) unit vectors.

    Returns:
        (..., 16) basis values.
    """
    original_shape = directions.shape[:-1]
    dirs_flat = directions.reshape(-1, 3)
    n = dirs_flat.shape[0]
    x = dirs_flat[:, 0]; y = dirs_flat[:, 1]; z = dirs_flat[:, 2]
    sh = np.zeros((n, 16), dtype=np.float32)
    # l=0
    sh[:, 0] = 0.28209479177387814
    # l=1
    sh[:, 1] = -0.4886025119029199 * y
    sh[:, 2] = 0.4886025119029199 * z
    sh[:, 3] = -0.4886025119029199 * x
    # l=2
    xy = x*y; yz = y*z; xz = x*z
    sh[:, 4] = 1.0925484305920792 * xy
    sh[:, 5] = -1.0925484305920792 * yz
    sh[:, 6] = 0.31539156525251999 * (3*z*z - 1)
    sh[:, 7] = -1.0925484305920792 * xz
    sh[:, 8] = 0.5462742152960395 * (x*x - y*y)
    # l=3
    sh[:, 9]  = -0.5900435899266435 * y * (3*x*x - y*y)
    sh[:, 10] =  2.890611442640554 * xy * z
    sh[:, 11] = -0.4570457994644658 * y * (5*z*z - 1)
    sh[:, 12] =  0.3731763325901154 * z * (5*z*z - 3)
    sh[:, 13] = -0.4570457994644658 * x * (5*z*z - 1)
    sh[:, 14] =  1.445305721320277  * z * (x*x - y*y)
    sh[:, 15] = -0.5900435899266435 * x * (x*x - 3*y*y)
    return sh.reshape(*original_shape, 16)


def eval_sh_color(sh_dc, sh_rest, directions, view_clip=True):
    """Evaluate SH-modulated RGB.

    The DC chunk in our v25/v28 loaders is ALREADY in linear [0,1] color space
    (decoded from u8 via x/255). So we use it directly as the base color and add
    higher-degree contributions on top, NOT the raw 3DGS f_dc convention.
    Caller can pass sh_dc as (n, 3) color in [0,1].

    Args:
        sh_dc: (n, 3) base RGB color in [0,1].
        sh_rest: (n, 45) or None — degrees 1..3 SH coefficients in raw float space.
                 Layout: 15 coefficients per channel, channel-major.
        directions: (n, 3) unit view directions in WORLD space.
        view_clip: clamp final RGB to [0,1].

    Returns:
        (n, 3) float32 RGB in [0,1] (after optional clip).
    """
    sh_dc = np.asarray(sh_dc, dtype=np.float32)
    if sh_dc.ndim != 2 or sh_dc.shape[1] != 3:
        raise ValueError(f"sh_dc must be (n,3); got {sh_dc.shape}")
    n = sh_dc.shape[0]
    directions = np.asarray(directions, dtype=np.float32).reshape(n, 3)

    rgb = sh_dc.copy()

    if sh_rest is not None:
        sh_rest = np.asarray(sh_rest, dtype=np.float32)
        if sh_rest.shape != (n, 45):
            raise ValueError(f"sh_rest must be (n,45); got {sh_rest.shape}")
        basis = eval_sh_basis(directions)            # (n, 16)
        basis_higher = basis[:, 1:16]                # (n, 15)
        # 3DGS layout: f_rest[i] is laid out channel-major: 15 coeffs for R, 15 for G, 15 for B.
        sh_coeffs = sh_rest.reshape(n, 3, 15)        # (n, channel, coeff)
        # contribution[i, c] = sum_k sh_coeffs[i, c, k] * basis_higher[i, k]
        contribution = np.einsum("nck,nk->nc", sh_coeffs, basis_higher).astype(np.float32)
        rgb = rgb + contribution

    if view_clip:
        rgb = np.clip(rgb, 0.0, 1.0)
    return rgb.astype(np.float32)
