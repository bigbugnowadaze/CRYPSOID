"""v33 Phase-2 — multi-view photometric floater detection (GS-2M-style).

The Phase-1 heuristic in material_codec.derive_material_hints uses single-frame
signals (SH magnitudes, kNN edge length, opacity, κ). It catches *structural*
floaters but misses the dense halo because dense halo splats look like real
surface from a single angle.

Phase 2 catches more floaters by checking *photometric agreement with neighbors
across multiple viewing angles*. The hypothesis (per GS-2M):

    A real surface splat's apparent color (SH-decoded at a view direction)
    should correlate strongly with its k-nearest neighbors' decoded colors at
    the same view directions. A floater's color sequence is uncorrelated with
    its neighbors — it's "noise" relative to the local surface.

Algorithm (single CPU pass, no source images required — uses the SH itself):

    1. Generate K=12 view directions on a Fibonacci sphere.
    2. For each splat, decode its SH at all K directions → (N, K, 3) RGB array.
    3. For each splat, gather neighbors' decoded RGB → (N, k, K, 3).
    4. Compute per-splat correlation between its color sequence and the
       MEAN of its neighbors' color sequences (across views).
    5. Low correlation (or negative) → photometric outlier → floater candidate.

This requires `sb.sh_dc + sb.sh_rest`, the v31 kNN edges chunk, and the SH eval
basis (already in crypsorender.math.sh).
"""

from __future__ import annotations
import numpy as np


def fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n approximately uniform unit vectors via Fibonacci spiral."""
    indices = np.arange(0, n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * indices
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def decode_sh_at_directions(sh_dc: np.ndarray,
                            sh_rest: np.ndarray,
                            directions: np.ndarray) -> np.ndarray:
    """Evaluate per-splat SH at K view directions.

    Args:
        sh_dc: (N, 3) DC coefficients
        sh_rest: (N, 45) bands 1-3 (15 coefs × 3 channels), packed
        directions: (K, 3) unit view directions

    Returns:
        rgb: (N, K, 3) decoded RGB at each direction, in approximate [0, 1] domain.
    """
    # SH basis (degrees 0-3) for K directions
    K = directions.shape[0]
    x = directions[:, 0]; y = directions[:, 1]; z = directions[:, 2]

    # Real SH basis up to degree 3 (16 basis functions)
    SH_C0 = 0.28209479177387814   # 1 / (2 * sqrt(pi))
    SH_C1 = 0.4886025119029199    # sqrt(3 / (4*pi))
    SH_C2 = [
        1.0925484305920792,       # sqrt(15/pi)/2
        -1.0925484305920792,
        0.31539156525252005,      # sqrt(5/pi)/4
        -1.0925484305920792,
        0.5462742152960396,
    ]
    SH_C3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435,
    ]

    # (K, 16) basis matrix
    basis = np.zeros((K, 16), dtype=np.float32)
    basis[:, 0] = SH_C0
    basis[:, 1] = -SH_C1 * y
    basis[:, 2] = SH_C1 * z
    basis[:, 3] = -SH_C1 * x
    basis[:, 4] = SH_C2[0] * x * y
    basis[:, 5] = SH_C2[1] * y * z
    basis[:, 6] = SH_C2[2] * (3 * z*z - 1)
    basis[:, 7] = SH_C2[3] * x * z
    basis[:, 8] = SH_C2[4] * (x*x - y*y)
    basis[:, 9]  = SH_C3[0] * y * (3*x*x - y*y)
    basis[:,10] = SH_C3[1] * x * y * z
    basis[:,11] = SH_C3[2] * y * (5*z*z - 1)
    basis[:,12] = SH_C3[3] * z * (5*z*z - 3)
    basis[:,13] = SH_C3[4] * x * (5*z*z - 1)
    basis[:,14] = SH_C3[5] * z * (x*x - y*y)
    basis[:,15] = SH_C3[6] * x * (x*x - 3*y*y)

    # sh_rest is (N, 45) = 15 coefs × 3 channels, packed channel-major
    # Convention from existing code: f_rest[0..14] = R band1-3, [15..29] = G, [30..44] = B
    N = sh_dc.shape[0]
    sh_rest_per_channel = sh_rest.reshape(N, 3, 15)   # (N, channel, coef_idx)
    # Combine band 0 (DC) + bands 1-3
    sh_full = np.concatenate([sh_dc[:, :, None], sh_rest_per_channel], axis=2)  # (N, 3, 16)
    # Evaluate: rgb[n, k, c] = sum_b basis[k, b] * sh_full[n, c, b]
    rgb = np.einsum('kb,ncb->nkc', basis, sh_full)
    # Convert from SH-space to [0, 1] color (offset by 0.5, clamp)
    rgb = (rgb + 0.5).clip(0.0, 1.0)
    return rgb.astype(np.float32)


def photometric_floater_score(sh_dc: np.ndarray,
                               sh_rest: np.ndarray,
                               neighbors: np.ndarray,
                               K: int = 12) -> np.ndarray:
    """Per-splat: correlation of color-sequence-across-views with mean of neighbors'.

    Args:
        sh_dc: (N, 3)
        sh_rest: (N, 45)
        neighbors: (N, k) uint32 indices into the same arrays
        K: number of view directions to sample

    Returns:
        floater_score: (N,) float32 in [0, 1] — 0 = strong agreement with neighbors
                       (real surface), 1 = uncorrelated (likely floater).
    """
    N, k = neighbors.shape
    dirs = fibonacci_sphere(K)
    # Decode all splats at K directions
    rgb_self = decode_sh_at_directions(sh_dc, sh_rest, dirs)        # (N, K, 3)
    # Mean of neighbors' decoded colors at each direction
    rgb_nbr_mean = rgb_self[neighbors].mean(axis=1)                 # (N, K, 3)

    # Per-splat correlation: flatten (K, 3) into a 3K-vector
    a = rgb_self.reshape(N, K * 3)
    b = rgb_nbr_mean.reshape(N, K * 3)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    a_norm = np.linalg.norm(a, axis=1).clip(min=1e-9)
    b_norm = np.linalg.norm(b, axis=1).clip(min=1e-9)
    corr = np.einsum('ni,ni->n', a, b) / (a_norm * b_norm)         # in [-1, 1]

    # Floater score: 1 - corr (high when uncorrelated)
    score = (1.0 - corr) * 0.5                                      # [0, 1]
    return score.astype(np.float32)
