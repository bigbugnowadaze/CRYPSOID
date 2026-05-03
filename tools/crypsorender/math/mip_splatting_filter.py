"""Bar 2.4 — Mip-Splatting pre-filter (Yu et al. 2024).

When a 3D Gaussian splat projects to less than ~1 pixel on screen, it aliases
badly: tiny splats flicker as the camera moves, because they undersample the
pixel grid. The Mip-Splatting fix is to widen the 2D screen-space covariance
of small splats up to a minimum size, while attenuating their opacity to keep
energy roughly conserved.

Per the paper:

    Build sigma_screen_pre  : the splat's natural 2D covariance from EWA projection
    Add a 2D Gaussian filter: sigma_filter² = (kernel_radius_pixels)²
    Final covariance        : sigma_screen + sigma_filter*I
    Opacity attenuation     : alpha *= sqrt( det(sigma_screen) / det(sigma_screen + sigma_filter) )

We don't need the full antialiasing 3D-Gaussian filter for our purposes; the
screen-space 2D pre-filter alone removes the worst flicker.

The kernel_radius is per-splat: derived from the v33 mip_zoom byte we
populated this session. Splats whose recovered sigma is small relative to the
camera setup get a bigger filter.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple

# Bar-1 default minimum filter radius in pixels — empirical tradeoff.
# Smaller -> sharper but more aliasing on tiny splats.
# Larger  -> softer but eliminates flicker entirely.
DEFAULT_MIN_FILTER_PX = 0.5


def apply_mip_splatting_filter(cov_2d: np.ndarray,
                               opa: np.ndarray,
                               radii: np.ndarray,
                               min_filter_px: float = DEFAULT_MIN_FILTER_PX,
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add a 2D pre-filter to all splats; widen tiny ones, attenuate opacity.

    Args:
        cov_2d:  (M, 2, 2) screen-space 2D covariance (from EWA project).
        opa:     (M,) per-splat opacity.
        radii:   (M,) bounding radii (pixels). Will be re-computed for the
                 filtered covariance and returned.
        min_filter_px: minimum pre-filter Gaussian standard deviation, in pixels.
                 Acts uniformly on all splats; small splats see relatively
                 more widening, large splats almost none.

    Returns:
        (cov_2d_filtered, opa_filtered, radii_filtered).
    """
    M = cov_2d.shape[0]
    sigma2 = float(min_filter_px ** 2)

    # Add isotropic 2D Gaussian to covariance: sigma + sigma_filter² * I
    cov_filtered = cov_2d.copy().astype(np.float32)
    cov_filtered[:, 0, 0] += sigma2
    cov_filtered[:, 1, 1] += sigma2

    # Determinant ratio (energy-conservation alpha attenuation)
    det_pre = (cov_2d[:, 0, 0] * cov_2d[:, 1, 1]
               - cov_2d[:, 0, 1] * cov_2d[:, 1, 0]).astype(np.float32)
    det_post = (cov_filtered[:, 0, 0] * cov_filtered[:, 1, 1]
                - cov_filtered[:, 0, 1] * cov_filtered[:, 1, 0]).astype(np.float32)
    # Sqrt ratio is the alpha multiplier
    ratio = np.sqrt(np.maximum(det_pre, 1e-12) / np.maximum(det_post, 1e-12))
    opa_filtered = (opa.astype(np.float32) * ratio).astype(np.float32)

    # Recompute bounding radii (3-sigma) for the widened covariance
    # Eigenvalues of the new 2x2 → max -> sqrt -> 3*
    a = cov_filtered[:, 0, 0]
    b = cov_filtered[:, 0, 1]
    d = cov_filtered[:, 1, 1]
    tr = a + d
    diff = np.sqrt(np.maximum((a - d) ** 2 + 4.0 * b * b, 0.0))
    lam_max = 0.5 * (tr + diff)
    radii_filtered = (3.0 * np.sqrt(np.maximum(lam_max, 0.0))).astype(np.float32)

    return cov_filtered.astype(np.float32), opa_filtered, radii_filtered


def per_splat_filter_radius(mip_zoom_u8: np.ndarray,
                            depths: np.ndarray,
                            focal: float,
                            min_px: float = 0.5,
                            ) -> np.ndarray:
    """Per-splat filter radius in pixels, from the v33 mip_zoom byte.

    Decoder of mip_zoom (matches material_codec.decode_mip_zoom):
        sigma_world = 2^((u8/8) - 8) / 1024

    Splat's projected pixel size: (focal * sigma_world / depth).
    Filter radius ≥ max(min_px, target_min_px - projected_size).
    """
    # Decode sigma per Mip-Splatting LOD
    sigma_world = np.power(2.0,
                           mip_zoom_u8.astype(np.float32) / 8.0 - 8.0) / 1024.0
    pix = focal * sigma_world / np.maximum(np.abs(depths), 0.05)
    radius = np.maximum(min_px, min_px * 1.5 - pix)
    return radius.astype(np.float32)


def invert_2x2_batch(M: np.ndarray) -> np.ndarray:
    """Invert (M, 2, 2). Returns (M, 2, 2)."""
    a = M[:, 0, 0]; b = M[:, 0, 1]; c = M[:, 1, 0]; d = M[:, 1, 1]
    det = a * d - b * c
    inv_det = 1.0 / np.where(np.abs(det) < 1e-12, 1e-12, det)
    out = np.empty_like(M, dtype=np.float32)
    out[:, 0, 0] =  d * inv_det
    out[:, 0, 1] = -b * inv_det
    out[:, 1, 0] = -c * inv_det
    out[:, 1, 1] =  a * inv_det
    return out
