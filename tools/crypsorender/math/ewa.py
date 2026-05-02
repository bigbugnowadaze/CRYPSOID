"""EWA (Elliptical Weighted Average) 3D->2D covariance projection.

Standard 3DGS EWA formula. Returns 2D centers IN PIXEL SPACE and the 2D
covariance in pixel^2 units (so the rasterizer can operate directly in pixels).
"""

from __future__ import annotations
import numpy as np
from .quat import quat_to_rot_matrix


def build_3d_cov(scales: np.ndarray, quats: np.ndarray) -> np.ndarray:
    """Build 3D covariance matrices from log-space scales and unit quaternions.

    3DGS convention: stored "scales" are log-space, so linear sigma = exp(scale).
    Sigma = R @ diag(exp(2*scales)) @ R^T  (a 3x3 SPD matrix).
    """
    scales_sq = np.exp(2.0 * scales).astype(np.float32)        # (n, 3)
    rot = quat_to_rot_matrix(quats)                            # (n, 3, 3)
    # Sigma = R @ diag(scales_sq) @ R^T
    scaled_RT = scales_sq[:, :, None] * rot.transpose(0, 2, 1) # (n, 3, 3)  diag(s) @ R^T
    cov_3d = rot @ scaled_RT                                   # (n, 3, 3)
    return cov_3d.astype(np.float32)


def ewa_project(
    cam_to_pixel_jac: np.ndarray,    # (n, 2, 3) Jacobian of (x_cam, y_cam, z_cam) -> (px, py)
    world_to_cam_rot: np.ndarray,    # (n, 3, 3) world->camera rotation (broadcast or per-splat)
    centers_3d_cam: np.ndarray,      # (n, 3) splat centers in camera space
    cov_3d: np.ndarray,              # (n, 3, 3) world-space splat covariances
    focal: float,
    size: int,
):
    """Project covariance to pixel space and centers to pixel coordinates.

    Sigma_pix = J . W . Sigma_world . W^T . J^T

    The returned centers are PIXEL coordinates (origin top-left), with Y flipped
    so positive world-Y becomes upward = smaller pixel-Y.
    """
    n = centers_3d_cam.shape[0]

    # World -> camera frame for the covariance: Sigma_cam = W . Sigma_world . W^T
    if world_to_cam_rot.ndim == 2:
        # broadcast a single (3,3) to per-splat
        W = np.broadcast_to(world_to_cam_rot[None], (n, 3, 3)).astype(np.float32)
    else:
        W = world_to_cam_rot.astype(np.float32)
    cov_cam = W @ cov_3d @ W.transpose(0, 2, 1)               # (n, 3, 3)

    # Camera -> pixel projection of covariance: Sigma_pix = J . Sigma_cam . J^T
    cov_pix = cam_to_pixel_jac @ cov_cam @ cam_to_pixel_jac.transpose(0, 2, 1)  # (n, 2, 2)

    # Add a small isotropic dilation so single-pixel splats anti-alias smoothly
    # (matches Inria reference's "low-pass filter" trick).
    cov_pix[:, 0, 0] += 0.1
    cov_pix[:, 1, 1] += 0.1

    # Centers in pixel space.  Camera looks down +z (depths > 0 = in front).
    # Image convention: origin top-left, +x right, +y down -> Y flip on world up.
    x = centers_3d_cam[:, 0]
    y = centers_3d_cam[:, 1]
    # Camera looks down -z (so cam_z is negative for in-front splats); use +depth = -z.
    depth = -centers_3d_cam[:, 2]
    depth = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    px =  focal * (x / depth) + 0.5 * size
    py = -focal * (y / depth) + 0.5 * size  # Y flip for image-coord convention
    centers_2d = np.stack([px, py], axis=1).astype(np.float32)

    return centers_2d, cov_pix.astype(np.float32)


def invert_2x2(cov_2d: np.ndarray) -> np.ndarray:
    a = cov_2d[:, 0, 0]; b = cov_2d[:, 0, 1]; d = cov_2d[:, 1, 1]
    det = a * d - b * b
    det = np.where(np.abs(det) < 1e-10, 1e-10, det)
    inv = np.zeros_like(cov_2d)
    inv[:, 0, 0] =  d / det
    inv[:, 0, 1] = -b / det
    inv[:, 1, 0] = -b / det
    inv[:, 1, 1] =  a / det
    return inv.astype(np.float32)


def get_radius_2d(cov_2d: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """3-sigma footprint radius from a 2x2 SPD covariance, in pixels."""
    a = cov_2d[:, 0, 0]; b = cov_2d[:, 0, 1]; d = cov_2d[:, 1, 1]
    trace = a + d
    det = a * d - b * b
    det = np.maximum(det, 0.0)
    disc = np.maximum(trace * trace - 4.0 * det, 0.0)
    lambda_max = 0.5 * (trace + np.sqrt(disc))
    lambda_max = np.maximum(lambda_max, 0.0)
    return (sigma * np.sqrt(lambda_max)).astype(np.float32)
