"""Camera transforms: world→camera→NDC→pixel."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CameraParams:
    """Camera parameters for rendering."""
    yaw_deg: float = 35.0
    pitch_deg: float = 18.0
    distance: float = 2.4  # as fraction of scene radius
    fov_deg: float = 42.0
    size: int = 1024  # (size, size) pixels
    near: float = 0.01
    far: float = 1000.0


class Camera:
    """Camera with world→camera→NDC transforms."""

    def __init__(self, xyz: np.ndarray, params: CameraParams):
        """Initialize camera for a scene.

        Args:
            xyz: (n, 3) world positions of splats (used to compute framing)
            params: CameraParams
        """
        self.params = params
        self.size = params.size

        # Compute scene center and radius
        self.center = xyz.mean(axis=0)
        pts_rel = xyz - self.center
        self.radius = np.linalg.norm(pts_rel, axis=1).max()

        # Compute eye position from yaw, pitch, distance
        yaw = math.radians(params.yaw_deg)
        pitch = math.radians(params.pitch_deg)
        eye_dir = np.array(
            [
                math.cos(pitch) * math.sin(yaw),
                math.sin(pitch),
                math.cos(pitch) * math.cos(yaw),
            ],
            dtype=np.float32,
        )
        self.eye = self.center + eye_dir * self.radius * params.distance

        # Build view matrix (world→camera)
        forward = self.center - self.eye
        forward = forward / (np.linalg.norm(forward) + 1e-9)
        world_up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right = right / (np.linalg.norm(right) + 1e-9)
        up = np.cross(right, forward)

        # Camera basis (cols are right, up, -forward in camera frame)
        self.cam_right = right
        self.cam_up = up
        self.cam_forward = forward

        # View matrix: transform world→camera
        # cam_pos = (world @ right, world @ up, world @ -forward)
        self.view_rot = np.stack(
            [self.cam_right, self.cam_up, -self.cam_forward], axis=1
        ).astype(
            np.float32
        )  # (3, 3)

        # Compute focal length
        self.focal = 0.5 * params.size / math.tan(math.radians(params.fov_deg) / 2)

    def world_to_cam(self, xyz: np.ndarray) -> np.ndarray:
        """Transform world coords to camera space.

        Args:
            xyz: (n, 3) world positions

        Returns:
            (n, 3) camera-space positions
        """
        rel = xyz - self.eye
        cam_pos = rel @ self.view_rot  # (n, 3)
        return cam_pos.astype(np.float32)

    def projection_jacobian(self, centers_3d_cam: np.ndarray) -> np.ndarray:
        """Compute perspective projection Jacobian.

        For perspective projection (x, y, z) -> (f*x/z, f*y/z):
        J = [[f/z, 0, -f*x/z²],
             [0, f/z, -f*y/z²]]

        Args:
            centers_3d_cam: (n, 3) camera-space splat centers

        Returns:
            (n, 2, 3) Jacobian matrices
        """
        n = centers_3d_cam.shape[0]
        x = centers_3d_cam[:, 0]
        y = centers_3d_cam[:, 1]
        z = centers_3d_cam[:, 2]
        z_sq = z * z
        z_safe = np.where(np.abs(z) < 1e-6, 1e-6, z)

        jac = np.zeros((n, 2, 3), dtype=np.float32)
        # (px, py) = (focal*x/(-z) + cx, -focal*y/(-z) + cy) where cam looks down -z.
        # J = d(px,py)/d(x,y,z)
        jac[:, 0, 0] = -self.focal / z_safe
        jac[:, 0, 2] =  self.focal * x / z_sq
        jac[:, 1, 1] =  self.focal / z_safe
        jac[:, 1, 2] = -self.focal * y / z_sq

        return jac

    def world_to_pixel(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project world coords to pixel space.

        Args:
            xyz: (n, 3) world positions

        Returns:
            (px, py): (n,) pixel coordinates (may be out-of-bounds)
        """
        cam_pos = self.world_to_cam(xyz)
        px = (cam_pos[:, 0] / cam_pos[:, 2]) * self.focal + self.size / 2
        py = -(cam_pos[:, 1] / cam_pos[:, 2]) * self.focal + self.size / 2
        return px.astype(np.float32), py.astype(np.float32)
