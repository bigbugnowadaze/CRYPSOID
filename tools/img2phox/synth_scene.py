"""Synthetic textured-scene renderer for Phase F testing.

Builds a small ground-truth scene (cube + sphere + plane), renders it from N
camera poses to produce "photos" + ground-truth (CameraBundle, PointCloud).
This lets us validate every Phase F stage without needing real photos.

The scene is rendered as a thin point cloud (vertex colors) — we don't need a
proper rasterizer, just a debug projector that shows colored dots for each
point. That's enough to feed SfM and the blob optimizer.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Tuple

from .data_classes import (
    PhotoSet, CameraIntrinsics, CameraExtrinsics, CameraBundle, PointCloud,
)
from .load_photos import photoset_from_arrays


def build_ground_truth_scene(n_cube_pts: int = 800,
                             n_sphere_pts: int = 600,
                             n_plane_pts: int = 400,
                             seed: int = 7) -> PointCloud:
    """A textured cube + sphere + ground plane, all in a 2-unit world."""
    rng = np.random.default_rng(seed)
    parts = []

    # CUBE: textured with a checker pattern
    side = rng.uniform(-0.5, 0.5, size=(n_cube_pts, 3)).astype(np.float32)
    face_axis = rng.integers(0, 6, size=n_cube_pts)
    for i in range(n_cube_pts):
        ax = face_axis[i] // 2
        sign = 1.0 if face_axis[i] % 2 == 0 else -1.0
        side[i, ax] = 0.5 * sign
    side[:, 1] += 0.5  # sit on the ground plane
    chk = ((np.floor(side[:, 0] * 8) + np.floor(side[:, 2] * 8)) % 2.0).astype(np.float32)
    cube_color = np.stack([0.8 + 0.15*chk,                  # red varies with checker
                           0.3 + 0.20*chk,                  # green
                           0.25*np.ones_like(chk)], axis=1)  # blue constant
    parts.append((side, cube_color))

    # SPHERE: smoothly-shaded vertical-rainbow ball at (1.5, 0.5, 0.5)
    phi = rng.uniform(0, 2*np.pi, n_sphere_pts).astype(np.float32)
    theta = np.arccos(rng.uniform(-1, 1, n_sphere_pts)).astype(np.float32)
    r = 0.4
    pts = np.stack([r*np.sin(theta)*np.cos(phi) + 1.5,
                    r*np.cos(theta) + 0.5,
                    r*np.sin(theta)*np.sin(phi) + 0.5], axis=1).astype(np.float32)
    rainbow = np.stack([0.5 + 0.5*np.sin(theta*3),
                        0.5 + 0.5*np.sin(theta*3 + 2),
                        0.5 + 0.5*np.sin(theta*3 + 4)], axis=1).astype(np.float32)
    parts.append((pts, rainbow))

    # PLANE: ground (y=0) with checker
    px = rng.uniform(-2.0, 2.0, n_plane_pts).astype(np.float32)
    pz = rng.uniform(-2.0, 2.0, n_plane_pts).astype(np.float32)
    py = np.zeros_like(px)
    plane_chk = ((np.floor(px * 2) + np.floor(pz * 2)) % 2.0).astype(np.float32)
    plane_pts = np.stack([px, py, pz], axis=1)
    plane_color = np.stack([0.35 + 0.20*plane_chk,
                            0.35 + 0.20*plane_chk,
                            0.30 + 0.18*plane_chk], axis=1).astype(np.float32)
    parts.append((plane_pts, plane_color))

    xyz = np.concatenate([p[0] for p in parts], axis=0)
    rgb = np.concatenate([p[1] for p in parts], axis=0)
    return PointCloud(xyz=xyz, colors=rgb,
                      visibility=[set() for _ in range(xyz.shape[0])])


def make_orbit_cameras(n_cams: int = 8,
                       distance: float = 3.5,
                       height: float = 1.2,
                       fov_deg: float = 50.0,
                       width: int = 320,
                       height_px: int = 240,
                       look_at=(0.5, 0.5, 0.5)) -> CameraBundle:
    """Orbit `n_cams` cameras around `look_at` at `distance` and `height`."""
    intr = CameraIntrinsics.from_fov(fov_deg, width, height_px)
    look_at = np.asarray(look_at, dtype=np.float32)
    up = np.array([0, 1, 0], dtype=np.float32)
    extrinsics = []
    for k in range(n_cams):
        theta = 2*np.pi * k / n_cams
        eye = np.array([look_at[0] + distance*np.sin(theta),
                        height,
                        look_at[2] + distance*np.cos(theta)], dtype=np.float32)
        forward = look_at - eye
        forward /= np.linalg.norm(forward) + 1e-9
        right = np.cross(forward, up); right /= np.linalg.norm(right) + 1e-9
        cam_up = np.cross(right, forward)
        # World-to-camera rotation: rows are camera basis in world coords.
        # We need a *proper* rotation (det = +1) so that _mat_to_rotvec /
        # _rotvec_to_mat (used by bundle adjustment) round-trips correctly.
        # Convention: camera +x = right, +y = -cam_up (image-y goes down),
        # +z = forward (into scene).  This is right-handed with det(R)=+1
        # and matches the renderer's "z > 0 means in front" convention.
        R = np.stack([right, -cam_up, forward], axis=0).astype(np.float32)
        t = -R @ eye
        extrinsics.append(CameraExtrinsics(R=R, t=t))
    return CameraBundle(intrinsics=intr, extrinsics=extrinsics)


def render_pointcloud_to_photo(point_cloud: PointCloud,
                               intr: CameraIntrinsics,
                               extr: CameraExtrinsics,
                               point_radius_px: int = 2,
                               bg=(0.10, 0.12, 0.15)) -> np.ndarray:
    """Project + splat a sparse point cloud as colored dots. Returns (H, W, 3) float32 in [0, 1]."""
    H, W = intr.height, intr.width
    img = np.full((H, W, 3), bg, dtype=np.float32)

    cam_pts = extr.world_to_cam(point_cloud.xyz)
    z = cam_pts[:, 2]
    valid = z > 0.05
    cam_pts = cam_pts[valid]
    colors = point_cloud.colors[valid] if point_cloud.colors is not None else \
             np.full((cam_pts.shape[0], 3), 0.7, dtype=np.float32)

    px = (cam_pts[:, 0] / cam_pts[:, 2]) * intr.focal_x + intr.cx
    py = (cam_pts[:, 1] / cam_pts[:, 2]) * intr.focal_y + intr.cy
    # No flip: with R = [right, -cam_up, forward] the y axis is already image-down.

    # Sort back-to-front so closer points overwrite
    order = np.argsort(-cam_pts[:, 2])
    px, py, colors = px[order], py[order], colors[order]

    pxi = px.astype(np.int32); pyi = py.astype(np.int32)
    r = point_radius_px
    for i in range(len(pxi)):
        x0, x1 = max(0, pxi[i]-r), min(W, pxi[i]+r+1)
        y0, y1 = max(0, pyi[i]-r), min(H, pyi[i]+r+1)
        if x1 <= x0 or y1 <= y0: continue
        img[y0:y1, x0:x1] = colors[i]
    return img


def render_synthetic_scene(point_cloud: PointCloud,
                            cameras: CameraBundle) -> PhotoSet:
    """Render the synthetic scene from each camera in the bundle."""
    arrays = []
    arrays = []
    for k, extr in enumerate(cameras.extrinsics):
        arrays.append(render_pointcloud_to_photo(point_cloud, cameras.intrinsics, extr))
    return photoset_from_arrays(arrays, names=[f"cam_{k:02d}" for k in range(len(cameras))])
