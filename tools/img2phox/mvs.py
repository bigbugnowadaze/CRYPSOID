"""F.6 — Dense Multi-View Stereo via OpenCV StereoSGBM.

Strategy:
  For each pair (i, j) of cameras with sufficient baseline + overlap:
    1. Rectify the pair (compute rectifying homographies + reprojection matrix Q).
    2. Run StereoSGBM (semi-global block matching) on the rectified pair.
    3. Reproject disparity to 3D via Q -> dense point cloud for that pair.
    4. Filter by depth range + reprojection confidence.
  Fuse all per-pair clouds: voxel-grid downsample to remove duplicates.

This is the cheap-but-working version of MVS. It's not PatchMatch-MVS quality,
but for "good enough to seed a dense BlobBundle" it works.
"""
from __future__ import annotations
import time
import numpy as np
import cv2
from typing import List

from .data_classes import (
    PhotoSet, CameraBundle, CameraIntrinsics, CameraExtrinsics, PointCloud,
)


def select_pairs_for_stereo(cameras: CameraBundle,
                              max_pairs: int = 12,
                              min_baseline: float = 0.05,
                              max_baseline_ratio: float = 0.6) -> List[tuple]:
    """Pick the (i, j) camera pairs with healthy baseline for stereo.

    Healthy = enough translation difference to produce parallax, but not so much
    that there's no overlap. Sorted by score (best pairs first).
    """
    N = len(cameras)
    eyes = np.array([e.cam_position for e in cameras.extrinsics])
    scene_radius = np.linalg.norm(eyes - eyes.mean(axis=0), axis=1).mean()
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            b = np.linalg.norm(eyes[i] - eyes[j])
            if b < min_baseline: continue
            if b > max_baseline_ratio * scene_radius * 4: continue
            score = -abs(b - 0.4 * scene_radius)
            pairs.append(((i, j), score))
    pairs.sort(key=lambda x: -x[1])
    return [p[0] for p in pairs[:max_pairs]]


def rectify_and_compute_disparity(photo_l, photo_r,
                                    extr_l: CameraExtrinsics,
                                    extr_r: CameraExtrinsics,
                                    intr: CameraIntrinsics,
                                    sgbm_window: int = 5,
                                    min_disp: int = 0,
                                    num_disp: int = 64) -> tuple:
    """Rectify L/R pair + compute disparity. Returns (disparity, Q, valid_mask)."""
    H, W = photo_l.height, photo_l.width
    K = intr.K
    # Relative pose: cam_R_in_cam_L frame
    R_lr = extr_r.R @ extr_l.R.T
    t_lr = extr_r.t - R_lr @ extr_l.t
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K, np.zeros(5), K, np.zeros(5),
        (W, H), R_lr, t_lr,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    map_lx, map_ly = cv2.initUndistortRectifyMap(K, np.zeros(5), R1, P1, (W, H), cv2.CV_32FC1)
    map_rx, map_ry = cv2.initUndistortRectifyMap(K, np.zeros(5), R2, P2, (W, H), cv2.CV_32FC1)

    img_l8 = (photo_l.image.clip(0, 1) * 255).astype(np.uint8)
    img_r8 = (photo_r.image.clip(0, 1) * 255).astype(np.uint8)
    rect_l = cv2.remap(img_l8, map_lx, map_ly, cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r8, map_rx, map_ry, cv2.INTER_LINEAR)
    gray_l = cv2.cvtColor(rect_l, cv2.COLOR_RGB2GRAY)
    gray_r = cv2.cvtColor(rect_r, cv2.COLOR_RGB2GRAY)

    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp, numDisparities=num_disp, blockSize=sgbm_window,
        P1=8 * 3 * sgbm_window * sgbm_window,
        P2=32 * 3 * sgbm_window * sgbm_window,
        disp12MaxDiff=2, uniquenessRatio=8,
        speckleWindowSize=50, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp16 = sgbm.compute(gray_l, gray_r)        # int16 disparity * 16
    disp = disp16.astype(np.float32) / 16.0
    valid = (disp > 0)
    return disp, Q, valid, rect_l, R1


def disparity_to_world_points(disp, Q, valid_mask, rect_l_color,
                                R1_l_world: np.ndarray,
                                extr_l: CameraExtrinsics,
                                max_depth: float = 50.0) -> tuple:
    """Reproject SGBM disparity through Q to 3D in the rectified-left frame,
    then transform back to world. Returns (xyz_world, rgb)."""
    pts3d_rect = cv2.reprojectImageTo3D(disp, Q)
    pts3d_rect[~valid_mask] = 0
    # Filter sane depths
    z = pts3d_rect[..., 2]
    finite = np.isfinite(z) & valid_mask & (z > 0.05) & (z < max_depth)
    pts = pts3d_rect[finite]
    if rect_l_color.dtype == np.uint8:
        rgb = rect_l_color[finite].astype(np.float32) / 255.0
    else:
        rgb = rect_l_color[finite]

    # The "rectified" frame is the left camera's orientation rotated by R1.
    # To get back to the original left-camera frame: R1.T @ p_rect.
    pts_cam_l = (R1_l_world.T @ pts.T).T

    # Then world: world_point = R_l.T @ (p_cam_l - t_l)
    R = extr_l.R
    t = extr_l.t
    pts_world = (R.T @ (pts_cam_l - t).T).T

    return pts_world.astype(np.float32), rgb.astype(np.float32)


def voxel_downsample(xyz: np.ndarray, rgb: np.ndarray, voxel_size: float = 0.02
                      ) -> tuple:
    """Voxel-grid downsample. Average xyz + rgb within each voxel cell."""
    if len(xyz) == 0:
        return xyz, rgb
    keys = np.floor(xyz / voxel_size).astype(np.int64)
    flat_keys = (keys[:, 0].astype(np.int64) * 1000003
                 ^ keys[:, 1].astype(np.int64) * 9176899
                 ^ keys[:, 2].astype(np.int64) * 50331653)
    order = np.argsort(flat_keys)
    flat_keys = flat_keys[order]; xyz = xyz[order]; rgb = rgb[order]
    breaks = np.where(np.diff(flat_keys) != 0)[0] + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(flat_keys)]])
    out_xyz = np.zeros((len(starts), 3), dtype=np.float32)
    out_rgb = np.zeros((len(starts), 3), dtype=np.float32)
    for i, (s, e) in enumerate(zip(starts, ends)):
        out_xyz[i] = xyz[s:e].mean(axis=0)
        out_rgb[i] = rgb[s:e].mean(axis=0)
    return out_xyz, out_rgb


def run_dense_mvs(photoset: PhotoSet, cameras: CameraBundle,
                    sparse_cloud: PointCloud,
                    max_pairs: int = 8,
                    voxel_size: float = 0.03,
                    sgbm_window: int = 5,
                    num_disp: int = 64,
                    verbose: bool = True) -> PointCloud:
    """Run dense MVS on selected pairs, fuse via voxel downsample, return PointCloud.

    Includes the sparse cloud as a baseline so we never lose points the SfM
    found that MVS missed.
    """
    t0 = time.perf_counter()
    pairs = select_pairs_for_stereo(cameras, max_pairs=max_pairs)
    if verbose:
        print(f"  [F.6] selected {len(pairs)} stereo pairs from {len(cameras)} cameras")

    intr = cameras.intrinsics
    all_xyz = [sparse_cloud.xyz]
    all_rgb = [sparse_cloud.colors if sparse_cloud.colors is not None
               else np.full((len(sparse_cloud), 3), 0.5, dtype=np.float32)]

    for k, (i, j) in enumerate(pairs):
        if i >= len(photoset) or j >= len(photoset): continue
        try:
            disp, Q, valid, rect_l, R1 = rectify_and_compute_disparity(
                photoset.photos[i], photoset.photos[j],
                cameras.extrinsics[i], cameras.extrinsics[j],
                intr, sgbm_window=sgbm_window, num_disp=num_disp,
            )
            xyz_w, rgb_w = disparity_to_world_points(
                disp, Q, valid, rect_l, R1, cameras.extrinsics[i],
            )
            if verbose:
                print(f"  [F.6]   pair {i}-{j}: {len(xyz_w):,} dense points "
                      f"(disp valid {100*valid.mean():.1f}%)")
            if len(xyz_w) > 0:
                all_xyz.append(xyz_w)
                all_rgb.append(rgb_w)
        except cv2.error as e:
            if verbose:
                print(f"  [F.6]   pair {i}-{j} failed: {e}")

    fused_xyz = np.concatenate(all_xyz, axis=0) if len(all_xyz) > 0 else np.zeros((0, 3), dtype=np.float32)
    fused_rgb = np.concatenate(all_rgb, axis=0) if len(all_rgb) > 0 else np.zeros((0, 3), dtype=np.float32)
    n_before = len(fused_xyz)
    fused_xyz, fused_rgb = voxel_downsample(fused_xyz, fused_rgb, voxel_size=voxel_size)
    if verbose:
        print(f"  [F.6] fused {n_before:,} -> {len(fused_xyz):,} points "
              f"(voxel size {voxel_size})  in {time.perf_counter()-t0:.2f}s")

    return PointCloud(xyz=fused_xyz, colors=fused_rgb,
                      visibility=[set() for _ in range(len(fused_xyz))])
