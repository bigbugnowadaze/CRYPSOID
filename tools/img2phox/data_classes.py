"""Data classes shared across all img2phox stages.

The contracts here are intentionally simple — each stage's input and output
types are these dataclasses, no deeper coupling. This makes it easy to swap
a stage out (e.g. plug in COLMAP for SfM) without touching the rest.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set, Optional

import numpy as np


@dataclass
class Photo:
    """A single input image + EXIF metadata."""
    path: Path
    image: np.ndarray                # (H, W, 3) float32 in [0, 1]
    exif: dict = field(default_factory=dict)

    @property
    def height(self) -> int: return self.image.shape[0]
    @property
    def width(self) -> int:  return self.image.shape[1]


@dataclass
class PhotoSet:
    """Collection of input photos."""
    photos: List[Photo] = field(default_factory=list)

    def __len__(self): return len(self.photos)
    def __getitem__(self, i): return self.photos[i]


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics (no distortion in F.0..F.4)."""
    focal_x: float
    focal_y: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: tuple = ()        # (k1, k2, p1, p2) Brown-Conrady, deferred to F.5

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.focal_x, 0,            self.cx],
                         [0,            self.focal_y, self.cy],
                         [0,            0,            1.0]], dtype=np.float32)

    @classmethod
    def from_fov(cls, fov_deg: float, width: int, height: int):
        import math
        focal = 0.5 * width / math.tan(math.radians(fov_deg) / 2)
        return cls(focal_x=focal, focal_y=focal, cx=width/2, cy=height/2,
                   width=width, height=height)


@dataclass
class CameraExtrinsics:
    """World-to-camera rigid transform."""
    R: np.ndarray   # (3, 3)
    t: np.ndarray   # (3,)

    def world_to_cam(self, xyz: np.ndarray) -> np.ndarray:
        """xyz (N, 3) world -> (N, 3) camera."""
        return (xyz @ self.R.T + self.t[None, :]).astype(np.float32)

    @property
    def cam_position(self) -> np.ndarray:
        """Eye position in world coords."""
        return (-self.R.T @ self.t).astype(np.float32)


@dataclass
class CameraBundle:
    """Bundle of N cameras (shared intrinsics for now)."""
    intrinsics: CameraIntrinsics
    extrinsics: List[CameraExtrinsics] = field(default_factory=list)

    def __len__(self): return len(self.extrinsics)


@dataclass
class PointCloud:
    """Sparse or dense 3D point cloud + per-point camera-visibility."""
    xyz: np.ndarray                                     # (M, 3)
    colors: Optional[np.ndarray] = None                 # (M, 3) RGB in [0, 1]
    visibility: List[Set[int]] = field(default_factory=list)   # per-point sets of camera indices

    def __len__(self): return self.xyz.shape[0]

    @classmethod
    def empty(cls):
        return cls(xyz=np.zeros((0, 3), dtype=np.float32))


@dataclass
class BlobBundle:
    """Per-blob params suitable for .3dphox encoding."""
    xyz: np.ndarray              # (N, 3)
    scales: np.ndarray           # (N, 3) log-sigma
    quats: np.ndarray            # (N, 4) wxyz
    opacity: np.ndarray          # (N,) sigmoid logit
    sh_dc: np.ndarray            # (N, 3) RGB DC (already as albedo, NOT raw SH coef)
    sh_rest: Optional[np.ndarray] = None        # (N, 45)
    tier: Optional[np.ndarray] = None           # (N,) uint8 — A/B/C tier label

    def __len__(self): return self.xyz.shape[0]
