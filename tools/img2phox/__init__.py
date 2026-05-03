"""Phase F — image → .3dphox compiler.

5-stage pipeline:
    PhotoSet -> CameraBundle + PointCloud -> BlobBundle -> .3dphox

See docs/img2phox_spec.md.
"""
from .data_classes import (
    Photo, PhotoSet,
    CameraIntrinsics, CameraExtrinsics, CameraBundle,
    PointCloud, BlobBundle,
)

__all__ = [
    'Photo', 'PhotoSet',
    'CameraIntrinsics', 'CameraExtrinsics', 'CameraBundle',
    'PointCloud', 'BlobBundle',
]
