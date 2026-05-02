"""CRYPSOID crypsorender v0.1 — CPU-only Gaussian splat renderer with tier awareness."""

__version__ = "0.1.0"

from .io.splat_buffer import SplatBuffer
from .pipeline.camera import Camera, CameraParams

__all__ = ["SplatBuffer", "Camera", "CameraParams"]
