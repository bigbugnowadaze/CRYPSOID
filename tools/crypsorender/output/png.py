"""Save framebuffers to PNG files."""

from pathlib import Path

import numpy as np
from PIL import Image


def save_png(framebuffer: np.ndarray, path: Path, alpha: np.ndarray | None = None):
    """Save a framebuffer to PNG.

    Args:
        framebuffer: (H, W, 3) float32 array in [0, 1] or (H, W) grayscale
        path: output file path
        alpha: optional (H, W) alpha channel
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Clamp and convert to uint8
    if framebuffer.dtype == np.float32 or framebuffer.dtype == np.float64:
        data_uint8 = np.clip(framebuffer * 255, 0, 255).astype(np.uint8)
    else:
        data_uint8 = framebuffer

    if framebuffer.ndim == 3 and framebuffer.shape[2] == 3:
        img = Image.fromarray(data_uint8, "RGB")
    elif framebuffer.ndim == 2:
        img = Image.fromarray(data_uint8, "L")
    else:
        raise ValueError(f"Unsupported shape: {framebuffer.shape}")

    img.save(path)
