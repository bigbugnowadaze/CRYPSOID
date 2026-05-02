"""Compute image metrics (PSNR, SSIM, MSE, MAE)."""

import json
import math
from pathlib import Path
from typing import Dict

import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False


def compute_metrics(img_a: np.ndarray, img_b: np.ndarray) -> Dict[str, float]:
    """Compute MSE, MAE, PSNR, and SSIM between two images.

    Args:
        img_a, img_b: (H, W, 3) uint8 arrays

    Returns:
        dict with keys: mse, mae, psnr_db, ssim
    """
    a = img_a.astype(np.float32) / 255.0
    b = img_b.astype(np.float32) / 255.0

    mse = float(np.mean((a - b) ** 2))
    mae = float(np.mean(np.abs(a - b)))

    if mse <= 1e-12:
        psnr = 99.0
    else:
        psnr = float(-10.0 * math.log10(mse))

    ssim_val = None
    if SSIM_AVAILABLE:
        try:
            ssim_val = float(ssim(a, b, channel_axis=2, data_range=1.0))
        except Exception:
            pass

    return {"mse": mse, "mae": mae, "psnr_db": psnr, "ssim": ssim_val}


def save_metrics(
    metrics: Dict[str, float],
    path: Path,
):
    """Save metrics to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(metrics, f, indent=2)
