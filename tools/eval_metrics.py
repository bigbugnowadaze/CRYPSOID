"""Image-quality metrics with optional foreground-mask support.

Tier 1.5 item 4: PSNR / SSIM computed only over alpha > threshold pixels, so
that a constant-black background can't inflate the numbers.

Usage:
    python3 tools/eval_metrics.py --a path/to/render_a.png --b path/to/render_b.png
                                  --mask path/to/alpha.png --threshold 0.05

If --mask is omitted, full-image metrics are computed (matches Tier 1 behaviour).

Returns metrics as JSON to stdout AND writes alongside the images:
    foo_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
except ImportError:
    structural_similarity = None
    peak_signal_noise_ratio = None


def _to_float01(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32)


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return _to_float01(np.array(img))


def load_alpha_mask(path: Path, threshold: float) -> np.ndarray:
    """Load an alpha PNG (or alpha channel of an RGBA) and threshold to bool."""
    img = Image.open(path)
    if img.mode == "RGBA":
        alpha = _to_float01(np.array(img))[:, :, 3]
    elif img.mode in ("L", "I"):
        alpha = _to_float01(np.array(img))
    else:
        # Convert to grayscale and use that as alpha proxy
        alpha = _to_float01(np.array(img.convert("L")))
    return alpha > threshold


def masked_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """PSNR computed only over pixels where mask is True. Both inputs in [0,1]."""
    if mask.sum() == 0:
        return float("nan")
    diff = (a - b)
    if diff.ndim == 3:
        # mean over channels
        sq = (diff * diff).mean(axis=-1)
    else:
        sq = diff * diff
    mse = float(sq[mask].mean())
    if mse <= 0:
        return float("inf")
    return -10.0 * float(np.log10(mse))


def masked_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """SSIM weighted by mask; we evaluate full-image SSIM map, then average
    only over masked pixels. This is the standard 'foreground SSIM' approach
    used in 3DGS papers like Mip-NeRF 360 evaluations."""
    if structural_similarity is None:
        raise RuntimeError("scikit-image required (skimage.metrics.structural_similarity)")
    # full_map=True returns the per-pixel SSIM map
    _, ssim_map = structural_similarity(
        a, b, channel_axis=-1, data_range=1.0, full=True,
    )
    # collapse channels
    if ssim_map.ndim == 3:
        ssim_map = ssim_map.mean(axis=-1)
    if mask.sum() == 0:
        return float("nan")
    return float(ssim_map[mask].mean())


def masked_mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    diff = np.abs(a - b)
    if diff.ndim == 3:
        diff = diff.mean(axis=-1)
    return float(diff[mask].mean())


def alpha_from_render(render: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    """Derive a foreground mask from a rendered image when no separate alpha
    channel exists: any pixel with luminance > threshold is foreground."""
    lum = render.mean(axis=-1)
    return lum > threshold


def compute_all(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Return dict of all metrics. If mask is None, full-image; else masked."""
    out = {}
    full_psnr = peak_signal_noise_ratio(a, b, data_range=1.0)
    full_ssim = structural_similarity(a, b, channel_axis=-1, data_range=1.0)
    out["full_psnr_db"] = float(full_psnr)
    out["full_ssim"] = float(full_ssim)
    out["full_mae"] = float(np.abs(a - b).mean())
    out["full_mse"] = float(((a - b) ** 2).mean())
    if mask is not None:
        out["mask_pixel_count"] = int(mask.sum())
        out["mask_fraction"] = float(mask.mean())
        out["masked_psnr_db"] = masked_psnr(a, b, mask)
        out["masked_ssim"] = masked_ssim(a, b, mask)
        out["masked_mae"] = masked_mae(a, b, mask)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True, help="reference image")
    ap.add_argument("--b", type=Path, required=True, help="comparison image")
    ap.add_argument("--mask", type=Path,
                    help="optional alpha mask PNG (RGBA or grayscale)")
    ap.add_argument("--auto-mask-from", type=Path,
                    help="instead of --mask, derive a foreground mask from this image's luminance")
    ap.add_argument("--threshold", type=float, default=0.02,
                    help="alpha or luminance threshold (default 0.02)")
    ap.add_argument("--out", type=Path, help="optional JSON output path")
    args = ap.parse_args()

    a = load_image(args.a)
    b = load_image(args.b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    mask = None
    if args.mask is not None:
        mask = load_alpha_mask(args.mask, args.threshold)
    elif args.auto_mask_from is not None:
        ref = load_image(args.auto_mask_from)
        if ref.shape[:2] != a.shape[:2]:
            raise ValueError(f"auto-mask source shape {ref.shape} doesn't match image shape {a.shape}")
        mask = alpha_from_render(ref, args.threshold)

    metrics = compute_all(a, b, mask)
    metrics["a"] = str(args.a); metrics["b"] = str(args.b)
    metrics["mask_threshold"] = args.threshold
    out_path = args.out or args.b.with_name(args.b.stem + "_metrics.json")
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
