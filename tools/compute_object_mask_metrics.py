"""Tier 1.5 item 4 — Object-mask metrics.

Recompute PSNR/SSIM only over object pixels (mask out the constant-black
background). Background pixels otherwise inflate SSIM and PSNR since both
renders trivially agree on "this is empty space".

Usage:
    python3 tools/compute_object_mask_metrics.py
    python3 tools/compute_object_mask_metrics.py --reference path/to/ply.png \\
            --pairs name=path/to/render.png ...

Default pairings use the camera-aligned 512x512 trio that has been the
project's reference comparison since v28: PLY side / v28 archive side /
v28 VQ render side.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

ROOT = Path(__file__).parent.parent

DEFAULT_REFERENCE = 'renders/crypsorender_v01/ply_200k_side.png'
DEFAULT_PAIRS = {
    'v28 EXACT archive': 'renders/crypsorender_v01/v28_archive_200k_side.png',
    'v28 VQ render':     'renders/crypsorender_v01/v28_render_200k_side.png',
}


def load(path):
    return np.asarray(Image.open(ROOT / path).convert('RGB'), dtype=np.float32) / 255.0


def make_mask(rgb, threshold=0.02):
    """Binary mask: True for pixels brighter (luma) than threshold."""
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return luma > threshold


def masked_psnr(a, b, mask):
    diff = (a - b)[mask]
    mse = float(np.mean(diff ** 2))
    return float('inf') if mse < 1e-12 else float(10.0 * np.log10(1.0 / mse))


def masked_ssim(a, b, mask):
    """Crop to mask bounding box, compute SSIM on the crop."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return float('nan')
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return float(ssim_fn(a[y0:y1, x0:x1], b[y0:y1, x0:x1],
                         channel_axis=2, data_range=1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', default=DEFAULT_REFERENCE)
    ap.add_argument('--threshold', type=float, default=0.02,
                    help='luma threshold for object mask (default 0.02)')
    ap.add_argument('--out-json', type=Path,
                    default=ROOT / 'reports/TIER_1.5_object_mask_metrics.json')
    ap.add_argument('--pairs', nargs='*', default=None,
                    help='name=path pairs (default: built-in v28 trio)')
    args = ap.parse_args()

    pairs = DEFAULT_PAIRS
    if args.pairs:
        pairs = dict(p.split('=', 1) for p in args.pairs)

    ref = load(args.reference)
    H, W = ref.shape[:2]
    mask = make_mask(ref, args.threshold)
    mask_frac = float(mask.mean())
    print(f'Reference: {args.reference} ({H}x{W})')
    print(f'Object mask: {mask_frac:.1%} of frame (luma > {args.threshold})')

    results = []
    for name, path in pairs.items():
        other = load(path)
        if other.shape != ref.shape:
            print(f'  shape mismatch on {name}: {other.shape}')
            continue
        psnr_full = float(psnr_fn(ref, other, data_range=1.0))
        ssim_full = float(ssim_fn(ref, other, channel_axis=2, data_range=1.0))
        psnr_obj = masked_psnr(ref, other, mask)
        ssim_obj = masked_ssim(ref, other, mask)
        results.append({
            'pair': f'reference vs {name}',
            'image_size': [H, W],
            'object_mask_fraction': mask_frac,
            'full_frame': {'psnr_db': round(psnr_full, 3), 'ssim': round(ssim_full, 5)},
            'object_only': {'psnr_db': round(psnr_obj, 3), 'ssim': round(ssim_obj, 5)},
            'ssim_inflation_from_background': round(ssim_full - ssim_obj, 5),
            'psnr_change_when_masked_db': round(psnr_obj - psnr_full, 3),
        })
        print(f'\n{name}:')
        print(f'  full-frame  PSNR {psnr_full:6.2f} dB   SSIM {ssim_full:.5f}')
        print(f'  object-only PSNR {psnr_obj:6.2f} dB   SSIM {ssim_obj:.5f}')
        print(f'  SSIM background inflation: +{ssim_full - ssim_obj:.5f}')
        print(f'  PSNR delta when masked:    {psnr_obj - psnr_full:+.2f} dB')

    args.out_json.write_text(json.dumps({
        'note': 'Tier 1.5 item 4 — object-mask metrics (background masked out).',
        'reference': args.reference,
        'mask_threshold_luma': args.threshold,
        'image_size': [H, W],
        'object_mask_fraction': mask_frac,
        'comparisons': results,
    }, indent=2))
    print(f'\nwrote {args.out_json}')


if __name__ == '__main__':
    main()
