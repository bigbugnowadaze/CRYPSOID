"""Convert any HDR-ish image to the .npy format HDRIEnvironment loves.

No external optional installs required if PIL can read the file (works for
.hdr in modern Pillow, .exr if openexr is installed, plus .png/.jpg as
8-bit-rescaled fallbacks).

Usage:
    python3 tools/hdr_to_npy.py path/to/whatever.hdr [output.npy]

If output.npy is omitted, it's saved next to the input with the .npy suffix.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np


def load_hdr_anyway(path: Path) -> np.ndarray:
    """Try every available backend in order until one returns float32 (H, W, 3)."""
    ext = path.suffix.lower()

    # 1) imageio with format auto-detection (handles .hdr if FreeImage IS installed)
    try:
        import imageio
        for fmt_arg in [None, 'HDR-FI', 'HDR', 'EXR-FI']:
            try:
                arr = imageio.imread(str(path), format=fmt_arg) if fmt_arg else imageio.imread(str(path))
                arr = np.asarray(arr).astype(np.float32)
                if arr.ndim == 2:
                    arr = arr[:, :, None].repeat(3, axis=2)
                return arr
            except Exception:
                continue
    except ImportError:
        pass

    # 2) PIL (no HDR but works for .png/.jpg as 8-bit, also .hdr in newer Pillow)
    try:
        from PIL import Image
        img = Image.open(path)
        arr = np.asarray(img).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, None].repeat(3, axis=2)
        if ext in ('.png', '.jpg', '.jpeg', '.tiff'):
            arr = arr / 255.0
        return arr
    except Exception:
        pass

    raise SystemExit(f"Could not read {path} with any available backend. "
                     f"Install imageio[freeimage] or convert manually.")


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    src = Path(sys.argv[1])
    if not src.exists():
        raise SystemExit(f"input not found: {src}")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix('.npy')

    arr = load_hdr_anyway(src)
    np.save(out, arr)
    print(f"  in:  {src}  ({src.stat().st_size:,} B)")
    print(f"  out: {out}  ({out.stat().st_size:,} B)")
    print(f"  shape={arr.shape}  dtype={arr.dtype}  range=[{arr.min():.3f}, {arr.max():.3f}]")
    print(f"  load via:  HDRIEnvironment(Path('{out}'))")


if __name__ == '__main__':
    main()
