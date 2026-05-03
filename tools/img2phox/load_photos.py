"""Load a folder of photos into a PhotoSet."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image

from .data_classes import Photo, PhotoSet


SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp')


def load_photo(path: Path, max_dim: Optional[int] = None) -> Photo:
    """Load a single photo. EXIF grabbed BEFORE resize (PIL.resize strips EXIF)."""
    img = Image.open(path).convert('RGB')
    # Grab EXIF first
    exif = {}
    try:
        raw_exif = dict(img.getexif() or {})
        for k, v in raw_exif.items():
            exif[str(k)] = v
    except Exception:
        pass
    if max_dim and max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return Photo(path=path, image=arr, exif=exif)


def load_photoset(folder: Path, max_dim: Optional[int] = None,
                  pattern: str = '*') -> PhotoSet:
    folder = Path(folder)
    paths = sorted(p for p in folder.glob(pattern)
                   if p.suffix.lower() in SUPPORTED_EXT)
    if not paths:
        raise ValueError(f"no photos found in {folder} (looked for {SUPPORTED_EXT})")
    photos = [load_photo(p, max_dim=max_dim) for p in paths]
    return PhotoSet(photos=photos)


def photoset_from_arrays(arrays: Iterable[np.ndarray],
                         names: Optional[Iterable[str]] = None) -> PhotoSet:
    photos = []
    arrs = list(arrays)
    nms = list(names) if names is not None else [f"synth_{i:04d}" for i in range(len(arrs))]
    for arr, name in zip(arrs, nms):
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        photos.append(Photo(path=Path(f"<memory>/{name}"), image=arr, exif={}))
    return PhotoSet(photos=photos)
