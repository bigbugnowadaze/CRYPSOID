"""F.7 — Preprocessing: EXIF parsing, distortion correction, exposure normalization.

Real photos come with three nuisances that the synthetic pipeline doesn't have:

1. **Lens distortion.** Phone cameras have noticeable Brown-Conrady radial+tangential
   distortion. SfM expects pinhole. We undistort once at load time.

2. **Unknown intrinsics.** EXIF *sometimes* gives focal length (in mm) which we can
   convert to pixels if we know the sensor size. Otherwise we use the FOV prior.

3. **Exposure variation.** Photos taken seconds apart have different auto-exposure /
   white balance. Multi-view photometric loss only makes sense after normalization.

This module ships:
    parse_exif_focal(photo)            — read focal length / focal_35mm / sensor info
    estimate_intrinsics_from_exif(...) — best guess at intrinsics
    undistort_photo(photo, K, D)       — apply Brown-Conrady undistortion via cv2
    normalize_exposure(photoset)       — gamma + per-channel scale to a reference
"""
from __future__ import annotations
import numpy as np
import cv2
from typing import Optional, Tuple

from .data_classes import Photo, PhotoSet, CameraIntrinsics


# ------------------ EXIF parsing ------------------

def parse_exif_focal(photo: Photo) -> dict:
    """Pull whatever focal-length info EXIF has. Returns dict with possibly:
       focal_mm, focal_35mm, sensor_width_mm, image_width_px.
    """
    exif = photo.exif or {}
    out = {}
    # PIL EXIF tags (numeric IDs from PIL.ExifTags.TAGS):
    #   37386 = FocalLength
    #   41989 = FocalLengthIn35mmFilm
    if '37386' in exif:
        v = exif['37386']
        if isinstance(v, tuple):
            out['focal_mm'] = float(v[0]) / max(float(v[1]), 1e-6)
        else:
            out['focal_mm'] = float(v)
    if '41989' in exif:
        out['focal_35mm'] = float(exif['41989'])
    out['image_width_px'] = photo.width
    out['image_height_px'] = photo.height
    return out


def estimate_intrinsics_from_exif(photo: Photo,
                                    fov_deg_fallback: float = 50.0) -> CameraIntrinsics:
    """Best-effort intrinsics from EXIF, fallback to FOV prior."""
    info = parse_exif_focal(photo)
    W, H = photo.width, photo.height

    # If we have focal_35mm: image diagonal in 35mm-equivalent space is 43.27mm
    # focal_px = focal_35mm * (image_diag_px / 43.27mm)
    if 'focal_35mm' in info:
        diag_px = (W**2 + H**2) ** 0.5
        focal_px = info['focal_35mm'] * (diag_px / 43.266615)
        return CameraIntrinsics(focal_x=focal_px, focal_y=focal_px,
                                  cx=W / 2, cy=H / 2,
                                  width=W, height=H)
    # Fall back to FOV prior
    return CameraIntrinsics.from_fov(fov_deg_fallback, W, H)


# ------------------ Distortion correction ------------------

def undistort_photo(photo: Photo, K: np.ndarray, D: np.ndarray) -> Photo:
    """Apply Brown-Conrady undistortion. D = (k1, k2, p1, p2, [k3])."""
    img8 = (photo.image.clip(0, 1) * 255).astype(np.uint8)
    undist = cv2.undistort(img8, K.astype(np.float64), D.astype(np.float64))
    return Photo(path=photo.path,
                  image=(undist.astype(np.float32) / 255.0),
                  exif=photo.exif)


def undistort_photoset(photoset: PhotoSet, K: np.ndarray, D: np.ndarray) -> PhotoSet:
    """Undistort every photo. Same K and D applied (single-camera assumption)."""
    return PhotoSet(photos=[undistort_photo(p, K, D) for p in photoset.photos])


# ------------------ Exposure normalization ------------------

def normalize_exposure(photoset: PhotoSet,
                        method: str = 'mean_match',
                        reference_idx: int = 0) -> PhotoSet:
    """Normalize per-photo exposure so multi-view photometric loss is meaningful.

    method:
      'mean_match'   — scale each photo's per-channel mean to match reference.
      'histogram'    — match the cumulative histogram per channel (more robust
                       to highlight clipping but distorts colors slightly).
      'gamma'        — match overall luminance via a gamma curve.
    """
    if len(photoset) == 0:
        return photoset
    ref = photoset.photos[reference_idx].image
    if method == 'mean_match':
        ref_mean = ref.reshape(-1, 3).mean(axis=0) + 1e-6
        out = []
        for k, p in enumerate(photoset.photos):
            if k == reference_idx:
                out.append(p); continue
            this_mean = p.image.reshape(-1, 3).mean(axis=0) + 1e-6
            scale = ref_mean / this_mean
            new_img = (p.image * scale[None, None, :]).clip(0, 1).astype(np.float32)
            out.append(Photo(path=p.path, image=new_img, exif=p.exif))
        return PhotoSet(photos=out)
    elif method == 'gamma':
        ref_lum = (ref[..., 0] * 0.2126 + ref[..., 1] * 0.7152 + ref[..., 2] * 0.0722).mean()
        out = []
        for k, p in enumerate(photoset.photos):
            if k == reference_idx:
                out.append(p); continue
            this_lum = (p.image[..., 0] * 0.2126 + p.image[..., 1] * 0.7152 + p.image[..., 2] * 0.0722).mean()
            this_lum = max(this_lum, 1e-6)
            gamma = np.log(max(ref_lum, 1e-6)) / np.log(this_lum) if this_lum != 1 else 1.0
            new_img = np.power(p.image.clip(1e-6, 1), gamma).astype(np.float32)
            out.append(Photo(path=p.path, image=new_img, exif=p.exif))
        return PhotoSet(photos=out)
    elif method == 'histogram':
        # Per-channel CDF matching against ref
        out = [photoset.photos[reference_idx]]
        ref_cdfs = []
        for c in range(3):
            ref_c = (ref[..., c] * 255).astype(np.uint8).flatten()
            hist, _ = np.histogram(ref_c, bins=256, range=(0, 256))
            cdf = np.cumsum(hist).astype(np.float64)
            cdf = cdf / cdf[-1]
            ref_cdfs.append(cdf)
        for k, p in enumerate(photoset.photos):
            if k == reference_idx: continue
            new_img = np.zeros_like(p.image)
            for c in range(3):
                src_c = (p.image[..., c] * 255).astype(np.uint8)
                src_hist, _ = np.histogram(src_c.flatten(), bins=256, range=(0, 256))
                src_cdf = np.cumsum(src_hist).astype(np.float64); src_cdf /= src_cdf[-1]
                # LUT: for each src bin, find ref bin with closest CDF
                lut = np.zeros(256, dtype=np.float32)
                for s in range(256):
                    lut[s] = float(np.argmin(np.abs(ref_cdfs[c] - src_cdf[s]))) / 255.0
                new_img[..., c] = lut[src_c]
            out.append(Photo(path=p.path, image=new_img.astype(np.float32), exif=p.exif))
        return PhotoSet(photos=out)
    else:
        raise ValueError(f"unknown exposure normalization method: {method}")


# ------------------ Combined preprocessing pipeline ------------------

def preprocess_photoset(photoset: PhotoSet,
                          fov_deg_fallback: float = 50.0,
                          distortion_coeffs: Optional[np.ndarray] = None,
                          exposure_method: str = 'mean_match',
                          auto_distortion: bool = True,
                          verbose: bool = False,
                          ) -> Tuple[PhotoSet, CameraIntrinsics]:
    """Full F.7+ preprocessing: EXIF -> intrinsics, undistort, exposure normalize.

    Returns (cleaned_photoset, intrinsics).

    Distortion handling:
      1. If `distortion_coeffs` is supplied explicitly, use those.
      2. Else if `auto_distortion=True`, look up the camera in camera_db and
         use those coefficients if recognized.
      3. Else assume pinhole (no undistortion).
    """
    intr = estimate_intrinsics_from_exif(photoset.photos[0], fov_deg_fallback=fov_deg_fallback)
    if distortion_coeffs is None and auto_distortion:
        from .camera_db import auto_distortion_for_photoset, explain_lookup
        distortion_coeffs = auto_distortion_for_photoset(photoset)
        if verbose:
            print(f"  [F.7+] {explain_lookup(photoset.photos[0])}")
    if distortion_coeffs is not None:
        photoset = undistort_photoset(photoset, intr.K, distortion_coeffs)
        if verbose:
            print(f"  [F.7+] applied undistortion to {len(photoset)} photos")
    if exposure_method:
        photoset = normalize_exposure(photoset, method=exposure_method)
    return photoset, intr
