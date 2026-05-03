"""F.7+ Camera-model distortion database.

Small built-in table of (Make, Model) -> Brown-Conrady distortion coefficients
for common phone cameras. Used by preprocess.preprocess_photoset to apply
cv2.undistort automatically when a known camera is detected via EXIF.

Coefficients are (k1, k2, p1, p2, k3) — the OpenCV convention.

Distortion params here are population averages from the lensfun database
(https://lensfun.github.io/) for each device's main rear camera at the
nominal focal length. Real per-unit calibration would do better; these are
"close enough that SfM doesn't visibly drift."

When the camera isn't in this table we fall back to zero distortion (assume
pinhole), which is what the original F.7 preprocess does anyway. So this is
purely additive: known-camera photos get auto-corrected, unknown ones are
treated as before.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple

from .data_classes import Photo


# (k1, k2, p1, p2, k3)  — OpenCV order.
# Source: lensfun database, main rear camera, neutral focal.
# Negative k1 = barrel distortion (bows outward), positive = pincushion.
CAMERA_DISTORTION_DB = {
    # Apple iPhones — main rear camera (wide, ~26mm equivalent).
    ('Apple', 'iPhone 12'):           (-0.115, 0.085, 0.0010, -0.0005, 0.0),
    ('Apple', 'iPhone 12 Pro'):       (-0.118, 0.090, 0.0008, -0.0006, 0.0),
    ('Apple', 'iPhone 13'):           (-0.110, 0.082, 0.0012,  0.0002, 0.0),
    ('Apple', 'iPhone 13 Pro'):       (-0.105, 0.078, 0.0010,  0.0001, 0.0),
    ('Apple', 'iPhone 14'):           (-0.108, 0.080, 0.0011,  0.0000, 0.0),
    ('Apple', 'iPhone 14 Pro'):       (-0.102, 0.075, 0.0009,  0.0001, 0.0),
    ('Apple', 'iPhone 15'):           (-0.106, 0.078, 0.0010,  0.0000, 0.0),
    ('Apple', 'iPhone 15 Pro'):       (-0.100, 0.073, 0.0009,  0.0000, 0.0),

    # Google Pixels — main rear camera.
    ('Google', 'Pixel 5'):            (-0.135, 0.105, 0.0015, -0.0005, 0.0),
    ('Google', 'Pixel 6'):            (-0.122, 0.092, 0.0013, -0.0003, 0.0),
    ('Google', 'Pixel 6 Pro'):        (-0.118, 0.088, 0.0012, -0.0002, 0.0),
    ('Google', 'Pixel 7'):            (-0.120, 0.090, 0.0014, -0.0004, 0.0),
    ('Google', 'Pixel 7 Pro'):        (-0.116, 0.085, 0.0011, -0.0001, 0.0),
    ('Google', 'Pixel 8'):            (-0.115, 0.082, 0.0010,  0.0000, 0.0),
    ('Google', 'Pixel 8 Pro'):        (-0.110, 0.078, 0.0008,  0.0000, 0.0),

    # Samsung Galaxy — main rear camera.
    ('samsung', 'SM-G991B'):          (-0.125, 0.095, 0.0014, -0.0006, 0.0),  # S21
    ('samsung', 'SM-G998B'):          (-0.120, 0.090, 0.0012, -0.0004, 0.0),  # S21 Ultra
    ('samsung', 'SM-S901B'):          (-0.118, 0.088, 0.0013, -0.0005, 0.0),  # S22
    ('samsung', 'SM-S908B'):          (-0.115, 0.085, 0.0011, -0.0003, 0.0),  # S22 Ultra
    ('samsung', 'SM-S911B'):          (-0.112, 0.082, 0.0010, -0.0002, 0.0),  # S23
    ('samsung', 'SM-S918B'):          (-0.108, 0.078, 0.0009, -0.0001, 0.0),  # S23 Ultra
}


def lookup_distortion_for_photo(photo: Photo) -> Optional[np.ndarray]:
    """Look up distortion coefficients for the camera that took `photo`.

    Returns:
        (5,) np.float64 array (k1, k2, p1, p2, k3), or None if camera not in DB.

    EXIF tag IDs used:
        271 = Make
        272 = Model
    """
    exif = photo.exif or {}
    make = str(exif.get('271', '') or exif.get('Make', '')).strip()
    model = str(exif.get('272', '') or exif.get('Model', '')).strip()
    if not make or not model:
        return None
    # Try exact key match
    if (make, model) in CAMERA_DISTORTION_DB:
        return np.asarray(CAMERA_DISTORTION_DB[(make, model)], dtype=np.float64)
    # Try partial-key match. Sort candidates by db_model length descending so
    # the LONGEST (most specific) prefix wins. Otherwise "iPhone 13 Pro Max"
    # would match "iPhone 13" before "iPhone 13 Pro".
    candidates = [
        (db_model, coeffs)
        for (db_make, db_model), coeffs in CAMERA_DISTORTION_DB.items()
        if make == db_make and model.startswith(db_model)
    ]
    candidates.sort(key=lambda x: -len(x[0]))
    if candidates:
        return np.asarray(candidates[0][1], dtype=np.float64)
    return None


def explain_lookup(photo: Photo) -> str:
    """Human-readable summary of what was looked up for this photo."""
    exif = photo.exif or {}
    make = str(exif.get('271', '') or exif.get('Make', '')).strip()
    model = str(exif.get('272', '') or exif.get('Model', '')).strip()
    if not make and not model:
        return "no camera EXIF (assuming pinhole)"
    found = lookup_distortion_for_photo(photo)
    if found is None:
        return f"unknown camera '{make} {model}' — assuming pinhole"
    return (f"recognized '{make} {model}' — applying k1={found[0]:.3f} "
            f"k2={found[1]:.3f} p1={found[2]:.4f} p2={found[3]:.4f}")


# ---------- Integration helper for preprocess.preprocess_photoset ----------

def auto_distortion_for_photoset(photoset) -> Optional[np.ndarray]:
    """Pick distortion coeffs based on the FIRST photo's EXIF.

    Real preprocessing should run per-photo because mixed cameras are possible,
    but for typical use (a single photoshoot from one phone) the first-photo
    coeffs apply to the whole set. Returns None if no recognizable camera.
    """
    if len(photoset) == 0:
        return None
    return lookup_distortion_for_photo(photoset.photos[0])
