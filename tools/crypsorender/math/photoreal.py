"""Photoreal preset — studio 3-point lighting + ACES tonemap + photographic env.

Sits on top of bar2_lighting. Adds:
    - 3-point lighting rig (key + fill + rim) instead of single sun
    - ACES filmic tonemap (replaces gamma + smoothstep)
    - Studio-photographic environment (bright photographic grey + soft top-down sun)
    - Subtle vignette
    - Color-graded final output

Use this when you want "publish on a marketing page" quality, not "physically
correct." It's still single-bounce direct + IBL — no path tracing — but the
lighting setup is the one a 3D artist would actually use for a hero render.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from .bar2_lighting import apply_bar2_lighting
from .environment import ProceduralEnvironment


# --------------- ACES filmic tonemap ---------------
# Approximation of the ACES Reference Rendering Transform (RRT + ODT) by
# Krzysztof Narkowicz. Maps unbounded HDR to [0, 1] with film-like rolloff.

def aces_filmic(x: np.ndarray) -> np.ndarray:
    """Narkowicz ACES filmic tonemap. Input HDR linear [0, ∞), output sRGB linear [0, 1]."""
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Standard sRGB encoding gamma. Input linear [0, 1], output sRGB."""
    a = 0.055
    return np.where(x <= 0.0031308,
                    12.92 * x,
                    (1.0 + a) * np.power(np.maximum(x, 1e-12), 1.0 / 2.4) - a)


def color_grade(rgb: np.ndarray,
                exposure_stops: float = 0.0,
                contrast: float = 1.05,
                saturation: float = 1.10,
                lift=(0.005, 0.005, 0.012),
                gamma_rgb=(1.00, 1.00, 0.98),
                gain=(1.02, 1.00, 0.97),
                ) -> np.ndarray:
    """Lift / gamma / gain color grading + exposure + contrast + saturation.

    Subtle defaults: 1-stop-ish exposure handle, slight cool shadows, slight
    warm highlights, gentle s-curve contrast, bumped saturation.
    """
    out = rgb.astype(np.float32) * (2.0 ** exposure_stops)
    # Lift: add to shadows
    out = out + np.array(lift, dtype=np.float32)[None, None, :]
    # Gamma per-channel
    out = np.power(np.maximum(out, 1e-9), 1.0 / np.array(gamma_rgb, dtype=np.float32))
    # Gain
    out = out * np.array(gain, dtype=np.float32)[None, None, :]
    # Contrast around 0.5
    out = (out - 0.5) * contrast + 0.5
    # Saturation
    luma = (0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2])[..., None]
    out = luma + (out - luma) * saturation
    return out.astype(np.float32)


# --------------- Studio environment (photographic grey background) ---------------

class StudioEnvironment(ProceduralEnvironment):
    """Override of ProceduralEnvironment with a photographic-studio look:
       big bright softbox sun, neutral grey backdrop, subtle ground shadow.
    """

    def __init__(self, sun_dir: np.ndarray, intensity: float = 1.0):
        super().__init__(
            sun_dir=sun_dir,
            sun_rgb=(2.20, 2.05, 1.85),               # bright softbox key
            sky_zenith=(0.78, 0.80, 0.85),            # neutral light grey top
            sky_horizon=(0.92, 0.92, 0.92),           # neutral grey at horizon
            ground_dark=(0.30, 0.30, 0.30),           # darker grey below
            ground_light=(0.55, 0.55, 0.55),          # light grey checker
            sun_disc_size=0.992,
            sun_glow_sharpness=10.0,
            horizon_haze_band=0.04,
            intensity=intensity,
        )


# --------------- 3-point lighting helpers ---------------

def three_point_directions(camera_eye, scene_center,
                           key_az=35.0, key_el=25.0, key_dist_factor=2.0,
                           fill_az=-50.0, fill_el=10.0, fill_dist_factor=2.5,
                           rim_az=170.0, rim_el=35.0, rim_dist_factor=2.5,
                           up=(0.0, 1.0, 0.0)):
    """Compute three light directions in world space relative to the camera.

    All angles in degrees. Each light direction points FROM source TO scene
    center (i.e. the "L" you'd plug into N·(-L) to compute incidence).

    Returns dict of (key_dir, fill_dir, rim_dir) — each a unit (3,) np.float32.
    """
    import math
    eye = np.asarray(camera_eye, dtype=np.float32)
    ctr = np.asarray(scene_center, dtype=np.float32)
    up_v = np.array(up, dtype=np.float32)
    forward = (ctr - eye)
    forward /= (np.linalg.norm(forward) + 1e-9)
    right = np.cross(forward, up_v); right /= (np.linalg.norm(right) + 1e-9)
    realup = np.cross(right, forward); realup /= (np.linalg.norm(realup) + 1e-9)

    def dir_for(az_deg, el_deg):
        az = math.radians(az_deg); el = math.radians(el_deg)
        # Build a unit vector in camera-relative coordinates, then rotate to world
        v = (math.cos(el) * math.sin(az) * right
             + math.sin(el) * realup
             + math.cos(el) * math.cos(az) * forward)
        v = v.astype(np.float32)
        return -v / (np.linalg.norm(v) + 1e-9)   # FROM source TO scene

    return {
        'key_dir':  dir_for(key_az, key_el),
        'fill_dir': dir_for(fill_az, fill_el),
        'rim_dir':  dir_for(rim_az, rim_el),
    }


# --------------- Top-level photoreal compose ---------------

def apply_photoreal_lighting(albedo, metallic, roughness, F0, kd,
                             normals, xyz, eye,
                             environment,
                             key_dir, key_rgb,
                             fill_dir, fill_rgb,
                             rim_dir, rim_rgb,
                             shadow_factor, ao_factor=None,
                             curvature_visibility=None,
                             cusp_norm=None, max_pixel_size=None,
                             env_ambient_strength=0.55,
                             env_reflection_strength=1.0,
                             cusp_glint_strength=0.4,
                             ):
    """Three-light photoreal composer.

    Each of (key, fill, rim) is treated as an independent directional light
    that goes through the same Bar 2 PBR composer with proportional strengths.

    Returns (N, 3) HDR linear (will be tone-mapped after rasterization).
    """
    # Run Bar 2 once per light, sum HDR contributions.
    out = np.zeros_like(albedo, dtype=np.float32)

    # KEY
    out = out + apply_bar2_lighting(
        albedo=albedo, metallic=metallic, roughness=roughness, F0=F0, kd=kd,
        normals=normals, xyz=xyz, eye=eye,
        sun_dir=key_dir, sun_rgb=key_rgb,
        environment=environment,
        shadow_factor=shadow_factor, ao_factor=ao_factor,
        curvature_visibility=curvature_visibility,
        cusp_norm=cusp_norm, max_pixel_size=max_pixel_size,
        sun_strength=1.6,
        env_ambient_strength=env_ambient_strength,
        env_reflection_strength=env_reflection_strength,
        cusp_glint_strength=cusp_glint_strength,
    )

    # FILL — half-strength, no env (already counted by key) and no shadow modulation
    out = out + 0.55 * apply_bar2_lighting(
        albedo=albedo, metallic=metallic, roughness=roughness, F0=F0, kd=kd,
        normals=normals, xyz=xyz, eye=eye,
        sun_dir=fill_dir, sun_rgb=fill_rgb,
        environment=environment,
        shadow_factor=np.ones_like(shadow_factor),    # fill is ambient-like, ignore shadow
        ao_factor=ao_factor,
        curvature_visibility=None,
        cusp_norm=None,                                # no cusp glint from fill
        max_pixel_size=None,
        sun_strength=0.6,
        env_ambient_strength=0.0,                     # don't double-count env
        env_reflection_strength=0.0,
        cusp_glint_strength=0.0,
    )

    # RIM — narrow rim/back light for edge highlights only (specular-heavy)
    out = out + 0.40 * apply_bar2_lighting(
        albedo=albedo, metallic=metallic, roughness=roughness, F0=F0, kd=kd,
        normals=normals, xyz=xyz, eye=eye,
        sun_dir=rim_dir, sun_rgb=rim_rgb,
        environment=environment,
        shadow_factor=np.ones_like(shadow_factor),
        ao_factor=None,
        curvature_visibility=None,
        cusp_norm=cusp_norm,
        max_pixel_size=max_pixel_size,
        sun_strength=1.0,
        env_ambient_strength=0.0,
        env_reflection_strength=0.0,
        cusp_glint_strength=cusp_glint_strength * 0.5,
    )
    return out.astype(np.float32)


# --------------- Vignette ---------------

def vignette(img: np.ndarray, strength: float = 0.20, falloff: float = 1.6) -> np.ndarray:
    """Subtle radial vignette darken. img (H, W, 3) in [0, 1]."""
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, cx = (H - 1) * 0.5, (W - 1) * 0.5
    r2 = ((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2
    mask = 1.0 - strength * np.power(np.clip(r2, 0.0, 1.0), falloff)
    return (img * mask[..., None]).astype(np.float32)
