"""Bar 2.2 — Environment sampler for ambient + reflections.

Two backends:
    1. ProceduralEnvironment — analytic sky + ground + sun, no asset required.
       Richer than Bar 1's flat sky-ground gradient: includes a procedural
       checkered ground plane, horizon haze, sun disc + glow.
    2. HDRIEnvironment       — loads an equirectangular .hdr file via imageio
       and samples it spherically. Drop-in replacement.

Both expose the same interface:
    env.sample(directions: (N, 3)) -> (N, 3) RGB in linear (HDR) range.
    env.sample_blurred(directions, roughness) -> (N, 3) RGB with mip-chain blur.

The blurred sample approximates pre-filtered importance-sampled IBL: at high
roughness we average a small kernel of nearby directions. This is the cheap
real-time approximation; a true split-sum BRDF integration is overkill for
our needs.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional


# ---------------------- Procedural environment ----------------------

class ProceduralEnvironment:
    """Procedural sky-ground-sun environment (no external asset)."""

    def __init__(self,
                 sun_dir: np.ndarray,
                 sun_rgb=(1.4, 1.30, 1.10),
                 sky_zenith=(0.45, 0.62, 0.92),
                 sky_horizon=(0.78, 0.85, 0.95),
                 ground_dark=(0.18, 0.16, 0.14),
                 ground_light=(0.45, 0.42, 0.38),
                 sun_disc_size: float = 0.985,    # cos(angle) above which a hit is "on the sun"
                 sun_glow_sharpness: float = 16.0,
                 horizon_haze_band: float = 0.08,
                 intensity: float = 1.0):
        self.sun_dir = (sun_dir / (np.linalg.norm(sun_dir) + 1e-9)).astype(np.float32)
        self.sun_rgb = np.array(sun_rgb, dtype=np.float32)
        self.sky_zenith  = np.array(sky_zenith,  dtype=np.float32)
        self.sky_horizon = np.array(sky_horizon, dtype=np.float32)
        self.ground_dark = np.array(ground_dark, dtype=np.float32)
        self.ground_light= np.array(ground_light,dtype=np.float32)
        self.sun_disc_size = sun_disc_size
        self.sun_glow_sharpness = sun_glow_sharpness
        self.horizon_haze_band = horizon_haze_band
        self.intensity = intensity

    def sample(self, directions: np.ndarray) -> np.ndarray:
        """Sample environment in unit directions. (M, 3) -> (M, 3) RGB."""
        D = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-9)
        cos_up   = np.clip(D[:, 1], -1.0, 1.0)
        cos_sun  = np.clip(D @ -self.sun_dir, -1.0, 1.0)   # +1 = looking at sun

        out = np.zeros_like(D, dtype=np.float32)

        # --- Sky hemisphere (cos_up > 0)
        sky_mask = cos_up > 0
        # Vertical mix
        zenith_w  = np.power(cos_up, 0.5)                       # 1 at top, ~0 at horizon
        horizon_w = np.maximum(0.0, 1.0 - cos_up)               # 1 at horizon, 0 at top
        sky_color = (zenith_w[:, None]  * self.sky_zenith[None, :]
                     + horizon_w[:, None] * self.sky_horizon[None, :])
        # Sun glow (additive)
        glow = np.power(np.clip(cos_sun, 0.0, 1.0), self.sun_glow_sharpness)[:, None] * \
               self.sun_rgb[None, :] * 0.45
        sky_color = sky_color + glow
        # Sharp sun disc (very small region)
        on_sun = cos_sun > self.sun_disc_size
        if on_sun.any():
            sky_color[on_sun] = sky_color[on_sun] + self.sun_rgb[None, :] * 4.0

        out[sky_mask] = sky_color[sky_mask]

        # --- Horizon haze (a thin warm band right at the horizon, both above + below)
        haze_mask = np.abs(cos_up) < self.horizon_haze_band
        haze_w = (1.0 - np.abs(cos_up) / self.horizon_haze_band)[haze_mask]
        out[haze_mask] = (out[haze_mask] * (1.0 - 0.5 * haze_w[:, None])
                          + (self.sky_horizon * 1.15)[None, :] * (0.5 * haze_w[:, None]))

        # --- Ground hemisphere (cos_up < 0): procedural checker + base color
        ground_mask = cos_up < -1e-3
        if ground_mask.any():
            # Project ray to a virtual ground plane at y=-1 (eye is at origin).
            # Hit point: H = D * (-1 / D.y) for D.y < 0
            t = -1.0 / D[ground_mask, 1]
            hit = D[ground_mask] * t[:, None]
            checker = ((np.floor(hit[:, 0] * 1.7) + np.floor(hit[:, 2] * 1.7)) % 2.0).astype(np.float32)
            # Falloff with distance from origin (atmospheric)
            dist = np.linalg.norm(hit, axis=1)
            falloff = np.exp(-dist * 0.08)
            base = (self.ground_dark[None, :] * (1.0 - checker[:, None])
                    + self.ground_light[None, :] * checker[:, None])
            base = base * falloff[:, None] + self.sky_horizon[None, :] * (1.0 - falloff)[:, None]
            out[ground_mask] = base

        return (self.intensity * out).astype(np.float32)

    def sample_blurred(self, directions: np.ndarray, roughness: np.ndarray,
                       n_taps: int = 6, seed: int = 0) -> np.ndarray:
        """Approximated pre-filtered sampling. Per-direction blur radius from roughness.

        For each direction D and per-splat roughness r, take n_taps random taps
        within a cone of half-angle ~ r * pi/2 around D, average the results.
        Cheap mip-chain stand-in.
        """
        D = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-9)
        r = np.clip(roughness, 0.0, 1.0).astype(np.float32)
        # Sharp roughness -> single tap, only the central direction
        sharp_mask = r < 0.05
        out = np.zeros_like(D, dtype=np.float32)
        # Central tap
        center = self.sample(D)
        out[sharp_mask] = center[sharp_mask]

        # Off-tap accumulation for non-sharp splats
        rough_mask = ~sharp_mask
        if not rough_mask.any():
            return out
        rng = np.random.RandomState(seed)
        idx_rough = np.where(rough_mask)[0]
        D_r = D[idx_rough]
        r_r = r[idx_rough]
        # Build local frame around each direction for cone sampling
        # Pick an arbitrary up vector that's not parallel to D
        ref_up = np.where(np.abs(D_r[:, 1])[:, None] < 0.99,
                          np.array([0, 1, 0], dtype=np.float32),
                          np.array([1, 0, 0], dtype=np.float32))
        T = np.cross(ref_up, D_r); T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
        B = np.cross(D_r, T)

        accum = self.sample(D_r).copy()
        weight = np.ones((len(D_r),), dtype=np.float32)
        cone_half = r_r * (np.pi * 0.5)   # roughness=1 → 90° cone half-angle
        for k in range(1, n_taps):
            # Uniformly sample within the cone
            u1 = rng.random(len(D_r))
            u2 = rng.random(len(D_r))
            phi = 2.0 * np.pi * u1
            theta = cone_half * np.sqrt(u2)
            sin_t, cos_t = np.sin(theta), np.cos(theta)
            local = (np.cos(phi) * sin_t)[:, None] * T \
                  + (np.sin(phi) * sin_t)[:, None] * B \
                  + cos_t[:, None] * D_r
            tap = self.sample(local)
            # Cosine weighting along the cone
            w = cos_t.astype(np.float32)
            accum = accum + w[:, None] * tap
            weight = weight + w
        accum = accum / weight[:, None]
        out[idx_rough] = accum
        return out


# ---------------------- HDRI .hdr file backend ----------------------

class HDRIEnvironment:
    """Equirectangular HDR environment.

    Accepts:
        - a Path to an .hdr / .exr / .npy file
        - a numpy array (H, W, 3) directly

    For .hdr files, requires either imageio[freeimage] or imageio v3 with HDR
    support. Falls back gracefully if neither is available; pass an .npy file
    or a numpy array directly to bypass.
    """

    def __init__(self, hdr_source, intensity: float = 1.0):
        if isinstance(hdr_source, np.ndarray):
            self.hdr = hdr_source.astype(np.float32)
        else:
            p = Path(hdr_source)
            ext = p.suffix.lower()
            if ext == '.npy':
                self.hdr = np.load(p).astype(np.float32)
            elif ext in ('.hdr', '.exr', '.png', '.jpg', '.jpeg'):
                import imageio
                # Try several format names, fall back to default
                arr = None
                for fmt in (None, 'HDR-FI', 'HDR', 'EXR-FI', 'PNG-PIL'):
                    try:
                        arr = imageio.imread(str(p), format=fmt) if fmt else imageio.imread(str(p))
                        break
                    except Exception:
                        continue
                if arr is None:
                    raise RuntimeError(
                        f"Could not load {p} with any imageio backend. "
                        f"For .hdr support: `pip install imageio[freeimage]` or save as .npy."
                    )
                self.hdr = arr.astype(np.float32)
                # 8-bit images: rescale to ~1.0 max
                if self.hdr.max() > 5.0 and ext in ('.png', '.jpg', '.jpeg'):
                    self.hdr = self.hdr / 255.0
            else:
                raise ValueError(f"Unsupported HDR file extension: {ext}")
        if self.hdr.ndim == 2:
            self.hdr = self.hdr[:, :, None].repeat(3, axis=2)
        self.H, self.W = self.hdr.shape[:2]
        self.intensity = intensity

    def sample(self, directions: np.ndarray) -> np.ndarray:
        D = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-9)
        # Equirectangular projection: u = atan2(x, -z) / 2pi + 0.5; v = asin(y) / pi + 0.5
        u = np.arctan2(D[:, 0], -D[:, 2]) / (2.0 * np.pi) + 0.5
        v = np.arcsin(np.clip(D[:, 1], -1, 1)) / np.pi + 0.5
        u_px = np.clip((u * self.W).astype(np.int32), 0, self.W - 1)
        v_px = np.clip(((1.0 - v) * self.H).astype(np.int32), 0, self.H - 1)
        return (self.hdr[v_px, u_px] * self.intensity).astype(np.float32)

    def sample_blurred(self, directions: np.ndarray, roughness: np.ndarray,
                       n_taps: int = 6, seed: int = 0) -> np.ndarray:
        # Same cone-tap approximation as procedural; could be replaced with a
        # pre-built mip pyramid for speed.
        D = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-9)
        r = np.clip(roughness, 0.0, 1.0).astype(np.float32)
        sharp_mask = r < 0.05
        out = np.zeros_like(D, dtype=np.float32)
        out[sharp_mask] = self.sample(D[sharp_mask])
        rough_mask = ~sharp_mask
        if not rough_mask.any():
            return out
        rng = np.random.RandomState(seed)
        idx_rough = np.where(rough_mask)[0]
        D_r = D[idx_rough]; r_r = r[idx_rough]
        ref_up = np.where(np.abs(D_r[:, 1])[:, None] < 0.99,
                          np.array([0, 1, 0], dtype=np.float32),
                          np.array([1, 0, 0], dtype=np.float32))
        T = np.cross(ref_up, D_r); T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
        B = np.cross(D_r, T)
        accum = self.sample(D_r).copy()
        weight = np.ones((len(D_r),), dtype=np.float32)
        cone_half = r_r * (np.pi * 0.5)
        for k in range(1, n_taps):
            u1, u2 = rng.random(len(D_r)), rng.random(len(D_r))
            phi = 2 * np.pi * u1; theta = cone_half * np.sqrt(u2)
            sin_t, cos_t = np.sin(theta), np.cos(theta)
            local = (np.cos(phi) * sin_t)[:, None] * T \
                  + (np.sin(phi) * sin_t)[:, None] * B \
                  + cos_t[:, None] * D_r
            tap = self.sample(local)
            w = cos_t.astype(np.float32)
            accum = accum + w[:, None] * tap
            weight = weight + w
        out[idx_rough] = accum / weight[:, None]
        return out
