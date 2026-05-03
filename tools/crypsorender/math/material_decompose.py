"""Bar 2.1 — Per-splat PBR material decomposition from SH bands.

Replaces the heuristic v33-material_hint-driven roughness/F0 with a proper
decomposition that extracts:
    albedo    (N, 3)  view-independent base color (from SH DC)
    metallic  (N,)    [0..1] metallic-ness (how concentrated SH bands are around mirror dir)
    roughness (N,)    [0..1] specular lobe width (from SH angular spread)
    F0        (N, 3)  Fresnel reflectance at normal incidence (derived from albedo + metallic)

The intuition:

  - SH band 0 (DC)  encodes the *view-independent* color → that's the albedo.
  - SH bands 1..3   encode *view-dependent* variation. If the variation is
                    concentrated in a tight lobe around the mirror direction
                    (R = reflect(-V, N)) for *most* viewing angles, the splat
                    is metallic-glossy. If the variation is broad and diffuse,
                    it's matte. If band magnitudes are tiny, it's flat diffuse.
  - Roughness       inversely related to the angular concentration. We use
                    the ratio of band-3 magnitude (most directional) to band-1
                    magnitude (smoothest directional variation) as a proxy.
  - F0              For non-metals: 0.04 dielectric default. For metals: F0 = albedo
                    (metals reflect their characteristic color at normal incidence).

This decomposition runs once at load (or on visible-only at render). The only
inputs are the existing SH coefficients we already decode for color — no new
chunks needed in the file format.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


SH_C0 = 0.28209479177387814   # SH band 0 normalization

# SH band slicing (45 = 9 + 16 + 20 ... actually 3*1 + 3*3 + 3*5 + 3*7 = 3 + 9 + 15 + 21,
# but stored as bands 1..3 = 3*(3 + 5 + 7) = 45, that's 9 (band1) + 15 (band2) + 21 (band3).
SH_BAND1_SLICE = slice(0,  9)     # 3 channels * 3 coefs
SH_BAND2_SLICE = slice(9,  24)    # 3 channels * 5 coefs
SH_BAND3_SLICE = slice(24, 45)    # 3 channels * 7 coefs


def decompose_pbr(sh_dc: np.ndarray,
                  sh_rest: Optional[np.ndarray],
                  opacities: Optional[np.ndarray] = None,
                  ) -> dict:
    """Per-splat PBR material decomposition.

    Args:
        sh_dc:    (N, 3) DC SH coefficients (already in linear-color form,
                  i.e. dc * SH_C0 + 0.5 -> albedo).
        sh_rest:  (N, 45) SH bands 1..3 coefficients, or None (treats as fully diffuse).
        opacities: (N,) optional, used only to weight the metallic prior down for
                  highly transparent splats.

    Returns:
        dict with keys 'albedo' (N,3), 'metallic' (N,), 'roughness' (N,),
        'F0' (N, 3). All in [0, 1].
    """
    n = sh_dc.shape[0]

    # --- Albedo ---
    # The DC SH coefficient * C0 + 0.5 is the splat's "average color across all
    # viewing directions" — by construction view-independent. That's the albedo.
    # But we already get a "color" by passing through eval_sh_color; here we
    # want JUST the DC. Most 3DGS files store sh_dc that already had the
    # +0.5 offset applied during decode (see ply_loader / phox_loader). To be
    # robust we clamp.
    albedo = np.clip(sh_dc.astype(np.float32), 0.0, 1.0)

    # --- Bands 1..3 magnitude analysis ---
    if sh_rest is not None and sh_rest.shape[1] >= 45:
        rest = sh_rest.astype(np.float32)
        # Per-band per-splat L2 magnitude
        m_band1 = np.linalg.norm(rest[:, SH_BAND1_SLICE], axis=1)   # (N,)
        m_band2 = np.linalg.norm(rest[:, SH_BAND2_SLICE], axis=1)
        m_band3 = np.linalg.norm(rest[:, SH_BAND3_SLICE], axis=1)
        m_total = m_band1 + m_band2 + m_band3
        m_dc    = np.linalg.norm(albedo, axis=1) + 1e-3
    else:
        m_band1 = m_band2 = m_band3 = m_total = np.zeros(n, dtype=np.float32)
        m_dc = np.linalg.norm(albedo, axis=1) + 1e-3

    # --- Metallic-ness ---
    # Splats that have STRONG higher-band SH AND most of that variation is in
    # band-3 (the most directional band) are likely metallic-glossy: their
    # appearance changes sharply with viewing angle, and the change is
    # spatially-localized in a specular lobe.
    # Conversely, broad bands-1+2 dominance with weak band-3 = matte/diffuse.
    rest_to_dc_ratio = m_total / m_dc
    band3_concentration = m_band3 / (m_total + 1e-6)   # 0..1
    # Combine: metallic = sigmoid(rest_to_dc * band3_concentration boost)
    metallic_raw = rest_to_dc_ratio * (0.4 + 1.6 * band3_concentration)
    # Sigmoid with center 0.6 and width 0.5 → maps [0..2] roughly to [0.05..0.9]
    metallic = 1.0 / (1.0 + np.exp(-(metallic_raw - 0.6) / 0.25))
    metallic = np.clip(metallic, 0.0, 0.95).astype(np.float32)

    # --- Roughness ---
    # Tight angular variation (high band3 relative to band1) = sharp specular
    # = LOW roughness. Smooth or broad variation = HIGH roughness.
    band_spread = m_band1 / (m_band3 + 1e-6)             # high = spread, low = sharp
    # Map to roughness: spread=4 -> 0.85, spread=0.25 -> 0.15
    roughness_raw = 0.5 + 0.20 * np.log(np.clip(band_spread, 0.05, 20.0))
    roughness = np.clip(roughness_raw, 0.05, 0.95).astype(np.float32)
    # Splats with NO higher-band SH at all are diffuse → high roughness
    fully_flat = m_total < 0.02 * m_dc
    roughness[fully_flat] = 0.9
    metallic[fully_flat]  = 0.05

    # --- Optional opacity tempering: very transparent splats are likely
    # haze / floaters; reduce metallic prior so they don't grab errant glints.
    if opacities is not None:
        opa = np.clip(opacities.astype(np.float32), 0.0, 1.0)
        metallic *= opa                                  # drop metallic on translucents

    # --- F0 (base reflectance) ---
    # Dielectrics: ~0.04 grey. Metals: F0 = albedo color.
    F0_dielectric = np.full((n, 3), 0.04, dtype=np.float32)
    F0_metallic   = albedo.astype(np.float32)
    F0 = (F0_dielectric * (1.0 - metallic[:, None])
          + F0_metallic * metallic[:, None])
    F0 = np.clip(F0, 0.02, 1.0).astype(np.float32)

    # --- Diffuse-attenuation factor for the energy-conserving PBR model ---
    # When a surface is metallic, the diffuse term should approach zero (metals
    # don't have a meaningful Lambertian part). kd = 1 - metallic.
    kd = (1.0 - metallic).astype(np.float32)

    return {
        'albedo':    albedo.astype(np.float32),
        'metallic':  metallic.astype(np.float32),
        'roughness': roughness.astype(np.float32),
        'F0':        F0,
        'kd':        kd,
    }


def decompose_summary(decomp: dict) -> str:
    """Pretty-print a summary of a decomposition (for logging)."""
    n = decomp['metallic'].shape[0]
    metal = decomp['metallic']
    rough = decomp['roughness']
    return (f"  N={n:,} splats decomposed:\n"
            f"    metallic:  mean={metal.mean():.3f} median={np.median(metal):.3f} "
            f"p90={np.percentile(metal, 90):.3f}  (>0.5 ≈ {(metal > 0.5).sum():,} splats)\n"
            f"    roughness: mean={rough.mean():.3f} median={np.median(rough):.3f} "
            f"p10={np.percentile(rough, 10):.3f}  (sharp <0.2 ≈ {(rough < 0.2).sum():,} splats)\n"
            f"    F0:        mean={decomp['F0'].mean():.3f}  (over RGB)\n"
            f"    kd:        mean={decomp['kd'].mean():.3f}")
