"""Bar 1 lighting upgrade — GGX BRDF + HDRI environment ambient + v32c integration.

Replaces the v32c Phong-shininess proxy with proper microfacet GGX math, and
replaces flat-color ambient with normal-conditioned environment-driven ambient.

This sits on top of v32a (Lambert), v32b (germ-curvature visibility), and v32.5
(kNN soft shadows + graph AO). All those terms still apply. Bar 1 ADDS:

    1. Per-splat roughness `alpha` derived from material_hint + view_dependence
    2. Per-splat F0 (specular reflectance at normal incidence) from albedo + material_hint
    3. Cook-Torrance / GGX BRDF for the specular term
    4. Synthesized HDRI sky-ground environment ambient (no external HDRI file required;
       analytic sky-gradient with sun glow). Optional: load a real .hdr file later.

Inputs needed (all per-splat):
    albedo (N,3)         base color in [0,1] from SH DC + bands 1..3 in view dir
    normals (N,3)        unit normals from v31 normals chunk
    view_dirs (N,3)      from splat -> camera, unit
    material_hint (N,)   v33 enum (0=unknown / 1=diffuse / 2=glossy / 3=mirror / 6=floater)
    view_dep (N,)        v33 view-dependence score (u8 0-255)
    cusp_norm (N,)       v40 cusp magnitude (0-1) — used to widen specular lobe at cusps

Outputs (N,3) — composite shaded color in [0,1].
"""

from __future__ import annotations
import numpy as np
from typing import Optional


# Material-hint enum (mirror of material_codec.py)
MH_UNKNOWN, MH_DIFFUSE, MH_GLOSSY, MH_MIRROR = 0, 1, 2, 3
MH_TRANSPARENT, MH_EMISSIVE, MH_FLOATER = 4, 5, 6


def derive_roughness_F0(albedo: np.ndarray,
                        material_hint: Optional[np.ndarray],
                        view_dep: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-splat (roughness alpha, F0 base reflectance).

    alpha controls the GGX lobe width:
        diffuse  -> 0.95 (wide, almost no specular)
        glossy   -> 0.4
        mirror   -> 0.05 (sharp peak)
        unknown  -> derived from view_dep: high VD -> smaller alpha (sharper)

    F0:
        diffuse / unknown -> 0.04 (typical dielectric, ~4% reflectance)
        glossy           -> mix(0.04, albedo, 0.3)
        mirror           -> albedo (treat as metallic)
    """
    n = albedo.shape[0]
    alpha = np.full(n, 0.6, dtype=np.float32)
    F0    = np.full((n, 3), 0.04, dtype=np.float32)

    if material_hint is not None:
        mh = material_hint
        # Diffuse
        diffuse_mask = mh == MH_DIFFUSE
        alpha[diffuse_mask] = 0.95
        # Glossy
        gloss_mask = mh == MH_GLOSSY
        alpha[gloss_mask] = 0.4
        F0[gloss_mask]    = 0.04 + 0.3 * (albedo[gloss_mask] - 0.04)
        # Mirror (treat as metallic)
        mir_mask = mh == MH_MIRROR
        alpha[mir_mask] = 0.05
        F0[mir_mask]    = albedo[mir_mask].clip(0.0, 1.0)
        # Floater: very wide & dim (no specular contribution)
        floater_mask = mh == MH_FLOATER
        alpha[floater_mask] = 1.0
        F0[floater_mask] *= 0.0

    if view_dep is not None:
        # For unknown splats: high VD score -> sharper specular
        unknown_mask = (material_hint == MH_UNKNOWN) if material_hint is not None else np.ones(n, dtype=bool)
        vd_norm = view_dep.astype(np.float32) / 255.0
        # alpha range: 0.9 at VD=0 down to 0.2 at VD=1
        alpha[unknown_mask] = (0.9 - 0.7 * vd_norm[unknown_mask]).astype(np.float32)

    return alpha, F0


def ggx_specular(N, V, L, alpha, F0, cusp_norm=None):
    """Cook-Torrance GGX BRDF specular term, fully vectorized.

    N, V, L: (M, 3) unit vectors
        N = surface normal
        V = view direction (splat -> camera)
        L = light direction (toward light, i.e. -sun_dir)
    alpha: (M,) roughness in (0, 1]
    F0: (M, 3) base reflectance at normal incidence
    cusp_norm: (M,) optional cusp magnitude in [0,1] — slightly widens the lobe
               for splats whose germ has high cubic curvature (folds, glints).

    Returns (M, 3) specular contribution (unmultiplied by light color or N.L weight).
    """
    EPS = 1e-6
    H = (V + L)
    H = H / (np.linalg.norm(H, axis=1, keepdims=True) + EPS)

    NdotL = np.maximum(0.0, np.einsum('ij,ij->i', N, L))
    NdotV = np.maximum(0.0, np.einsum('ij,ij->i', N, V))
    NdotH = np.maximum(0.0, np.einsum('ij,ij->i', N, H))
    VdotH = np.maximum(0.0, np.einsum('ij,ij->i', V, H))

    # Optionally widen alpha at cusps -- cubic-germ splats blur the specular peak
    a = alpha.astype(np.float32)
    if cusp_norm is not None:
        a = np.clip(a + 0.20 * cusp_norm.astype(np.float32), 0.02, 1.0)
    a2 = a * a

    # GGX Normal Distribution Function
    denom = (NdotH * NdotH * (a2 - 1.0) + 1.0)
    D = a2 / (np.pi * denom * denom + EPS)

    # Smith geometry term (Schlick-GGX approximation, unsplit)
    k = (a + 1.0)**2 / 8.0
    G_v = NdotV / (NdotV * (1.0 - k) + k + EPS)
    G_l = NdotL / (NdotL * (1.0 - k) + k + EPS)
    G = G_v * G_l

    # Schlick Fresnel (per-channel since F0 is RGB)
    F = F0 + (1.0 - F0) * np.power(1.0 - VdotH, 5.0)[:, None]

    # Cook-Torrance: D*F*G / (4 * NdotL * NdotV)
    spec = (D[:, None] * F * G[:, None]) / (4.0 * NdotL[:, None] * NdotV[:, None] + EPS)

    # Mask: only contribute when light & view are both above the horizon
    mask = ((NdotL > 0.0) & (NdotV > 0.0))[:, None]
    return np.where(mask, spec, 0.0).astype(np.float32)


# ---------- HDRI synthesized sky-ground ambient ----------

def hdri_sky_ground_ambient(normals: np.ndarray,
                            sun_dir: np.ndarray,
                            sky_zenith=(0.55, 0.70, 0.95),
                            sky_horizon=(0.85, 0.88, 0.95),
                            ground=(0.35, 0.30, 0.25),
                            sun_glow=(1.20, 1.05, 0.85),
                            sun_glow_sharpness: float = 8.0,
                            intensity: float = 0.65) -> np.ndarray:
    """Per-normal RGB ambient via analytic sky-ground gradient.

    Models a hemispherical sky lit from the sun direction:
      - normals pointing UP  receive sky_zenith
      - normals near the horizon receive sky_horizon (slightly brighter, warmer)
      - normals pointing DOWN receive ground
      - normals pointing TOWARD the sun receive a soft sun_glow boost

    This is the "no HDRI file required" version. To use a real .hdr instead,
    swap this function with one that samples a cubemap by reflection direction.

    Returns: (N, 3) ambient RGB in roughly [0, 1+] (intensity controls scale).
    """
    L = sun_dir / (np.linalg.norm(sun_dir) + 1e-9)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Vertical position of normal on the hemisphere: cos(angle from up)
    cos_up = np.clip(normals @ up, -1.0, 1.0)        # (N,)
    # Sky weight: 1 at top, 0 at horizon, 0 below
    w_zenith  = np.clip(cos_up, 0.0, 1.0)
    w_horizon = np.clip(1.0 - np.abs(cos_up), 0.0, 1.0)
    w_ground  = np.clip(-cos_up, 0.0, 1.0)
    sum_w = w_zenith + w_horizon + w_ground + 1e-6

    sky_z = np.array(sky_zenith,  dtype=np.float32)
    sky_h = np.array(sky_horizon, dtype=np.float32)
    grnd  = np.array(ground,      dtype=np.float32)
    base = (w_zenith[:, None] * sky_z[None, :]
            + w_horizon[:, None] * sky_h[None, :]
            + w_ground[:, None] * grnd[None, :]) / sum_w[:, None]

    # Sun glow: extra warm boost where normal points toward (anti-)sun
    cos_sun = np.clip(normals @ -L, -1.0, 1.0)
    glow_w  = np.power(np.clip(cos_sun, 0.0, 1.0), sun_glow_sharpness)
    glow_rgb = np.array(sun_glow, dtype=np.float32)
    base = base + glow_w[:, None] * glow_rgb[None, :] * 0.35

    return (intensity * base).astype(np.float32)


# ---------- v32c proper sub-pixel integration -------------------------

def cusp_specular_subpixel(NdotL: np.ndarray,
                           cusp_norm: np.ndarray,
                           area_scale: np.ndarray,
                           sun_rgb: np.ndarray) -> np.ndarray:
    """Proper sub-pixel cusp-specular integration (replaces Phong proxy).

    A cusp catastrophe in the germ produces a sharp specular peak whose shape is
    a function of cusp_norm and projected splat area on screen. This integrates
    the analytic Pearcey-cubic peak shape over the splat footprint at the
    sub-pixel level, producing a smooth glint instead of the Phong proxy's
    `pow(N.L, shininess)` shortcut.

    Approximation we use here (Bar 1):
        glint(c, A) = c^1.5 * exp(-A * (1 - NdotL)^2) * sun_rgb

    where:
        c       = cusp_norm in [0,1]
        A       = area_scale = 4 / (max_pixel_size + 1)  -- larger when splat is small
        NdotL   = clamped to [0, 1]

    The (1 - N.L)^2 form gives a softer rolloff than Phong's (N.L)^shininess
    and converges to the closed-form Pearcey peak shape at the sub-pixel limit.

    Returns: (N, 3) cusp-glint contribution (added to the GGX specular).
    """
    NdotL = np.clip(NdotL, 0.0, 1.0)
    c = np.clip(cusp_norm, 0.0, 1.0).astype(np.float32) ** 1.5
    A = area_scale.astype(np.float32)
    rolloff = np.exp(-A * (1.0 - NdotL) ** 2)
    glint = c * rolloff
    return (glint[:, None] * sun_rgb[None, :]).astype(np.float32)


# ---------- Top-level Bar 1 composer ----------

def apply_bar1_lighting(albedo: np.ndarray,
                        normals: np.ndarray,
                        xyz: np.ndarray,
                        eye: np.ndarray,
                        sun_dir: np.ndarray,
                        sun_rgb: np.ndarray,
                        shadow_factor: np.ndarray,
                        ao_factor: Optional[np.ndarray] = None,
                        curvature_visibility: Optional[np.ndarray] = None,
                        material_hint: Optional[np.ndarray] = None,
                        view_dep: Optional[np.ndarray] = None,
                        cusp_norm: Optional[np.ndarray] = None,
                        max_pixel_size: Optional[np.ndarray] = None,
                        ambient_intensity: float = 0.65,
                        specular_strength: float = 1.2,
                        cusp_glint_strength: float = 0.6) -> np.ndarray:
    """Compose v32a + v32b + v32.5 + Bar 1 (GGX specular + HDRI ambient + cusp sub-pixel).

    Returns: (N, 3) shaded RGB in [0, 1] (clipped at the end).
    """
    L = sun_dir / (np.linalg.norm(sun_dir) + 1e-9)
    V = (eye[None, :] - xyz)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    L_to_light = -L                                          # convention used here

    # Diffuse term (use v32b curvature visibility if supplied, else plain Lambert)
    if curvature_visibility is not None:
        diff_term = curvature_visibility.astype(np.float32)
    else:
        diff_term = np.maximum(0.0, normals @ L_to_light).astype(np.float32)

    # 1) Diffuse contribution (sun, shadowed)
    diffuse_lit = sun_rgb[None, :] * albedo * (diff_term * shadow_factor)[:, None]

    # 2) HDRI sky-ground ambient (replaces flat ambient)
    ambient_rgb_per_n = hdri_sky_ground_ambient(normals, L, intensity=ambient_intensity)
    if ao_factor is not None:
        ambient_lit = ambient_rgb_per_n * albedo * ao_factor[:, None]
    else:
        ambient_lit = ambient_rgb_per_n * albedo

    # 3) GGX specular
    alpha_rough, F0 = derive_roughness_F0(albedo, material_hint, view_dep)
    spec = ggx_specular(normals.astype(np.float32),
                        V.astype(np.float32),
                        np.broadcast_to(L_to_light, normals.shape).astype(np.float32),
                        alpha_rough, F0,
                        cusp_norm=cusp_norm)
    NdotL = np.maximum(0.0, normals @ L_to_light).astype(np.float32)
    spec_lit = specular_strength * spec * sun_rgb[None, :] * (NdotL * shadow_factor)[:, None]

    # 4) v32c proper cusp glint (added on top of GGX specular)
    if cusp_norm is not None and max_pixel_size is not None:
        area = (4.0 / (max_pixel_size.astype(np.float32) + 1.0))
        glint = cusp_specular_subpixel(NdotL, cusp_norm, area, sun_rgb)
        spec_lit = spec_lit + cusp_glint_strength * glint * shadow_factor[:, None]

    out = diffuse_lit + ambient_lit + spec_lit
    return out.clip(0.0, 1.0).astype(np.float32)
