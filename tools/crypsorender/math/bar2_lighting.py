"""Bar 2 — Full PBR composer.

Uses:
    - bar1_lighting.ggx_specular        — same Cook-Torrance / GGX BRDF
    - material_decompose.decompose_pbr  — proper per-splat (albedo, metallic, roughness, F0, kd)
    - environment.ProceduralEnvironment / HDRIEnvironment — env sampling for ambient + reflections
    - cusp sub-pixel glint (carried over from Bar 1)

Adds:
    - Environment-cubemap reflections: glossy / mirror splats reflect their surroundings
    - Energy conservation: kd * diffuse + (1 - kd) * specular  (handles metallics correctly)
    - Tinted ambient: ambient term derived from environment normal-direction sample
                      rather than fixed sky-ground gradient
    - Multi-scale roughness sampling: blurred env reads for high-roughness splats
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from .bar1_lighting import ggx_specular, cusp_specular_subpixel
from .environment import ProceduralEnvironment, HDRIEnvironment


def reflect_directions(N: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Reflection of -V about N. (M,3) -> (M,3) unit."""
    NdotV = np.einsum('ij,ij->i', N, V)
    R = 2.0 * NdotV[:, None] * N - V
    return (R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def schlick_fresnel(F0: np.ndarray, VdotH: np.ndarray) -> np.ndarray:
    """Schlick Fresnel per-channel. F0 (M,3), VdotH (M,) -> (M,3)."""
    return (F0 + (1.0 - F0) * np.power(1.0 - VdotH.clip(0, 1), 5.0)[:, None]).astype(np.float32)


def apply_bar2_lighting(albedo: np.ndarray,
                        metallic: np.ndarray,
                        roughness: np.ndarray,
                        F0: np.ndarray,
                        kd: np.ndarray,
                        normals: np.ndarray,
                        xyz: np.ndarray,
                        eye: np.ndarray,
                        sun_dir: np.ndarray,
                        sun_rgb: np.ndarray,
                        environment: ProceduralEnvironment,
                        shadow_factor: np.ndarray,
                        ao_factor: Optional[np.ndarray] = None,
                        curvature_visibility: Optional[np.ndarray] = None,
                        cusp_norm: Optional[np.ndarray] = None,
                        max_pixel_size: Optional[np.ndarray] = None,
                        sun_strength: float = 1.4,
                        env_ambient_strength: float = 0.55,
                        env_reflection_strength: float = 0.85,
                        cusp_glint_strength: float = 0.55,
                        ) -> np.ndarray:
    """Compose v32a Lambert + v32b curvature + v32.5 shadows/AO + Bar 2 PBR + env reflections.

    Returns (N, 3) shaded RGB clamped to [0, 1].
    """
    EPS = 1e-9
    L = sun_dir / (np.linalg.norm(sun_dir) + EPS)
    L_to_light = -L                                    # convention: toward the light
    V = (eye[None, :] - xyz)
    V = (V / (np.linalg.norm(V, axis=1, keepdims=True) + EPS)).astype(np.float32)

    # ---------------- Direct sun (diffuse + GGX specular) ----------------
    if curvature_visibility is not None:
        diff_term = curvature_visibility.astype(np.float32)
    else:
        diff_term = np.maximum(0.0, normals @ L_to_light).astype(np.float32)
    NdotL = np.maximum(0.0, normals @ L_to_light).astype(np.float32)

    # Direct sun GGX specular (per Bar 1 module)
    spec_direct = ggx_specular(
        normals.astype(np.float32),
        V,
        np.broadcast_to(L_to_light, normals.shape).astype(np.float32),
        alpha=roughness, F0=F0, cusp_norm=cusp_norm,
    )

    # Diffuse Lambert, energy-conserved by kd  (kd is essentially 1 - metallic)
    diff_lit_sun = sun_strength * sun_rgb[None, :] * (kd[:, None] * albedo) \
                   * (diff_term * shadow_factor)[:, None]

    # Specular sun: weighted by NdotL (the BRDF was already energy-normalized)
    spec_lit_sun = sun_strength * sun_rgb[None, :] * spec_direct \
                   * (NdotL * shadow_factor)[:, None]

    # ---------------- Environment ambient (diffuse) ----------------
    # Per-splat ambient: sample environment in the normal direction, weighted by AO.
    env_ambient = environment.sample(normals.astype(np.float32))   # cheap normal-direction read
    if ao_factor is not None:
        env_ambient = env_ambient * ao_factor[:, None]
    diff_lit_env = env_ambient_strength * env_ambient * (kd[:, None] * albedo)

    # ---------------- Environment reflections (specular) ----------------
    # Sample environment in the reflection direction, blurred by roughness.
    R = reflect_directions(normals.astype(np.float32), V)
    env_refl = environment.sample_blurred(R, roughness, n_taps=4, seed=0)
    # Fresnel weight: how much of the incoming light goes to specular vs diffuse
    NdotV = np.maximum(0.0, np.einsum('ij,ij->i', normals, V))
    F = schlick_fresnel(F0, NdotV)                # (M, 3)
    # Energy of the env reflection: F * env_refl, gated by AO so reflections
    # don't punch through occluded splats.
    if ao_factor is not None:
        env_refl = env_refl * (0.5 + 0.5 * ao_factor[:, None])
    else:
        env_refl = env_refl * 0.85
    spec_lit_env = env_reflection_strength * F * env_refl

    # ---------------- Cusp sub-pixel glint (Bar 1, carried over) ----------------
    if cusp_norm is not None and max_pixel_size is not None:
        area = (4.0 / (max_pixel_size.astype(np.float32) + 1.0))
        glint = cusp_specular_subpixel(NdotL, cusp_norm, area, sun_rgb)
        glint_lit = cusp_glint_strength * glint * shadow_factor[:, None]
    else:
        glint_lit = np.zeros_like(albedo)

    # ---------------- Sum everything ----------------
    out = diff_lit_sun + spec_lit_sun + diff_lit_env + spec_lit_env + glint_lit
    return out.clip(0.0, 1.0).astype(np.float32)
    return out.clip(0.0, 1.0).astype(np.float32)
