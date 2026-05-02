"""v32.5 — kNN-graph soft shadows + graph ambient occlusion.

Per docs/v32_5_shadows_spec.md.

For each phoxoid P with normal N, kNN neighbors {N_1..N_k}, and light dir L:

    Per neighbor:
        d   = N_i.pos - P.pos
        t   = dot(d, L)               # signed distance along L
        if t <= 0: continue            # neighbor is behind P relative to light
        perp = d - t * L
        r_i  = max(N_i.scale)         # effective support radius
        occlusion_i = exp(-|perp|^2 / (2*r_i^2)) * N_i.opacity
        shadow *= (1 - occlusion_i)

    shadow_factor = clamp(shadow, 0, 1)

Compose with diffuse term (multiply); ambient is unaffected by per-light shadow
but optionally darkened by graph-AO (separate function below).

Graph AO uses the same neighbor walk against the upper hemisphere defined by N:

    ao_total = sum over neighbors:
        d = N_i.pos - P.pos
        t = dot(d, N)
        if t <= 0: continue            # neighbor below the surface
        ao_i = exp(-|d|^2 / (2*R^2)) * N_i.opacity
    ao_factor = exp(-gamma * ao_total)

This is fully vectorized across all phoxoids (numpy einsum + broadcast).
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


def knn_shadow_factor(xyz: np.ndarray,
                      neighbors: np.ndarray,
                      neighbor_scales: np.ndarray,
                      neighbor_opacities: np.ndarray,
                      light_dir: np.ndarray,
                      strength: float = 1.0) -> np.ndarray:
    """Compute per-phoxoid shadow factor in [0, 1] for a single directional light.

    Args:
        xyz: (N, 3) splat positions in world space
        neighbors: (N, k) uint32 indices into the same array (kNN of each splat)
        neighbor_scales: (N, k, 3) per-neighbor scale (log-space, will be expd here)
        neighbor_opacities: (N, k) per-neighbor sigmoid opacity in [0, 1]
        light_dir: (3,) world-space LIGHT direction (FROM source toward scene)
                  shadow tests against -light_dir (toward the light)
        strength: multiplier on occlusion; default 1.0 matches the spec math

    Returns:
        shadow_factor: (N,) float32 in [0, 1] — multiplier on the diffuse term
    """
    N, k = neighbors.shape
    L = -light_dir / (np.linalg.norm(light_dir) + 1e-12)   # toward-light direction
    L = L.astype(np.float32)

    # Gather neighbor positions: (N, k, 3)
    npos = xyz[neighbors]
    # Offset from P to each neighbor
    d = (npos - xyz[:, None, :]).astype(np.float32)        # (N, k, 3)
    t = np.einsum('nki,i->nk', d, L)                       # (N, k) signed dist along L
    # Perpendicular component: d - t * L
    perp = d - t[..., None] * L[None, None, :]
    perp_sq = np.einsum('nki,nki->nk', perp, perp)         # |perp|^2

    # Effective neighbor radius: max scale axis (in linear, not log) per neighbor
    nsc_lin = np.exp(neighbor_scales).max(axis=2)          # (N, k)
    # Avoid division by zero
    r2 = np.maximum(nsc_lin * nsc_lin, 1e-8)               # (N, k)

    occlusion = np.exp(-0.5 * perp_sq / r2) * neighbor_opacities
    occlusion = occlusion * strength
    # Mask: only neighbors in the +L direction occlude
    in_front = t > 0.0
    occlusion = np.where(in_front, occlusion, 0.0)
    occlusion = np.clip(occlusion, 0.0, 1.0)

    # Multiplicative composition: shadow = product of (1 - occlusion_i)
    shadow = np.prod(1.0 - occlusion, axis=1)              # (N,)
    return shadow.astype(np.float32)


def knn_graph_ao(xyz: np.ndarray,
                 normals: np.ndarray,
                 neighbors: np.ndarray,
                 neighbor_opacities: np.ndarray,
                 ao_radius: float | None = None,
                 gamma: float = 1.0) -> np.ndarray:
    """Compute per-phoxoid ambient-occlusion factor in (0, 1] from kNN graph.

    Each neighbor that sits "above" the surface (in the +N hemisphere) blocks
    a fraction of the ambient hemisphere proportional to its proximity and
    opacity. Sum over neighbors, exponentiate to a multiplicative factor.

    Args:
        xyz, normals, neighbors, neighbor_opacities: as above
        ao_radius: scalar R (the falloff scale). If None, set to 2x median
                   per-splat neighbor distance.
        gamma: contrast knob (default 1.0)

    Returns:
        ao_factor: (N,) float32 in (0, 1] — multiplier on ambient term
    """
    N, k = neighbors.shape
    npos = xyz[neighbors]
    d = (npos - xyz[:, None, :]).astype(np.float32)        # (N, k, 3)
    d_len2 = np.einsum('nki,nki->nk', d, d)                # (N, k)

    if ao_radius is None:
        # Auto: 2x median neighbor distance
        ao_radius = 2.0 * float(np.sqrt(np.median(d_len2)))
    R2 = max(ao_radius * ao_radius, 1e-8)

    # Hemisphere test: neighbor must be in +N direction
    t = np.einsum('nki,ni->nk', d, normals.astype(np.float32))
    in_hemi = t > 0.0
    weight = np.exp(-0.5 * d_len2 / R2) * neighbor_opacities
    weight = np.where(in_hemi, weight, 0.0)
    ao_total = weight.sum(axis=1)                          # (N,)
    return np.exp(-gamma * ao_total).astype(np.float32)


def apply_v32_5_lighting(albedo: np.ndarray,
                         normals: np.ndarray,
                         shadow_factor: np.ndarray,
                         ao_factor: np.ndarray | None,
                         light_dir: np.ndarray,
                         ambient_rgb: np.ndarray,
                         sun_rgb: np.ndarray,
                         curvature_visibility: np.ndarray | None = None,
                         curvature_ambient_factor: np.ndarray | None = None) -> np.ndarray:
    """Compose v32a Lambert + (optional) v32b curvature + v32.5 shadows + AO.

    Args:
        albedo: (N, 3) base color in [0, 1]
        normals: (N, 3) unit normals
        shadow_factor: (N,) from knn_shadow_factor
        ao_factor: (N,) from knn_graph_ao, or None to skip
        light_dir: (3,) light direction
        ambient_rgb, sun_rgb: per-channel intensities
        curvature_visibility: optional v32b N.L curvature term (overrides plain N.L)
        curvature_ambient_factor: optional v32b ambient AO factor (multiplied with graph-AO)

    Returns:
        shaded: (N, 3) in [0, 1]
    """
    L = light_dir / (np.linalg.norm(light_dir) + 1e-12)
    if curvature_visibility is None:
        # Plain Lambert
        diffuse_term = np.maximum(0.0, normals @ (-L)).astype(np.float32)
    else:
        diffuse_term = curvature_visibility.astype(np.float32)

    # Apply shadow to diffuse term (sun is blocked by neighbors, ambient isn't)
    diffuse_lit = sun_rgb[None, :] * albedo * (diffuse_term * shadow_factor)[:, None]

    # Ambient: combine v32b curvature AO and v32.5 graph AO if present
    if curvature_ambient_factor is not None and ao_factor is not None:
        amb_factor = curvature_ambient_factor * ao_factor
    elif ao_factor is not None:
        amb_factor = ao_factor
    elif curvature_ambient_factor is not None:
        amb_factor = curvature_ambient_factor
    else:
        amb_factor = np.ones_like(shadow_factor)
    ambient_lit = ambient_rgb[None, :] * albedo * amb_factor[:, None]

    return (diffuse_lit + ambient_lit).clip(0.0, 1.0).astype(np.float32)
