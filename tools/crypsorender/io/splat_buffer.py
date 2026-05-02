"""Canonical in-memory splat representation."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GermBuffer:
    """Phoxoidal germ data (v0.2+)."""
    k1: np.ndarray  # (n,) principal curvature 1
    k2: np.ndarray  # (n,) principal curvature 2
    # Higher-order terms reserved for v0.2+ (chi, omega, zeta, support_pot)


@dataclass
class CorrBuffer:
    """Tier B exact-residual correction (v0.2+)."""
    sh_residual: Optional[np.ndarray] = None  # (n, 45) float32, Tier B only


@dataclass
class SplatBuffer:
    """The canonical in-memory splat representation.

    All loaders (PLY, .3dphox v25/v27/v28) produce this.
    Renderer is agnostic to source format.
    """
    n: int                       # number of splats
    xyz: np.ndarray              # (n, 3) float32 — world position
    scales: np.ndarray           # (n, 3) float32 — log-space scales (3DGS convention)
    quats: np.ndarray            # (n, 4) float32 — unit quaternion (wxyz)
    opacities: np.ndarray        # (n,)   float32 — sigmoid logit
    sh_dc: np.ndarray            # (n, 3) float32 — degree-0 SH coefficients (RGB)
    sh_rest: Optional[np.ndarray] = None   # (n, 45) float32 — degrees 1–3 SH coefficients
    tier: Optional[np.ndarray] = None      # (n,)   uint8 — 0=A, 1=B, 2=C
    germ: Optional[GermBuffer] = None      # phoxoidal germ data
    correction: Optional[CorrBuffer] = None  # Tier B exact-residual correction
    source: str = ""             # source file path
    scene_format: str = ""       # "ply", "3dphox_v25", "3dphox_v28_render", etc.

    def __post_init__(self):
        """Validate shapes."""
        assert self.xyz.shape == (self.n, 3), f"xyz shape mismatch: {self.xyz.shape}"
        assert self.scales.shape == (self.n, 3), f"scales shape mismatch"
        assert self.quats.shape == (self.n, 4), f"quats shape mismatch"
        assert self.opacities.shape == (self.n,), f"opacities shape mismatch"
        assert self.sh_dc.shape == (self.n, 3), f"sh_dc shape mismatch"
        if self.sh_rest is not None:
            assert self.sh_rest.shape == (self.n, 45), f"sh_rest shape mismatch: {self.sh_rest.shape}"
        if self.tier is not None:
            assert self.tier.shape == (self.n,), f"tier shape mismatch"
