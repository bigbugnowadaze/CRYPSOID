"""PhoxBench Tier 0 — synthetic stress scenes.

Each generator returns:
    pts:    (n, 3) float32 ground-truth point cloud
    colors: (n, 3) float32 in [0,1] for visualization
    name:   short scene identifier

The scenes isolate specific geometric conditions where phoxoidal blobs
should structurally beat ellipsoidal Gaussians (or where they shouldn't).
Scenes 1-6 cover the spec §4.1 list.

Why each one matters:
    1. plane:        sanity baseline (both should match)
    2. sphere:       smooth curvature (gentle phoxoid advantage)
    3. saddle:       opposite-sign principal curvatures
    4. fold:         smooth fold surface; cubic germ should win
    5. cusp:         Pearcey cusp; Gaussians fundamentally cannot represent this
    6. thin sheet:   two parallel layers; opacity blending test
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Scene:
    name: str
    pts: np.ndarray         # (n, 3) world points
    colors: np.ndarray      # (n, 3) [0,1] RGB
    description: str        # one-line human description


def _color_by_height(z: np.ndarray) -> np.ndarray:
    """Simple height-mapped color for visualization."""
    z_n = (z - z.min()) / (z.max() - z.min() + 1e-9)
    rgb = np.stack([0.3 + 0.6 * z_n, 0.5 + 0.3 * (1 - z_n), 0.6 + 0.4 * z_n], axis=1)
    return np.clip(rgb, 0, 1).astype(np.float32)


def scene_plane(n: int = 10000, seed: int = 0) -> Scene:
    rng = np.random.default_rng(seed)
    s = rng.uniform(-1, 1, n).astype(np.float32)
    t = rng.uniform(-1, 1, n).astype(np.float32)
    pts = np.stack([s, t, np.zeros_like(s, dtype=np.float32)], axis=1)
    return Scene("plane", pts, _color_by_height(pts[:, 2]),
                 "z = 0 plane over [-1,1]^2")


def scene_sphere(n: int = 10000, seed: int = 1) -> Scene:
    rng = np.random.default_rng(seed)
    # Sample upper hemisphere by uniform spherical coords
    phi = rng.uniform(0, 2 * np.pi, n)
    cos_theta = rng.uniform(0.05, 1.0, n)         # avoid pole degeneracy
    sin_theta = np.sqrt(1 - cos_theta * cos_theta)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = cos_theta
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    return Scene("sphere", pts, _color_by_height(pts[:, 2]),
                 "upper hemisphere of unit sphere")


def scene_saddle(n: int = 10000, seed: int = 2) -> Scene:
    rng = np.random.default_rng(seed)
    s = rng.uniform(-1, 1, n).astype(np.float32)
    t = rng.uniform(-1, 1, n).astype(np.float32)
    z = (s * s - t * t).astype(np.float32)
    pts = np.stack([s, t, z], axis=1)
    return Scene("saddle", pts, _color_by_height(pts[:, 2]),
                 "z = s^2 - t^2  (opposite-sign curvatures)")


def scene_fold(n: int = 10000, seed: int = 3) -> Scene:
    """Fold caustic surface: parametrise a fold by (s, t), z = s^3 - 3 s t^2 / 5.
    This is a genuine Whitney-fold-like surface that vanilla quadratic Gaussians
    smear through.  Sample s in [-1, 1], t in [-0.8, 0.8].
    """
    rng = np.random.default_rng(seed)
    s = rng.uniform(-1, 1, n).astype(np.float32)
    t = rng.uniform(-0.8, 0.8, n).astype(np.float32)
    z = (s * (s * s - 3.0 * t * t) * 0.2).astype(np.float32)  # scale down so |z|<1
    pts = np.stack([s, t, z], axis=1)
    return Scene("fold", pts, _color_by_height(pts[:, 2]),
                 "fold-like surface z = 0.2 s (s^2 - 3 t^2)")


def scene_cusp(n: int = 10000, seed: int = 4) -> Scene:
    """Pearcey cusp surface from the thesis: z = (s^3 - 3 s t^2) / 4.
    The cubic Pearcey term IS the s(s^2 - 3 t^2) generator.  This should be
    the place where the cubic phoxoidal germ visibly beats Gaussians.
    """
    rng = np.random.default_rng(seed)
    s = rng.uniform(-1, 1, n).astype(np.float32)
    t = rng.uniform(-1, 1, n).astype(np.float32)
    z = (s * (s * s - 3.0 * t * t) * 0.25).astype(np.float32)
    pts = np.stack([s, t, z], axis=1)
    return Scene("cusp", pts, _color_by_height(pts[:, 2]),
                 "Pearcey cusp surface z = 0.25 s (s^2 - 3 t^2)")


def scene_thin_sheet(n: int = 10000, seed: int = 5, gap: float = 0.04) -> Scene:
    """Two parallel planes at z = +/- gap/2.  Tests whether the renderer can
    keep them resolved instead of blurring them into one fat sheet."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(-1, 1, n).astype(np.float32)
    t = rng.uniform(-1, 1, n).astype(np.float32)
    sign = rng.choice([-1, 1], n).astype(np.float32)
    z = sign * (gap / 2)
    pts = np.stack([s, t, z], axis=1)
    colors = np.where(sign[:, None] > 0, np.array([1, 0.4, 0.4]), np.array([0.4, 0.4, 1])).astype(np.float32)
    return Scene("thin_sheet", pts, colors,
                 f"two parallel planes z = +/- {gap/2}")


SCENES = {
    "plane": scene_plane,
    "sphere": scene_sphere,
    "saddle": scene_saddle,
    "fold": scene_fold,
    "cusp": scene_cusp,
    "thin_sheet": scene_thin_sheet,
}


def make(name: str, n: int = 10000, seed: int = 0) -> Scene:
    """Factory."""
    if name not in SCENES:
        raise ValueError(f"unknown scene {name!r}; choices: {sorted(SCENES)}")
    return SCENES[name](n=n, seed=seed)


if __name__ == "__main__":
    import argparse, json
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=sorted(SCENES), required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("phoxbench_scene.npz"))
    args = ap.parse_args()
    sc = make(args.scene, n=args.n, seed=args.seed)
    np.savez(args.out, pts=sc.pts, colors=sc.colors)
    print(f"{sc.name}: {sc.pts.shape[0]} pts, bbox = {sc.pts.min(0).tolist()} -> {sc.pts.max(0).tolist()}")
    print(f"  saved to {args.out}")
