"""PhoxBench Tier 1 — run phoxoid-vs-Gaussian on real meshes.

Wraps `phoxbench.run_scene.run_scene()` but uses real PLY pointclouds as input.
Reuses the same fit + render + killer-ratio harness, so results are directly
comparable to Tier 0 numbers in `summary.json`.

Usage:
    python3 -m phoxbench.run_mesh --ply inputs/stanford/happy_stand/happyStandRight_0.ply --name happy --budget 32
    python3 -m phoxbench.run_mesh --all
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from phoxbench.scenes_mesh import load_ply_pointcloud, normalize_pointcloud
from phoxbench.scenes import Scene, _color_by_height
from phoxbench import scenes as scenes_mod
from phoxbench import run_scene as run_scene_mod


def make_mesh_scene(name, ply_path, max_points=10000):
    pts = load_ply_pointcloud(ply_path, max_points=None)
    pts, _, _ = normalize_pointcloud(pts)
    if pts.shape[0] > max_points:
        import numpy as np
        rng = np.random.default_rng(0)
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    colors = _color_by_height(pts[:, 1])
    return Scene(name, pts, colors,
                 f"PLY mesh ({Path(ply_path).name}, {pts.shape[0]} pts after subsample)")


def run(name, ply_path, budgets, n_pts, out_root, do_killer=True):
    """Inject our scene into scenes.SCENES so run_scene.run_scene picks it up."""
    sc = make_mesh_scene(name, ply_path, max_points=n_pts)
    # Monkey-patch SCENES so run_scene_mod.make_scene returns our scene
    scenes_mod.SCENES[name] = lambda n=10000, seed=0, _sc=sc: _sc
    results = []
    for b in budgets:
        m = run_scene_mod.run_scene(name, b, n_pts, 0, out_root, do_killer=do_killer)
        results.append(m)
    return results


CANONICAL_TIER1 = [
    # (name, ply_path)
    ("happy",      Path("/sessions/ecstatic-sleepy-curie/mnt/Crypsoid/inputs/stanford/happy_stand/happyStandRight_0.ply")),
    ("armadillo",  Path("/sessions/ecstatic-sleepy-curie/mnt/Crypsoid/inputs/stanford/Armadillo_scans/ArmadilloBack_0.ply")),
    ("doom",       Path("/sessions/ecstatic-sleepy-curie/mnt/Crypsoid/inputs/Doom combat scene.ply")),
    ("audi",       Path("/tmp/audi_scene.ply")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", type=Path)
    ap.add_argument("--name", type=str, default="mesh")
    ap.add_argument("--all", action="store_true",
                    help="run every entry in CANONICAL_TIER1")
    ap.add_argument("--budgets", type=int, nargs="+", default=[32])
    ap.add_argument("--n-pts", type=int, default=10000)
    ap.add_argument("--out", type=Path,
                    default=Path("/sessions/ecstatic-sleepy-curie/mnt/Crypsoid/phoxbench/runs"))
    ap.add_argument("--no-killer", action="store_true")
    args = ap.parse_args()

    summary = []
    if args.all:
        for name, p in CANONICAL_TIER1:
            if not p.exists():
                print(f"  skip {name}: {p} not found")
                continue
            print(f"\n=== Tier 1: {name} ({p.name}) ===")
            summary.extend(run(name, p, args.budgets, args.n_pts, args.out, do_killer=not args.no_killer))
    else:
        if not args.ply or not args.ply.exists():
            raise SystemExit("--ply or --all required")
        summary = run(args.name, args.ply, args.budgets, args.n_pts, args.out, do_killer=not args.no_killer)

    out_path = args.out / "tier1_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nTier 1 summary -> {out_path}")
    print(f"\n{'name':<14} {'B':>5}  {'g_rmse':>9}  {'p_rmse':>9}  {'adv':>6}  {'killer':>7}  {'ratio':>7}")
    for m in summary:
        kr = m.get("killer_ratio_gaussian_blobs_to_match_phoxoid", -1)
        adv = m.get("fit_advantage_phoxoid_over_gaussian") or 0
        print(f"  {m['scene']:<12} {m['blob_budget']:>5d}  {m['fit_rmse']['gaussian']:>9.5f}  "
              f"{m['fit_rmse']['phoxoid']:>9.5f}  {adv:>6.2f}  {kr:>7d}  "
              f"{(kr/m['blob_budget']) if kr > 0 else 0:>7.2f}x")


if __name__ == "__main__":
    main()
