"""End-to-end PhoxBench scene runner.

For a given (scene, blob_budget):
    1. Generate the synthetic scene point cloud (10k pts).
    2. Cluster into B blobs.
    3. Fit Gaussian baseline (PCA-only) and Phoxoid (PCA + 5-coeff germ).
    4. Render: ground-truth scatter, Gaussian-blob reconstruction, Phoxoid reconstruction.
    5. Compute metrics:
        - per-cluster fit RMSE for both
        - image PSNR phoxoid vs Gaussian
        - "killer ratio" = how many Gaussian blobs needed to match phoxoid RMSE at budget B.
    6. Write outputs per spec §4.4.

Pure CPU.  No GPU dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure crypsorender is importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from phoxbench.scenes import make as make_scene
from phoxbench.fit import fit_blobs

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None


def _project_orthographic(pts: np.ndarray, size: int = 256, view: str = "iso"):
    """Cheap orthographic projection for visualization.
    view: 'iso' (default), 'top', 'front', 'side'.
    """
    if view == "top":
        x, y = pts[:, 0], -pts[:, 1]
    elif view == "front":
        x, y = pts[:, 0], -pts[:, 2]
    elif view == "side":
        x, y = pts[:, 1], -pts[:, 2]
    else:  # iso: a fixed isometric-ish projection
        ang = 30 * np.pi / 180
        cs, sn = np.cos(ang), np.sin(ang)
        x = pts[:, 0] - 0.5 * pts[:, 1]
        y = -0.5 * pts[:, 0] - 0.5 * pts[:, 1] - pts[:, 2]
    bb_min = np.array([x.min(), y.min()])
    bb_max = np.array([x.max(), y.max()])
    span = (bb_max - bb_min).max() * 1.1 + 1e-6
    cx, cy = bb_min + (bb_max - bb_min) * 0.5
    px = (x - cx) / span * size + size / 2
    py = (y - cy) / span * size + size / 2
    return px, py


def render_pointcloud(pts: np.ndarray, colors: np.ndarray, size: int = 256, view: str = "iso") -> np.ndarray:
    """Splat-free dot rendering for the scenes."""
    img = np.zeros((size, size, 3), dtype=np.float32)
    px, py = _project_orthographic(pts, size, view)
    pix_x = np.clip(px.astype(np.int32), 0, size - 1)
    pix_y = np.clip(py.astype(np.int32), 0, size - 1)
    np.add.at(img, (pix_y, pix_x), colors)
    # Normalize so brightest hit doesn't blow out
    if img.max() > 1:
        img = img / img.max()
    return (img * 255).astype(np.uint8)


def render_blobs(blobs: list, kind: str, scene_pts: np.ndarray, size: int = 256, view: str = "iso") -> np.ndarray:
    """Render a set of fitted blobs by sampling points from each blob's surface
    and drawing them.  For Gaussian blobs the surface is the tangent plane; for
    Phoxoid blobs it's the germ surface.

    This is NOT the same as the splat rasterizer — it's a 'reconstruction
    surface' visualization so we can see how well each blob's local model
    explains the scene.
    """
    pts_all = []
    cols_all = []
    rng = np.random.default_rng(42)
    for blob in blobs:
        # Sample (s, t) within the blob's lateral extent
        sa = max(blob.sigma[0], 1e-3); sb = max(blob.sigma[1], 1e-3)
        n_per = 64
        s = rng.uniform(-2 * sa, 2 * sa, n_per).astype(np.float32)
        t = rng.uniform(-2 * sb, 2 * sb, n_per).astype(np.float32)
        n_local = blob.predict_z(s, t)
        local = np.stack([s, t, n_local], axis=1)
        world = blob.center[None, :] + local @ blob.R.T
        pts_all.append(world)
        # Color: heat-mapped by which blob
        col = np.tile(np.array([0.7, 0.6, 0.5], dtype=np.float32), (n_per, 1))
        cols_all.append(col)
    pts_all = np.concatenate(pts_all, axis=0)
    cols_all = np.concatenate(cols_all, axis=0)
    return render_pointcloud(pts_all, cols_all, size=size, view=view)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    if Image is None:
        return img
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    pad = Image.new("RGB", (pil.width, pil.height + 20), (0, 0, 0))
    pad.paste(pil, (0, 0))
    d2 = ImageDraw.Draw(pad)
    d2.text((4, pil.height + 3), text, fill=(220, 220, 220), font=font)
    return np.array(pad)


def killer_ratio_search(scene_pts: np.ndarray, target_rmse: float,
                        max_budget: int = 4096, seed: int = 0) -> int:
    """Find the smallest Gaussian budget B_G whose RMSE <= target_rmse.
    Search by doubling: 16, 32, 64, ... up to max_budget."""
    budget = 16
    last_rmse = None
    while budget <= max_budget:
        gauss, _, g_rmse, _ = fit_blobs(scene_pts, n_blobs=budget, seed=seed)
        if g_rmse <= target_rmse:
            return budget
        last_rmse = g_rmse
        budget *= 2
    # Couldn't reach target within max_budget
    return -1


def run_scene(scene_name: str, budget: int, n_pts: int, seed: int, out_root: Path,
              do_killer: bool = True, image_size: int = 384):
    sc = make_scene(scene_name, n=n_pts, seed=seed)
    print(f"[{scene_name}] {sc.pts.shape[0]} pts; budget={budget}", flush=True)
    t0 = time.perf_counter()
    g_blobs, p_blobs, g_rmse, p_rmse = fit_blobs(sc.pts, n_blobs=budget, seed=seed)
    fit_time = time.perf_counter() - t0
    print(f"  fit: gaussian RMSE={g_rmse:.5f}, phoxoid RMSE={p_rmse:.5f}  ({fit_time:.1f}s)", flush=True)

    t0 = time.perf_counter()
    img_input = _label(render_pointcloud(sc.pts, sc.colors, size=image_size), "input scene")
    img_gauss = _label(render_blobs(g_blobs, "gauss", sc.pts, size=image_size), f"Gaussian blobs ({budget})")
    img_phox  = _label(render_blobs(p_blobs, "phoxoid", sc.pts, size=image_size), f"Phoxoid blobs ({budget})")
    render_time = time.perf_counter() - t0

    # Side-by-side
    sbs = np.concatenate([img_input, img_gauss, img_phox], axis=1)
    img_diff = np.abs(img_gauss.astype(np.int16) - img_phox.astype(np.int16)).clip(0, 255).astype(np.uint8)
    img_diff = _label(img_diff, "|gauss - phoxoid| diff")

    # Image PSNR phoxoid vs Gaussian
    a = img_gauss[:, :, :].astype(np.float32) / 255
    b = img_phox[:, :, :].astype(np.float32) / 255
    mse = float(((a - b) ** 2).mean())
    psnr = -10 * np.log10(mse + 1e-12)

    # Killer ratio
    killer = -1
    if do_killer and p_rmse > 0:
        killer = killer_ratio_search(sc.pts, target_rmse=p_rmse, seed=seed)

    out_dir = out_root / f"{scene_name}_b{budget}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if Image is not None:
        Image.fromarray(img_input).save(out_dir / "input_preview.png")
        Image.fromarray(img_gauss).save(out_dir / "gaussian_render.png")
        Image.fromarray(img_phox).save(out_dir / "phoxoidal_render.png")
        Image.fromarray(sbs).save(out_dir / "side_by_side.png")
        Image.fromarray(img_diff).save(out_dir / "error_heatmap.png")

    metrics = {
        "scene": scene_name,
        "scene_description": sc.description,
        "blob_budget": budget,
        "n_pts": int(sc.pts.shape[0]),
        "fit_rmse": {"gaussian": g_rmse, "phoxoid": p_rmse},
        "fit_advantage_phoxoid_over_gaussian": (g_rmse / p_rmse) if p_rmse > 0 else None,
        "image_psnr_gauss_vs_phoxoid_db": psnr,
        "killer_ratio_gaussian_blobs_to_match_phoxoid": killer,
        "killer_ratio_normalized": (killer / budget) if killer > 0 else None,
        "fit_seconds": fit_time,
        "render_seconds": render_time,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    md = [
        f"# PhoxBench: {scene_name} (B={budget})",
        f"",
        f"{sc.description}",
        f"",
        f"- Gaussian RMSE: {g_rmse:.5f}",
        f"- Phoxoid RMSE:  {p_rmse:.5f}",
        f"- Phoxoid advantage on fit RMSE: {g_rmse/p_rmse:.3f}x" if p_rmse > 0 else "",
        f"- Image PSNR (gauss vs phoxoid): {psnr:.2f} dB",
        f"- Killer ratio (Gaussian budget to match phoxoid RMSE @ {budget}): {killer}{' ('+str(killer/budget)+'x)' if killer > 0 else ''}",
    ]
    (out_dir / "report.md").write_text("\n".join(md))
    print(f"  -> {out_dir}", flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="all", help="scene name or 'all'")
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--budgets", type=int, nargs="+",
                    help="optional list of budgets; overrides --budget")
    ap.add_argument("--n-pts", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("phoxbench/runs"))
    ap.add_argument("--no-killer", action="store_true",
                    help="skip the (expensive) killer-ratio search")
    args = ap.parse_args()

    from phoxbench.scenes import SCENES
    scene_list = sorted(SCENES) if args.scene == "all" else [args.scene]
    budgets = args.budgets if args.budgets else [args.budget]

    all_metrics = []
    for sn in scene_list:
        for b in budgets:
            m = run_scene(sn, b, args.n_pts, args.seed, args.out,
                          do_killer=not args.no_killer)
            all_metrics.append(m)

    summary_path = args.out / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nSummary saved to {summary_path}")

    # Print compact table
    print(f"\n{'scene':<14} {'B':>5}  {'g_rmse':>9}  {'p_rmse':>9}  {'adv':>6}  {'killer':>7}")
    for m in all_metrics:
        adv = m.get("fit_advantage_phoxoid_over_gaussian")
        kr  = m.get("killer_ratio_gaussian_blobs_to_match_phoxoid", -1)
        print(f"  {m['scene']:<14} {m['blob_budget']:>5d}  {m['fit_rmse']['gaussian']:>9.5f}  "
              f"{m['fit_rmse']['phoxoid']:>9.5f}  {adv if adv else 0:>6.2f}  {kr:>7d}")


if __name__ == "__main__":
    main()
