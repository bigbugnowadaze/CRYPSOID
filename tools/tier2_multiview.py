"""Multi-view renderer + per-view metrics — Tier 1.5 item 5 + Tier 2.

For each camera in a JSON list, render Audi at v28 EXACT archive, compute
PSNR/SSIM (full + masked) vs the same scene rendered from the original PLY at
the same camera. Aggregate over all views: mean / median / worst.

Designed to be sandbox-friendly: each per-view render is one
`render_phox_chunked.py` invocation, so the bash sandbox sees small batches.
This script is the orchestrator.

Usage:
    python3 tools/tier2_multiview.py \
        --cameras renders/crypsorender_v01/multiview_cams.json \
        --out    renders/crypsorender_v01/multiview \
        --max-points 60000 --size 384
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def _state_dir(out: Path, kind: str, cam_id: str) -> Path:
    return Path(f"/tmp/state_mv_{kind}_{cam_id}")


def render_one(scene_path: Path, is_phox: bool, cam: dict, kind: str,
               state_dir: Path, out_png: Path,
               size: int = 384, max_points: int = 60000,
               batch_size: int = 80000):
    """Render a single view in chunks, then finalize."""
    state_dir.mkdir(parents=True, exist_ok=True)
    init_args = [
        "python3", "tools/render_phox_chunked.py",
        "--scene", str(scene_path),
        "--size", str(size),
        "--max-points", str(max_points),
        "--use-sh",
        "--yaw", str(cam["yaw_deg"]),
        "--pitch", str(cam["pitch_deg"]),
        "--distance", str(cam["distance"]),
        "--fov", str(cam["fov_deg"]),
        "--state-dir", str(state_dir),
        "--init",
    ]
    if is_phox:
        init_args.append("--is-phox")
    subprocess.run(init_args, check=True, capture_output=True)
    # Render in repeated batches until done
    for _ in range(20):
        r = subprocess.run([
            "python3", "tools/render_phox_chunked.py",
            "--state-dir", str(state_dir),
            "--batch", str(batch_size),
            "--mode", "gaussian",   # for fair PLY-vs-archive comparison; gaussians on both
        ], check=True, capture_output=True, text=True)
        if "already done" in r.stdout or "already done" in r.stderr:
            break
    subprocess.run([
        "python3", "tools/render_phox_chunked.py",
        "--state-dir", str(state_dir),
        "--finalize", "--out", str(out_png),
    ], check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ply", type=Path,
                    default=Path("/sessions/ecstatic-sleepy-curie/mnt/Crypsoid/inputs/audi/Audi A5 Sportback.zip"))
    ap.add_argument("--archive", type=Path,
                    default=Path("/sessions/ecstatic-sleepy-curie/mnt/Crypsoid/outputs/v28_sh_vq_exact_archive_container.3dphox"))
    ap.add_argument("--render", type=Path,
                    default=Path("/sessions/ecstatic-sleepy-curie/mnt/Crypsoid/outputs/v28_sh_vq_render_container.3dphox"))
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--max-points", type=int, default=60000)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cams: List[dict] = json.loads(args.cameras.read_text())
    print(f"rendering {len(cams)} cameras x 3 (PLY, v28-archive, v28-render)", flush=True)

    sys.path.insert(0, str(Path(__file__).parent))
    from eval_metrics import load_image, alpha_from_render, compute_all

    rows = []
    for c in cams:
        cid = c["cam_id"]
        ply_png = args.out / f"ply_{cid}.png"
        arc_png = args.out / f"archive_{cid}.png"
        ren_png = args.out / f"render_{cid}.png"
        if not ply_png.exists():
            print(f"  [{cid}] PLY ...", flush=True)
            render_one(args.ply, False, c, "ply", _state_dir(args.out, "ply", cid),
                       ply_png, size=args.size, max_points=args.max_points)
        if not arc_png.exists():
            print(f"  [{cid}] archive ...", flush=True)
            render_one(args.archive, True, c, "arc", _state_dir(args.out, "arc", cid),
                       arc_png, size=args.size, max_points=args.max_points)
        if not ren_png.exists():
            print(f"  [{cid}] render ...", flush=True)
            render_one(args.render, True, c, "ren", _state_dir(args.out, "ren", cid),
                       ren_png, size=args.size, max_points=args.max_points)

        ply_img = load_image(ply_png)
        arc_img = load_image(arc_png)
        ren_img = load_image(ren_png)
        mask = alpha_from_render(ply_img, threshold=0.05)
        m_arc = compute_all(ply_img, arc_img, mask)
        m_ren = compute_all(ply_img, ren_img, mask)
        rows.append({
            "cam_id": cid, **{f"yaw_deg pitch_deg distance fov_deg".split()[i]: c[k] for i, k in enumerate(["yaw_deg","pitch_deg","distance","fov_deg"])},
            "archive": m_arc, "render": m_ren,
        })
        print(f"    archive masked PSNR={m_arc['masked_psnr_db']:.2f}dB SSIM={m_arc['masked_ssim']:.4f}  "
              f"render masked PSNR={m_ren['masked_psnr_db']:.2f}dB SSIM={m_ren['masked_ssim']:.4f}", flush=True)

    # Aggregate
    import numpy as np
    arc_p = np.array([r["archive"]["masked_psnr_db"] for r in rows])
    arc_s = np.array([r["archive"]["masked_ssim"] for r in rows])
    ren_p = np.array([r["render"]["masked_psnr_db"] for r in rows])
    ren_s = np.array([r["render"]["masked_ssim"] for r in rows])
    summary = {
        "n_views": len(rows),
        "v28_archive_vs_ply": {
            "psnr_mean_db": float(arc_p.mean()), "psnr_median_db": float(np.median(arc_p)), "psnr_worst_db": float(arc_p.min()),
            "ssim_mean": float(arc_s.mean()),  "ssim_median": float(np.median(arc_s)),  "ssim_worst": float(arc_s.min()),
        },
        "v28_render_vs_ply": {
            "psnr_mean_db": float(ren_p.mean()), "psnr_median_db": float(np.median(ren_p)), "psnr_worst_db": float(ren_p.min()),
            "ssim_mean": float(ren_s.mean()),  "ssim_median": float(np.median(ren_s)),  "ssim_worst": float(ren_s.min()),
        },
        "rows": rows,
    }
    (args.out / "multiview_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
