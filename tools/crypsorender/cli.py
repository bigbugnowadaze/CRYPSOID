"""Command-line interface for crypsorender."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

from .output.contact_sheet import make_contact_sheet_3panel
from .output.metrics import compute_metrics
from .output.png import save_png
from .pipeline.camera import CameraParams
from .render import render_and_save


def cmd_render_comparison(args):
    """Render original PLY and CRYPSOID container side-by-side."""

    # Parse arguments
    original_path = Path(args.original_ply)
    phox_path = Path(args.crypsoid)
    out_dir_base = Path(args.out)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scene_name = original_path.stem.replace(" ", "_").replace(".zip", "")
    config_str = f"{args.size}x{args.size}"
    out_dir = out_dir_base / f"{timestamp}_{scene_name}_{config_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Camera parameters
    camera_params = CameraParams(
        yaw_deg=args.yaw,
        pitch_deg=args.pitch,
        distance=args.distance,
        fov_deg=args.fov,
        size=args.size,
    )

    print(f"Output directory: {out_dir}")

    # Render original PLY (Gaussian-only, all Tier C)
    print("Rendering original PLY...")
    result_ply = render_and_save(
        original_path,
        is_phox=False,
        out_dir=out_dir / "ply_render",
        camera_params=camera_params,
        use_sh=True,
        max_points=args.max_points,
    )
    print(f"  Loaded {result_ply['splat_count']} splats in {result_ply['load_time_s']:.3f}s")
    print(f"  Rendered in {result_ply['render_time_s']:.3f}s")

    # Render v28 as "truth" (Gaussian-only, all tiers via Gaussian path)
    print("Rendering v28 as truth (DC-only, Gaussian fallback)...")
    result_truth = render_and_save(
        phox_path,
        is_phox=True,
        out_dir=out_dir / "truth_render",
        camera_params=camera_params,
        use_sh=False,  # DC-only for numerical anchor
        max_points=args.max_points,
    )
    print(f"  Loaded {result_truth['splat_count']} splats in {result_truth['load_time_s']:.3f}s")
    print(f"  Rendered in {result_truth['render_time_s']:.3f}s")
    print(f"  Tier counts: {result_truth['tier_counts']}")

    # Render v28 with full SH (synthetic germ version for now)
    print("Rendering v28 with full SH (synthetic-germ preview)...")
    result_synthetic = render_and_save(
        phox_path,
        is_phox=True,
        out_dir=out_dir / "synthetic_germ_render",
        camera_params=camera_params,
        use_sh=True,  # Full SH evaluation
        max_points=args.max_points,
    )
    print(f"  Rendered in {result_synthetic['render_time_s']:.3f}s")

    # Save individual frames
    Image.fromarray(result_ply["framebuffer"]).save(out_dir / "frame_original_ply.png")
    Image.fromarray(result_truth["framebuffer"]).save(out_dir / "frame_truth.png")
    Image.fromarray(result_synthetic["framebuffer"]).save(out_dir / "frame_synthetic_germ.png")
    print("Saved individual frames")

    # Compute metrics
    print("Computing metrics...")
    metrics_ply_vs_truth = compute_metrics(result_ply["framebuffer"], result_truth["framebuffer"])
    metrics_ply_vs_synthetic = compute_metrics(result_ply["framebuffer"], result_synthetic["framebuffer"])
    metrics_truth_vs_synthetic = compute_metrics(result_truth["framebuffer"], result_synthetic["framebuffer"])

    print(f"  PLY vs Truth:       PSNR={metrics_ply_vs_truth['psnr_db']:.2f} dB, SSIM={metrics_ply_vs_truth['ssim']:.4f}")
    print(f"  PLY vs Synthetic:   PSNR={metrics_ply_vs_synthetic['psnr_db']:.2f} dB, SSIM={metrics_ply_vs_synthetic['ssim']:.4f}")
    print(f"  Truth vs Synthetic: PSNR={metrics_truth_vs_synthetic['psnr_db']:.2f} dB, SSIM={metrics_truth_vs_synthetic['ssim']:.4f}")

    # Create contact sheet
    print("Creating contact sheet...")
    img_ply = Image.fromarray(result_ply["framebuffer"])
    img_truth = Image.fromarray(result_truth["framebuffer"])
    img_synthetic = Image.fromarray(result_synthetic["framebuffer"])

    contact = make_contact_sheet_3panel(
        img_ply,
        img_truth,
        img_synthetic,
        label_a="Original PLY (Gaussian SH)",
        label_b="v28 Truth (DC-only Gaussian)",
        label_c="v28 Synthetic Germ (SH)",
        path=out_dir / "contact_sheet.png",
    )

    # Save manifest
    manifest = {
        "renderer_version": "0.1.0",
        "timestamp": timestamp,
        "scene_format": result_truth["scene_format"],
        "camera": {
            "yaw_deg": args.yaw,
            "pitch_deg": args.pitch,
            "distance": args.distance,
            "fov_deg": args.fov,
            "size": args.size,
        },
        "tier_dispatch_counts": {
            "truth": result_truth["tier_counts"],
            "synthetic_germ": result_synthetic["tier_counts"],
        },
        "sh_degree_used": 3 if result_truth["use_sh"] else 0,
        "code_paths_exercised": ["gaussian_inner_loop", "ewa_projection", "tile_compositing"],
        "honesty_caveat": "Synthetic germ auto-fitted by renderer (not from data).",
        "render_times_seconds": {
            "original_ply": result_ply["render_time_s"],
            "truth": result_truth["render_time_s"],
            "synthetic_germ": result_synthetic["render_time_s"],
        },
        "image_metrics": {
            "ply_vs_truth": metrics_ply_vs_truth,
            "ply_vs_synthetic": metrics_ply_vs_synthetic,
            "truth_vs_synthetic": metrics_truth_vs_synthetic,
        },
    }

    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nRender complete. Output at: {out_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="CRYPSOID crypsorender v0.1")
    subparsers = parser.add_subparsers(dest="command", help="command to run")

    # render-comparison subcommand
    cmd = subparsers.add_parser("render-comparison", help="Render PLY and .3dphox side-by-side")
    cmd.add_argument("--original-ply", required=True, help="Path to original .ply or .zip")
    cmd.add_argument("--crypsoid", required=True, help="Path to .3dphox render container")
    cmd.add_argument("--out", default="renders", help="Output directory")
    cmd.add_argument("--size", type=int, default=1024, help="Image size (pixels)")
    cmd.add_argument("--max-points", type=int, default=0, help="Max splats to render (0 = all)")
    cmd.add_argument("--yaw", type=float, default=35, help="Camera yaw (degrees)")
    cmd.add_argument("--pitch", type=float, default=18, help="Camera pitch (degrees)")
    cmd.add_argument("--distance", type=float, default=2.4, help="Camera distance (as fraction of radius)")
    cmd.add_argument("--fov", type=float, default=42, help="Field of view (degrees)")
    cmd.set_defaults(func=cmd_render_comparison)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
