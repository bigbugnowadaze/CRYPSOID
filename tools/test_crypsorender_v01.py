#!/usr/bin/env python3
"""Test script for crypsorender v0.1 — runs a small test render."""

import sys
import time
from pathlib import Path

# Add crypsorender to path
sys.path.insert(0, str(Path(__file__).parent))

from crypsorender.io.ply_loader import load_ply
from crypsorender.io.phox_loader import load_3dphox
from crypsorender.pipeline.camera import CameraParams
from crypsorender.render import render_and_save


def main():
    # Paths
    audi_ply = Path(__file__).parent.parent / "inputs" / "audi" / "Audi A5 Sportback.zip"
    v28_phox = Path(__file__).parent.parent / "outputs" / "v28_sh_vq_render_container.3dphox"
    out_base = Path(__file__).parent.parent / "renders"

    if not audi_ply.exists():
        print(f"Error: {audi_ply} not found")
        return 1
    if not v28_phox.exists():
        print(f"Error: {v28_phox} not found")
        return 1

    print("=" * 70)
    print("CRYPSORENDER v0.1 TEST")
    print("=" * 70)

    # Camera params
    params = CameraParams(
        yaw_deg=35,
        pitch_deg=18,
        distance=2.4,
        fov_deg=42,
        size=1024,
    )

    # Test 1: Load PLY
    print("\n[TEST 1] Loading PLY...")
    t0 = time.perf_counter()
    ply_scene = load_ply(audi_ply)
    print(f"  Loaded {ply_scene.n} splats in {time.perf_counter() - t0:.3f}s")
    print(f"  XYZ range: [{ply_scene.xyz.min():.3f}, {ply_scene.xyz.max():.3f}]")

    # Test 2: Load .3dphox
    print("\n[TEST 2] Loading .3dphox...")
    t0 = time.perf_counter()
    phox_scene = load_3dphox(v28_phox)
    print(f"  Loaded {phox_scene.n} splats in {time.perf_counter() - t0:.3f}s")
    print(f"  Tier distribution: A={int((phox_scene.tier == 0).sum())}, B={int((phox_scene.tier == 1).sum())}, C={int((phox_scene.tier == 2).sum())}")

    # Test 3: Render small subset (to test pipeline)
    print("\n[TEST 3] Rendering small test (5000 splats, 512x512)...")
    params_small = CameraParams(yaw_deg=35, pitch_deg=18, distance=2.4, fov_deg=42, size=512)
    t0 = time.perf_counter()
    result = render_and_save(audi_ply, is_phox=False, out_dir=out_base / "test_small", camera_params=params_small, use_sh=True, max_points=5000)
    elapsed = time.perf_counter() - t0
    print(f"  Completed in {elapsed:.3f}s")
    print(f"  Result: {result['image_path']}")
    print(f"  Framebuffer range: [{result['framebuffer'].min()}, {result['framebuffer'].max()}]")

    print("\n" + "=" * 70)
    print("TESTS PASSED")
    print("=" * 70)
    print("\nTo run the full v0.1 deliverable:")
    print(f"  python3 -m crypsorender.cli render-comparison \\")
    print(f"    --original-ply {audi_ply} \\")
    print(f"    --crypsoid {v28_phox} \\")
    print(f"    --out {out_base} \\")
    print(f"    --size 1024 \\")
    print(f"    --max-points 0")

    return 0


if __name__ == "__main__":
    sys.exit(main())
