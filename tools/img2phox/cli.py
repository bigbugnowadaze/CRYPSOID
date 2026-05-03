"""Top-level driver. End-to-end synthetic test for Phase F.4.

Pipeline:
    build synthetic scene  ->  render N "photos"  ->  SfM (synthetic mode)  ->
    quick-seed BlobBundle  ->  optional photometric refine  ->  encode .3dphox  ->
    re-render through CRYPSOID renderer  ->  compare PSNR vs ground truth.

Usage:
    python3 -m img2phox.cli --synth --n-cams 6 --refine 0
    python3 -m img2phox.cli --synth --n-cams 6 --refine 30   # with photometric refinement
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .synth_scene import build_ground_truth_scene, make_orbit_cameras, render_synthetic_scene
from .sfm import run_sfm_synthetic, pose_error
from .optimize import quick_seed_from_pointcloud, photometric_refine, render_blobs_to_photo
from .encode import encode_blobbundle_to_3dphox


def psnr(a, b, peak=1.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return 10 * np.log10(peak * peak / max(mse, 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--synth', action='store_true', help='Run synthetic round-trip')
    p.add_argument('--n-cams', type=int, default=6)
    p.add_argument('--width', type=int, default=200)
    p.add_argument('--height', type=int, default=150)
    p.add_argument('--refine', type=int, default=0,
                    help='Number of photometric refinement iterations (0 = quick-seed only)')
    p.add_argument('--out', type=Path,
                    default=Path('/sessions/ecstatic-sleepy-curie/mnt/Crypsoid/outputs/img2phox_synth_demo.3dphox'))
    p.add_argument('--out-render', type=Path,
                    default=Path('/sessions/ecstatic-sleepy-curie/mnt/Crypsoid/renders/crypsorender_v01/SHOWCASE_IMG2PHOX_synth.png'))
    args = p.parse_args()

    if not args.synth:
        raise SystemExit("Phase F.4: only --synth mode is implemented. Real photos = F.5+.")

    print("=" * 70)
    print("  Phase F.4 — synthetic image -> .3dphox round-trip")
    print("=" * 70)

    print(f"\n[1/6] Build synthetic ground-truth scene ...")
    t0 = time.perf_counter()
    cloud_gt = build_ground_truth_scene(n_cube_pts=300, n_sphere_pts=200, n_plane_pts=150)
    cams = make_orbit_cameras(n_cams=args.n_cams, distance=3.5, height=1.2,
                                fov_deg=50, width=args.width, height_px=args.height)
    print(f"  {len(cloud_gt)} ground-truth points, {len(cams)} cameras")

    print(f"\n[2/6] Render N synthetic 'photos' ...")
    photoset = render_synthetic_scene(cloud_gt, cams)
    print(f"  rendered {len(photoset)} photos at {args.width}x{args.height}")

    print(f"\n[3/6] Synthetic SfM (BA disabled for speed) ...")
    rec_cams, rec_cloud, cost = run_sfm_synthetic(
        cloud_gt, cams, noise_px=0.5,
        initial_pose_perturbation=0.0, initial_point_perturbation=0.0,
        run_ba=False,
    )
    errs = [pose_error(p.R, p.t, t.R, t.t)
            for p, t in zip(rec_cams.extrinsics, cams.extrinsics)]
    print(f"  pose error max  rot={max(e[0] for e in errs):.3f} deg, "
          f"trans={100*max(e[1] for e in errs):.2f} %")
    pt_err = np.linalg.norm(rec_cloud.xyz - cloud_gt.xyz, axis=1)
    print(f"  point error mean={pt_err.mean():.4f}, max={pt_err.max():.4f}")

    print(f"\n[4/6] Quick-seed BlobBundle from sparse cloud ...")
    blobs = quick_seed_from_pointcloud(rec_cloud, opacity=0.7, scale_kappa=0.5, k_neighbors=6)
    print(f"  {len(blobs)} blobs seeded")
    print(f"  scale range (sigma): "
          f"[{np.exp(blobs.scales.min()):.4f}, {np.exp(blobs.scales.max()):.4f}]")

    if args.refine > 0:
        print(f"\n[4b] Photometric refinement, {args.refine} iters ...")
        blobs = photometric_refine(blobs, photoset, rec_cams, n_iters=args.refine, verbose=True)

    print(f"\n[5/6] Encode -> {args.out} ...")
    sz = encode_blobbundle_to_3dphox(blobs, args.out)
    print(f"  wrote {sz:,} bytes")

    print(f"\n[6/6] Re-render via img2phox renderer + measure PSNR ...")
    rendered = render_blobs_to_photo(blobs, rec_cams, cam_idx=0)
    gt_photo = photoset.photos[0].image
    rec_psnr = psnr(rendered, gt_photo)
    print(f"  PSNR (camera 0, reconstruction vs ground-truth photo) = {rec_psnr:.2f} dB")

    # Side-by-side panel: gt | reconstructed | abs-diff
    diff = np.abs(rendered - gt_photo)
    panel = np.concatenate([gt_photo, rendered, diff * 4.0], axis=1).clip(0, 1)
    args.out_render.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((panel * 255).astype(np.uint8)).save(args.out_render)
    print(f"  saved side-by-side: {args.out_render}")

    print("\n" + "=" * 70)
    gate = rec_psnr >= 20.0
    print(f"  Phase F.4 acceptance gate: PSNR >= 20 dB ?  "
          f"{'PASS' if gate else 'FAIL'}  ({rec_psnr:.2f} dB)")
    print(f"  Total wall time: {time.perf_counter() - t0:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
