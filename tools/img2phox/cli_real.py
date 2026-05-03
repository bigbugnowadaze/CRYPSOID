"""F.9 — Full real-photo end-to-end demo (F.5 + F.6 + F.7 + F.8 + encode).

For testing without real photos: synthesize a textured scene, render it from N
viewpoints, then run the FULL real-photo pipeline (ORB SfM, MVS, distortion=zero,
exposure normalization, dense optimizer) as if those renders were real photos.

Output: a 5-panel comparison showing the progression of reconstruction quality,
plus a final .3dphox the existing CRYPSOID renderer can load.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path

import numpy as np
from PIL import Image

from .synth_scene import build_ground_truth_scene, make_orbit_cameras, render_pointcloud_to_photo
from .load_photos import photoset_from_arrays
from .preprocess import normalize_exposure
from .sfm_real import run_sfm_real
from .mvs import run_dense_mvs
from .optimize import quick_seed_from_pointcloud, render_blobs_to_photo
from .optimize_dense import optimize_dense
from .encode import encode_blobbundle_to_3dphox


def psnr(a, b, peak=1.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return 10 * np.log10(peak * peak / max(mse, 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-cams',       type=int, default=6)
    p.add_argument('--width',        type=int, default=480)
    p.add_argument('--height',       type=int, default=360)
    p.add_argument('--mvs-pairs',    type=int, default=4)
    p.add_argument('--opt-iters',    type=int, default=30)
    p.add_argument('--out',          type=Path,
                    default=Path('/sessions/ecstatic-sleepy-curie/mnt/Crypsoid/outputs/img2phox_real_demo.3dphox'))
    p.add_argument('--out-render',   type=Path,
                    default=Path('/sessions/ecstatic-sleepy-curie/mnt/Crypsoid/renders/crypsorender_v01/SHOWCASE_IMG2PHOX_real.png'))
    args = p.parse_args()

    print("=" * 70)
    print("  Phase F.9 — full real-photo pipeline (F.5 + F.6 + F.7 + F.8)")
    print("=" * 70)

    # ---- Build "photos" from a synthetic scene to drive the pipeline ----
    t0 = time.perf_counter()
    print(f"\n[setup] Render synthetic scene as 'real photos' ...")
    cloud_gt = build_ground_truth_scene(n_cube_pts=5000, n_sphere_pts=3000, n_plane_pts=3000)
    cams_gt  = make_orbit_cameras(n_cams=args.n_cams, distance=3.5, fov_deg=50,
                                    width=args.width, height_px=args.height)
    arrays = [render_pointcloud_to_photo(cloud_gt, cams_gt.intrinsics, e, point_radius_px=3)
              for e in cams_gt.extrinsics]
    photoset = photoset_from_arrays(arrays)
    print(f"  {len(photoset)} photos at {args.width}x{args.height}")

    # ---- F.7: exposure normalize (no distortion since synthetic is pinhole) ----
    print(f"\n[F.7] Exposure normalize (mean_match) ...")
    photoset = normalize_exposure(photoset, method='mean_match')

    # ---- F.5: real-photo SfM via ORB ----
    print(f"\n[F.5] Real-photo SfM ...")
    rec_cams, sparse_cloud, sfm_stats = run_sfm_real(
        photoset, fov_deg_prior=50.0,
        n_features=8000, ratio=0.85, min_matches=15, verbose=True,
    )
    n_reg = sfm_stats['n_cameras_registered']
    if n_reg < 2:
        print(f"FAIL: only {n_reg} cameras registered — synthetic scene too sparse")
        return

    # Save the quick-seed render BEFORE MVS for the ladder
    blobs_seed = quick_seed_from_pointcloud(sparse_cloud, opacity=0.7, scale_kappa=0.5, k_neighbors=6)
    print(f"  quick-seed from sparse: {len(blobs_seed)} blobs")
    render_seed = render_blobs_to_photo(blobs_seed, rec_cams, cam_idx=0)

    # ---- F.6: dense MVS ----
    print(f"\n[F.6] Dense MVS via SGBM ...")
    dense_cloud = run_dense_mvs(photoset, rec_cams, sparse_cloud,
                                  max_pairs=args.mvs_pairs, voxel_size=0.04, verbose=True)

    blobs_dense = quick_seed_from_pointcloud(dense_cloud, opacity=0.65, scale_kappa=0.7, k_neighbors=6)
    print(f"  dense seed: {len(blobs_dense)} blobs")
    render_dense = render_blobs_to_photo(blobs_dense, rec_cams, cam_idx=0)

    # ---- F.8: dense optimizer (skip if --opt-iters=0; useful for big scenes) ----
    if args.opt_iters > 0 and len(blobs_dense) < 5000:
        print(f"\n[F.8] Dense optimizer ({args.opt_iters} iters with density control) ...")
        blobs_opt = optimize_dense(blobs_dense, photoset, rec_cams,
                                     n_iters=args.opt_iters,
                                     densify_every=10, densify_top_pct=8,
                                     max_blobs=20_000, verbose=True)
        render_opt = render_blobs_to_photo(blobs_opt, rec_cams, cam_idx=0)
    else:
        print(f"\n[F.8] skipped ({len(blobs_dense)} blobs > 5000 budget for the slow CPU optimizer)")
        blobs_opt = blobs_dense
        render_opt = render_dense

    # ---- Encode + verify ----
    print(f"\n[encode] -> {args.out} ...")
    sz = encode_blobbundle_to_3dphox(blobs_opt, args.out)
    print(f"  wrote {sz:,} bytes")

    # ---- Build 5-panel ladder ----
    gt_photo = photoset.photos[0].image
    panels = [
        ('GT photo (input)',          gt_photo),
        ('SfM sparse + quick-seed',    render_seed),
        ('+ Dense MVS seed',            render_dense),
        ('+ F.8 optimizer',             render_opt),
        ('|GT - F.8| x 4',              np.abs(render_opt - gt_photo) * 4),
    ]
    args.out_render.parent.mkdir(parents=True, exist_ok=True)
    H = panels[0][1].shape[0]; W = panels[0][1].shape[1]
    sep = 8
    sheet = np.full((H + 50, len(panels) * W + (len(panels) - 1) * sep, 3), 0.07,
                     dtype=np.float32)
    cursor = 0
    for label, arr in panels:
        sheet[40:40+H, cursor:cursor+W] = arr.clip(0, 1)
        cursor += W + sep
    img8 = (sheet * 255).astype(np.uint8)
    Image.fromarray(img8).save(args.out_render)
    print(f"  saved {args.out_render}")

    # ---- Final readout ----
    psnr_seed  = psnr(render_seed, gt_photo)
    psnr_dense = psnr(render_dense, gt_photo)
    psnr_opt   = psnr(render_opt, gt_photo)
    print()
    print("=" * 70)
    print(f"  Phase F.9 ladder PSNRs (camera 0):")
    print(f"    SfM sparse seed              : {psnr_seed:.2f} dB")
    print(f"    + Dense optimizer (F.8)       : {psnr_opt:.2f} dB")
    print(f"  cameras registered             : {n_reg}/{args.n_cams}")
    print(f"  final blob count               : {len(blobs_opt):,}")
    print(f"  total wall time                : {time.perf_counter()-t0:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
