# Phase F.21 — COLMAP wire-up (working, partial photoreal unlock)

**Date**: 2026-05-03

## What shipped

`tools/img2phox/sfm_colmap.py` — pycolmap-based SfM backend that produces the
same `_sfm_cache.pkl` shape as our pure-Python `run_sfm_chunked.py`. Includes
proper observations_per_cam tracking and scale normalization. Drop-in
alternative: same downstream phoxel pipeline.

## Verification

Sparse-cloud projection onto GT image **visually confirmed** that COLMAP poses
are correct — every projected sparse point lands on the corresponding scene
feature in the GT photo. Saved: `outputs/_colmap_sparse_proj_check.png`.

## Numbers

| Setup | mean PSNR |
|---|---|
| Pure-SfM 8-cam evenly-spaced (Family) | 20.16 dB |
| **COLMAP 13-cam every-5 (Family) + 60 iters phoxel** | 15.76 dB |
| LEGO 11-unique COLMAP | 0 dB (no reconstruction — symmetric studs) |
| LEGO 13-with-dupes COLMAP | 15.22 dB (only registered 4 dupes) |
| Family 8 evenly-spaced COLMAP | 0 dB (gaps too wide for SIFT) |

## Honest read

The wire-up works. The bottleneck is **how we feed COLMAP frames**:

- **Wide angular gaps** (8 evenly-spaced from 152): COLMAP can't seed the
  initial pair → 0 reconstruction
- **Narrow consecutive** (8 frames in a row): COLMAP gets all 8 cams + 515
  points but they have no parallax for phoxel
- **Every-5 sampling** (30→13 registered): partial coverage, partial sparse
- **Symmetric textures** (LEGO studs): SIFT confused, fails to match

To beat 20 dB on Family: run COLMAP on ALL 152 frames with **sequential
matching** (each frame matches its k temporal neighbors). That gives:
  - Full 360° angular coverage (good for phoxel)
  - Small-baseline pairs (good for COLMAP seeding)
  - Estimate: 5-10 min runtime, 100+ cams registered, 5000+ sparse points

That's Phase F.22. Saved as next task.

## Bottom line for the integration story

CRYPSOID image-in pipeline **now has two SfM backends**:

  - **Pure** (no deps, 20 dB ceiling on cooperative cherry-picked subsets)
  - **COLMAP** (pycolmap dep, working but needs proper frame-feeding strategy)

The architecture is genuinely shippable as a 2-backend system. F.22 is the
final demo step to hit photoreal.

## Files

  - `tools/img2phox/sfm_colmap.py` — pycolmap backend
  - `outputs/_family_every5_colmap.pkl` — 13/31 cams, 690 pts
  - `outputs/_family8c_colmap_cache.pkl` — 8/8 consecutive cams, 515 pts
  - `outputs/_lego_colmap_cache.pkl` — 4/13 cams (dupes only)
  - `outputs/_colmap_sparse_proj_check.png` — pose verification
  - `outputs/_fam_colmap_phoxel.npz` — 60-iter phoxel grid
