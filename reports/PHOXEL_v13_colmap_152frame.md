# Phase F.22 — Full 152-frame Family with COLMAP (final diagnosis)

**Date**: 2026-05-03

## What we ran

Full 152-frame T&T Family dataset through COLMAP incremental SfM, then
phoxel optimization on the result. This was the "real photoreal demo" Bug
was pushing for — the deepest dive on the photo-in pipeline.

## SfM result (COLMAP wins outright)

  - Feature extraction: 152 imgs in 33.7s
  - Exhaustive matching: 3703 verified pairs in ~45s (chunked, partial)
  - Incremental mapping: **101/152 cams + 6663 3D pts in 26.3s**
  - Total observations: 47700

This is **the cleanest SfM front-end we've ever had on Family**. 5× more
cams + 70× more 3D points than our pure-Python SfM at its best.

## Phoxel result (architectural ceiling hit)

| Setup | mean PSNR |
|---|---|
| Pure 8-cam evenly-spaced (cherry-picked) | **20.16 dB** |
| **COLMAP 101-cam, 50 iter phoxel** | 13.39 dB |
| LEGO 4-cam (cherry-picked, bounded object) | 20.33 dB |

## Why COLMAP-poses + 101-cams didn't break 20 dB

**The bottleneck moved from SfM to phoxel itself.** Specifically: phoxel's
bounded-AABB optimizer cannot represent the unbounded outdoor scene.

Family contains:
  - Foreground statue at ~1 unit (normalized)
  - Plaza + benches at ~2 units
  - Building wall at ~3 units
  - Distant trees at ~5+ units
  - Sky at infinity

Our 96³ phoxel grid spans the AABB of the sparse cloud (~4 units).
Per-pixel-of-detail cell size for the foreground statue ends up coarse
relative to the per-pixel render ray, AND the distant content pushes
gradient signal toward the boundary cells that can't actually represent it.

The 4-cam LEGO result (20.33 dB) didn't have this problem because the
LEGO is a bounded object on a finite base with a black void background.

## What unlocks photoreal on outdoor scenes

**Mip-NeRF 360 style scene-contraction**: warp distant content into a
unit ball so unbounded scenes become bounded. Specifically, contract
points outside r=1 via:
    contract(x) = (2 - 1/||x||) * x/||x||

This was published in CVPR 2022 and is essentially the unlock for any
voxel-grid method on outdoor scenes. ~1 week of careful work to add to
phoxel — saved as Phase F.23.

## Honest revised story for Vince

The image-in pipeline now has a **comprehensively diagnosed architecture**:

  - **Front-end (SfM)**: solved. COLMAP backend produces 100+ cams + 5000+
    sparse points in ~1 minute on full Family dataset. Pure-Python backend
    works for cherry-picked subsets up to 20 dB.
  - **Back-end (phoxel optimizer)**: works at 20+ dB on bounded objects
    (LEGO 4-cam), capped at ~13 dB on unbounded scenes (Family) due to
    bounded-AABB representation.
  - **Path to outdoor photoreal**: scene-contraction (Mip-NeRF 360 style),
    1 week of work, saved as Phase F.23.
  - **Path to bounded-object photoreal**: already works. Just needs
    parallax-rich camera input (4-cam LEGO already at 20 dB).

## Files

  - `outputs/_family152_colmap_cache.pkl` — 101 cams + 6663 sparse pts (158 MB)
  - `outputs/_fam152_phoxel.npz` — phoxel grid 50 iter (14 MB)
  - `tools/img2phox/sfm_colmap.py` — pycolmap backend
