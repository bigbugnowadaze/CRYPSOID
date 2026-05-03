# Phase F.13.2 — Scaling phoxel to more views (the honest answer)

**Date**: 2026-05-03
**Question**: Can we use the 152 views Family provides?

## TL;DR

Yes we can run more views. **But more views with budget-limited SfM gave WORSE
results than fewer views with high-quality SfM.** The SfM front-end is the
real bottleneck, not the phoxel optimization.

## Numbers (all at 96³ uniform grid, ~30-50 iters per camera)

| Views | SfM mode                            | AABB | Mean PSNR |
|-------|-------------------------------------|------|-----------|
|     8 | full-pair + global BA (300 nfev)    | 0.67 | **20.16** |
|    12 | full-pair + global BA (200 nfev)    | 0.67 |   15.97   |
|    20 | temporal-window=8, no BA            | 0.13 |   13.73   |
|    30 | temporal-window=4, no BA            | 0.65 |   14.18   |

## Why scaling-up failed (the diagnosis)

To fit 30-cam SfM in the 45s shell timeout, we used:
  - **Temporal-window matching** (only pairs (i,j) with |i-j|≤window) — cuts
    O(N²) matching cost, but means cams far apart in index never share features
  - **No global BA** — saves 30+s, but leaves pose errors un-refined
  - **Lower per-cam features** (1500 vs 8000) — fewer keypoints means fewer
    triangulatable points

Result: only 6 sparse 3D points survived global triangulation across the
20-cam graph. With so few constraints, the pose accuracy degrades. The
phoxel optimizer then has to fit images to noisy poses — the surfaces it
finds are inconsistent across views, and PSNR drops.

For comparison, the 8-cam SfM had 95 sparse points and full BA. Pose quality
was much higher → phoxel could find consistent surfaces → better PSNR.

## The actual ceiling

This isn't a phoxel problem — it's a SfM problem. To break past 20 dB we need:
  - Background-process SfM (longer than shell budget) for proper full-pair BA
  - OR cheaper essential-matrix verification (currently 0.23s/pair)
  - OR a different SfM library entirely (COLMAP, OpenSfM)

We **can scale views** (proven by 30/30 registrations). What we **cannot do
in 45s** is scale views WITH the SfM accuracy needed to make those views useful.

## What was shipped

  - `tools/img2phox/run_sfm_chunked.py` — two-stage chunked SfM (graph cache
    + pose solve) with temporal-window matching
  - `outputs/_phoxel_sfm_cache_30.pkl` — 30/30 cams registered
  - `outputs/_phoxel_sfm_cache_20.pkl` — 20/20 cams registered
  - `outputs/_phoxel_sfm_cache_12.pkl` — 12/12 cams registered with proper BA
  - `outputs/_phoxel_grid_30cam.npz`   — 14.18 dB result
  - `outputs/_phoxel_grid_20cam.npz`   — 13.73 dB result

## Honest recommendation

Stay on the 8-cam result (20.16 dB) for any showcase use. The Phoxel pipeline
itself is sound — it's the SfM front-end that's the next-attack target, and
that's a meaningful chunk of work (Phase F.13.3 — proper BA on 30+ cams).
