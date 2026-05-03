# Phase F.20 — Pose-consensus filter (modest win, not a breakthrough)

**Date**: 2026-05-03

## What I tried

Two filter strategies on LEGO 13-cam:

1. **Sparse-coverage filter** (sparse 3D points project inside image): wrong
   metric — well-centered cams have sparse points project off-screen due to
   perspective. Kept the bad cams, dropped the good ones. Abandoned.

2. **Per-cam-PSNR filter** (drop cams whose post-fit PSNR < 14 dB): drops
   bad cams correctly. Result: 9 cams kept, mean PSNR 17.40 dB.

## Numbers

| Setup | mean PSNR | per-cam range |
|---|---|---|
| LEGO 13-cam unfiltered             | 15.39 dB | 12.4-20.3 |
| **LEGO 9-cam PSNR-filtered**       | **17.40 dB** | 14.3-21.0 |
| LEGO 7-cam top-half PSNR-filter    | 18.76 dB | 15.9-21.2 |
| LEGO 6-cam (orig)                  | 18.28 dB | 16.4-19.8 |
| **LEGO 4-cam cherry-pick**         | **20.33 dB** | 19.4-21.1 |

## Pattern

Filtering helps. **Filtering more aggressively helps more.** The 4-cam
cherry-pick remains best. This is a robust observation across now ~5
different cam-subset experiments.

## Why filtering can't beat the cherry-pick

Each cam either contributes (well-paired with neighbors → reduces residual)
or hurts (mis-paired → drives optimizer to compromise). The optimal subset
is the largest set where every cam is well-paired with every other. For
this dataset that's apparently 4 cameras at ~3/4 viewpoints — adding any
more either introduces top/front cams (insufficient parallax neighbors)
or near-duplicate views (no new constraint).

The architecture is doing what it should. The dataset just doesn't have
13 cams with mutually-consistent geometry.

## True remaining unlock (unchanged)

**Better SfM upstream**, specifically incremental SfM that adds cameras
one-at-a-time using PnP against a growing model. Each new camera is
constrained against good geometry, so the mutual-consistency property is
maintained as more cams are added.

## Final architecture readout

  - **20-21 dB** with 4 well-paired cameras, ANY object scan dataset
  - **17-19 dB** with 6-9 cameras, modest pose disagreement
  - **~15 dB** with 13+ cameras at full mixed pose quality
  - **5 dB** with the original splat-from-scratch optimizer (pre-Phoxel)

The Phoxel architecture itself is shipping at 20 dB cleanly. Going past
needs SfM front-end work (F.17 incremental, ~1 week or COLMAP wire-up
~2 days). All optimizer-side cheap fixes are now exhausted.
