# Phase F.19 — Per-camera weighting (negative result, sharper diagnosis)

**Date**: 2026-05-03

## What I tried

Compute per-cam L1 loss each iter, EMA-smooth it, build per-cam weights via
softmin (temp=4) so well-fitting cams contribute more to the gradient sum.

## Result

**Negative: 15.31 dB (vs 15.39 unweighted) — no improvement.**

| Setup                                  | mean PSNR |
|----------------------------------------|-----------|
| LEGO 13-cam unweighted                 | 15.39 dB |
| LEGO 13-cam loss-weighted (temp=4)     | 15.31 dB |

## Why it didn't work

Per-cam L1 losses were all in 0.10-0.15 range — only 1.5× spread. Softmin
with temp=4 distributes weights too gently to actually mute the bad cams.

But more fundamentally: **the bad cams aren't producing noisy-but-correctly-pointed
gradients**. Their POSES are wrong, so they're trying to write density to the
wrong cells entirely. Down-weighting reduces the magnitude of that wrong push
but doesn't fix the direction. The bad cams keep voting for cells the good
cams don't see.

## Sharpened final diagnosis

The 13-cam → 4-cam regression is **not** a "noisy gradient averaging" problem.
It's a "pose-disagreement" problem:

  - 4 well-paired cams have CONSISTENT pose triangulation → optimizer converges to one shared geometry → 20 dB
  - 13 cams include some with bad poses → those cams disagree about which cell
    contains the LEGO body → the optimizer has to compromise → no cell becomes
    sharp → 15 dB

## What this rules out + leaves on the table

**Ruled out**: gradient weighting, view filtering, more iters, higher resolution.

**Still on the table** (in increasing cost):
1. **Coarse robust pose pre-filter** (~half day): drop cams whose pose
   disagrees with the consensus by > threshold BEFORE running phoxel.
   This is per-cam pose-trust, which is different from per-cam gradient-trust.
2. **Incremental SfM** (~1 week): fix the pose problem at the source.
3. **COLMAP wire-up** (~2 days): same fix, different vendor.

## True ceiling restated

**Architecture's honest ceiling on real-photo SfM-derived poses: ~20 dB.**

This holds across: Family (20.16), LEGO 4-cam (20.33), LEGO 7-cam-filtered
(18.76), LEGO 13-cam-weighted (15.31). Cell budget is fine; per-iter
optimization is fine; the gate is **pose accuracy upstream**.

## For Vince

Final story doesn't change: photo-in working at 20 dB on cooperative cams,
photoreal gated on the SfM pose-accuracy unlock. We've now ruled out all
the cheap optimizer-side fixes — the unlock genuinely is upstream pose work.
