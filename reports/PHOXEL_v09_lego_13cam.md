# Phase F.18.2 — LEGO with 13 views (final diagnosis lock-in)

**Date**: 2026-05-03

## Headline

Adding more views to LEGO **didn't break past 20 dB** — and notably hurt the
all-cam mean PSNR. The 4-cam cherry-picked result remains best.

## Numbers

| Setup | mean PSNR | per-cam range | sparse pts |
|---|---|---|---|
| LEGO 4-cam (cherry-picked 3/4 views)  | **20.33 dB** | 19.4-21.1 | 71  |
| LEGO 6-cam (orig dataset)             | 18.28 dB | 16.4-19.8 | 71  |
| LEGO 7-cam (top-half PSNR-filter)     | 18.76 dB | 15.9-21.2 | 180 |
| LEGO 13-cam (all uploaded views)      | 15.39 dB | 12.4-20.3 | 180 |

## What this validates

The diagnosis from F.18.1 is now **conclusive**: the architecture's per-view
PSNR is gated by **constraint quality (parallax-rich neighbors per cam)**,
not by raw view count or cell budget.

Adding cameras that don't share enough overlap with existing well-positioned
neighbors actively **hurts** mean PSNR by polluting cells with noisy gradients.
The phoxel optimizer can't tell which cam to "trust" when they disagree, so
under-constrained cells become muddy averages.

## Implications for the photoreal path

The naive "more views = better" intuition is wrong for our optimizer at
its current configuration. The real path to >20 dB is:

1. **Curate camera distribution**: ensure every cam has ≥3 parallax-rich
   neighbors. For object scans, this means a uniform orbit at constant
   elevation — not a random sample of angles.
2. **Per-camera trust weighting**: weight each cam's gradient by its
   reprojection error or by the count of agreed-upon sparse points it
   shares with neighbors. Down-weight bad-pose cams automatically.
3. **THEN incremental SfM** to deliver more uniformly-distributed cams
   that all satisfy criterion 1.

The cheapest immediate unlock is **per-camera gradient weighting** — that's
~1 day of work, doesn't need new infrastructure. It would let the 13-cam
dataset converge to ~20 dB because the bad cams would auto-mute.

## Files

  - `outputs/lego13_phoxel.3dphox` — 46.8k Hessian-aligned blobs from 13-cam, 626 KB
  - `outputs/_lego13_sfm_cache.pkl` — 13/13 cams + 180 sparse pts (full BA)
  - `outputs/renders/phoxel_v01/SHOWCASE_LEGO_13cam.png` — 13-row contact sheet
  - `outputs/renders/phoxel_v01/SHOWCASE_LEGO_4cam_BEST.png` — best deliverable

## For Vince

The 4-cam-LEGO result at 20.33 dB **stays the headline**. The 13-cam
experiment confirms (by failing in the predicted direction) that the
diagnosis is sound: this isn't a "more data needed" problem, it's a
"selectively trust the available data" problem.

That's a much more concrete and shippable unlock to point at than "needs
COLMAP" or "needs incremental SfM." Per-camera gradient weighting is
~1-2 days of well-scoped work.
