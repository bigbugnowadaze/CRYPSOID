# Phase F.13.3 + F.15 — Final state of the photo-in pipeline

**Date**: 2026-05-03

## Verdict

**The 8-camera result at 20.16 dB IS the honest ceiling at our current SfM
accuracy.** Multiple optimization paths confirmed it:

| Attempt | PSNR | Notes |
|---|---|---|
| 8 cams, 96³, 130 iter (v03)             | **20.16 dB** | best |
| 8 cams, 96³ + 15 iter at full res lr=0.3 | 20.06 dB | ~no change |
| 8 cams, octree from downsampled v03      | 17.63 dB | info-lost in downsample |
| 8 cams + TV reg (5e-6)                   | 15.91 dB | over-smoothed |
| 12 cams, full-pair + BA (200 nfev)       | 15.97 dB | wider AABB hurt cell res |
| 20 cams, window=8, no BA (218 edges)     | 13.73 dB | budget SfM noisy |
| 30 cams, window=10, no BA (218 edges)    | 12.37 dB | even noisier |
| 30 cams, drop bad cam0                   | 12.16 dB | one bad cam not the sole cause |

## What actually got built (resumable chunked SfM)

`tools/img2phox/run_sfm_chunked.py` — six stages, each fits 45s budget,
each saves atomic checkpoints. Resume from any stage:

  - features    — ORB detection
  - match       — temporal-window k-NN matching
  - verify      — essential-matrix RANSAC (the slow one)
  - pose        — rotation + translation averaging + triangulation
  - ba          — sparse bundle adjustment (interface mismatch with current sfm.py — TODO)
  - all         — run them all

This let us push 30-cam SfM across 5+ shell calls without losing progress.
**218 verified edges (vs 95 in earlier attempts)** confirms the chunking
works. The bottleneck became the **gauge-stability of the resulting poses**,
not the matching count.

## Diagnosis: why more cams + bigger graph didn't help

The 30-cam graph has 218 edges but `global_triangulate` only produced
**6 → 64 3D points** (after I patched in cheirality diagnostics + flip
detection). The reason: the spanning-tree rotation init plus the LUD
translation refinement leave per-camera sign ambiguities that make some
cameras' triangulated points fall behind their image plane.

The 8-cam version of the same code path produces 95 sparse 3D points
because there are FEWER pairs that can disagree about gauge.

The **real fix** is bundle adjustment refining poses against re-projection
error. Our `bundle_adjust_sparse` exists but expects an
`observations_per_cam` argument that the global-SfM path doesn't currently
build. Wiring this is ~2-3 days of careful work — not a 45-second-shell job.

## What's shippable

- **Best phoxel result**: `outputs/family_phoxel_BEST.3dphox`
  (8 cams, 187k Hessian-aligned phoxoidal blobs, 20.16 dB)
- **Full pipeline code**: `tools/img2phox/phoxel.py`, `phoxel_octree.py`,
  `phoxel_hessian.py`, `cli_phoxel.py`, `run_sfm_chunked.py`,
  `run_phoxel_chunk.py`, `run_phoxel_octree.py`
- **Validation renders**: `outputs/renders/phoxel_v01/SHOWCASE_PHOXEL_BEST_4cam.png`,
  `SHOWCASE_PHOXOIDAL_LIT_3panel.png`, `SHOWCASE_BUNNY_PHOXOIDAL_LIT.png`

## What's needed to break past 20 dB → 25-30 dB photoreal

Two genuinely necessary pieces of work, neither a shell-budget item:

1. **Wire `bundle_adjust_sparse` into the global SfM path** with proper
   observation tracking. ~3 days. Unlocks 30+ cam usage.
2. **Multi-resolution phoxel** with progressive subdivision starting from
   a converged lower-res grid (not from coarse-averaged downsample).
   ~2 days. Unlocks 192³-equivalent surface detail.

Together these should hit 25-30 dB on Family. Without them, **20 dB is
the honest plateau**.

## For Vince

The image-in pipeline is **working end-to-end at proof-of-concept quality**.
We can scale views (proven: 30/30 register), but doing so productively
requires the BA wire-up that's documented above. Don't pitch this as
photoreal yet.
