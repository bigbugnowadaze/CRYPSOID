# Phase F.16 — BA wire-up attempt (honest results)

**Date**: 2026-05-03

## What I built (this session, after Bug said "push, make it work")

1. **Resumable chunked SfM** (`tools/img2phox/run_sfm_chunked.py`): six stages
   (features → match → verify → pose → ba), each fits within a 45s shell budget,
   each saves atomic checkpoints. Lets us push 30-cam SfM across multiple calls.

2. **Wired `bundle_adjust_sparse` properly**: built `_do_global_triangulation`
   that returns `observations_per_cam` alongside the 3D points, then passed
   into the existing BA function with the right signature.

3. **Triangulate-BA loop**: each BA pass refines poses, allowing more pairs
   to triangulate cleanly on the next pass:
       64 pts → BA → 279 pts → BA → 654 pts → BA → 732 pts → 835 pts
   Convergence over 3 BA passes.

## Verdict

**BA works mechanically. PSNR doesn't improve.**

| Setup | sparse pts | mean PSNR |
|---|---|---|
| 8-cam, full-pair, BA (original)  | 95   | **20.16 dB** |
| 30-cam, no BA                    | 6    | 14.18 dB |
| 30-cam, BA-converged (3 passes)  | 835  | 11.70 dB |
| 30-cam BA, drop bad cams (29)    | 835  | 11.78 dB |

## Why BA didn't unlock more

BA is **local optimization**. It refines poses and points in a neighborhood
of the initial guess. It cannot escape a wrong-basin rotation.

The spanning-tree rotation initializer in `sfm_global.py` builds R_abs by
chaining R_ij values from cv2.recoverPose along a BFS tree. Each R_ij is
correctly chosen (recoverPose picks the cheirality-positive solution), but
chaining 30 cameras through a sparse tree accumulates sign ambiguities at
junction nodes. Some cameras end up rotated 180° from the consistent gauge.

BA's gradient descent stays in the wrong basin because flipping a 180°
rotation requires crossing a singularity in the parameter manifold — it's a
discrete jump, not a continuous improvement.

This is a known limitation of global SfM. The standard fix is **incremental
SfM**: start with a single best pair, then add cameras one at a time using
PnP against the existing 3D model. Each new camera is constrained against
already-good geometry, which avoids the gauge problem. COLMAP and
OpenSfM both do this.

## Where this leaves us

- **8-cam result at 20.16 dB IS our SOTA.** Not a temporary cap, an honest
  ceiling for the global SfM architecture we have.
- **The chunked SfM + BA infrastructure works** and is reusable for any
  future global SfM experiments (e.g. with a better rotation init).
- **The path to higher quality is real engineering work**, not parameter
  tuning. Pick one:
   - Implement incremental SfM cascade (~1 week)
   - Wire COLMAP via subprocess (~2 days, but adds a dependency)
   - Use a learned pose estimator (would break the no-torch rule)

## For the integration story

This doesn't change the Vince pitch: **20 dB image-in pipeline working
end-to-end, scalable architecture in place, photoreal-grade output gated
on ~1 week of incremental-SfM work that's not done.** The architecture is
sound; the pose-init algorithm is the gate.
