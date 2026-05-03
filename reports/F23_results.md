# F.23 — Scene contraction on Family: results & honest verdict

**Date:** 2026-05-03
**Owner:** Crypsoid pipeline

## TL;DR

The Mip-NeRF 360 contraction implementation works (Gate 1 PASS) but does **not**
beat the bounded `PhoxelGrid` on the real Tanks-and-Temples *Family* scene with
the iteration / sampling / lr budget we have. Bounded ~14.6 dB; contracted ~8.6
dB after parity iters. **Delta = −6 dB** in the wrong direction.

This is the unblock we hoped F.23 would be. It isn't. We close out F.23 with
the implementation shipped and a clear list of what would need to change.

## Setup

| Item | Value |
|---|---|
| SfM cache | `outputs/_family152_colmap_cache.pkl` (F.22 result) |
| Registered cams | 101 of 152 |
| Sparse pts | 6,663 |
| Train resolution | 67 × 120 (0.25 × original 270 × 480) |
| Bounded grid | 64³ over camera AABB, 48 ray samples |
| Contracted grid | 96³ in (-2, +2)³ contracted space, 64 ray samples, t_far_norm=30 |
| Optimizer | RMSProp, lr_density=2.0, lr_color=0.3 |

## Numbers

| Run | Iters | Final L1 | Per-cam PSNR mean | min | max |
|---|---:|---:|---:|---:|---:|
| Bounded | ~40 | 0.143 | **14.62 dB** | 6.05 | 16.92 |
| Contracted | ~22 | ~0.28 (oscillating) | **8.60 dB** | 6.92 | 11.41 |
| **Δ** |   |   | **−6.02 dB** |   |   |

(Iter counts differ because contracted is ~3× slower per iter; we ran more
bounded chunks within the shell budget. Even so, contracted is not converging
faster — its L1 actively *oscillates* iter-to-iter, suggesting the optimizer
is overshooting.)

## What's wrong

Three independent issues compound:

1. **Optimizer instability.** L1 jumps `0.21 → 0.36 → 0.25 → 0.21 → 0.35` —
   that's not noise, that's RMSProp overshooting. The contracted-shell cells
   are densely shared across cameras (foreground rays from many directions
   converge to the same outer-shell cells), so per-cell gradients have higher
   variance than in the bounded grid where each cell is touched by ~one
   camera's rays. The same lr that's stable for bounded oscillates for
   contracted.

2. **Sampling density at the bg shell is too low.** With disparity sampling
   covering t∈[0.05, 30] in normalised world units (≈ [0.16, 100] meters in
   Family's scale), the bg shell at contracted r ≈ 1.93 gets only a handful
   of samples per ray. With grid res 96, the outer shell is ~1 cell thick.
   Many rays' bg samples fall in the same cell as their neighbours' — gradient
   contention again.

3. **No bg model separation.** Mip-NeRF 360 in its full form has a separate
   *background* density+colour parameterisation (and in practice an
   importance-MLP that biases samples). We have neither.

## What would need to change to make this work

In rough order:

- **Lower lr for contracted, especially density** — cut to lr_density=0.5 and
  re-run. This is the lowest-effort fix and might immediately stabilise the
  oscillation.
- **Separate bg model** — a dedicated 2D directional grid (lat/lon → RGB)
  for the contracted shell, jointly optimised. This is what 3DGS, Mip-NeRF
  360, Plenoxels all do for outdoor scenes.
- **Importance sampling** — even a heuristic version that biases samples
  toward the unit-ball boundary (where most opaque surfaces live in
  normalised space) would help.
- **Higher contracted res** — 128³ or 192³, with the bg shell having more
  cells. Slow but mechanical.

Any one of these in isolation is unlikely to flip the sign. Together they'd
give us the Mip-NeRF-360-quality unlock the spec promised. Estimated effort
to do all four: ~3-5 days.

## What this means for the project

The big question — *can our CPU phoxel pipeline match Mip-NeRF 360 on
unbounded outdoor scenes?* — is **NOT yet answered**. Today's answer is "the
naive contraction port loses; the full Mip-NeRF 360 recipe (contraction +
bg model + importance sampling + lower lr) might win, but is several days
of work."

For the deliverable arc, this means:

- **Bounded phoxel** remains the best we can do on Family at 14.6 dB. That's
  the honest ceiling.
- **Outdoor photoreal** is a research-bet, not a "wire it up" task.
- **Indoor / object-centric** scenes (LEGO, Audi, Family-foreground-only)
  can already hit much higher PSNR with bounded — these are where the
  pipeline's strength is.

## Files shipped this phase

| File | What |
|---|---|
| `docs/scene_contraction_spec.md` | Spec from F.23.1 |
| `tools/img2phox/phoxel_contracted.py` | ContractedPhoxelGrid + JIT kernels |
| `tools/img2phox/test_contraction_unit.py` | Gate 1 (PASS) |
| `tools/img2phox/test_contraction_outdoor.py` | Gate 2 (FAIL on synthetic) |
| `tools/run_family_contracted.py` | Driver for Gate 4 |
| `outputs/_family_bounded_grid.npz` | Bounded baseline grid (64³) |
| `outputs/_family_contracted_grid.npz` | Contracted grid (96³) |
| `outputs/_family_F23_results.json` | Per-cam PSNR for both runs |
| `reports/F23_2_status.md` | Phase 2 status |
| `reports/F23_results.md` | This doc |

## Honest verdict for Bug

We thought F.23 might be the photoreal-outdoor unlock. It isn't, with the
budget we have. The pipeline path is clear (the four-part fix above), but it
is a 3-5 day investment with uncertain payoff, not a one-session wire-up.

Three options for next:

1. **Park F.23, move to F.27 (CGI v2)** — guaranteed deliverable, half-day,
   companion piece to F.26.
2. **Park F.23, move to F.28 (inverse rendering)** — research-grade, 1-2 days,
   targets a different problem (re-lighting baked-in lighting).
3. **Continue F.23 with the 4-part fix** — 3-5 days, betting that the
   stable+importance-sampled+bg-model contracted phoxel will actually beat
   bounded on Family. Highest-value if it works.

My read: 1 first (lock down a deliverable), then 3 (back to F.23 with a
full week of focused work).
