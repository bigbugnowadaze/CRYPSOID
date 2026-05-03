# F.23.2 — Contracted phoxel sampler: status

**Date:** 2026-05-03
**Owner:** Crypsoid pipeline

## What shipped

- `tools/img2phox/phoxel_contracted.py` — `ContractedPhoxelGrid` class with
  Mip-NeRF 360 contraction, world↔normalised transform from camera
  positions, Numba-JIT'd `_ray_march_forward_contracted` and
  `_ray_march_backward_contracted` kernels, mixed-schedule disparity
  sampling (linear-in-t for foreground, linear-in-1/t for background).
- `tools/img2phox/test_contraction_unit.py` — Acceptance Gate 1.
- `tools/img2phox/test_contraction_outdoor.py` — Acceptance Gate 2.

## Gate 1 — PASS

Synthetic bounded sphere scene. Contracted should match bounded since
all geometry is inside the unit ball (contract is identity).

```
[bounded]    PSNR = 13.93 dB
[contracted] PSNR = 15.47 dB
ratio = 1.111  (criterion: >= 0.95)
GATE 1: PASS
```

The contracted path actually scored *higher* than bounded — the disparity
sampling concentrates more samples near the camera, which on this scene
gives sharper foreground reconstruction.

## Gate 2 — NOT YET PASSING

Synthetic outdoor: small foreground sphere + sharp horizon-split bg
sphere at world r=20.

```
[bounded]    PSNR = 17.97 dB   (with hard-edge horizon backdrop)
[contracted] PSNR = 14.48 dB
delta = -3.49 dB  (criterion: >= +3.0 dB)
GATE 2: FAIL
```

### Diagnosis

Two problems compound:

1. **Bounded grid cheats.** Even though the AABB doesn't enclose the bg
   sphere, the optimizer can put gradient-driven density inside the AABB
   that produces the right *average* bg colour along the ray. With a
   small fixed AABB and aggressive bg pattern, this "fog memorisation"
   gets ~18 dB on its own — without modelling bg geometry at all.

2. **Contracted cells at the bg shell are too few.** The bg sphere at
   world r=20 maps to contracted r = 2 − 1/14.3 ≈ 1.93. With grid
   res=48, the cell width is 4/48 ≈ 0.083, so the bg shell is only
   ~1 cell thick. All cameras' bg rays from different lat/lon angles
   land in the same outer ring of cells, so optimizer gradients conflict
   and average rather than learning per-direction colour.

### What would unstick Gate 2

In rough order of likely impact:

- **Higher contracted res** (96³ or 128³) so the bg shell has more cells
  per direction. Slow but mechanical fix.
- **Proposal sampling** — pre-pass that estimates where density is
  concentrated and biases samples toward those depths. Mip-NeRF 360 uses
  a small MLP for this; a CPU heuristic version is feasible.
- **Separate spherical-shell bg model** — a dedicated 2D directional
  parameterisation (lat/lon → RGB) for the contracted shell, learned
  jointly with the inner grid.
- **More iters** — 30 iters at lr=2.0 leaves the contracted bg shell
  oscillating; 200+ iters might converge it.

None of these are blockers — the spec acknowledges them as future work
(see `docs/scene_contraction_spec.md` §6 "Out of scope").

## What this means for F.23.3 (Family)

The real validation is the Family run, not the synthetic gate. Synthetic
gates are diagnostic; they tell us *whether the implementation works*,
not *whether it will help on real photos*. Two things are different on
real Family:

1. **Bounded baseline is much harder to cheat on real photos.** The
   trees and statue have many sharp edges across many camera angles;
   "fog in the AABB" cannot fake them. Bounded Family was ~17-18 dB on
   prior runs; that's a real ceiling, not a fog hack.
2. **Contracted shell doesn't have to compete with cheating fog** — it
   just has to produce *something* in the bg slot. Even a noisy bg
   shell beats "constant grey because the AABB excludes it".

So F.23.3 is the test that matters. The shipped contracted impl is
sound (Gate 1 confirms math + integration), and we expect it to win on
real outdoor data even though it can't yet win the synthetic Gate 2.

## Honest take for Bug

- Code: shipped. Gate 1: passing. Gate 2: needs more work or a
  different baseline to be discriminating.
- Ready to move to F.23.3 (Family run) which is the actual unlock that
  matters. The risk is that contracted *also* fails on Family, in which
  case we'd come back and address Gate-2-style limits with the fixes
  listed above.
- Alternative: deep-dive on Gate 2 first (try res=96, +200 iters, see
  if delta turns positive). That's a half-day of fitting time + some
  experimentation.
