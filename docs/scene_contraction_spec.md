# F.23 — Scene contraction for unbounded outdoor phoxel scenes

**Status:** SPEC — phase 1 of three (F.23.1)
**Owner:** Crypsoid pipeline
**Date:** 2026-05-03

## 1. Why this exists

The current `PhoxelGrid` (tools/img2phox/phoxel.py) is a dense voxel cube
defined by `(origin, size, res)`. Every cell maps to a physical world cube
of size `cell_size = size / res`. Outside that cube there is nothing.

This works for **bounded scenes** (LEGO excavator on a table, the synthetic
"sphere on plane" smoke test, anything that fits in a box) but it falls
apart on **unbounded outdoor scenes** like Tanks-and-Temples Family. There,
the camera looks at a foreground subject (the statue) but the photo also
captures background (trees, sky, distant buildings) extending to infinity.

What goes wrong on Family with the bounded grid:

- The AABB is sized to fit the cameras + the sparse cloud. To cover the
  trees behind the statue we'd need the box to be ~10× larger than the
  statue itself. With fixed res (say 96³), each cell becomes ~10× coarser.
  Statue detail vanishes.
- If we keep the box tight on the statue, distant background pixels never
  hit any cell, so background photons fall through to `bg_color`. The
  optimizer then tries to "explain" background pixels by adding spurious
  density along the foreground rays — classic floater behavior, low PSNR.

This is exactly the problem **Mip-NeRF 360** (Barron et al. 2022) solved
for unbounded NeRFs, and it's the same fix Plenoxels and the original
3DGS use for outdoor data.

## 2. The fix — Mip-NeRF 360 contraction

Squash all of unbounded R³ into a finite ball of radius 2:

```
contract(x) = x                                if ||x|| <= 1
            = (2 - 1/||x||) * x / ||x||        if ||x|| > 1
```

Properties:
- The unit ball maps to itself (foreground stays at 1× resolution)
- Infinity maps to the surface of the radius-2 sphere
- The transform is C¹ continuous
- Cell size in WORLD space grows as ||x||² in the contracted region —
  exactly matching the screen-space size growth of distant geometry, so
  rays through the background hit cells of consistent screen-pixel size

The phoxel grid then lives in **contracted space**, not world space. Every
sample point along a ray is contracted before the trilinear lookup.

## 3. Coordinate convention

Two coordinate frames in the renderer once contraction is on:

| Frame | Where it lives | Used by |
|---|---|---|
| **World** | metres, raw SfM/COLMAP output | rays, camera poses, splats |
| **Contracted** | inside the radius-2 ball | the phoxel grid |

The grid's `origin` and `size` describe the contracted ball, not the
original world AABB. We use `origin = (-2, -2, -2)`, `size = (4, 4, 4)`
so the entire contracted ball fits inside the cube.

Before that, world coordinates are **scene-normalised** so the foreground
subject sits at ||x|| ≈ 1. We compute a normalisation transform from the
camera positions:

```
center = mean(camera positions)
scale  = median(||cam_i - center||)
normalise(x) = (x - center) / scale
```

so that the typical camera distance becomes 1.0. Foreground geometry then
falls inside the unit ball; background ends up at ||x|| > 1 and gets
squashed.

## 4. Where this slots into the existing code

Three integration points, kept **opt-in** so the bounded path stays
unchanged for LEGO and the smoke tests:

### 4.1 `class ContractedPhoxelGrid` (new)

Mirrors `PhoxelGrid` shape (origin, size, res, density, color) but the
forward/backward kernels apply `contract()` to the world-space sample
point before computing grid coords. Same trilinear interpolation, same
analytic gradient — just composed with contraction.

### 4.2 New JIT'd kernels

`_ray_march_forward_contracted` and `_ray_march_backward_contracted` —
identical to the existing kernels except for one extra inlined block:

```python
# Inside the per-sample loop, replace
gx = (wx - origin[0]) / cell_size[0] - 0.5
# with
r = sqrt(wx*wx + wy*wy + wz*wz)
if r > 1.0:
    s = (2.0 - 1.0/r) / r
    cx = wx*s; cy = wy*s; cz = wz*s
else:
    cx = wx; cy = wy; cz = wz
gx = (cx - origin[0]) / cell_size[0] - 0.5
# (same for gy, gz)
```

Numba can JIT this without trouble — same scalar math, no allocation.

### 4.3 Backward gradient correction

The trilinear backward pass computes `dL/dgrid` from `dL/dsample`. Adding
contraction inserts a Jacobian `dx_contracted/dx_world` between the two.
For the optimizer to converge correctly we DO NOT need to divide by this
Jacobian (we're not back-propagating through coordinates, just accumulating
into cells). The existing accumulate-into-corners pattern is correct as-is
because gradient flows from the loss into the cell value at the sampled
location, not through coordinate transforms. **Verify this in a unit test
before trusting the result.** (Acceptance gate 1.)

### 4.4 Sampling schedule

Outside the unit ball, equal `t` steps in world space pile up many
samples in nearly the same contracted cell. Mip-NeRF 360's fix is to
sample with `t` step proportional to expected screen-space size. For our
phoxel CPU code, simpler equivalent: sample uniformly in `s = 1/||x_eye -
sample||` space, which gives more samples near the eye (foreground) and
fewer at infinity. Implement as a pre-computed `t_samples` array per ray
instead of changing the loop.

## 5. Acceptance gates

Each gate must be a script in `tools/img2phox/` or `tools/` that prints
PASS/FAIL.

| # | Test | What it proves |
|---|---|---|
| 1 | `test_contraction_unit.py`: trilinear gradient on contracted grid round-trips on the synthetic sphere scene to within 5% of bounded-grid PSNR | The Jacobian-omitted backward pass is still convergent |
| 2 | `test_contraction_outdoor.py`: synthetic outdoor scene (foreground sphere + textured background sphere at r=20) hits >25 dB; bounded baseline stuck at <18 dB | Contraction actually represents background |
| 3 | `tools/run_phoxel_chunk.py --contracted` flag works end-to-end | Existing chunked runner keeps working |
| 4 | `reports/F23_results.md` shows Family per-cam PSNR with bounded vs contracted, with at least 2 dB mean improvement | The real-data win |

## 6. Out of scope (future work)

- **Multi-resolution / hash grid** (Instant-NGP style). Plenoxel-style
  dense-grid + contraction is simpler and CPU-friendly; we punt the hash
  grid until we hit the resolution wall on contracted dense.
- **Proposal-MLP** for sample-importance scheduling. Mip-NeRF 360 uses one;
  we approximate with the `1/r` sampling above.
- **Spherical-harmonic background**. Background-only SH model with separate
  parameters. Could be added if `bg_color` constant becomes a bottleneck.

## 7. File-by-file delta

| File | Change | Phase |
|---|---|---|
| `docs/scene_contraction_spec.md` | new (this file) | F.23.1 |
| `tools/img2phox/phoxel.py` | add `ContractedPhoxelGrid`, `_ray_march_*_contracted` | F.23.2 |
| `tools/img2phox/test_contraction_unit.py` | new — gate 1 | F.23.2 |
| `tools/img2phox/test_contraction_outdoor.py` | new — gate 2 | F.23.2 |
| `tools/img2phox/run_phoxel_chunk.py` | add `--contracted` flag | F.23.2 |
| `tools/run_family_contracted.py` | new — driver for Family run | F.23.3 |
| `reports/F23_results.md` | new — bounded-vs-contracted PSNR | F.23.3 |

## 8. Time estimate

- F.23.1 (this spec): done
- F.23.2 (implement + unit tests): ~1 day of focused work
- F.23.3 (Family run + results doc): ~half day, plus shell time for the run

Total: ~2 days of work over potentially several sessions, since each phase
gives Bug a reviewable artifact (this doc, then the unit tests passing,
then the Family numbers).

## 9. References

- Barron et al., **Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance
  Fields**, CVPR 2022. The contract() function is §3.2.
- Müller et al., **Instant Neural Graphics Primitives**, SIGGRAPH 2022.
  Multi-res hash grid as the alternative to dense + contraction.
- Fridovich-Keil et al., **Plenoxels: Radiance Fields without Neural
  Networks**, CVPR 2022. The dense-voxel-grid baseline our PhoxelGrid is
  modelled after.
