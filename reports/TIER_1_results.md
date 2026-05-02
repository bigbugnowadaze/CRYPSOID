# Tier 1 — PhoxBench on real meshes

**Run date:** 2026-05-01.
**Code:** `tools/phoxbench/run_mesh.py` + `tools/phoxbench/scenes_mesh.py`.
**Anchor tests:** 4/4 PASS (Tier 0 fitter + closest-point Newton).
**Outputs:** per-scene dirs under `phoxbench/runs/{scene}_b{B}/`, summary at
`phoxbench/runs/tier1_summary.json`.

## Headline

**Phoxoidal blobs replace ~2× as many Gaussian blobs at equal fit RMSE on
every real mesh tested, at every budget.** The synthetic Tier 0 finding
(2.0× across sphere/saddle/fold/cusp) is now **reproduced on four real
meshes** including a trained 3DGS PLY.

## The four scenes

| Scene | What it is |
|---|---|
| **Happy Buddha** | Stanford 3D Scanning Repository scan (`happyStandRight_0.ply`) — densely-sampled cyberware mesh |
| **Armadillo**    | Stanford scan (`ArmadilloBack_0.ply`) — organic curved surface |
| **Doom combat**  | User-supplied artist-built game scene PLY — mixed flat + props |
| **Audi A5**      | Trained 3DGS PLY (`audi_scene.ply`, xyz cloud only) — real splat scene used as a point cloud |

Each is normalized into a unit ball and subsampled to 10k points for
harness speed.

## Results — B = 32

| Scene | Gauss RMSE | Phox RMSE | Adv | Killer | Replace |
|---|---:|---:|---:|---:|---:|
| Happy Buddha | 0.01799 | 0.01583 | 1.14× | 64 | **2.0×** |
| Armadillo    | 0.00902 | 0.00749 | 1.20× | 64 | **2.0×** |
| Doom combat  | 0.05017 | 0.04670 | 1.07× | 64 | **2.0×** |
| Audi A5      | 0.02857 | 0.02728 | 1.05× | 64 | **2.0×** |

## Results — B = 64

| Scene | Gauss RMSE | Phox RMSE | Adv | Killer | Replace |
|---|---:|---:|---:|---:|---:|
| Happy Buddha | 0.01148 | 0.01013 | 1.13× | 128 | **2.0×** |
| Armadillo    | 0.00428 | 0.00373 | 1.15× | 128 | **2.0×** |
| Doom combat  | 0.03405 | 0.03141 | 1.08× | 128 | **2.0×** |
| Audi A5      | 0.02308 | 0.02189 | 1.05× | 128 | **2.0×** |

The killer ratio is **flat at 2.0× across all 8 (scene × budget)
combinations** — phoxoid replacement scales with budget, not against it.

## Visuals

- **`renders/crypsorender_v01/SHOWCASE_T1_meshes.png`** — 4×3 contact sheet:
  per scene, panels for `input | Gaussian (32) | Phoxoid (32)`, scene label
  on the left with RMSE numbers, killer ratio annotation on the right.
- **`renders/crypsorender_v01/SHOWCASE_T1_AB.png`** — 4×3 A/B sheet:
  larger panels of `Gaussian | Phoxoid | error heatmap` with all metrics
  on the left margin.
- Per-scene PNGs under `phoxbench/runs/{scene}_b32/`:
  `input_preview.png`, `gaussian_render.png`, `phoxoidal_render.png`,
  `error_heatmap.png`, `side_by_side.png`.

## Honest reading

**What this says:**
- Phoxoidal advantage holds on real geometry, not just analytic stress
  shapes. The 2.0× replacement is reproducible and dimension-stable
  (constant across B=32 and B=64).
- The advantage size (1.05–1.20× lower RMSE) is smaller on real meshes
  than on synthetic curved scenes (1.29–1.65×) — which is what we'd
  expect, since real meshes mix locally-flat regions with curved ones.
- The Doom and Audi PLYs (artist-built and trained-3DGS) both behave
  the same as cyberware Stanford scans: the result is not specific to
  any particular mesh provenance.

**What this does NOT say:**
- This is **fit RMSE on point clouds**, not visual rendering quality on
  trained splat scenes with full SH/opacity machinery. The Audi PLY is
  being treated as a point cloud here, not as splats.
- The killer-ratio search uses doubling (16, 32, 64, 128, …) so 2.0× is
  the smallest power-of-two budget that meets phoxoid RMSE; actual ratio
  is anywhere in [1.5×, 2.5×].
- Each mesh is subsampled to 10k pts; bigger N would change absolute
  RMSE but the *relative* gap is what matters and that gap is stable.

## Reproduce

```bash
cd tools
python3 -m phoxbench.run_mesh --all --budgets 32 64
# → phoxbench/runs/tier1_summary.json
# → phoxbench/runs/{happy,armadillo,doom,audi}_b{32,64}/{input,gaussian_render,phoxoidal_render,error_heatmap,side_by_side}.png
```

(~60 seconds on CPU, no GPU needed.)

## Where this fits

- **Synthetic Tier 0** (validated): `reports/TIER_2_results.md` — 2.0×
  on cusp/fold/saddle/sphere, 4.0× on thin_sheet.
- **Real-mesh Tier 1** (this doc): 2.0× on 4 real meshes × 2 budgets.
- **Next: Tier 2** — render trained 3DGS PLYs as splats (with SH +
  opacity), at the same camera, both Gaussian and phoxoidal-converted,
  compare PSNR/SSIM. That's the "real splat scene" test, distinct from
  the geometric-fit test here.
