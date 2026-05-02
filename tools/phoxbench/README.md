# PhoxBench Tier 0 — synthetic stress benchmark

CPU-only, no GPU dependencies. Tests phoxoidal blobs against vanilla Gaussian
blobs on six analytic surfaces where the difference should and shouldn't matter.

## Files

- `scenes.py` — 6 ground-truth scene generators (plane, sphere, saddle, fold, cusp, thin_sheet)
- `fit.py` — clustering + per-cluster Gaussian and 5-coefficient phoxoid fit
- `run_scene.py` — end-to-end harness (scene -> fit -> render -> metrics)
- `tests.py` — Tier 2 numerical-correctness anchors (run before benchmarking)

## Running it (when sandbox is back)

Step 1: anchor tests (must pass before trusting any numbers):
```bash
cd tools && python3 -m phoxbench.tests
```

Step 2: a single scene at one budget (smoke test):
```bash
cd tools && python3 -m phoxbench.run_scene --scene cusp --budget 64 --no-killer
```

Step 3: the full sweep — 6 scenes x 3 budgets = 18 runs:
```bash
cd tools && python3 -m phoxbench.run_scene --scene all --budgets 64 128 256
```
This includes the killer-ratio search (find smallest Gaussian budget matching
each phoxoid RMSE), which adds ~50% compute time. Expect ~10 minutes total
on CPU.

Outputs land in `tools/phoxbench/runs/<scene>_b<budget>/`:
- `input_preview.png` — scene point cloud
- `gaussian_render.png` — Gaussian-blob reconstruction surface
- `phoxoidal_render.png` — Phoxoid-blob reconstruction surface
- `side_by_side.png` — three-panel comparison
- `error_heatmap.png` — |gaussian - phoxoid| pixel diff
- `metrics.json` — per-scene RMSE, image PSNR, killer ratio

A summary table writes to `tools/phoxbench/runs/summary.json`.

## What "killer ratio" means

For each (scene, budget B), the harness asks: how many Gaussian blobs `B_G`
do you need to match the phoxoid RMSE achieved with B blobs?

- `killer_ratio = 1.0` -> phoxoids don't help on this scene.
- `killer_ratio > 1.0` -> phoxoids replace multiple Gaussians at equal quality.
- `killer_ratio < 0` -> Gaussians can't reach phoxoid RMSE within max_budget=4096.

The thesis predicts `killer_ratio > 2x` on cusp/fold scenes and ~1x on plane.

## Honest expectations (pre-run)

| Scene | Predicted killer_ratio | Reason |
|---|---:|---|
| plane | ~1.0 | flat; nothing for the germ to model |
| sphere | 1.1-1.3 | smooth curvature, mild advantage |
| saddle | 1.5-2 | both curvatures captured by k1, k2 |
| fold | 2-4 | cubic germ catches the fold; Gaussians smear |
| cusp | 3-8 | Pearcey term is the cusp generator |
| thin_sheet | 1.0-1.5 | mostly an opacity test, less a germ test |

If we get >2x on cusp, the thesis has been validated in measurable terms.
If all ratios cluster near 1, we have a project-shaking honest finding.
