# Phase C — Mip-NeRF 360 trained-3DGS scene benchmark readiness

**Status:** infrastructure ready; awaiting input PLY.

## What Phase C is supposed to prove

PhoxBench Tier 0 (synthetic stress scenes) and Tier 1 (real meshes + Audi 3DGS PLY) both measured the **2.0× killer-ratio** for phoxoidal blobs replacing Gaussians at equal RMSE. Phase C asks: **does the killer-ratio hold on properly trained 3DGS scenes from a published benchmark?** (Mip-NeRF 360 is the standard.)

## What we already have

| Scene | Type | Tier 1 killer ratio |
|---|---|---:|
| Audi A5 | **trained 3DGS PLY** (763k splats, full 45-coef SH) | **2.0×** |
| Happy Buddha | Stanford scan (point cloud) | 2.0× |
| Armadillo | Stanford scan (point cloud) | 2.0× |
| Doom combat | artist mesh (point cloud) | 2.0× |

The Audi line *already counts as a trained-3DGS data point.* Phase C is "confirm with at least one more trained 3DGS scene that's not Audi."

## Why this is gated on user action

Trained 3DGS PLY files are not typically distributed:

1. **The 3DGS authors release training code, not pre-trained weights.** Researchers train per-scene from the Mip-NeRF source images (dozens of GBs).
2. **CRYPSOID's hard rule forbids GPU dependencies.** So we cannot train new 3DGS scenes inside this codebase. (Doing so would require `gsplat`, `diff-gaussian-rasterization`, or similar — all banned per [`README.md`](../README.md).)
3. **Public mirrors of pre-trained PLYs exist but are scattered.** No single canonical CDN hosts them.

So Phase C requires: either Bug downloads a pre-trained PLY from a public source, or someone trains one externally and provides the resulting PLY.

## Where to find a trained Mip-NeRF 360 3DGS PLY

Reasonable options (Bug's call):

| Source | Notes |
|---|---|
| [NerfBaselines](https://nerfbaselines.github.io/) | Reproducible NeRF/3DGS evaluation framework. Supports `nerfbaselines download-dataset external://mipnerf360/garden` for source images; pretrained PLY export depends on which method runs. |
| [Mip-Splatting repo](https://github.com/autonomousvision/mip-splatting) | Provides scripts (`create_fused_ply.py`) to export trained PLYs after training. Doesn't publish PLYs directly. |
| [Inria 3DGS repo](https://github.com/graphdeco-inria/gaussian-splatting) | Original implementation; same situation as Mip-Splatting. |
| [Hugging Face datasets](https://huggingface.co/datasets) | A few user-uploaded trained 3DGS scenes exist — search for "3D Gaussian Splatting" + scene name. Quality varies. |
| [Polycam](https://poly.cam/) / [PostShot](https://www.jawset.com/) | Commercial tools that produce trained 3DGS PLYs from user captures. |
| Personal capture | Any 3DGS-trained PLY of a real scene (your own iPhone capture trained externally) works equally well. |

The benchmark only needs **one additional trained 3DGS PLY**, not all 9 Mip-NeRF scenes.

## What to do once a PLY is dropped in

Drop the PLY at any path (e.g. `inputs/mipnerf/bicycle.ply`). Then:

```bash
cd tools
python3 -m phoxbench.run_mesh \
    --ply ../inputs/mipnerf/bicycle.ply \
    --name bicycle \
    --budgets 32 64
```

This reuses the existing Tier 1 harness ([`tools/phoxbench/run_mesh.py`](../tools/phoxbench/run_mesh.py)) and produces:
- `phoxbench/runs/bicycle_b32/{input_preview,gaussian_render,phoxoidal_render,error_heatmap,side_by_side}.png`
- `phoxbench/runs/bicycle_b32/metrics.json` with the killer ratio
- Same for B=64

Expected runtime: ~30 seconds per (scene, budget) pair.

## Expected outcome

If the killer-ratio thesis holds on properly trained 3DGS scenes, you should see:

- Killer ratio = **2.0×** at both B=32 and B=64 (consistent with Audi).
- Per-blob RMSE advantage somewhere in 1.05–1.20× (typical for real surface vs synthetic stress scenes).
- Visible-quality side-by-side render shows phoxoid clearly cleaner than Gaussian on curved/cusp regions.

If killer ratio drops below 1.5× or above 3.0× on the new scene, that's an interesting finding requiring investigation — but the Audi result strongly predicts it'll land at 2.0×.

## Output integration

Once a result is in, [`reports/TIER_1_results.md`](TIER_1_results.md) should be amended with the new row in the B=32 and B=64 tables, and `renders/crypsorender_v01/SHOWCASE_T1_meshes.png` should be regenerated to include the 5th scene.

The script handles all of this automatically once the PLY is in place.

## Honest framing for the paper

If we get one Mip-NeRF 360 scene benchmarked, the [paper](../paper/CRYPSOID_paper_draft.md)'s Phase C row in the next-steps section becomes a measured row in the Tier 1 table instead. The current paper is honest about the gap: "Phase C — Mip-NeRF 360 trained-3DGS scene benchmark... ~1 week." If Bug provides a PLY, that becomes 30 minutes.
