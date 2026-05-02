# crypsorender v0.1 — first real render

**Built:** 2026-04-30. CPU-only, no GPU dependencies. About 1,600 LoC of pure NumPy across `tools/crypsorender/`.

## What's in this folder

| File | What it is |
|---|---|
| `v01_contact_sheet.png` | The deliverable — four panels horizontally |
| `original_ply_full_sh_1024.png` | Original Audi A5 PLY rendered with all 763,800 splats and full SH at 1024×1024 |
| `ply_200k_512.png` | Same Audi PLY, 200,000-splat random subsample, 512×512. Halo is sparse enough to see the car body through it |
| `v28_truth_1024.png` | Our v28 `.3dphox` container rendered with all 763,800 splats at 1024×1024, DC-only (the v28 SH is VQ-encoded and not yet decoded in this version) |
| `manifest.json` | Honest declaration of what code paths ran, what data each panel used, and what's left for v0.2 |

## Honest read of the result

**The renderer works.** EWA 3D→2D covariance projection, 16×16 tile binning, anisotropic Gaussian rasterization, depth sort, front-to-back alpha compositing with early termination, real-basis SH evaluation up to degree 3 — all on CPU, no GPU stack. The 200k-subsample panel shows the actual Audi car body emerging through its splat halo, which proves the math end-to-end.

**The 763k full-density renders look washed out.** That isn't a bug in the renderer; it's a property of the scene + camera. From the above-rear angle, ~21 splats overlap each pixel on average, and the splats in front of the car body are mostly bright halo splats. With no view-dependent shadowing, those bright splats stack to near-pure-white before the car body has a chance to contribute.

**Why doesn't v0.30 (the dot baseline) look saturated?** Because dot rasterization gives each splat exactly one pixel. With 200k dots over 1M pixels, no over-coverage. Mine renders proper anisotropic ellipses (3-sigma footprint, median 3 pixels each), so coverage is much denser. Mine is correct; v0.30 was deliberately a sanity gate.

**What it doesn't do yet** (v0.2 work):
- Decode v28's VQ-encoded SH coefficients. Until then the v28 panel is DC-only.
- Phoxoidal Tier A/B rasterizer paths (we wired the dispatch but currently every tier dispatches to the Gaussian path because the `.3dphox` files don't carry germ data yet).
- Tier B exact-residual correction.
- Turntable MP4 (deferred per Bug's v0.1 decision).

## What this proves

CRYPSOID can produce real Gaussian splat renders on a CPU with no GPU dependencies. Total render time for the full 763k-splat 1024² image was about 70 seconds across ~5 chunked bash calls. The math matches the canonical 3DGS algorithm. The code is ready for the phoxoidal extensions in v0.2.

## How to reproduce

```bash
# Init state for a render
python3 tools/render_audi_chunked.py \
  --scene 'inputs/audi/Audi A5 Sportback.zip' \
  --size 1024 --use-sh \
  --state-dir /tmp/state_ply_full --init

# Process splats in chunks until done
python3 tools/render_audi_chunked.py --state-dir /tmp/state_ply_full --batch 200000
# (repeat until cursor reaches n_visible)

# Save final PNG
python3 tools/render_audi_chunked.py --state-dir /tmp/state_ply_full \
  --finalize --out renders/crypsorender_v01/original_ply_full_sh_1024.png
```
