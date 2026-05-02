# CRYPSOID v0.29 Phase Plan — Render-Gated Residual Debt Burndown

## Current checkpoint

v0.28 separated the file into two honest modes:

1. **Render mode**: compact SH-VQ render core around 17.93 MiB.
2. **Archive mode**: exact q8 SH reconstruction around 30.18 MiB using the best tested global correction stream.

The negative result matters: per-tier correction was more conceptually aligned with CRYPSOID, but it compressed worse on the Audi scene. So v0.29 should not keep splitting correction streams just because it feels semantically right. It should let measured entropy decide.

## v0.29 target

Reduce the exact-correction debt while adding a real render gate:

- Render original PLY/ZIP and v28 from the same camera path.
- Measure image error, not only chunk byte size.
- Test multiple SH residual models in parallel.
- Promote only residual transforms that improve both byte size and rendered-view fidelity.

## Phase order

### Phase 1 — Re-render baseline truth

Build a repeatable render harness with:

- original PLY render
- v25 q8 render
- v27/v28 VQ render core
- v28 exact archive render
- contact sheet output
- image metrics: MAE, MSE, PSNR, SSIM if available

Acceptance gate:

- Same camera, same resolution, same max splat budget.
- No compression claim accepted without a visual contact sheet.

### Phase 2 — Residual transform sweep

Run residual compression variants over the SH error stream:

- raw global residual
- Morton-ordered residual
- delta-over-Morton residual
- channel-major vs splat-major layout
- coefficient-band grouping: low-order SH separate from high-order SH
- sign-magnitude or zigzag remap before entropy coding
- small learned/VQ codebook plus second-stage residual

Acceptance gate:

- Beat v28 global correction payload of ~12.25 MiB.
- Preserve exact q8 reconstruction for archive mode.

### Phase 3 — Context model that earns its keep

Borrow HAC/HAC++ direction, but keep it CPU/prototype-sized:

- hash-grid / Morton-neighborhood context IDs
- per-context residual histogram model
- adaptive quantization flags only where render error allows it
- arithmetic or range coding candidate after zlib/brotli baselines

Acceptance gate:

- Context split must beat global coding; otherwise it gets rejected.

### Phase 4 — Hybrid semantic/native path

Revisit SARC/phoxoidal chunks only after the renderer is locked:

- smooth panels: chart/phoxoid native
- hard edges, wheels, trim, silhouettes: exception splats
- view-dependent/specular regions: preserve SH or residualized SH

Acceptance gate:

- Visual parity first, byte savings second.

## v0.29 immediate build command shape

```bash
python render_v28_vs_original.py \
  --original "/mnt/data/Audi A5 Sportback.zip" \
  --v28 "/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_28/outputs/v28_sh_vq_render_container.3dphox" \
  --out "/mnt/data/crypsoid_v28_render_check" \
  --size 1024 \
  --max-points 0
```

Then start the residual sweep only after the visual check exists.
