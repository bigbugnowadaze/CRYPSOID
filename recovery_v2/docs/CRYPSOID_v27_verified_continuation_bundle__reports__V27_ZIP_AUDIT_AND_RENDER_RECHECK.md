# CRYPSOID v0.27 Zip Audit + Render Recheck

## Verdict
The uploaded v0.27 zip is a **usable continuation anchor**, but it is **not a fully self-contained source-rebuild package**.

It contains the v0.27 build script, reports, size chart, and the actual v0.27 `.3dphox` render container. The embedded container is byte-identical to the separately uploaded v27 container.

It does **not** contain the original Audi PLY/ZIP or the v0.25 input container/report needed to rerun `build_v27_fast.py` from scratch. So it is usable to continue from as a checked v27 artifact, but not enough alone to regenerate v27 without the earlier v25 artifacts.

## Uploaded v27 zip contents
- `outputs/v27_attribute_group_sh_vq_render_container.3dphox`
- `outputs/v27_sh_vq_size_bars.svg`
- `reports/PHOXBENCH_V27_SH_DEBT_REPORT.json`
- `reports/RESEARCH_BUILD_TEST_CYCLE_V27.md`
- `tools/build_v27_fast.py`

## Container integrity
- Embedded container hash equals separate upload: `True`
- CRC readback according to v27 report: `True`
- Local CRC verification: `True`

## Container identity
- Format: `CRYPSOID_3DPHOX_ATTRIBUTE_GROUP_V27_SH_VQ_RENDER`
- Size: `18,796,089` bytes / `17.93` MiB
- Source splats: `763,800`

## Render recheck
Renderer: `Fast CPU DC/opacity preview renderer; not a full anisotropic SH splat renderer`

This renderer is useful for geometry/DC-color/opacity preview and regression checks, but it is not the final anisotropic SH-aware Gaussian renderer.

### Metrics
- PSNR: `54.631795` dB
- SSIM: `0.999977430`
- MSE: `0.223821`
- MAE: `0.033333`

### Counts
- Original splats: `763,800`
- v27 splats: `763,800`

### Tier counts
- Tier 0 / A residual-phoxoid regions: `94,006`
- Tier 1 / B native-exact phoxoid regions: `144,271`
- Tier 2 / C exact splat-stream debt: `525,523`

## v27 technical meaning
v27 is the **SH debt breaker**. It preserves the v25 non-SH attribute groups and replaces the large `sh_rest_q8_global` stream with:
- `sh_vq128_idx_u8`
- `sh_vq128_codebook_i8`

The v27 report states the v25 SH global compressed chunk was `12,563,450` bytes, about `41.88%` of v25. v27 reduced the full attribute render container to `17.93` MiB, a `-37.34%` change vs v25.

## Missing for a complete source-rebuild archive
To be fully self-contained, the archive should also include:
- original Audi source zip or a declared external input path
- v25 render container, because `build_v27_fast.py` reads `/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_25/outputs/v25_attribute_group_render_container.3dphox`
- v25 report, because the script reads `/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_25/reports/PHOXBENCH_V25_ATTRIBUTE_GROUP_REPORT.json`
- requirements/dependency notes, especially `numpy` and `scikit-learn`
- a top-level README with exact run commands

## Recommended continuation state
Continue from v27 as the latest stable render artifact, then fold in the v0.29/v0.30 render-gate work.

Next practical phase: **v0.30 verified render truth gate**.
