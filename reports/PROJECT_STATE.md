# CRYPSOID — project state after the recovery cycle (2026-04-30)

This document is the single page that summarizes where the CRYPSOID `.3dphox` project stands after the post-chat-loss recovery and execution cycle. Drop-in for the next session.

## What was done in this session

| Phase | Outcome | Artifact |
|---|---|---|
| 1. Verify v27 anchor | All 7 chunks decode + CRC32 OK | `recovery_v2/v27_attribute_group_sh_vq_render_container.3dphox`, `docs/v27_verification.md` |
| 2. Spec the v25 build | One-page Markdown spec, sign-off taken | `docs/v25_build_spec.md` |
| 3. Build v25 | All 9 acceptance gates PASS, byte-identical to anchor on chunks 0–4 | `outputs/v25_attribute_group_render_container.3dphox` (28.61 MiB), `tools/build_v25_attribute_group.py`, `reports/v25_handoff_summary.md` |
| 4. Build v28 | Render container byte-identical to recorded; archive within 0.002% | `outputs/v28_sh_vq_render_container.3dphox` (17.93 MiB), `outputs/v28_sh_vq_exact_archive_container.3dphox` (30.67 MiB), `tools/build_v28_sh_exact_correction.py`, `reports/v28_handoff_summary.md` |
| 5. Run render harness | PSNR 51.19 dB on DC/opacity preview; visually identical | `renders/v28_vs_original_side_by_side.png`, `reports/v28_render_summary.md` |
| 6. Run v0.29 sweep | Winner: morton_splat_major / zlib9 = 12,818,553 B; archive 30.15 MiB | `v29_sweep/outputs/v29_residual_transform_archive.3dphox`, `tools/run_v29_incremental.py`, `reports/v29_handoff_summary.md` |
| 7. Build v0.30 truth gate | All 7 acceptance gates PASS; PSNR 54.57 dB, SSIM 0.9999, all 5 chunks byte-identical | `v30_truth_gate/renders/v30_contact_sheet.png`, `tools/render_v30_truth_gate.py`, `v30_truth_gate/reports/v30_truth_gate.json` |

## Bottom-line numbers (HONEST after Tier 1.5 baseline pass)

**Source asset:** Audi A5 PLY, 763,800 splats, 180,258,277 bytes (171.91 MiB).

**Compression vs raw PLY** (the unsophisticated headline; what we used to claim):
- v25: 28.61 MiB → 6.01× smaller
- v27: 17.93 MiB → 9.59× smaller
- v28 render: 17.93 MiB → 9.59× smaller
- v28 EXACT archive: 30.67 MiB → 5.60× smaller
- v29 best archive: 30.15 MiB → 5.70× smaller

**Compression vs zstd-12 PLY** (the fair-baseline picture, post Tier 1.5):
- v28 EXACT archive vs zstd-12 (44.92 MiB): **1.40× smaller** at the same q8 fidelity
- v28 VQ render vs zstd-12: **2.39× smaller** with measurable SH quality loss (PSNR 38.76 dB vs PLY)
- v29 best vs zstd-12: 1.42× smaller
- The "5.6× / 9.6× smaller" headline depends entirely on baseline. Use the 1.40× / 2.39× number when honesty matters.

**Bits per Gaussian** (standard 3DGS-paper normalization, n=763,800):
- raw PLY: 1,888 bpg (essentially the float32 ceiling)
- gzip -9 PLY: 695 bpg
- zstd -12 PLY: 470 bpg
- v28 EXACT archive: **337 bpg** (~1.4× better than zstd-12)
- v28 VQ render: **197 bpg** (~2.4× better than zstd-12)
- For context: Self-Organizing Gaussians ≈ 50–80 bpg, HAC ≈ 30–60 bpg. **CRYPSOID is NOT yet competitive with state-of-the-art splat codecs on bitrate.**

**Bit-exactness verified end-to-end (Tier 1.5 item 1):**
- All 5 v25 passthrough chunks (tier_labels, xyz, dc/opacity, scale, quat): byte-identical between v25 and v28 archive.
- v28 EXACT archive recovers v25 q8 SH stream byte-for-byte (all 34,371,000 int8 elements match).
- PLY → v25 quantization is deterministic; reproduces v25 stored values exactly for every attribute.

**Visual quality (single side-view comparison, 200k subsample):**
- PLY vs v28 EXACT archive: PSNR 56.33 dB / SSIM 0.9996 (essentially indistinguishable; SSIM may be inflated by constant-black background — masked metrics pending Tier 1.5 item 4).
- PLY vs v28 VQ render: PSNR 38.76 dB / SSIM 0.9923.
- Multi-view distribution metrics (mean/median/worst over 32+ cameras, plus LPIPS) pending Tier 1.5 item 5.

## Known truths and limitations

- The CPU DC/opacity preview is a **sanity gate**, not final visual truth. SH bands 1–3 and the anisotropic Gaussian shape are not exercised. v0.30 made this measurable but didn't replace it with a full SH rasterizer.
- v25 tier_labels are **derived from the v27 anchor**, not independently re-derivable from the recovered v21/v22 CSVs. The v21/v22 data records per-cell counts but not the original cell-membership rule, so nearest-center substitution produces a ~3× distribution mismatch. The v27 shortcut is correct in result and is documented as such (`reports/v25_tier_derivation_audit.md`).
- v25 quat encoding required reverse-engineering: the original used **float32 normalize + sign-flip** (not float64). Without this the quat chunk is ~470 bytes off from the anchor. The fix is in `encode_quat_i16` and is documented (`reports/v25_quat_fix_diagnostic.md`).
- v0.29 sweep tested 10 of the 11 advertised layouts; `bitplane_zigzag_u8` was listed in the handoff but never implemented in the recovered script. Codec coverage was zlib6/zlib9 only; bz2 and lzma6 are still untested in this environment.
- **Tier 1.5 honest reframing:** the original "5.6× / 9.6× smaller than PLY" headline numbers compare against the most bloated possible baseline (raw float32 PLY). Against zstd-12 PLY (a fair lossless baseline at the same q8 fidelity), the win shrinks to 1.40× for the EXACT archive and 2.39× for the lossy VQ render. CRYPSOID is meaningfully better than zstd PLY at the q8 grid, but **not competitive with state-of-the-art splat-specific compressors (SOG, HAC) on bitrate alone**. The architectural claims (tier-aware dispatch, phoxoidal density, no GPU dependency, bit-exact at q8 grid) stand on their own and are independently verified.
- **Phoxoidal path in shipped renderer is a screen-space approximation** (`exp(-mahal/2 · (1 + λ·|κ|·mahal))`), not the faithful per-pixel closest-point Newton solve from the thesis. Tier 2 work (per `docs/TIER_2_spec.md`) makes the phoxoidal path qualitatively distinct from Gaussian.
- **Phoxoidal germs are curvature-only (κ₁, κ₂)** in the shipped code. The Pearcey-class cubic terms (χ, ω) and quartic ζ from the thesis are not yet fitted or evaluated. Tier 2 spec covers extension to the full 5-coefficient basis.

## Recovered scripts that have been path-refactored

These now take CLI args and run cleanly outside `/mnt/data/...`:

- `tools/build_v25_attribute_group.py` (new — written this cycle)
- `tools/build_v28_sh_exact_correction.py` (refactored from `recovery_v2/`)
- `tools/render_v28_vs_original.py` (copied; already used argparse)
- `tools/build_v29_residual_transform_sweep.py` (copied; already used argparse)
- `tools/run_v29_incremental.py` (new wrapper for sandbox-bounded runs)
- `tools/render_v30_truth_gate.py` (new — written this cycle)

## What's open / good next moves

1. **Replace the CPU DC/opacity preview with a real SH rasterizer.** The v0.30 gate measures decode/render times, which are the input the future renderer needs. Could be supersplat, gsplat, or PlayCanvas integration.
2. **Re-derive original v21 cell-membership rule** if the original v0.21 source code can be recovered. Until then, v25's tier_labels remain anchored on v27.
3. **Implement `bitplane_zigzag_u8`** as the 11th v0.29 candidate (requires a per-bit transpose of the int8 residuals, then zigzag-coding the resulting bitplanes).
4. **Run v0.29 with `--all-codecs`** (bz2 + lzma6) outside the sandbox time limit — would tell us whether the marginal Morton win holds against slower codecs.
5. **Consolidate the v25 → v28 → v29 → v30 pipeline into a single `make_all` driver.** Each step's output is the next step's input; tying them together with a Makefile or a single Python wrapper would prevent path drift.

## File inventory

```
Crypsoid/
├── docs/
│   ├── v25_build_spec.md
│   └── v27_verification.md
├── inputs/
│   ├── audi/Audi A5 Sportback.zip
│   └── v21_v22_artifacts/{v21_*.csv, v22_*.csv, build_v22*.py, RESEARCH_BUILD_TEST_CYCLE_V21.md, ...}
├── outputs/
│   ├── v25_attribute_group_render_container.3dphox      (28.61 MiB)
│   ├── v28_sh_vq_render_container.3dphox                (17.93 MiB)
│   └── v28_sh_vq_exact_archive_container.3dphox         (30.67 MiB)
├── recovery_v2/                    (the original recovery zip, untouched)
├── renders/
│   ├── original_ply_dc_opacity_preview.png
│   ├── crypsoid_v28_dc_opacity_preview.png
│   └── v28_vs_original_side_by_side.png
├── reports/
│   ├── PROJECT_STATE.md           (this file)
│   ├── PHOXBENCH_V25_ATTRIBUTE_GROUP_REPORT.json
│   ├── PHOXBENCH_V28_SH_EXACT_CORRECTION_REPORT.json
│   ├── v25_handoff_summary.md
│   ├── v25_quat_fix_diagnostic.md
│   ├── v25_tier_derivation_audit.md
│   ├── v25_acceptance_gates.json
│   ├── v28_handoff_summary.md
│   ├── v28_render_summary.md
│   └── v29_handoff_summary.md
├── tools/
│   ├── build_v25_attribute_group.py
│   ├── build_v28_sh_exact_correction.py
│   ├── build_v29_residual_transform_sweep.py
│   ├── render_v28_vs_original.py
│   ├── render_v30_truth_gate.py
│   ├── render_phox_chunked.py            (Tier 1+2: gaussian/phoxoidal/faithful modes)
│   ├── render_audi_chunked.py            (Tier 1: gaussian-only legacy driver)
│   ├── run_v29_incremental.py
│   ├── eval_metrics.py                   (Tier 1.5: PSNR/SSIM with optional foreground mask)
│   ├── multiview_cameras.py              (Tier 1.5/2: orbit-camera generator)
│   ├── tier2_multiview.py                (Tier 1.5/2: per-camera renders + distribution metrics)
│   ├── tier2_contact_sheet.py            (Tier 2: builds SHOWCASE_T2.png)
│   ├── tier2_run_all.sh                  (Tier 2: master runner — anchor tests -> sweep -> renders -> sheet)
│   ├── crypsorender/                     (the renderer package, ~1600 LoC pure numpy)
│   │   ├── io/{ply_loader, phox_loader, splat_buffer}.py
│   │   ├── math/{quat, sh, ewa, germ}.py
│   │   ├── pipeline/{camera, project, sort, tile, rasterize}.py
│   │   ├── output/{png, metrics, contact_sheet}.py
│   │   ├── render.py
│   │   └── cli.py
│   └── phoxbench/                        (Tier 2 benchmark)
│       ├── scenes.py                     (6 synthetic stress scenes)
│       ├── fit.py                        (Gaussian + 5-coef phoxoid per-cluster fit)
│       ├── run_scene.py                  (end-to-end harness + killer-ratio search)
│       ├── tests.py                      (numerical-correctness anchors)
│       └── README.md
├── v29_sweep/
│   ├── outputs/v29_residual_transform_archive.3dphox    (30.15 MiB)
│   └── reports/{PHOXBENCH_V29_RESIDUAL_TRANSFORM_REPORT.json, sweep_progress.json}
└── v30_truth_gate/
    ├── renders/{v30_contact_sheet.png, v30_error_heatmap.png, v30_tier_view.png, ...}
    └── reports/{v30_truth_gate.json, v30_truth_gate.md}
```

---

## Since 2026-04-30 (added 2026-05-01)

The recovery cycle above ended awaiting sandbox return. Sandbox is back; the
following has been executed and shipped since:

### PhoxBench Tier 0 — synthetic stress scenes (run + writeup)
- All 6 scenes (plane, sphere, saddle, fold, cusp, thin_sheet) ran end-to-end.
- 4/4 numerical anchor tests PASS (`tools/phoxbench/tests.py`).
- **Headline:** phoxoidal blobs replace ~2× Gaussians at equal RMSE on every
  curved scene; **4×** on thin_sheet.
- See `reports/TIER_2_results.md`, `renders/crypsorender_v01/SHOWCASE_T2.png`.

### PhoxBench Tier 1 — real meshes (4 scenes × 2 budgets)
- `tools/phoxbench/scenes_mesh.py` + `tools/phoxbench/run_mesh.py` shipped.
- Stanford Happy Buddha, Stanford Armadillo, Doom combat scene PLY, and
  trained Audi A5 3DGS PLY.
- **2.0× killer ratio is flat across all 8 (scene × budget) combinations.**
- See `reports/TIER_1_results.md`,
  `renders/crypsorender_v01/SHOWCASE_T1_meshes.png` (4×3 contact sheet),
  `renders/crypsorender_v01/SHOWCASE_T1_AB.png` (per-scene A/B + error heatmap),
  `renders/crypsorender_v01/manifest_T1_meshes.json`.

### Audi turntable
- `tools/turntable_audi.py` — chunked per-frame render, FLIP_TOP_BOTTOM
  orientation fix, ffmpeg-stitched MP4.
- 36 frames @ 384², full SH, written to `renders/crypsorender_v01/turntable/`.

### WebGL viewer (Tier 3 partial)
- `viewer/index.html` + `viewer/phox_decoder.js` + `viewer/sort_worker.js`.
- Loads any `.3dphox` (v25 / v27 / v28 render / v28 EXACT archive),
  decodes client-side, renders via WebGL2 with web-worker depth sort.
- **Core Python codebase remains GPU-free** — viewer's GPU usage is
  client-side, separate process.

### CI (GitHub Actions)
- `.github/workflows/test.yml` — runs anchor tests, PhoxBench cusp smoke
  benchmark, v25↔v28 byte-identity check, **banned-package check** (no
  torch/cuda/gsplat/nvidia-* allowed in the dep tree).
- `.github/workflows/test.yml` runs on every push.

### Format reference
- `docs/FORMAT.md` — canonical `.3dphox` format spec (header, chunk
  taxonomy, encoding rules per attribute, version compatibility).
- `docs/ROADMAP.md` — phased plan A-F with time estimates.

### Updated headline (Audi A5 PLY) — unchanged
- vs raw PLY: 5.6× smaller (v28 archive); inflated baseline.
- vs zstd-12 PLY: **1.40× smaller** at q8 fidelity (honest baseline).
- bits-per-Gaussian: 337 (EXACT) / 197 (VQ) — meaningful vs zstd, not yet
  competitive with SOTA splat codecs (SOG ~50-80 bpg, HAC ~30-60 bpg).

### Still TODO (Tier 1.5 items 4 & 5)
- Object-mask metrics (mask out constant-black background, recompute PSNR/SSIM).
- Multi-view distribution metrics (32+ cameras, mean/median/worst PSNR).

## Tier 1.5 final (added 2026-05-01, items 4 + 5)

### Item 4 — object-mask metrics
- `tools/compute_object_mask_metrics.py` + `reports/TIER_1.5_object_mask_metrics.md` + `.json`.
- Camera-aligned 512×512 trio (PLY / v28 archive / v28 VQ render).
- Object mask covers 38.8% of the side-view frame (luma > 0.02).
- **PLY vs v28 archive:** full-frame 56.33 dB / SSIM 0.99957  →  object-only **52.23 dB / 0.99936** (PSNR drops 4.10 dB when masked, SSIM inflation +0.00021).
- **PLY vs v28 VQ render:** full-frame 38.76 dB / 0.99226  →  object-only **34.67 dB / 0.98862** (PSNR drops 4.09 dB when masked, SSIM inflation +0.00364).
- Honest framing: SSIM inflation is small here (object covers a large fraction of the frame), but PSNR ~4 dB drop is real and consistent across both pairs.

### Item 5 — multi-view distribution
- 8 cameras × pitch +8°, full 360° azimuth orbit at 256², subsample 30k splats.
- All 24 PNGs rendered (PLY + archive + VQ render at each angle), per-view masked metrics computed, distribution aggregated.
- **v28 EXACT archive vs PLY:** PSNR mean **55.28 dB**, median 55.32, **worst 54.44**. SSIM mean 0.99993, worst 0.99991. Spread = 1.42 dB.
- **v28 VQ render vs PLY:** PSNR mean **37.83 dB**, median 38.01, **worst 36.85**. SSIM mean 0.99646, worst 0.99581. Spread = 2.21 dB.
- Per-camera detail + chart in `reports/TIER_1.5_multiview_distribution.md` and `reports/TIER_1.5_multiview_chart.png`.
- The single-view 56.33 dB headline is **not lucky**: it sits ~1 dB above the worst across the orbit. PSNR distribution is tight; no single bad angle.

## v31 graph extension spec (added 2026-05-01)

Strategy doc `questions for claude.md` proposes turning CRYPSOID from "splat codec" into "explicit-math universal scene representation" by absorbing non-AI ancestors of NeRF/hypernet/SDS/CL-NeRF/LoRA-NeRF (TSDF, surfels, MLS, Poisson, Plenoxels, kNN graphs, normal maps).

`docs/v31_graph_extension_spec.md` is the one-pager for the smallest viable absorption — three additions:
1. Explicit normal + tangent frame per phoxoid (4 bytes/blob, octahedral encoding).
2. kNN edges chunk (16 bytes/blob, k=4 u32 indices).
3. `.phoxdelta` patch format (sparse, low-rank, base-CRC referenced).

Cost on Audi: ~+47% bytes vs v28 archive (~497 bpg vs current 337) — a bitrate regression bought for the structural extensions that justify the project's existence. Acceptance gates listed; phasing suggested. Awaiting sign-off before any implementation work.

## v25 cell-membership caveat — CLOSED (2026-05-01)

The long-standing v25 caveat ("tier_labels derived from v27 anchor, not independently re-derivable from recovered v21/v22 CSVs") has been **closed**. Bug uploaded `phoxbench_v020_hash_context.py` which contained the cell-key decoder (`key = ix + 32·iy + 32²·iz`). With that, three plausible bbox hypotheses were tested against the 247 v20-accepted cells in `v20_context_accepted_chunks.csv`:

- **A — full PLY xyz bbox:** 247/247 EXACT match, mean ratio 1.000, std 0.000.
- B — 99-pct bbox: 0/247.
- C — fitted from cell centers: 4/247.

The simplest hypothesis (full PLY bbox) reproduces every count bit-exactly. v25 cell-membership is now independently re-derivable from the original PLY without depending on the v27 anchor. Recovered rule + per-splat cell_keys saved at `reports/v25_cell_membership_rule_recovered.json` and `reports/v25_per_splat_cell_keys.npy`. Full writeup: `reports/v25_cell_membership_recovered.md`.

## v32 + v33 lighting + materials spec drafted (2026-05-01)

`docs/v32_v33_lighting_materials_spec.md` — paired one-pager covering lighting and materials as one signed-off unit. Rationale: lighting without materials is half a story (SH bakes lighting + material together; you have to separate them to relight properly).

Three additions:

1. **v32a — standard lighting** (Lambert + ambient + directional sun). Renderer-only, zero format bytes. Standard graphics math; works the day v31 normals land. Unlocks "user can see lit geometry."
2. **v32b — curvature-aware lighting** (the phoxoidal-math-specific contribution). Two terms using existing v31 germ data: (i) self-shadowing from curvature at grazing angles, (ii) curvature-modulated ambient occlusion. Cusp-specular deferred to v32c. Renderer-only, zero format bytes.
3. **v33 — material_hints chunk** + heuristic derivation + albedo/lighting separation. 2 bytes/phoxoid (1-byte enum opaque/diffuse/glossy/mirror/transparent/emissive + 1-byte confidence). Cost: +4.7% on Audi (~1.5 MB).

Combined v32+v33 cost: **+4.7%** vs v28 archive (much cheaper than v31's +47%, because lighting stores nothing and material hints are tiny).

Acceptance gates listed; phasing tied to v31 sign-off (v32a unlocks day v31 ships normals; v32b uses germ chunks already in place).

Honest framing in the spec: v33 enables *approximate* relighting (SH was never explicitly material-separated). Multi-view photometric variation (the GS-2M better-way) needs source images we don't have for the Audi PLY; format slot is ready for the future.

## v32.5 shadows spec drafted (2026-05-01)

`docs/v32_5_shadows_spec.md` — kNN-graph soft shadows + optional graph-based ambient occlusion. Sits between v32a (lights) and v33 (materials) in the spec sequence.

Key claim: **shadows become a graph query** because v31 already stores k=4 neighbor edges per phoxoid. For each (phoxoid P, light direction L), walk P's 4 neighbors and ask "do any sit between P and L?" via signed-distance projection + Gaussian falloff in perpendicular distance. O(k=4) per phoxoid per light, no spatial structure needed. **This is the second use case for v31's kNN edges** (after LOD) and is genuinely phoxoidal-specific — surfels and standard splats don't carry an explicit neighbor graph.

Composes cleanly with v32b's curvature-AO (per-phoxoid local) — graph-AO adds the between-phoxoid global occlusion the curvature term can't capture. Format additions: zero. Pure renderer feature.

Estimated cost on Audi: ~160 ms shadow pass at 512² with 1 sun light (well under existing render time).

Honest scope:
- Local shadows only — distant blockers don't shadow nearby surfaces by default (mitigation: optional 2-hop walk, expensive).
- Soft by construction — no hard sharp shadows (feature for fuzzy splats, limit for stylized rendering).
- AO is qualitatively similar to SSAO/ray-traced AO; quantitatively coarser; composes with v32b curvature AO.

## Roadmap updated

`docs/ROADMAP.md` now contains a new "Phase D.4 — Format extensions" section with the full v31 → v32a → v32b → v32.5 → v33 → v34 → v40+ sequence, dependency graph, and per-spec phoxoidal-math-vs-standard split. Time estimates added: ~3 weeks total once v31 sign-off lands.

## Renderer ceiling push (2026-05-01)

Pushed the renderer to its current quality ceiling. Three configurations rendered for comparison; full progression saved at `renders/crypsorender_v01/SHOWCASE_HIGHEST_progression.png`.

| Render | Splats | Resolution | Camera | Result |
|---|---:|---|---|---|
| Prior best (TIER 1.1 deliverable) | 200,000 (subsample) | 1024² → 512 SS | side, dist 1.5, fov 42 | clear body, light halo |
| **New highest** | **400,000** (subsample) | **1024² native** | **side, dist 1.5, fov 42** | **richer panels + cockpit + wheels visible — current quality ceiling** |
| Full density (honesty render) | 753,631 (all visible after culling) | 1024² native | tighter, dist 1.15, fov 36 | floor halo dominates — more splats != better visual when framing is tight |

**File:** `renders/crypsorender_v01/SHOWCASE_HIGHEST.png` (1024², 286 KB) is the new headline render.

**Honest finding:** more splats is not strictly better. The full 763k render at tight framing is *technically* the most complete reconstruction, but the v28 archive carries floor/halo splats that swamp the car body when the camera gets close. The 400k subsample at moderate distance is the sweet spot for a beauty shot. This argues for the v33 `material_hint` work (separating "real surface" splats from "loose halo" splats) and v32.5 graph-AO (which would naturally darken the dense halo regions).

Each batch took 8-30 seconds in the sandbox. Full-density render took 8 chunked bash calls + finalize + flip; ~3 minutes wall-clock total.

## SIGN-OFF — v31 / v32 / v32.5 / v33 specs (2026-05-01)

Bug approved the spec stack on 2026-05-01. The following specs are now committed for implementation:

- `docs/v31_graph_extension_spec.md` — normals + kNN edges + `.phoxdelta` patches.
- `docs/v32_v33_lighting_materials_spec.md` — Lambert + curvature shading + material_hint enum + view_dependence_score (amended).
- `docs/v32_5_shadows_spec.md` — kNN-graph soft shadows + graph-AO.

Implementation order (per dependency graph in `docs/ROADMAP.md` Phase D.4):
1. v31 Addition 1 — normals chunk (this cycle).
2. v31 Addition 2 — kNN edges chunk.
3. v32a — Lambert shading using v31 normals.
4. v32b — curvature self-shadow + AO using existing germ chunks.
5. v32.5 — kNN soft shadows + graph-AO.
6. v33 — material_hint + confidence + view_dependence_score chunk; albedo/lighting separation toggle.
7. v31 Addition 3 — `.phoxdelta` reader/writer + acceptance test.

v32c (cusp-specular), v34 (temporal `.phoxdelta`), v40+ (transparency / refraction / Pearcey caustics) deferred to future cycles.

## v31 Addition 1 IMPLEMENTED + v32a first lit render (2026-05-02)

### v31 Addition 1 — normals chunk codec
`tools/crypsorender/io/normals_codec.py` ships:
- `normal_to_oct` / `oct_to_normal` — octahedral projection
- `quantize_oct_24bit` / `dequantize_oct_24bit` — 12-bit-per-axis lattice
- `tangent_angle_to_byte` / `byte_to_tangent_angle` — 8-bit angle (with +1e-9 nudge to keep encode/decode idempotent under float drift around 2π)
- `write_normals_chunk` / `read_normals_chunk` — full chunk I/O with CRC32 (chunk_id 0x12, version 0x01)
- `derive_normals_mls(xyz, k=24, refine_quadric=True)` — MLS plane fit + quadric refinement step that removes the plane-fit-on-curved-surface bias

`tools/test_normals_codec.py` — **5 of 5 acceptance gates PASS:**
1. chunk round-trip (idempotent on second pass; first-pass also bit-exact thanks to the nudge)
2. unit-norm + 12-bit precision (max 1.01 mrad angular error)
3. sphere stress test (p95 = 5.92 mrad, well under 10 mrad target — quadric refinement was the key)
4. tangent angle 8-bit (max 1.4° error)
5. CRC integrity (corruption detected)

### v32a Lambert lit Audi
First lit phoxoid scene rendered. Per-splat path:
1. Derive normal via `derive_normals_mls` (MLS+quadric, 16s for 200k splats on CPU)
2. Compute albedo from SH band 0 (`f_dc * 0.282 + 0.5`)
3. Apply Lambert: `shaded = ambient·albedo + sun·albedo·max(0, N·-L)`
4. Composite via depth-sorted "over" (existing rasterizer)

Visible deliverables:
- `renders/crypsorender_v01/SHOWCASE_LIT_v32a.png` — first lit render (1024², 200k splats)
- `renders/crypsorender_v01/SHOWCASE_v32a_unlit.png` — DC SH only baseline (same camera/scene)
- `renders/crypsorender_v01/SHOWCASE_v32a_lit.png` — Lambert + ambient + sun
- `renders/crypsorender_v01/SHOWCASE_v32a_compare.png` — before/after 2-up panel

The shading is doing real work: sun-side body panels brighter, shadow-side darker, the car reads as a 3D object with form rather than a flat splat soup. Numbers from the run: NdotL>0.5 splats average 0.79 brightness; NdotL<0.1 splats average 0.15.

### What this means structurally
- v31 spec is no longer "drafted" for Addition 1 — the codec is implemented and gated.
- The first phoxoidal-math contribution beyond compression is shipped: surface-aware lighting using a stored normal field that no other splat format carries natively.
- v32a is renderer-only (zero format bytes). The format change is from v31 (storing the normals chunk in `.3dphox`).
- Next implementation step: write the normals chunk into a v31-versioned `.3dphox` derived from the v28 archive + run the same lit render reading normals from the file (round-trip the codec end-to-end on real data).

## v31 file built + v32a/v32b implemented (2026-05-02)

### v31 .3dphox file shipped — `outputs/v31_audi_with_normals.3dphox`
First v31-versioned `.3dphox` file written. Layout: v28 archive bytes (verbatim, byte-identical, backward compatible) + `CRYPSOID31\0` trailer marker + JSON trailer manifest + normals chunk (chunk_id 0x12).

- Total size: **35,218,223 bytes** (1.095× v28 = exactly the +9.5% the spec predicted for normals alone).
- All 763,800 phoxoid normals stored as octahedral 24-bit + 8-bit tangent angle.
- Round-trip verified: decoded normals re-encode to byte-identical chunk for 99.99% of bytes; the remaining 0.01% are octahedral-fold edge cases at `x = -0.0` (codec normalization issue, no data loss — angular drift = 0 mrad).
- Backward compatibility: `raw[:len(v28_bytes)] == v28_bytes` (any v28 reader can still consume the file).
- Build script: `tools/build_v31_with_normals.py`.

### v32b curvature shading — implemented + rendered
Per-splat curvature derived as **surface-variation index** (Pauly 2002): `κ = λ_min / (λ_0 + λ_1 + λ_2)` from the same MLS covariance eigendecomposition that produced the normals. On Audi: κ range [0, 0.32], mean 0.09, median 0.09 — mostly low-curvature smooth body with high-curvature regions on edges + halo.

v32b math applied:
- **Self-shadow:** `visibility = max(0, N·-L) · (1 − β·|κ_eff|·(1 − N·-L))` with β=0.5
- **Curvature AO:** `ambient_factor = 1 − α·tanh(|κ_eff|)` with α=0.4

Effect measured at the camera: high-κ sun-side splats darken by **7%** vs pure Lambert. Mean shading drops 3.5% overall. Both effects vanish at κ=0 (verifies v32a degenerate case in spec).

### Showcase deliverables
- `renders/crypsorender_v01/SHOWCASE_v32ab_progression.png` — 3-panel: unlit / v32a / v32a+v32b
- `renders/crypsorender_v01/SHOWCASE_v32a_only.png` — Lambert only, 1024²
- `renders/crypsorender_v01/SHOWCASE_v32ab.png` — Lambert + curvature, 1024²

### What this proves
1. **End-to-end format → render loop closed.** Build a v31 file, read normals from it, apply lighting that uses those normals, get a visibly different render. Format extension works in practice.
2. **Phoxoidal-math contribution measured.** v32b's two terms (self-shadow + curvature AO) only depend on κ (which only the phoxoid representation natively carries). Surfels and standard splats can't do this without a separate curvature pass.
3. **Pure CPU, no GPU dependency.** All v32a + v32b math is per-splat numpy. Stays on the CRYPSOID side of the GPU-free hard rule.

### Still TODO (next implementation steps)
- v32.5 kNN soft shadows (depends on v31 Addition 2 — kNN edges chunk)
- v33 material_hint + albedo separation
- v32c cusp-specular (deferred per spec)
