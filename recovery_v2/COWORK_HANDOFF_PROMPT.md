# Cowork Handoff Prompt — CRYPSOID continuation (v2)

Paste everything below this line into Cowork after attaching this package.

---

You are continuing the CRYPSOID `.3dphox` project after a ChatGPT chat-loss recovery. The recovery turned out better than initially feared: the architecture documentation, the version history, the latest verified binary anchor, **and the build/render harness scripts** all survived. Read `RECOVERY_AUDIT.md` first — it covers what's present and what's still missing.

## Read in this order

1. `RECOVERY_AUDIT.md` — current state and gap list
2. `docs/CRYPSOID_BUILD_HANDOFF.md` — main strategy doc
3. `docs/NEXT_AGENT_PROMPT.md` — mission framing
4. `docs/CRYPSOID_V29_PHASE_PLAN.md` — phase plan with command shapes
5. `manifests/version_timeline.csv` — ground truth for what was built at each version
6. `reports/PHOXBENCH_V28_SH_EXACT_CORRECTION_REPORT.json` — confirms v28 was completed and verified before the chat ended

Treat `THESIS.txt` as v0–v0.5 era background only.

## Working anchors you have

- `v27_attribute_group_sh_vq_render_container.3dphox` (root of package, 17.93 MiB, all 7 chunks CRC-verified)
- `tools/build_v27_fast.py` — built the above
- `tools/build_v28_sh_exact_correction.py` — full v0.28 build (outputs render core + q8-exact archive)
- `tools/build_v29_residual_transform_sweep.py` — full v0.29 sweep (11 residual layouts × 4 codecs, real-or-synthetic mode)
- `tools/render_v28_vs_original.py` — CPU DC/opacity preview renderer with side-by-side contact sheet
- `tools/build_v23.py` — earlier no-external-fallback container scaffold
- `tools/phoxoid_convert.py` — v0-era research converter

## What still needs to be provided

- The original Audi A5 source PLY zip (~172 MB). Bug has it locally and will attach when needed.
- The v25 attribute-group container and its report. These are required inputs for `build_v27_fast.py`, `build_v28_sh_exact_correction.py`, and `build_v29_residual_transform_sweep.py` (real-input mode). They were not in the recovery zip.

## Mission

The previous mission ("rebuild the lost render harness") is **obsolete** — the harness was found. The new mission is one of these two paths, depending on whether v25 can be regenerated:

### Path A — Full continuation (preferred)

1. **Regenerate v25 from the Audi PLY.** Write a v25 build script using `tools/build_v27_fast.py` as the spec for v25's chunk layout (lines that read v25 chunks tell you exactly what v25 must contain: `tier_labels_u8`, `xyz_u24_fixed`, `dc_rgb_opacity_u8`, `scale_f16`, `quat_i16_norm4`, `sh_rest_q8_global`).
2. Run `tools/build_v28_sh_exact_correction.py` to regenerate the v28 render and q8-exact archive containers.
3. Run `tools/render_v28_vs_original.py` to produce the v28-vs-original contact sheet.
4. Run `tools/build_v29_residual_transform_sweep.py` in real mode (with v25 + v27 present) to sweep residual codecs against actual data.
5. Build v0.30 render truth gate by extending `tools/render_v28_vs_original.py`: add error heatmap output, tier visualization, SSIM, decode-time and render-time measurements, attribute parity checks.

### Path B — Validation-only (if v25 can't be regenerated)

1. Run `tools/build_v29_residual_transform_sweep.py` in synthetic mode to confirm the harness works in this environment.
2. Run `tools/render_v28_vs_original.py` against the v27 container alone (the v28 render core is byte-identical to v27 except for the magic string, so the harness produces a valid v27 render).
3. Stop. Wait for v25 regeneration before proceeding to Path A.

## Hard rules — do not relax these

- Do not claim a compression win without stating which attribute groups are carried (XYZ, DC/opacity, scale, quaternion, SH residuals, tier labels).
- Do not promote SARC or phoxoid replacement as primary render path until it passes visual gates against splat parity.
- The CPU DC/opacity preview renderer is a sanity gate, not final visual truth. Any output JSON must self-label it that way.
- Splatpack/native exact remains the master/fallback/parity path.
- v0.29 in synthetic mode is a harness validation, not an Audi compression result.

## Path note (read before running anything)

All recovered scripts hardcode `/mnt/data/...` paths from the original ChatGPT runtime. Either mount inputs at those paths in your environment, or refactor each script to accept `--input-root` and `--output-root`. The dependency graph is:

- `build_v27_fast.py` reads v25 → writes v27
- `build_v28_sh_exact_correction.py` reads v25 + v27 → writes v28 render + v28 archive
- `build_v29_residual_transform_sweep.py` reads v25 + v27 → writes v29 archive (real mode), or runs alone (synthetic mode)
- `render_v28_vs_original.py` reads v28 (or v27) container + Audi PLY → writes contact sheet + metrics

## Working mode

Bug is non-technical and prefers reviewable artifacts at each step. Decompose into phases that each produce one reviewable thing (a regenerated container, a metrics file, a contact sheet, an updated script). Pause for Bug's approval at each phase boundary. When you need raw coding work — for example, writing the v25 build script — hand that subtask off to a coding agent with explicit file inputs and acceptance criteria. Do not assume Bug will read the code.

## First three actions

1. Open the v27 `.3dphox` and confirm all 7 chunks decode with matching CRC32. Report the chunk sizes back to Bug as a sanity check.
2. Read `tools/build_v27_fast.py` and write a one-page Markdown spec for the v25 build script (`docs/v25_build_spec.md`). Get Bug's sign-off before any v25 code is written.
3. Once Bug approves the v25 spec and provides the Audi PLY, hand off the v25 build coding task with that spec as the brief.
