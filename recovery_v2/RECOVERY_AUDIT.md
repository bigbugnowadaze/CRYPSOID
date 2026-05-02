# CRYPSOID Recovery Audit — UPDATED after additional file recovery

**Audit version:** 2 (supersedes the original audit dated 2026-04-30 22:09)
**Updated:** 2026-04-30, after Bug recovered the v23/v28/v29 build scripts and the v28 render harness
**Verdict:** This is now a near-complete continuation package. The core code that was reported missing in audit v1 has been found. What remains missing is the v25 binary inputs (regenerable from the source PLY), the actual v28/v29 binary outputs (regenerable from the recovered scripts), and a handful of intermediate-version build scripts that are archaeological rather than load-bearing.

---

## 1. What changed since audit v1

### Major recoveries (load-bearing for continuation)

| File | Lines | Status |
|---|---:|---|
| `tools/render_v28_vs_original.py` | 251 | ✅ **Found.** Working CPU DC/opacity preview renderer with side-by-side contact sheet, camera math, and metrics output. Audit v1 told the next agent to rebuild this from spec — that is no longer necessary. |
| `tools/build_v28_sh_exact_correction.py` | 351 | ✅ **Found.** Full v0.28 build that produces both the SH-VQ render container (17.93 MiB) and the q8-exact archive container (30.18 MiB best, 30.67 MiB chosen). |
| `tools/build_v29_residual_transform_sweep.py` | 671 | ✅ **Found.** The complete render-gated residual debt burndown harness: 11 candidate residual layouts (splat-major raw, coefficient-major transpose, group-major 3×15, band-split low/mid/high, Morton splat-major, Morton-delta i16, tier-then-coefficient-major, zigzag splat-major u16, sign-magnitude planes, zero-mask values, bitplane zigzag u8) × 4 codecs (zlib, bz2, lzma, optionally brotli). Has both real-input mode (reads v25/v27, computes original_q8_SH − VQ_render_core_SH, writes v29 archive container) and synthetic mode. |
| `tools/build_v23.py` | 351 | ✅ **Found.** v0.23 native-container scaffold (the architectural pivot to no-external-fallback). |

### Documentation and metrics recoveries

| File | Status |
|---|---|
| `reports/RESEARCH_BUILD_TEST_CYCLE_V23.md` | ✅ Found |
| `reports/RESEARCH_BUILD_TEST_CYCLE_V28.md` | ✅ Found |
| `reports/PHOXBENCH_V28_SH_EXACT_CORRECTION_REPORT.json` | ✅ Found — full v28 metrics, including per-tier residual stats and all four correction encoding variants |

### What the v28 metrics actually say (this is new information)

The v28 work was completed and self-verified before the chat was lost. Per `PHOXBENCH_V28_SH_EXACT_CORRECTION_REPORT.json`:

- v28 render container: 18,795,838 bytes (17.93 MiB) — same render core as v27
- v28 q8-exact archive: 32,163,308 bytes (30.67 MiB) using the chosen `per_tier_group` correction encoding
- Best tested correction encoding: `global_full` at 12,848,777 bytes correction payload
- Archive q8 SH reconstruction is **exact** (verified at write time and on readback)
- All chunks pass CRC32

This means **two more `.3dphox` formats are documented, byte-counted, and CRC-verified, even though their binary outputs are not in the recovery zip.** The build scripts will regenerate them from v25 + v27 inputs.

---

## 2. Updated gap inventory

### Severity A — Still missing, blocking

| File / artifact | Why it matters | Workaround |
|---|---|---|
| Audi A5 source PLY zip (~172 MB) | Needed by `render_v28_vs_original.py` to produce the original-side render. Also needed if v25 must be regenerated. | Bug has it locally. Provide separately to whichever agent runs the harness. |
| v25 attribute-group container (`v25_attribute_group_render_container.3dphox`) | Required input by `build_v27_fast.py`, `build_v28_sh_exact_correction.py`, and `build_v29_residual_transform_sweep.py` (real-input mode). | Two paths: (a) regenerate v25 from the Audi PLY by writing a v25 build script using `build_v27_fast.py`'s decode logic as the spec; (b) have `build_v29` run in synthetic mode (already supported) for harness validation, then regenerate v25 separately. |
| v25 report (`PHOXBENCH_V25_ATTRIBUTE_GROUP_REPORT.json`) | Same scripts read this for source metadata. | Reconstructable from the Audi PLY as a side effect of regenerating v25. |

### Severity B — Missing but recoverable from existing scripts

| File | Why it's lower priority |
|---|---|
| v28 render container `.3dphox` binary | `build_v28_sh_exact_correction.py` reproduces it deterministically once v25 + v27 are present |
| v28 q8-exact archive `.3dphox` binary | Same |
| v29 archive container (if a real run completed) | `build_v29_residual_transform_sweep.py` will regenerate it; the synthetic-only run from the recovered docs probably wasn't a final result anyway |
| Intermediate render output PNGs (heatmap, tier view, original-only, etc.) | `render_v28_vs_original.py` regenerates the contact sheet; the other PNGs are easy to add |

### Severity C — Archaeological, not blocking

These are build scripts for intermediate versions. The architecture and outcome of each is captured in `manifests/version_timeline.csv`. Not having the source code means not being able to literally rerun those cycles, but the next phase of the project doesn't require them.

| Versions | Status |
|---|---|
| v0.6 through v0.22 (except v0.23 which we now have) | Source code not recovered |
| v0.24 hybrid decoder bootstrap | Source code not recovered; binary container also not in zip |
| v0.25 honest full-attribute container | Source code not recovered (this is the one above in Severity A) |
| v0.26 attribute-debt diagnosis | Source code not recovered |

### Severity D — Lost dialogue (irrecoverable from this package)

`THESIS.txt` covers v0–v0.5 (the thesis-era chat). The engineering dialogue from v0.6 onward is not in any recovered file. This is not a code problem — the code from key milestones is now mostly recovered — but if Bug ever needs to revisit *why* a particular design choice was made between v0.6 and v0.27, the conversation context is gone.

---

## 3. Where the project actually stands

The recovery package now contains:

**Working anchors:**
- v0.27 `.3dphox` binary (17.93 MiB, all CRCs verified, in repo root)
- v0.27 build script (76 lines)
- v0.28 build script (full, outputs render core + q8-exact archive)
- v0.29 sweep harness (full, 11 layouts × 4 codecs, real-or-synthetic)
- v0.28 render harness (full, CPU preview, side-by-side, metrics)

**Working spec:**
- Full architecture doctrine (5 branches)
- Hard rules (no SARC promotion without visual gates, label CPU preview as not-final, etc.)
- Version timeline (v0.1 → v0.30 with notes)
- Metrics report for v27 and v28

**The next agent's task is no longer "rebuild missing scripts."** It is:

1. Regenerate v25 from the Audi PLY (write a v25 build script using `build_v27_fast.py` as spec)
2. Run `build_v28_sh_exact_correction.py` to get the v28 binaries
3. Run `render_v28_vs_original.py` to produce the contact sheet against the original PLY
4. Run `build_v29_residual_transform_sweep.py` in real mode to sweep residual codecs against actual v25/v27 data
5. Build the v0.30 render truth gate as an extension of `render_v28_vs_original.py` (add error heatmap, tier view, SSIM, decode/render time, attribute parity checks)

That is real engineering work, but it is no longer reconstruction. It is execution against a recovered codebase.

---

## 4. Path notes (read this before running anything)

All recovered scripts hardcode `/mnt/data/...` paths from the original ChatGPT runtime. Whoever runs them will need to either:

- Mount or copy the inputs to `/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_25/...` etc. (cleanest if you can)
- Or refactor each script to take `--input-root` and `--output-root` arguments (about 10 lines of edits per script)

The scripts also reference each other through these paths — `build_v28` reads `/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_27/...` for instance. If you change one, change them consistently.

---

## 5. Updated handoff posture

**For Cowork:** the orchestration task is now mostly about driving the regenerate-v25 → run-v28-build → run-render-harness → run-v29-sweep sequence and reviewing each output. Cowork should stop after each step for Bug's approval. The Cowork prompt has been updated to reflect this.

**For Codex:** Codex now has a real codebase to extend, not a spec to implement from scratch. The Codex prompt has been updated to ask Codex to (a) write the missing v25 build script and (b) extend `render_v28_vs_original.py` into the v0.30 render truth gate. Both tasks now have concrete code to start from.

---

## 6. One-line summary

Audit v1 said: "the spec survived but the harness scripts were lost." Audit v2 says: "the harness scripts were also recovered, the v28 work was actually completed before the cutoff, and the next phase is execution rather than reconstruction." The v25 binary inputs are the only real remaining blocker, and they're regenerable from the Audi PLY plus the recovered build scripts that document v25's layout.
