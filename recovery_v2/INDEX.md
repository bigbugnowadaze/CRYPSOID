# CRYPSOID Recovery Package — v2

**Updated:** 2026-04-30, after recovery of v23/v28/v29 build scripts and the v28 render harness.
**Read first:** `RECOVERY_AUDIT.md`

This package is the consolidated continuation handoff for the CRYPSOID `.3dphox` Gaussian-splat compression project after a ChatGPT chat-loss event. It supersedes the original `crypsoid_recovery_lite__1_.zip`.

## What's in here

```
RECOVERY_AUDIT.md                    ← read first; covers what's recovered and what isn't
COWORK_HANDOFF_PROMPT.md             ← paste into Cowork for orchestration handoff
CODEX_HANDOFF_PROMPT.md              ← paste into Codex for the coding subtasks
INDEX.md                             ← this file
THESIS.txt                           ← v0–v0.5 era chat dump (background only)
v27_attribute_group_sh_vq_render_container.3dphox  ← the verified v0.27 binary anchor

docs/
├── CRYPSOID_BUILD_HANDOFF.md        ← main strategy doc; architecture, hard rules, metrics ladder
├── CRYPSOID_V29_PHASE_PLAN.md       ← phase plan with command shapes for v0.30
├── CRYPSOID_v27_verified_continuation_bundle__reports__V27_ZIP_AUDIT_AND_RENDER_RECHECK.md
└── NEXT_AGENT_PROMPT.md             ← original mission framing (still applicable, but see new prompts)

tools/
├── build_v23.py                          ← v0.23 no-external-fallback container scaffold
├── build_v27_fast.py                     ← built the v0.27 anchor binary
├── build_v28_sh_exact_correction.py      ← v0.28 render core + q8-exact archive
├── build_v29_residual_transform_sweep.py ← v0.29 11-layout × 4-codec residual sweep
├── render_v28_vs_original.py             ← CPU DC/opacity preview renderer
└── phoxoid_convert.py                    ← v0-era research converter

reports/
├── RESEARCH_BUILD_TEST_CYCLE_V23.md
├── RESEARCH_BUILD_TEST_CYCLE_V27.md
├── RESEARCH_BUILD_TEST_CYCLE_V28.md
├── PHOXBENCH_V27_SH_DEBT_REPORT.json
└── PHOXBENCH_V28_SH_EXACT_CORRECTION_REPORT.json   ← confirms v0.28 was completed before chat ended

manifests/
├── version_timeline.csv                ← v0.1 → v0.30 ledger (machine-readable)
├── version_timeline.json               ← same in JSON
└── local_artifact_manifest.tsv         ← what existed in the original /mnt/data

renders/
├── render_metrics.json                 ← v0.27 vs original metrics
└── v27_vs_original_side_by_side.png    ← reference contact sheet
```

## What's NOT in here that you'll need

- **The original Audi A5 source PLY zip (~172 MB).** Bug has it locally and will provide separately.
- **The v25 attribute-group container.** Required input for `build_v27_fast.py`, `build_v28_sh_exact_correction.py`, and `build_v29_residual_transform_sweep.py` real mode. **Regenerable** from the Audi PLY by writing a v25 build script — the spec for v25's chunk layout is in `tools/build_v27_fast.py` (see how it decodes v25 chunks). Codex prompt has detailed instructions for this as Task 1.

## Verdict (one line)

The hard part of the recovery is done: the build scripts, the verified anchor binary, the render harness, and the architecture spec all survived. Continuation is execution against a recovered codebase, not reconstruction from scratch.
