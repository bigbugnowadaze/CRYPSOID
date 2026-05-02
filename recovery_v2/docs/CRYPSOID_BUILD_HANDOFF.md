# CRYPSOID Recovery Build Handoff — recovered from pasted chat

Date: 2026-04-30  
Source material: `raw/Pasted_text_34_recovered_chat.txt` plus current local artifacts in `/mnt/data`.

## 0. Executive recovery verdict

The recovered project did **not** simply end at the highest version number. The newest **usable real artifact anchor** is:

```text
CRYPSOID v0.27 — SH attribute-debt breaker
Format: CRYPSOID_3DPHOX_ATTRIBUTE_GROUP_V27_SH_VQ_RENDER
Container: v27_attribute_group_sh_vq_render_container.3dphox
Size: 18,796,089 bytes / 17.93 MiB
Source splats: 763,800
Reduction vs logical source PLY: 89.57%
Ratio vs source PLY: 9.59×
```

v0.28/v0.29 are **not** the newest real render container. In this workspace:

- v0.28 is represented as a render harness / v0.29 phase plan package.
- v0.29 is a residual-transform sweep harness that ran a synthetic smoke test because the real v25/v27 inputs were missing.
- v0.30 is the recommended next phase: a render truth gate.

Safe continuation posture:

```text
Start from v0.27 verified container + Audi source.
Use v0.29 harness only after restoring real v25/v27 inputs.
Build v0.30 render truth gate before more primitive invention.
```

## 1. Current local files that matter

```text
/mnt/data/Audi A5 Sportback(1).zip
/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_27 (2).zip
/mnt/data/v27_attribute_group_sh_vq_render_container (1).3dphox
/mnt/data/CRYPSOID_v27_verified_continuation_bundle.zip
/mnt/data/CRYPSOID_v28_render_harness_and_v29_plan.zip
/mnt/data/CRYPSOID_phoxoidal_absorbed_v0_29.zip
/mnt/data/CRYPSOID_stage_bundle_current.zip
/mnt/data/v24_hybrid_decode_preview_container(2).3dphox
```

The Audi source is a Gaussian-splat-style binary PLY inside a zip:

```text
scene.ply
763,800 vertices/splats
logical PLY size: 180,258,277 bytes
fields: x y z, scale_0..2, f_dc_0..2, opacity, rot_0..3, f_rest_0..44
```

## 2. v27 verification result

The v27 audit says the uploaded v27 zip is usable to continue from, but not fully self-contained for rebuilding v27 from scratch.

It contains:

```text
outputs/v27_attribute_group_sh_vq_render_container.3dphox
outputs/v27_sh_vq_size_bars.svg
reports/PHOXBENCH_V27_SH_DEBT_REPORT.json
reports/RESEARCH_BUILD_TEST_CYCLE_V27.md
tools/build_v27_fast.py
```

It does **not** contain:

```text
original Audi PLY/ZIP
v25_attribute_group_render_container.3dphox
PHOXBENCH_V25_ATTRIBUTE_GROUP_REPORT.json
requirements/dependency notes
```

Fast CPU DC/opacity preview metrics:

```text
PSNR: 54.631795 dB
SSIM: 0.999977430
MSE: 0.223821
MAE: 0.033333
Original splats: 763,800
v27 splats: 763,800
```

Tier counts in v27:

```text
Tier 0 / A residual-phoxoid regions: 94,006
Tier 1 / B native-exact phoxoid regions: 144,271
Tier 2 / C exact splat-stream debt: 525,523
```

Important: this preview is **not** a full anisotropic SH-aware Gaussian splat render. It is useful for geometry/DC/opacity sanity and regression checks only.

## 3. Core architecture recovered

CRYPSOID evolved into layered architecture, not one single codec mode:

### A. Phoxel aggregation branch

```text
PLY → aggregated phoxels → small .3dphox → approximate preview
```

Good for LOD/future semantic compression. Not sufficient for splat parity.

### B. Splatpack / splatbin parity branch

```text
PLY → one Gaussian-compatible CRYPSOID record per source splat → renderable .3dphox
```

This is the master path for splat-usecase parity.

### C. Attribute-group native container branch

```text
tier_labels_u8
xyz_u24_fixed
dc_rgb_opacity_u8
scale_f16
quat_i16_norm4
SH residual stream
```

v25 made this honest but large. v27 made it smaller by attacking SH debt.

### D. SH compression branch

```text
SH16 → q8 SH → SH product VQ/codebooks → v27 SH-VQ stream
```

v27 replaced the v25 global SH stream with:

```text
sh_vq128_idx_u8
sh_vq128_codebook_i8
```

### E. Phoxoid/SARC branch

SARC and phoxoidal math are **not** allowed to replace the splat parity path until they pass visual/render gates.

Recovered rule:

```text
Splatpack/VQ/native exact = master/fallback/parity path
Phoxoids = quality-gated replacement/prediction atoms
SARC/QSARC = advisor/LOD/compression metadata
Residual/exact chunks = accuracy protection
```

## 4. Critical build lineage

See `manifests/version_timeline.csv` and `.json` for the full machine-readable version ledger.

Most important milestones:

| Version | Meaning | Status |
|---|---|---|
| v0.2 | Real Audi PLY ingestion | historic but important |
| v0.5 | Very small phoxel aggregation | non-parity, LOD only |
| v0.6 | Splatpack parity branch | architectural pivot |
| v0.7 | Anisotropic CPU renderer + SH eval | renderer foundation |
| v0.8 | q8 SH + Morton ordering | serious converter prototype |
| v0.9 | Explicit chunked splatbin | real file-format step |
| v0.10 | SH product VQ | first strong compression jump |
| v0.11 | VQ256 safe baseline | reference baseline |
| v0.12/v0.13 | SARC tested/rejected as active renderer | guardrail decision |
| v0.14 | New phoxoidal math absorbed | thesis rebase |
| v0.16 | Audi phoxoid replacement benchmark | partial, not global replacement |
| v0.18-v0.21 | Residual chunks/context sidecars | coverage grew to 12.31% |
| v0.23 | Internalized fallback into Tier C debt | structural milestone |
| v0.24 | Real PLY extraction → decoded preview stream | decoder bootstrap |
| v0.25 | Honest full attribute-group container | truth substrate |
| v0.26 | Attribute debt diagnosis | stopped bad loop |
| v0.27 | SH-VQ full-attribute render container | latest usable anchor |
| v0.29 | Residual transform harness | synthetic only here |
| v0.30 | Render truth gate | next phase |

## 5. Metrics ladder

Source baseline:

```text
Audi logical PLY: 180,258,277 bytes / 171.91 MiB
Splats: 763,800
```

| Build | Size | Ratio / reduction | Honest interpretation |
|---|---:|---:|---|
| v0.5 compact phoxel grid96 | 2.59 MiB | 66.31× / 98.49% | Very small, but aggregated/non-parity |
| v0.6/v0.7 u16 DC | 16.76 MiB | 10.26× / 90.25% | Drops SH residuals; not true parity |
| v0.6/v0.7 u16 SH16 | 44.90 MiB | 3.83× / 73.88% | Higher-preservation reference |
| v0.8 q8 SH Morton | 31.76 MiB | 5.41× / 81.52% | q8 middle path |
| v0.9 q8 chunked delta-XYZ | 31.69 MiB | 5.424× / 81.563% | Explicit chunked file format |
| v0.10 VQ256 | 18.26 MiB | 9.41× / 89.38% | Strong SH codebook direction |
| v0.11 VQ256 | 18.24 MiB | 9.43× / 89.39% | Safe practical baseline |
| v0.23 native container | 14.97 MiB | 11.48× / 91.29% | Structural but not full visual parity |
| v0.24 preview container | 11.05 MiB | n/a | Omitted full SH/rotations; decoder bootstrap only |
| v0.25 full attribute q8 | 28.61 MiB | 6.01× / 83.36% | Honest full-payload truth substrate |
| v0.27 SH-VQ full-attribute render | 17.93 MiB | 9.59× / 89.57% | Latest real usable anchor; not q8-exact SH |

## 6. What not to repeat

Do **not** call a build a win unless it explicitly states whether it carries:

```text
XYZ
DC color / opacity
scale
quaternion
SH residuals
tier labels
```

Do **not** promote SARC/phoxoid replacement as primary render path unless it passes visual/render gates.

Do **not** continue from v0.29 as if it were a real Audi compression result. It was a synthetic smoke run in this workspace.

Do **not** rely on CPU DC/opacity preview as final visual truth. It is only a sanity gate.

## 7. Immediate continuation plan

### Phase 1 — v0.30 Render Truth Gate

Goal: establish a real measuring instrument before further compression/primitive invention.

Build:

```text
Decode original Audi PLY/ZIP.
Decode v27 render container.
Decode v29 exact/archive candidate if real inputs are restored.
Render all through identical camera path.
Generate contact sheet:
  original
  v27 render core
  v29 exact/archive path if present
  error heatmap
Compute:
  MAE
  MSE
  PSNR
  SSIM
  count parity
  attribute parity checks
  decode time
  render time
```

Support container families:

```text
v24 hybrid preview
v25 attribute group
v27 SH-VQ render
future v28/v29 exact/archive
```

### Phase 2 — Exact SH correction / residual debt burndown

Goal: keep v27 size savings while adding exact or near-exact correction.

Work items:

```text
restore/use v25 full attribute container
restore/use v27 SH-VQ container
compute original_q8_SH - VQ_render_core_SH
run v29 residual transform sweep on real data
reject semantic splits that compress worse than global
promote only exact reversible layouts that beat global correction payload target
```

### Phase 3 — Browser/viewer parity

After render truth gate:

```text
JS/WebGPU .3dphox chunk decoder
orbit camera
depth sorting / tiled splat rendering
scale+quaternion projected covariance
SH DC first, then full view-dependent SH
compare against SuperSplat/PlayCanvas behavior
```

### Phase 4 — Photo/video-to-CRYPSOID pipeline

Later, not before file/viewer parity:

```text
OpenCV frame extraction/quality filtering
COLMAP or VGGT/DUSt3R/MASt3R geometry bootstrap
gsplat/Nerfstudio/Splatfacto training/export
PLY/SPZ/SOG → .3dphox converter
semantic side channels: SAM2/FastSAM/GroundingDINO/Florence/OpenCLIP/DINOv2
```

## 8. Package contents

```text
docs/CRYPSOID_BUILD_HANDOFF.md
docs/NEXT_AGENT_PROMPT.md
manifests/version_timeline.csv
manifests/version_timeline.json
manifests/local_artifact_manifest.tsv
render_outputs_v27/*
raw/Pasted_text_34_recovered_chat.txt
```
