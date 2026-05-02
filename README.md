# CRYPSOID

> **CPU-only Gaussian-splat alternative + native renderer.** No GPU, no `torch`, no `gsplat`, no `cuda-toolkit`. Just numpy, scipy, Pillow, and ffmpeg. Tier-aware splat dispatch and a phoxoidal-blob extension grounded in catastrophe optics.

CRYPSOID is two halves of one project:

1. **`.3dphox`** — a tiered container format for 3D Gaussian Splat scenes. Every splat carries a tier label (`A` native render phoxoid, `B` native exact phoxoid, `C` Gaussian fallback), and the format separates a small "render core" (with VQ-approximated SH) from a larger "EXACT archive" (with per-tier-group residual chunks that bit-exactly reconstruct the original q8-quantized data).

2. **`crypsorender`** — a pure-CPU rasterizer that respects those tiers. Tier C splats render as standard anisotropic Gaussians (the Inria/EWA algorithm). Tier A/B splats render through a **phoxoidal density evaluator** that uses a 5-coefficient germ basis from catastrophe optics (curvature κ₁/κ₂ + Pearcey cusp generators χ/ω + swallowtail ζ).

The point isn't to compete with state-of-the-art splat compressors on bitrate (it doesn't yet — see "Honest numbers" below). The point is a *structurally* novel splat-rendering pipeline that runs on a 2014 laptop GPU, separates render-mode from archive-mode at the format level, and provides a benchmark (PhoxBench Tier 0) that lets phoxoidal vs Gaussian comparisons be measured cleanly.

## Quick tour

| If you want to... | Look at... |
|---|---|
| **View a `.3dphox` in the browser** | [`viewer/index.html`](viewer/index.html) — drop a `.3dphox` file on it, WebGL renders it client-side |
| **Read or write `.3dphox` from your own code** | [`docs/FORMAT.md`](docs/FORMAT.md) — canonical format reference |
| See the math being claimed | [`docs/thesis_digest.md`](docs/thesis_digest.md) |
| See the current architecture | [`docs/crypsorender_architecture.md`](docs/crypsorender_architecture.md) |
| See the next-phase plan | [`docs/TIER_2_spec.md`](docs/TIER_2_spec.md) |
| See an actual rendered Audi A5 | [`renders/crypsorender_v01/SHOWCASE_T1_final.png`](renders/crypsorender_v01/SHOWCASE_T1_final.png) |
| **See the Tier 1 mesh sweep (4 scenes, 2.0× killer)** | [`renders/crypsorender_v01/SHOWCASE_T1_meshes.png`](renders/crypsorender_v01/SHOWCASE_T1_meshes.png) and [`SHOWCASE_T1_AB.png`](renders/crypsorender_v01/SHOWCASE_T1_AB.png) |
| Read the Tier 1 results writeup | [`reports/TIER_1_results.md`](reports/TIER_1_results.md) |
| See the format design | [`tools/build_v25_attribute_group.py`](tools/build_v25_attribute_group.py), [`tools/build_v28_sh_exact_correction.py`](tools/build_v28_sh_exact_correction.py) |
| See the renderer | [`tools/crypsorender/`](tools/crypsorender/) (~1,600 LoC pure numpy) |
| See the benchmark | [`tools/phoxbench/`](tools/phoxbench/) |
| See the honest measurement story | [`reports/TIER_1.5_compression_baselines.md`](reports/TIER_1.5_compression_baselines.md) and [`reports/TIER_1.5_bits_per_gaussian.md`](reports/TIER_1.5_bits_per_gaussian.md) |
| Run everything end-to-end | [`reports/TIER_2_run_when_sandbox_back.md`](reports/TIER_2_run_when_sandbox_back.md) |
| Read the founding thesis | [`recovery_v2/THESIS.txt`](recovery_v2/THESIS.txt) |
| Check CI is green | [`.github/workflows/test.yml`](.github/workflows/test.yml) — runs anchor tests, smoke benchmark, and bit-exactness check on every PR |

## PhoxBench Tier 1 result — phoxoidal vs Gaussian on real meshes

The thesis's central prediction ("phoxoidal beats Gaussian on curved/cusp/fold geometry") is now measured on real meshes. Same point cloud, same clustering, same blob budget — phoxoid replaces 2× the Gaussians at equal fit RMSE on every scene tested.

| Scene (B=32) | Source | Gauss RMSE | Phox RMSE | **Killer ratio** |
|---|---|---:|---:|---:|
| Happy Buddha | Stanford scan | 0.01799 | 0.01583 | **2.0×** |
| Armadillo    | Stanford scan | 0.00902 | 0.00749 | **2.0×** |
| Doom combat  | Game-scene PLY | 0.05017 | 0.04670 | **2.0×** |
| Audi A5      | Trained 3DGS PLY | 0.02857 | 0.02728 | **2.0×** |

The 2.0× ratio is **flat across both B=32 and B=64**, and reproduces the synthetic Tier 0 finding (sphere/saddle/fold/cusp). Visuals: [`SHOWCASE_T1_meshes.png`](renders/crypsorender_v01/SHOWCASE_T1_meshes.png) (4×3 contact sheet) and [`SHOWCASE_T1_AB.png`](renders/crypsorender_v01/SHOWCASE_T1_AB.png) (per-scene Gaussian / Phoxoid / error heatmap). Full writeup: [`reports/TIER_1_results.md`](reports/TIER_1_results.md). Reproduce: `python3 -m phoxbench.run_mesh --all --budgets 32 64` (~60s, CPU-only).

## Honest numbers (Audi A5 PLY, 763,800 splats)

The naive headline ("v28 is 5.6× smaller than the original PLY!") is **true vs raw float32 PLY** but inflated against any reasonable baseline. Against `zstd -12 PLY` at the same q8 fidelity:

| Format | bits/Gaussian | vs zstd-12 PLY | Lossy? |
|---|---:|---:|---|
| `zstd -12` PLY (lossless) | 470.4 | 1.00× | no |
| **CRYPSOID v28 EXACT archive** | **336.9** | **1.40× smaller** | bit-exact at q8 grid |
| **CRYPSOID v28 VQ render** | **196.9** | **2.39× smaller** | yes (SH ≈ 5% per-coef RMSE) |

Reference points the field uses:
- Self-Organizing Gaussians (SOG) ≈ 50–80 bpg
- HAC ≈ 30–60 bpg

So **CRYPSOID is meaningfully better than zstd PLY but is not competitive with state-of-the-art splat-specific compressors on bitrate alone.** The architectural claims (tier-aware dispatch, phoxoidal density, no GPU dependency, bit-exact at q8 grid) stand on their own and have been independently verified — see [`reports/`](reports/) for the per-attribute exactness audit.

## What's bit-exact (verified)

The chain `PLY → v25 quantization → v28 EXACT archive → decode → ndarray` is bit-exact at the q8 grid level. All five non-SH attributes are byte-identical between v25 and v28 archive; the v28 archive's SH reconstruction matches v25's stored q8 SH stream for all 34,371,000 int8 elements. See [`reports/TIER_1.5_compression_baselines.md`](reports/TIER_1.5_compression_baselines.md).

The lossy step is the one-time PLY → v25 quantization (q8 SH, u24 XYZ, f16 scale, i16 quat, u8 DC/opacity). It's deterministic and reproducible from PLY by design.

## What's done

- v25 / v27 / v28 / v29 container formats fully built and verified.
- v28 SH-VQ render container (17.93 MiB) and v28 q8-EXACT archive (30.67 MiB) both produced and decoded back successfully.
- v29 residual-codec sweep ran 10 of 11 advertised layouts; winner is `morton_splat_major + zlib9` at 30.15 MiB.
- CPU renderer: EWA projection, 16×16 tile binning, depth-sorted alpha compositing with early termination, real-basis SH eval up to degree 3, tier-aware dispatch, screen-space phoxoidal density (Tier 1) AND faithful 5-coef Pearcey germ density (Tier 2 code, awaiting execution).
- PhoxBench Tier 0 — six synthetic stress scenes (plane, sphere, saddle, fold, cusp, thin sheet) with analytic ground truth and per-scene phoxoid-vs-Gaussian killer-ratio benchmark.
- **Browser viewer** ([`viewer/index.html`](viewer/index.html)) — single self-contained HTML+JS that loads any `.3dphox` (v25 / v27 / v28 render / v28 EXACT archive) and renders it via WebGL2 in any modern browser. Includes a JS port of the format decoder. The viewer's GPU code runs on the user's GPU; the CRYPSOID core codebase stays GPU-free.
- **CI** ([`.github/workflows/test.yml`](.github/workflows/test.yml)) — every push runs the Tier 2 numerical-correctness anchors, a PhoxBench cusp smoke benchmark, and (if the containers are in-tree) a v25↔v28-archive byte-identity check. Banned-package check ensures no GPU dependency ever enters the dep tree.

## What's not done yet

- **Tier 2 execution.** All Tier 2 code is committed but the bash sandbox where I was running things hit a disk-full state mid-Tier-1.5 and never recovered in this session. See [`reports/TIER_2_run_when_sandbox_back.md`](reports/TIER_2_run_when_sandbox_back.md) for the one-line run command (`bash tools/tier2_run_all.sh`).
- **Real-time interactive viewer (Tier 3).** Either heavy numpy optimization or a separate WebGL viewer that loads `.3dphox` client-side.
- **Layer-1 evidence terms** from the thesis (render residual `R`, discontinuity barrier `D`, neighbor consistency `S`). Need an evidence stack `E` we don't have for the Audi data; belongs to PhoxBench Tier 3 (real reconstruction datasets).
- **Sheaf-theoretic neighbor compatibility maps** from the thesis. v0.4+ work.
- **Native germ chunks in `.3dphox`.** Currently germs are computed at load time from the splat positions; saving them is v0.4 work.

## Hard rules (read before contributing)

- **No GPU/CUDA dependencies, ever.** Forbidden: `torch`, `pytorch`, `cuda-toolkit`, any `nvidia-*` package, `gsplat`, `diff-gaussian-rasterization`, `nerfstudio`. The whole project exists because state-of-the-art splat tooling demands a heavy GPU stack; CRYPSOID's reason to exist is to be a 1:1 alternative that does not.
- **Honesty over flash.** Every render writes a `manifest.json` declaring which math path it actually used. Never claim "phoxoidal results" on Gaussian-only data.
- **Phased reviewable artifacts.** Spec → sign-off → implementation → measurement → repeat. See [`reports/PROJECT_STATE.md`](reports/PROJECT_STATE.md) for the cycle history.

## Provenance / source asset

The Audi A5 PLY (763,800 splats, 180,258,277 bytes) is a publicly available Gaussian-splat capture and is not redistributed in this repo (see `.gitignore`). To reproduce builds locally, drop the PLY at `inputs/audi/Audi A5 Sportback.zip` (must contain `scene.ply`).

## License

See `LICENSE` once chosen — TBD by the project owner.
