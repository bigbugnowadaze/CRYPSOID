# Contributing to CRYPSOID

Three rules first, then the rest.

## Rules

1. **No GPU/CUDA dependencies, ever.**
   The whole project exists to be a 1:1 alternative to splat tooling that demands a heavy GPU stack. A patch that brings in `torch`, `pytorch`, `cuda-toolkit`, any `nvidia-*` package, `gsplat`, `diff-gaussian-rasterization`, `nerfstudio`, or any GPU-bound library will be rejected.
   Allowed runtime: `numpy`, `scipy`, `scikit-image`, `scikit-learn`, `Pillow`, `imageio`, `opencv-python`, `ffmpeg` (binary), Python stdlib.
   Optional dev-only: `pytest`. CPU-only JIT (`numba`, `cython`) on a case-by-case basis if perf needs it.

2. **Honesty over flash.**
   Every render output writes a `manifest.json` declaring which math path actually ran and what the data actually was. Don't add a feature whose claim can't be measured. Don't compare against "raw PLY" without also comparing against `zstd -12 PLY` (see [`reports/TIER_1.5_compression_baselines.md`](reports/TIER_1.5_compression_baselines.md) for why).

3. **Phased reviewable artifacts.**
   Spec → sign-off → implementation → measurement → repeat. Big PRs should land as a series of small commits each with a deliverable. Match the existing `Tier N.M` numbering.

## Coding style

- Pure numpy operations preferred over Python loops over splats/pixels.
- Tile-batched inner loops where possible. Per-pixel-per-splat loops in Python are usually wrong (see `tools/render_phox_chunked.py` for the right pattern).
- Functions get a docstring with: what it does, what shape inputs it expects, what shape it returns, and any pre/post conditions.
- Test math changes with the anchor framework in `tools/phoxbench/tests.py`. Add an anchor for any new math you add.

## File-size hygiene

GitHub limits: 100 MB per file, recommended ≤1 GB per repo. Currently the repo sits around 150 MB with the .3dphox containers in-tree. If we add more containers, switch to git-lfs:

```bash
git lfs install
git lfs track "*.3dphox"
git add .gitattributes
```

Don't push the source PLY (Audi A5 Sportback.zip) — it's third-party and `.gitignore`d.

## Where things live

| Topic | Location |
|---|---|
| `.3dphox` format design | `tools/build_v25_attribute_group.py`, `tools/build_v28_sh_exact_correction.py` |
| The renderer | `tools/crypsorender/` |
| The benchmark | `tools/phoxbench/` |
| The orchestrators | `tools/tier2_*.{py,sh}` |
| The math (germ, SH, EWA, quat) | `tools/crypsorender/math/` |
| The thesis | `recovery_v2/THESIS.txt` |
| The state-of-the-project doc | `reports/PROJECT_STATE.md` |

## How to test changes

```bash
# Math anchors (must pass — covers Newton solver, reduce-to-Gaussian, etc.)
cd tools && python3 -m phoxbench.tests

# Single PhoxBench scene smoke test
cd tools && python3 -m phoxbench.run_scene --scene cusp --budget 64 --no-killer

# Full benchmark (~10 min)
cd tools && python3 -m phoxbench.run_scene --scene all --budgets 64 128 256

# Audi smoke test (small subsample, fast)
cd tools && python3 render_phox_chunked.py \
    --scene ../outputs/v28_sh_vq_exact_archive_container.3dphox \
    --is-phox --size 256 --max-points 20000 --use-sh \
    --yaw 90 --pitch 2 --distance 1.0 --fov 50 \
    --state-dir /tmp/state_smoke --init
cd tools && python3 render_phox_chunked.py --state-dir /tmp/state_smoke --batch 50000 --mode faithful
cd tools && python3 render_phox_chunked.py --state-dir /tmp/state_smoke --finalize --out /tmp/smoke.png
```

## How to claim something works

After running benchmarks, write a 1-page report in `reports/<phase>_<topic>.md` with:
- What you ran
- The numbers, including masked metrics where appropriate
- Honest caveats (what wasn't measured, where the comparison is unfair)
- The exact commands needed to reproduce

The existing `reports/TIER_1.5_*.md` are the format reference.

## Scope of phases (current as of 2026-04-30)

- **Tier 1**: Gaussian-baseline CPU renderer with tier-aware dispatch infrastructure. **DONE.**
- **Tier 1.5**: Measurement-integrity pass — per-attribute exactness, fair compression baselines, bits/Gaussian, mask metrics, multi-view distribution. **Items 1–3 done; 4 + 5 awaiting sandbox refresh.**
- **Tier 2**: Faithful per-pixel phoxoidal Newton solver + 5-coef Pearcey germ basis + PhoxBench Tier 0. **Code done; benchmark execution pending.**
- **Tier 3**: Real-time interactive viewer (CPU optimization or browser viewer). **Spec-only.**
- **Future**: Layer 1 evidence terms (`R, D, S`), sheaf-theoretic neighbor compatibility, native germ chunks in `.3dphox`.
