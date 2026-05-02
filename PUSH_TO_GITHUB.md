# Push CRYPSOID to github.com/<your-handle>/crypso1d

## Why I can't push from inside the agent

Two reasons:

1. **No credentials.** The agent doesn't have access to your GitHub auth (SSH key, personal access token, or `gh` CLI session). Even if I tried, every `git push` would fail with `Authentication failed`.
2. **Sandbox is currently disk-locked.** The agent's bash sandbox has `/etc/srt-settings: ENOSPC` (filled up during the Tier 1.5 compression baselines run) and won't init new processes — `git` itself can't run.

So this is a "you push it" rather than an "I push it." But everything's set up for a clean three-line first push.

## Before you start

1. **Create an empty repository on GitHub** at `github.com/<your-handle>/crypso1d`. Don't initialize it with a README, LICENSE, or .gitignore — those already live in this folder and would conflict.
2. **Pick a license.** I left `LICENSE` out of this folder so you can choose. Common picks for research code: `MIT`, `Apache-2.0`, or `CC-BY-4.0` (for the data + docs). If you're not sure, MIT is the safest "use it however" default. Drop a `LICENSE` file at the repo root before committing or right after.

## The four commands (run from this folder)

```bash
cd "C:\Users\TEETHBOX\Documents\Claude\Projects\Crypsoid"

git init -b main
git config user.name  "Donald"
git config user.email "Donald@Harrow.Haus"

git add -A
git commit -m "Initial CRYPSOID push: format + renderer + PhoxBench (Tier 1 done, Tier 2 awaiting bench run)"

git remote add origin https://github.com/<your-handle>/crypso1d.git
git push -u origin main
```

(Use SSH if you have a key configured: `git@github.com:<your-handle>/crypso1d.git`.)

## What's NOT in the push

The `.gitignore` excludes:
- `inputs/audi/Audi A5 Sportback.zip` — third-party 3D Gaussian Splat capture (172 MB, do not redistribute).
- `inputs/audi/scene.ply` — same content extracted, also third-party.
- `/tmp/state_*` and on-disk render-progress NPZ files — large + machine-specific.
- `__pycache__`, `.pyc`, `.vscode`, etc.

To reproduce a build after cloning, anyone (including future-you on a different machine) needs to put their own copy of `Audi A5 Sportback.zip` at `inputs/audi/Audi A5 Sportback.zip` and they get the same byte-exact builds.

## What IS in the push (~150 MB total)

- `docs/` — the spec docs (thesis digest, architecture, Tier 2 spec).
- `tools/` — all the build scripts, the `crypsorender` package, the `phoxbench` package, the orchestrator scripts.
- `outputs/` — the v25/v28 .3dphox containers (3 files, ~77 MB total).
- `recovery_v2/` — the original recovery zip's contents including the v27 anchor and the THESIS.txt.
- `inputs/v21_v22_artifacts/` — small CSVs/scripts from the v21/v22 era.
- `renders/crypsorender_v01/` — every shipped render image, contact sheets, manifests.
- `reports/` — every honest report including TIER_1.5_*.md.
- `v29_sweep/`, `v30_truth_gate/` — sub-run outputs.
- This file (PUSH_TO_GITHUB.md), README.md, .gitignore.

GitHub's per-file limit is 100 MB. The largest individual file in the push is the v28 q8-EXACT archive at 30.67 MiB — well under.

GitHub's recommended repo size is ≤1 GB; we're around 150 MB — comfortable.

## After the push

The README will render at the repo root. The headline images that GitHub renders inline are:
- `renders/crypsorender_v01/SHOWCASE_T1_final.png` — the Audi result
- `renders/crypsorender_v01/v28_tier_overlay_200k.png` — the project doctrine made visual
- `renders/crypsorender_v01/file_sizes.png` — the compression chart

If you want any of those in the README itself (not just linked), tell me and I'll edit the README to embed them.

## If you want CI / automation

Not set up yet. A reasonable first GitHub Action would be:
- run `python3 -m phoxbench.tests` on every PR to make sure the Newton math doesn't regress
- run `python3 tools/eval_metrics.py --a recovery_v2/v27_*.3dphox --b outputs/v28_*.3dphox` to verify bit-exactness on every commit

Both would run in <30 seconds and need only `numpy + scipy + scikit-image + pillow`. Let me know if you want me to write the workflow files.

## After-push to-do list (the real Tier 2 numbers)

Once pushed, the next task is to actually run `bash tools/tier2_run_all.sh` from a working environment (your local machine, a colab, an EC2 box — anywhere with bash + python3 + numpy/scipy/skimage/pillow). That produces:

- `phoxbench/runs/<scene>_b<budget>/` for 18 scenes — the killer-ratio numbers that validate or invalidate the thesis
- `renders/crypsorender_v01/T2_audi_*.png` — Audi at faithful Newton vs Gaussian baseline
- `renders/crypsorender_v01/SHOWCASE_T2.png` — the final headline image

Open a PR with the results and I'll write up the Tier 2 summary doc against the actual numbers.
