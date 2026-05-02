# Tier 2 — what to run when the sandbox is back

The sandbox is currently disk-locked (`/etc/srt-settings: ENOSPC`). All Tier 2
*code* is written and committed; only execution is blocked. This is the punch
list for whoever (future me, or anyone) gets a working bash next.

## In one command

```bash
bash /sessions/ecstatic-sleepy-curie/mnt/Crypsoid/tools/tier2_run_all.sh
```

That sequences anchor tests → PhoxBench sweep → Audi faithful + Gaussian
renders → object-mask metrics → multi-view distribution → final showcase
sheet.

## Step by step (if you want to gate at each stage)

### Step 0 — sanity-check the math first

```bash
cd /sessions/ecstatic-sleepy-curie/mnt/Crypsoid/tools
python3 -m phoxbench.tests
```

Should print 4 PASS lines. If anything fails, do NOT proceed — there's a bug
in the Newton solver or germ basis. The most likely failure mode is the cusp
asymmetry test (means the rasterizer's level sets are still elliptical —
Tier 2 hasn't actually produced phoxoidal geometry).

### Step 1 — PhoxBench Tier 0 sweep

```bash
cd /sessions/ecstatic-sleepy-curie/mnt/Crypsoid/tools
python3 -m phoxbench.run_scene --scene all --budgets 64 128 256
```

Expect ~10 minutes. 18 runs (6 scenes × 3 budgets) with killer-ratio search.
Output: `phoxbench/runs/<scene>_b<B>/{input_preview, gaussian_render,
phoxoidal_render, side_by_side, error_heatmap}.png + metrics.json`. A
summary lands at `phoxbench/runs/summary.json`.

The killer-ratio table is the headline number: how many Gaussian blobs do
you need to match phoxoid quality at the same budget. If we see ratios > 2×
on cusp/fold scenes and ≈ 1× on plane, the thesis is validated. If all
ratios cluster near 1, we have a project-shaking honest finding (also
valuable, just different).

### Step 2 — re-render Audi with the faithful path

This proves the per-pixel Newton path works on real splat data.

```bash
cd /sessions/ecstatic-sleepy-curie/mnt/Crypsoid/tools
mkdir -p /tmp/state_t2_faithful
python3 render_phox_chunked.py \
    --scene ../outputs/v28_sh_vq_exact_archive_container.3dphox \
    --is-phox --size 512 --max-points 200000 --use-sh \
    --yaw 90 --pitch 2 --distance 1.0 --fov 50 \
    --state-dir /tmp/state_t2_faithful --init
# 4 batches of 50k each:
for B in 50000 50000 50000 50000; do
    python3 render_phox_chunked.py --state-dir /tmp/state_t2_faithful --batch $B --mode faithful
done
python3 render_phox_chunked.py --state-dir /tmp/state_t2_faithful --finalize \
    --out ../renders/crypsorender_v01/T2_audi_faithful_512.png
```

Then the Gaussian baseline at the same camera (same splats, just different
density math), so we can A/B them:

```bash
mkdir -p /tmp/state_t2_gauss
python3 render_phox_chunked.py \
    --scene ../outputs/v28_sh_vq_exact_archive_container.3dphox \
    --is-phox --size 512 --max-points 200000 --use-sh \
    --yaw 90 --pitch 2 --distance 1.0 --fov 50 \
    --state-dir /tmp/state_t2_gauss --init
for B in 50000 50000 50000 50000; do
    python3 render_phox_chunked.py --state-dir /tmp/state_t2_gauss --batch $B --mode gaussian
done
python3 render_phox_chunked.py --state-dir /tmp/state_t2_gauss --finalize \
    --out ../renders/crypsorender_v01/T2_audi_gaussian_512.png
```

### Step 3 — Tier 1.5 item 4: object-mask metrics

```bash
cd /sessions/ecstatic-sleepy-curie/mnt/Crypsoid/tools
python3 eval_metrics.py \
    --a ../renders/crypsorender_v01/v28_archive_200k_side.png \
    --b ../renders/crypsorender_v01/v28_render_200k_side.png \
    --auto-mask-from ../renders/crypsorender_v01/ply_200k_side.png \
    --threshold 0.05
```

Should report both full-image and masked PSNR/SSIM. Expect masked SSIM to
drop noticeably from the inflated 0.9996 (background match) to something
in the 0.95–0.99 range that reflects only the car body.

### Step 4 — Tier 1.5 item 5: multi-view distribution + LPIPS-via-PSNR

```bash
cd /sessions/ecstatic-sleepy-curie/mnt/Crypsoid/tools
python3 multiview_cameras.py --n-azimuth 16 --elevations -2 18 \
    --out ../renders/crypsorender_v01/multiview_cams.json
python3 tier2_multiview.py \
    --cameras ../renders/crypsorender_v01/multiview_cams.json \
    --out ../renders/crypsorender_v01/multiview \
    --max-points 60000 --size 384
```

Slow — 32 cameras × 3 renders (PLY, archive, render) = 96 chunked renders.
Expect ~30 minutes. Outputs per-view PNGs plus a `multiview_summary.json`
with mean / median / worst PSNR + SSIM (full and masked) for both
v28-EXACT-vs-PLY and v28-render-vs-PLY across all views.

This is what replaces the cherry-picked single-view "PSNR 56.33 dB" claim
with a real distribution.

LPIPS proper would need torch, which is forbidden by the no-GPU-deps rule.
Skip LPIPS or implement a multi-scale-PSNR proxy if needed.

### Step 5 — final Tier 2 contact sheet

```bash
cd /sessions/ecstatic-sleepy-curie/mnt/Crypsoid/tools
python3 tier2_contact_sheet.py
```

Bundles everything into `renders/crypsorender_v01/SHOWCASE_T2.png`.
Missing inputs are placeholdered so you can run this iteratively.

### Step 6 — finish the compression baselines (Tier 1.5 item 1, partial)

The earlier Tier 1.5 run hit `zstd -15` and stopped. To finish:

```bash
# zstd 15 / 19 / 22 -- give each plenty of time:
PLY=/tmp/scene.ply  # extract from inputs/audi/*.zip if needed
unzip -p '/sessions/.../Audi A5 Sportback.zip' scene.ply > $PLY
for L in 15 19 22; do
    zstd -$L --long --ultra -k -q -f $PLY -o /tmp/scene.ply.zst$L
    ls -la /tmp/scene.ply.zst$L
done
xz -9 -k -c $PLY > /tmp/scene.ply.xz9 ; ls -la /tmp/scene.ply.xz9
# .npz baseline:
python3 -c "
import zipfile, numpy as np
from pathlib import Path
ply_zip = Path('/sessions/.../Audi A5 Sportback.zip')
# (same loader code from compression_baselines doc)
"
```

Then update `reports/TIER_1.5_compression_baselines.md` with the missing
rows. Almost certainly tightens CRYPSOID's compression edge a bit further;
won't change the qualitative story.

## Where things live

| Type | Path |
|---|---|
| The architecture spec | `docs/TIER_2_spec.md` |
| The thesis digest | `docs/thesis_digest.md` |
| The renderer code | `tools/crypsorender/` |
| The benchmark code | `tools/phoxbench/` |
| The metrics tool | `tools/eval_metrics.py` |
| The orchestrators | `tools/tier2_*.{py,sh}` |
| The Tier 1.5 reports | `reports/TIER_1.5_*.md` |
| The latest project state | `reports/PROJECT_STATE.md` |
| The headline showcase (Tier 1) | `renders/crypsorender_v01/SHOWCASE_T1_final.png` |
| The headline showcase (Tier 2, when run) | `renders/crypsorender_v01/SHOWCASE_T2.png` |

## Sanity checks before starting

1. **Disk has space** (`df -h /tmp` should show >5GB free).
2. **scene.ply extracted** at `/tmp/scene.ply` (180 MB) for the compression baselines.
3. **scikit-learn + scipy + scikit-image + PIL + numpy installed** (already are).
4. **Clear the workspace pycache** at the start: `find tools/crypsorender tools/phoxbench -name __pycache__ -exec rm -rf {} +` to avoid stale bytecode bugs we hit before.

## Honest expected outcomes (the numbers I'd bet on)

| What | Honest prediction |
|---|---|
| Anchor tests | PASS (the math is straightforward; main risk is some sign error) |
| PhoxBench plane killer-ratio | ≈ 1.0× |
| PhoxBench sphere | 1.1–1.3× |
| PhoxBench saddle | 1.5–2× |
| PhoxBench fold | 2–4× |
| **PhoxBench cusp** | **3–8× (the result that justifies the project)** |
| PhoxBench thin sheet | 1.0–1.5× |
| Audi faithful render vs Gaussian | Visually subtle on the body, more difference on edges and high-curvature splats. Per-tier dispatch counts will differ measurably (faithful path early-terminates more aggressively where germ is steep). |
| Multi-view archive vs PLY | Mean masked PSNR ≈ 50 dB, worst ≈ 40 dB. |
| Multi-view render vs PLY | Mean masked PSNR ≈ 36 dB, worst ≈ 28 dB. |
| Final SHOWCASE_T2.png | Should clearly show the cusp scene where phoxoids visibly beat Gaussians, plus the killer-ratio table making the project's reason-to-exist concrete. |

If cusp killer-ratio comes back ≈ 1.0×, the thesis hasn't been validated
on this benchmark — that's a real, reportable finding. The right response
is to either (a) verify the cubic Pearcey terms are actually being fit
non-trivially on the cusp scene, or (b) admit phoxoids don't help here.
