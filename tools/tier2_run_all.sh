#!/usr/bin/env bash
# Tier 2 master runner. Run this when the sandbox is back.
#
# Sequences:
#   1. Newton anchor tests (must pass before anything else)
#   2. PhoxBench Tier 0 sweep (6 scenes x 3 budgets)
#   3. Audi v28 archive re-render with --mode faithful
#   4. Audi v28 archive re-render with --mode gaussian (baseline)
#   5. Object-mask metrics on the Audi pair
#   6. Multi-view distribution: 32 cameras (16 az x 2 el) x 2 modes
#   7. Final Tier 2 contact sheet

set -e

ROOT=/sessions/ecstatic-sleepy-curie/mnt/Crypsoid
cd "$ROOT/tools"

echo "=========================================="
echo "Tier 2 Step 1 -- Newton anchor tests"
echo "=========================================="
python3 -m phoxbench.tests || { echo "ANCHOR TESTS FAILED -- aborting"; exit 1; }

echo
echo "=========================================="
echo "Tier 2 Step 2 -- PhoxBench Tier 0 sweep"
echo "=========================================="
python3 -m phoxbench.run_scene --scene all --budgets 64 128 256 \
    --out "$ROOT/phoxbench/runs"

echo
echo "=========================================="
echo "Tier 2 Step 3 -- Audi v28 archive @ faithful"
echo "=========================================="
mkdir -p /tmp/state_t2_faithful
python3 render_phox_chunked.py \
    --scene "$ROOT/outputs/v28_sh_vq_exact_archive_container.3dphox" \
    --is-phox --size 512 --max-points 200000 --use-sh \
    --yaw 90 --pitch 2 --distance 1.0 --fov 50 \
    --state-dir /tmp/state_t2_faithful --init
# Run in 50k chunks; 4 batches should cover 200k
for B in 50000 50000 50000 50000; do
    python3 render_phox_chunked.py --state-dir /tmp/state_t2_faithful --batch $B --mode faithful
done
python3 render_phox_chunked.py --state-dir /tmp/state_t2_faithful --finalize \
    --out "$ROOT/renders/crypsorender_v01/T2_audi_faithful_512.png"

echo
echo "=========================================="
echo "Tier 2 Step 4 -- Audi v28 archive @ gaussian baseline"
echo "=========================================="
mkdir -p /tmp/state_t2_gauss
python3 render_phox_chunked.py \
    --scene "$ROOT/outputs/v28_sh_vq_exact_archive_container.3dphox" \
    --is-phox --size 512 --max-points 200000 --use-sh \
    --yaw 90 --pitch 2 --distance 1.0 --fov 50 \
    --state-dir /tmp/state_t2_gauss --init
for B in 50000 50000 50000 50000; do
    python3 render_phox_chunked.py --state-dir /tmp/state_t2_gauss --batch $B --mode gaussian
done
python3 render_phox_chunked.py --state-dir /tmp/state_t2_gauss --finalize \
    --out "$ROOT/renders/crypsorender_v01/T2_audi_gaussian_512.png"

echo
echo "=========================================="
echo "Tier 2 Step 5 -- Object-mask metrics on the Audi pair"
echo "=========================================="
python3 eval_metrics.py \
    --a "$ROOT/renders/crypsorender_v01/T2_audi_gaussian_512.png" \
    --b "$ROOT/renders/crypsorender_v01/T2_audi_faithful_512.png" \
    --auto-mask-from "$ROOT/renders/crypsorender_v01/T2_audi_gaussian_512.png" \
    --threshold 0.05 \
    --out "$ROOT/renders/crypsorender_v01/T2_audi_masked_metrics.json"

echo
echo "=========================================="
echo "Tier 2 Step 6 -- Multi-view (16 az x 2 el = 32 cameras, both modes)"
echo "=========================================="
python3 multiview_cameras.py --n-azimuth 16 --elevations -2 18 \
    --distance 1.4 --fov 45 \
    --out "$ROOT/renders/crypsorender_v01/multiview_cams.json"
echo "(skipping per-camera renders here -- see tier2_multiview.py for incremental driver)"
python3 tier2_multiview.py \
    --cameras "$ROOT/renders/crypsorender_v01/multiview_cams.json" \
    --out "$ROOT/renders/crypsorender_v01/multiview"

echo
echo "=========================================="
echo "Tier 2 Step 7 -- Final contact sheet"
echo "=========================================="
python3 tier2_contact_sheet.py

echo
echo "Tier 2 run-all complete. See $ROOT/renders/crypsorender_v01/SHOWCASE_T2.png"
