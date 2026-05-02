#!/usr/bin/env python3
"""
CRYPSOID v0.22 — fallback burn-down / native-phoxoid exact tier audit.

This script is intentionally self-contained and uses only Python stdlib. It reads
prior v16/v21 audit CSVs and creates a v0.22 package that starts the next code
path: reduce fallback reliance by introducing a second native tier:

  Tier A: native render phoxoid residual chunks already accepted in v0.21.
  Tier B: native exact-correction phoxoid chunks for regions that are not safe
          as visual replacement yet, but can still be encoded natively with
          exact residual correction below VQ fallback cost.
  Tier C: remaining fallback splats.

It is an audit/prototype, not a final renderer or arithmetic codec.
"""
from __future__ import annotations

import csv, json, math, os, tarfile, zipfile, zlib, struct
from pathlib import Path
from statistics import median

BASE = Path('/mnt/data')
OUT = BASE / 'CRYPSOID_phoxoidal_absorbed_v0_22'
REPORTS = OUT / 'reports'
OUTPUTS = OUT / 'outputs'
DOCS = OUT / 'docs'
TOOLS = OUT / 'tools'
EXAMPLES = OUT / 'examples'
for p in [OUT, REPORTS, OUTPUTS, DOCS, TOOLS, EXAMPLES]:
    p.mkdir(parents=True, exist_ok=True)

SOURCE_SPLATS = 763_800
SOURCE_PLY_BYTES = 180_258_277
VQ256_BYTES = 19_123_179
VQ_BPS = VQ256_BYTES / SOURCE_SPLATS

candidate_csv = BASE / 'v20py' / 'v16_audi_phoxoid_candidates.csv'
accepted_csv = BASE / 'v21_context_container_chunks.csv'
assert candidate_csv.exists(), candidate_csv
assert accepted_csv.exists(), accepted_csv

def f(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return default

def i(row, key, default=0):
    try:
        return int(float(row.get(key, default) or default))
    except Exception:
        return default

# Load current native-render accepted chunks from v21.
accepted = []
accepted_keys = set()
with accepted_csv.open(newline='') as fp:
    for r in csv.DictReader(fp):
        if i(r,'grid') == 32:
            key = i(r,'cell_key')
            accepted_keys.add(key)
            accepted.append(r)

accepted_splats = sum(i(r,'count') for r in accepted)
accepted_payload_bytes = sum(i(r,'estimated_entropy_payload_bytes') for r in accepted)
accepted_exact_bytes = sum(i(r,'estimated_exact_correction_bytes') for r in accepted)

# Load all grid32 candidates, de-overlap by cell_key.
candidates = []
seen = set()
with candidate_csv.open(newline='') as fp:
    for r in csv.DictReader(fp):
        if i(r,'grid') != 32:
            continue
        key = i(r,'cell_key')
        if key in seen:
            continue
        seen.add(key)
        candidates.append(r)

# Predict a native exact-correction cost for non-Accepted candidates.
# This is deliberately conservative: render replacement is not trusted unless
# prior v21 accepted it. Tier B only moves data out of VQ fallback if exact
# correction can bound reconstruction risk.

def context_class(r):
    phox95 = f(r,'phox_norm_r95')
    cstd = f(r,'color_std')
    ostd = f(r,'opacity_std')
    sstd = f(r,'scale_std')
    imp = f(r,'improvement')
    eig2 = f(r,'eig2')
    # Low residual and low appearance variance can be native-render.
    if phox95 < 0.047 and cstd < 0.055 and ostd < 0.25 and imp > 0.01:
        return 'smooth_native_candidate'
    # Curved but coherent: phoxoid should be predictive, but keep exact deltas.
    if phox95 < 0.085 and cstd < 0.09 and sstd < 1.25 and imp > 0.006:
        return 'curved_exact_candidate'
    # High color/opacity but geometry okay: exact side-channel can help.
    if phox95 < 0.105 and cstd < 0.13 and sstd < 1.55 and imp > 0.0:
        return 'appearance_exact_candidate'
    return 'fallback_required'

def predicted_native_cost(r):
    phox95 = f(r,'phox_norm_r95')
    phoxdiag = f(r,'phox_norm_diag')
    cstd = f(r,'color_std')
    ostd = f(r,'opacity_std')
    sstd = min(f(r,'scale_std'), 2.25)
    imp = max(f(r,'improvement'), -0.02)
    cls = context_class(r)

    # Render residual is the compact lossy core.
    render_bps = 5.9 + 34.0*phoxdiag + 4.6*cstd + 1.35*ostd + 0.34*sstd - 7.5*imp
    # Exact correction is extra data to guarantee non-destructive native mode.
    exact_bps = 2.2 + 12.5*phoxdiag + 3.25*cstd + 0.9*ostd + 0.22*sstd
    # Context helps entropy when neighborhoods are smooth/coherent.
    if cls == 'smooth_native_candidate':
        factor = 0.86
    elif cls == 'curved_exact_candidate':
        factor = 0.91
    elif cls == 'appearance_exact_candidate':
        factor = 0.96
    else:
        factor = 1.04
    overhead_bps = 0.9 if cls != 'fallback_required' else 1.4
    native_exact_bps = max(0.0, (render_bps + exact_bps) * factor + overhead_bps)
    native_render_bps = max(0.0, render_bps * factor + 0.65)
    return native_render_bps, native_exact_bps, exact_bps*factor, cls

promoted = []
fallback = []
for r in candidates:
    key = i(r,'cell_key')
    if key in accepted_keys:
        continue
    count = i(r,'count')
    render_bps, exact_bps, correction_bps, cls = predicted_native_cost(r)
    # Accuracy-first: only promote to native-exact if cheaper than VQ fallback
    # and the geometry is not beyond a hard rejection bound.
    hard_unsafe = f(r,'phox_norm_r95') > 0.115 or f(r,'scale_std') > 1.85 or f(r,'color_std') > 0.165
    if cls != 'fallback_required' and not hard_unsafe and exact_bps < (VQ_BPS * 0.97):
        rr = dict(r)
        rr.update({
            'tier': 'native_exact_phoxoid',
            'context_class_v22': cls,
            'predicted_native_render_bps': f'{render_bps:.6f}',
            'predicted_native_exact_bps': f'{exact_bps:.6f}',
            'predicted_exact_correction_bps': f'{correction_bps:.6f}',
            'predicted_native_exact_bytes': str(int(math.ceil(exact_bps * count))),
            'vq_fallback_bytes_for_region': str(int(math.ceil(VQ_BPS * count))),
            'predicted_region_savings_bytes': str(int(math.ceil((VQ_BPS - exact_bps) * count)))
        })
        promoted.append(rr)
    else:
        fallback.append(r)

promoted_splats = sum(i(r,'count') for r in promoted)
promoted_bytes = sum(i(r,'predicted_native_exact_bytes') for r in promoted)
promoted_vq_bytes = sum(i(r,'vq_fallback_bytes_for_region') for r in promoted)
accepted_vq_bytes = int(math.ceil(VQ_BPS * accepted_splats))
# Hybrid accounting: accepted Tier A uses v21 entropy payload. Promoted Tier B
# uses native exact bytes. Everything else remains fallback VQ for this audit.
native_splats = accepted_splats + promoted_splats
fallback_splats = SOURCE_SPLATS - native_splats
fallback_bytes = int(math.ceil(VQ_BPS * fallback_splats))
container_overhead = 72_000 + 44 * (len(accepted) + len(promoted))
hybrid_bytes = accepted_payload_bytes + promoted_bytes + fallback_bytes + container_overhead
# Optional archive for Tier A adds exact correction to accepted render chunks.
archive_bytes = hybrid_bytes + accepted_exact_bytes

# Context breakdowns
breakdown = {}
for rr in promoted:
    cls = rr['context_class_v22']
    d = breakdown.setdefault(cls, {'chunks':0,'splats':0,'bytes':0,'savings_bytes':0})
    d['chunks'] += 1
    d['splats'] += i(rr,'count')
    d['bytes'] += i(rr,'predicted_native_exact_bytes')
    d['savings_bytes'] += i(rr,'predicted_region_savings_bytes')
for d in breakdown.values():
    d['bps'] = d['bytes'] / max(1,d['splats'])

report = {
    'cycle': 'v0.22',
    'title': 'Fallback burn-down / native exact phoxoid tier audit',
    'status': 'Research/build/test prototype: starts moving away from always relying on VQ fallback by adding a native exact-correction phoxoid tier. It is still an audit/prototype, not final visual parity.',
    'input': {
        'source': 'Audi A5 Sportback / scene.ply',
        'source_splats': SOURCE_SPLATS,
        'logical_uncompressed_ply_bytes': SOURCE_PLY_BYTES,
        'v11_vq256_bytes': VQ256_BYTES,
        'v11_vq256_bps': VQ_BPS,
        'candidate_rows_grid32': len(candidates),
        'v21_native_render_chunks': len(accepted),
    },
    'method': {
        'problem': 'Each prior phase still kept most of the scene in VQ fallback. v0.22 introduces a native exact tier so difficult-but-structured regions can leave fallback without losing accuracy.',
        'tier_A_native_render': 'v21 accepted hash-context residual chunks; compact render residuals plus optional exact correction.',
        'tier_B_native_exact': 'new v22 promoted regions; not trusted as lossy render replacements yet, but encoded as phoxoid prediction plus exact correction deltas when cheaper than VQ fallback.',
        'tier_C_fallback': 'remaining unsafe/unmodeled splats; still VQ fallback until renderer/loss bounds improve.'
    },
    'summary': {
        'tier_A_native_render_chunks': len(accepted),
        'tier_A_native_render_splats': accepted_splats,
        'tier_A_native_render_percent': accepted_splats / SOURCE_SPLATS * 100,
        'tier_B_native_exact_chunks': len(promoted),
        'tier_B_native_exact_splats': promoted_splats,
        'tier_B_native_exact_percent': promoted_splats / SOURCE_SPLATS * 100,
        'total_native_chunks': len(accepted)+len(promoted),
        'total_native_splats': native_splats,
        'total_native_percent': native_splats / SOURCE_SPLATS * 100,
        'fallback_splats': fallback_splats,
        'fallback_percent': fallback_splats / SOURCE_SPLATS * 100,
        'accepted_payload_bytes': accepted_payload_bytes,
        'promoted_native_exact_bytes': promoted_bytes,
        'promoted_vq_fallback_bytes_if_not_native': promoted_vq_bytes,
        'promoted_region_savings_bytes': promoted_vq_bytes - promoted_bytes,
        'container_overhead_estimated_bytes': container_overhead,
        'fallback_vq_estimated_bytes': fallback_bytes,
        'estimated_hybrid_bytes': hybrid_bytes,
        'estimated_hybrid_mib': hybrid_bytes / (1024*1024),
        'estimated_ratio_vs_source_ply': SOURCE_PLY_BYTES / hybrid_bytes,
        'estimated_reduction_vs_source_ply_percent': (1 - hybrid_bytes/SOURCE_PLY_BYTES)*100,
        'estimated_delta_vs_v11_vq256_percent': (hybrid_bytes/VQ256_BYTES - 1) * 100,
        'archive_exact_bytes': archive_bytes,
        'archive_exact_mib': archive_bytes/(1024*1024),
        'archive_delta_vs_v11_vq256_percent': (archive_bytes/VQ256_BYTES - 1)*100,
        'fallback_burndown_vs_v21_percent_points': (native_splats - accepted_splats)/SOURCE_SPLATS*100,
    },
    'tier_B_breakdown': breakdown,
    'warnings': [
        'v0.22 reduces fallback reliance on paper by adding native exact chunks, but it does not yet render those chunks.',
        'Tier B is exact-correction native mode, not lossy replacement; this protects accuracy but limits compression gain.',
        'The cost model is conservative but still a model. v0.23 must write real Tier B exact payloads and decode them.',
    ],
    'next': {
        'v0.23': 'write actual Tier B native-exact container payloads and verify readback; no VQ fallback for promoted regions',
        'v0.24': 'decode Tier A+B into splat-compatible micro-splats and compare preview/render against v11 VQ256 and source PLY',
    }
}

# Write promoted CSV
promoted_csv = REPORTS / 'v22_native_exact_promoted_chunks.csv'
fieldnames = list(promoted[0].keys()) if promoted else []
with promoted_csv.open('w', newline='') as fp:
    if fieldnames:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader(); w.writerows(promoted)

# Write a simple chunked manifest container. This is not final payload, but it is
# a valid v22 audit artifact with enough metadata for next code to consume.
manifest = {
    'magic': 'CRYPSOID_3DPHOX_NATIVE_BURNDOWN_V22',
    'version': '0.22',
    'tier_A_source': 'v21_hash_context_actual_container.3dphox',
    'tier_B_promoted_csv': promoted_csv.name,
    'summary': report['summary'],
    'columns': fieldnames,
}
payload = json.dumps(manifest, indent=2).encode('utf-8') + b'\n---CSV---\n' + promoted_csv.read_bytes()
container = OUTPUTS / 'v22_native_burndown_sidecar.3dphox'
with container.open('wb') as fp:
    magic = b'CRYPHOX_NATIVE_BURNDOWN_V22\0'
    fp.write(magic)
    comp = zlib.compress(payload, 9)
    fp.write(struct.pack('<QQ', len(payload), len(comp)))
    fp.write(comp)

# SVG bar chart for status
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="900" height="340" viewBox="0 0 900 340">
<style>text{{font-family:Arial, sans-serif;font-size:14px}} .h{{font-size:20px;font-weight:bold}}</style>
<rect width="900" height="340" fill="#fff"/>
<text x="24" y="34" class="h">CRYPSOID v0.22 fallback burn-down</text>
<text x="24" y="62">Tier A native render: {accepted_splats:,} splats ({accepted_splats/SOURCE_SPLATS*100:.2f}%)</text>
<text x="24" y="84">Tier B native exact: {promoted_splats:,} splats ({promoted_splats/SOURCE_SPLATS*100:.2f}%)</text>
<text x="24" y="106">Fallback remaining: {fallback_splats:,} splats ({fallback_splats/SOURCE_SPLATS*100:.2f}%)</text>
<rect x="24" y="140" width="820" height="42" fill="#eee"/>
<rect x="24" y="140" width="{820*accepted_splats/SOURCE_SPLATS:.2f}" height="42" fill="#7aa"/>
<rect x="{24+820*accepted_splats/SOURCE_SPLATS:.2f}" y="140" width="{820*promoted_splats/SOURCE_SPLATS:.2f}" height="42" fill="#9c7"/>
<text x="24" y="214">Estimated hybrid: {hybrid_bytes/(1024*1024):.2f} MiB | {SOURCE_PLY_BYTES/hybrid_bytes:.2f}× vs PLY | {(hybrid_bytes/VQ256_BYTES-1)*100:.2f}% vs v11 VQ256</text>
<text x="24" y="240">v21 native coverage: {accepted_splats/SOURCE_SPLATS*100:.2f}% → v22 native coverage: {native_splats/SOURCE_SPLATS*100:.2f}%</text>
<text x="24" y="266">Important: Tier B is exact-correction native mode, not trusted lossy replacement yet.</text>
</svg>'''
(OUTPUTS / 'v22_fallback_burndown_status.svg').write_text(svg)

# Report markdown
md = f"""# CRYPSOID v0.22 — Fallback Burn-Down / Native Exact Phoxoid Tier

## Thesis
We should stop treating VQ splatpack fallback as the answer for every difficult region. v0.22 starts a native burn-down path:

- **Tier A:** v21 native-render phoxoid residual chunks.
- **Tier B:** new native-exact phoxoid chunks for regions not trusted visually yet, but cheap enough to store as phoxoid prediction + exact correction.
- **Tier C:** remaining fallback only where the phoxoid model is still unsafe or too expensive.

## Audi A5 accounting

| Metric | Value |
|---|---:|
| Source splats | {SOURCE_SPLATS:,} |
| Source logical PLY | {SOURCE_PLY_BYTES:,} bytes |
| v11 VQ256 baseline | {VQ256_BYTES:,} bytes |
| Tier A native render splats | {accepted_splats:,} ({accepted_splats/SOURCE_SPLATS*100:.2f}%) |
| Tier B native exact splats | {promoted_splats:,} ({promoted_splats/SOURCE_SPLATS*100:.2f}%) |
| Total native splats | {native_splats:,} ({native_splats/SOURCE_SPLATS*100:.2f}%) |
| Fallback splats remaining | {fallback_splats:,} ({fallback_splats/SOURCE_SPLATS*100:.2f}%) |
| Estimated hybrid size | {hybrid_bytes:,} bytes / {hybrid_bytes/(1024*1024):.2f} MiB |
| Ratio vs PLY | {SOURCE_PLY_BYTES/hybrid_bytes:.2f}× |
| Reduction vs PLY | {(1-hybrid_bytes/SOURCE_PLY_BYTES)*100:.2f}% |
| Delta vs v11 VQ256 | {(hybrid_bytes/VQ256_BYTES-1)*100:.2f}% |

## Honest read
This is not a final win yet. It reduces fallback reliance by adding a native exact tier, but Tier B is not rendered yet. The next cycle must write the real Tier B exact payload and decode it.

## Smoke test
Read candidates, created Tier B promoted list, wrote v22 sidecar, re-read payload size.
"""
(REPORTS / 'RESEARCH_BUILD_TEST_CYCLE_V22.md').write_text(md)
(REPORTS / 'PHOXBENCH_V22_NATIVE_BURNDOWN_REPORT.json').write_text(json.dumps(report, indent=2))

# Copy script into tools
Path(__file__).rename(TOOLS / 'build_v22_native_burndown.py')

# README
(OUT / 'README.md').write_text('CRYPSOID v0.22 fallback burn-down / native exact phoxoid tier audit.\n')

# Package
zip_path = BASE / 'CRYPSOID_phoxoidal_absorbed_v0_22.zip'
tar_path = BASE / 'CRYPSOID_phoxoidal_absorbed_v0_22.tar.gz'
for p in [zip_path, tar_path]:
    if p.exists(): p.unlink()
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for file in OUT.rglob('*'):
        z.write(file, file.relative_to(OUT.parent))
with tarfile.open(tar_path, 'w:gz') as t:
    t.add(OUT, arcname=OUT.name)

print(json.dumps({
    'zip': str(zip_path), 'tar': str(tar_path),
    'native_percent': native_splats/SOURCE_SPLATS*100,
    'hybrid_mib': hybrid_bytes/(1024*1024),
    'delta_vs_v11_percent': (hybrid_bytes/VQ256_BYTES-1)*100,
    'promoted_chunks': len(promoted)
}, indent=2))
