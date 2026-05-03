"""Build a small v34 .phoxseq demo over the Audi base.

Synthetic "halo bloom" animation: the back-floor halo splats fade in, peak,
and fade back out over 24 frames (1 second @ 24 fps). Each frame's phoxdelta
modifies the opacity of ~10k halo splats with a sinusoidal envelope.

Output:
    outputs/v34_audi_halo_bloom.phoxseq
"""
from __future__ import annotations
import sys, time, struct, zlib
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np

from crypsorender.io.phox_loader import load_3dphox_v28_archive
from crypsorender.io.phoxdelta_codec import (
    encode_phoxdelta, decode_phoxdelta, base_crc as compute_base_crc,
)
from crypsorender.io.phoxseq_codec import (
    PhoxSeqFrame, encode_phoxseq, decode_phoxseq, apply_phoxseq_at_time,
)


BASE = ROOT / 'outputs' / 'v40_audi_full_mipfilled.3dphox'
OUT  = ROOT / 'outputs' / 'v34_audi_halo_bloom.phoxseq'

N_FRAMES = 24
FPS = 24.0
DURATION_MS = int(round(1000 * (N_FRAMES - 1) / FPS))


def main():
    t0 = time.time()
    print(f"[1/4] Loading base {BASE.name} ...")
    sb = load_3dphox_v28_archive(BASE)
    print(f"      base N = {sb.n:,}")

    base_bytes = BASE.read_bytes()
    bcrc = compute_base_crc(base_bytes)
    print(f"      base_crc = 0x{bcrc:08x}, size = {len(base_bytes):,} bytes")

    print("[2/4] Selecting halo splats (far from origin, lower-than-median opacity) ...")
    # Heuristic: take splats whose XZ-distance from origin is large and whose
    # opacity is below median — these are the back-floor halo splats.
    # NOTE: this file stores opacity already-sigmoid-decoded in [0,1].
    xyz = sb.xyz
    opa = sb.opacities                                 # already in [0, 1]
    opa_p50 = np.median(opa)
    r_xz = np.sqrt(xyz[:, 0]**2 + xyz[:, 2]**2)
    r_p80 = np.percentile(r_xz, 80)
    halo_mask = (r_xz > r_p80) & (opa < opa_p50)
    halo_ids = np.where(halo_mask)[0].astype(np.uint32)
    # Cap at 10000 for demo
    if len(halo_ids) > 10000:
        rng = np.random.default_rng(0)
        halo_ids = np.sort(rng.choice(halo_ids, 10000, replace=False).astype(np.uint32))
    print(f"      selected {len(halo_ids):,} halo splat ids "
          f"(r > {r_p80:.3f}, opa < {opa_p50:.3f})")

    base_opa_halo = sb.opacities[halo_ids].astype(np.float32)

    print(f"[3/4] Building {N_FRAMES} frames @ {FPS} fps "
          f"(duration {DURATION_MS} ms) ...")
    frames = []
    for k in range(N_FRAMES):
        t_ms = int(round(1000 * k / FPS))
        # Sinusoidal bloom envelope: 0 → 1 → 0 over the duration
        env = np.sin(np.pi * k / (N_FRAMES - 1))     # 0..1..0
        # Boost halo opacity by env * 0.4 (clamped to [0, 1])
        new_opa = np.clip(base_opa_halo + 0.4 * env, 0.0, 1.0)
        delta_raw = encode_phoxdelta(
            bcrc, sb.n, halo_ids,
            {'opacity': new_opa.reshape(-1, 1)},
        )
        delta = decode_phoxdelta(delta_raw)
        frames.append(PhoxSeqFrame(time_offset_ms=t_ms, delta=delta))

    print(f"      built {len(frames)} frames")

    print(f"[4/4] Encoding .phoxseq with zlib payload compression ...")
    raw = encode_phoxseq(base_bytes, sb.n, frames, fps=FPS, compress_payload=True)
    OUT.write_bytes(raw)
    print(f"      wrote {OUT.name}: {len(raw):,} bytes "
          f"(= {len(raw)/N_FRAMES:.0f} bytes/frame avg)")

    # Verify
    seq = decode_phoxseq(raw)
    assert seq.frame_count == N_FRAMES
    assert seq.base_crc == bcrc and seq.base_n == sb.n
    print(f"      verify: re-parsed {seq.frame_count} frames, fps={seq.fps:.2f}, "
          f"duration={seq.duration_ms} ms")

    # Cumulative apply at each timestamp — show opacity at peak (t = duration/2)
    sb_peak = apply_phoxseq_at_time(sb, seq, DURATION_MS // 2)
    delta_at_peak = sb_peak.opacities[halo_ids] - sb.opacities[halo_ids]
    delta_at_peak = sb_peak.opacities[halo_ids] - sb.opacities[halo_ids]
    print(f"      at peak t={DURATION_MS//2} ms: {len(halo_ids):,} halo splats")
    print(f"      mean opacity boost = +{delta_at_peak.mean():.3f}, max = +{delta_at_peak.max():.3f}")

    print()
    print("=" * 70)
    print(f"DONE  in {time.time()-t0:.2f}s   ->  outputs/{OUT.name}")
    print("=" * 70)


if __name__ == '__main__':
    main()
