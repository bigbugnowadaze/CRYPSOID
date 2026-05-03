"""Acceptance test for v34 .phoxseq temporal sequence codec."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np

from crypsorender.io.phoxdelta_codec import (
    PhoxDelta, encode_phoxdelta, decode_phoxdelta, base_crc as compute_base_crc,
)
from crypsorender.io.phoxseq_codec import (
    PhoxSeq, PhoxSeqFrame, encode_phoxseq, decode_phoxseq,
    SEQ_MAGIC, HEADER_SIZE, FRAME_INDEX_SIZE,
)
from crypsorender.io.splat_buffer import SplatBuffer


# ---------- Helpers ----------

def _make_base(n: int = 1000):
    """Synthetic base SplatBuffer + base file bytes for CRC."""
    rng = np.random.default_rng(42)
    sb = SplatBuffer(
        n=n,
        xyz=rng.standard_normal((n, 3)).astype(np.float32),
        scales=rng.uniform(0.001, 0.05, size=(n, 3)).astype(np.float32),
        quats=rng.standard_normal((n, 4)).astype(np.float32),
        opacities=np.full(n, 0.7, dtype=np.float32),
        sh_dc=rng.standard_normal((n, 3)).astype(np.float32) * 0.5,
        sh_rest=rng.standard_normal((n, 45)).astype(np.float32) * 0.1,
        tier=np.zeros(n, dtype=np.uint8),
    )
    base_bytes = b"FAKE_BASE_FILE_FOR_CRC\x00" * 100
    return sb, base_bytes


def _make_delta(base_crc: int, base_n: int, ids, vals_xyz=None, vals_opa=None):
    attrs = {}
    if vals_xyz is not None:
        attrs['xyz'] = np.asarray(vals_xyz, dtype=np.float32)
    if vals_opa is not None:
        attrs['opacity'] = np.asarray(vals_opa, dtype=np.float32).reshape(-1, 1)
    raw = encode_phoxdelta(base_crc, base_n,
                           np.asarray(ids, dtype=np.uint32), attrs)
    return decode_phoxdelta(raw)


# ---------- Gates ----------

def test_round_trip_byte_identical():
    sb, base_bytes = _make_base(500)
    bcrc = compute_base_crc(base_bytes)
    f0 = _make_delta(bcrc, sb.n, [0, 1, 2], vals_opa=[0.1, 0.2, 0.3])
    f1 = _make_delta(bcrc, sb.n, [10, 20], vals_opa=[0.5, 0.5])
    f2 = _make_delta(bcrc, sb.n, [100], vals_xyz=[[1.0, 2.0, 3.0]])
    frames = [
        PhoxSeqFrame(time_offset_ms=0,    delta=f0),
        PhoxSeqFrame(time_offset_ms=42,   delta=f1),
        PhoxSeqFrame(time_offset_ms=1000, delta=f2),
    ]
    raw = encode_phoxseq(base_bytes, sb.n, frames, fps=24.0, compress_payload=True)
    seq = decode_phoxseq(raw)
    raw2 = encode_phoxseq(base_bytes, sb.n,
                          [PhoxSeqFrame(time_offset_ms=f.time_offset_ms, delta=f.delta) for f in seq.frames],
                          fps=seq.fps, compress_payload=True)
    assert raw == raw2, "phoxseq round-trip not byte-identical"
    print(f"[round-trip] N={sb.n}, frames={len(frames)} -> {len(raw):,} bytes")
    print("  PASS gate 1 (round-trip byte-identical)")


def test_frame_index_offsets():
    sb, base_bytes = _make_base(200)
    bcrc = compute_base_crc(base_bytes)
    fs = [PhoxSeqFrame(time_offset_ms=k*10,
                       delta=_make_delta(bcrc, sb.n, [k, k+1], vals_opa=[k*0.01, k*0.02]))
          for k in range(8)]
    raw = encode_phoxseq(base_bytes, sb.n, fs, fps=30.0)
    seq = decode_phoxseq(raw)
    assert seq.frame_count == 8
    # Re-parse the raw frame index manually
    import struct
    p = HEADER_SIZE
    last_offset = HEADER_SIZE + FRAME_INDEX_SIZE * 8
    for i in range(8):
        offset = struct.unpack('<I', raw[p+8:p+12])[0]
        size   = struct.unpack('<I', raw[p+12:p+16])[0]
        assert offset >= last_offset, f"frame {i} offset {offset} < {last_offset}"
        assert offset + size <= len(raw), f"frame {i} extends past file end"
        last_offset = offset + size
        p += FRAME_INDEX_SIZE
    assert last_offset == len(raw), f"file ends at {len(raw)} but last frame end is {last_offset}"
    print(f"[frame index] {seq.frame_count} frames span [{HEADER_SIZE+FRAME_INDEX_SIZE*8}, {len(raw)})")
    print("  PASS gate 2 (frame index offsets are non-overlapping and exhaust file)")


def test_timeline_monotone():
    sb, base_bytes = _make_base(100)
    bcrc = compute_base_crc(base_bytes)
    fs = [PhoxSeqFrame(time_offset_ms=t,
                       delta=_make_delta(bcrc, sb.n, [0], vals_opa=[0.5]))
          for t in [0, 100, 100, 200, 999]]   # equal allowed, decreasing not
    encode_phoxseq(base_bytes, sb.n, fs, fps=24.0)  # should not raise
    bad = fs + [PhoxSeqFrame(time_offset_ms=500,
                             delta=_make_delta(bcrc, sb.n, [0], vals_opa=[0.5]))]
    try:
        encode_phoxseq(base_bytes, sb.n, bad, fps=24.0)
        raise AssertionError("expected non-monotone timeline to be rejected")
    except ValueError as e:
        assert 'monotone' in str(e)
    print("  PASS gate 3 (timeline monotone non-decreasing enforced)")


def test_apply_one_frame():
    sb, base_bytes = _make_base(300)
    bcrc = compute_base_crc(base_bytes)
    delta = _make_delta(bcrc, sb.n, [5, 10, 15], vals_opa=[0.0, 0.0, 0.0])
    seq = decode_phoxseq(encode_phoxseq(
        base_bytes, sb.n, [PhoxSeqFrame(time_offset_ms=0, delta=delta)], fps=24.0
    ))
    from crypsorender.io.phoxseq_codec import apply_phoxseq_frame, apply_phoxseq_at_time
    sb1 = apply_phoxseq_frame(sb, seq, 0)
    assert (sb1.opacities[[5, 10, 15]] == 0).all(), "phoxseq frame application didn't take"
    assert sb1.opacities[0] == sb.opacities[0], "non-targeted phoxoid was modified"
    sb2 = apply_phoxseq_at_time(sb, seq, 0)
    assert (sb2.opacities[[5, 10, 15]] == 0).all()
    print(f"[apply] frame applied: 3 phoxoids opacity = {sb1.opacities[[5,10,15]].tolist()}")
    print("  PASS gate 4 (apply produces expected SplatBuffer)")


def test_compose_equivalence():
    """Applying [f0..f5] cumulatively should match composing then applying."""
    sb, base_bytes = _make_base(400)
    bcrc = compute_base_crc(base_bytes)
    rng = np.random.default_rng(0)
    fs = []
    for k in range(6):
        ids = rng.choice(sb.n, 30, replace=False)
        opa = rng.uniform(0.0, 1.0, size=30).astype(np.float32)
        d = _make_delta(bcrc, sb.n, ids, vals_opa=opa)
        fs.append(PhoxSeqFrame(time_offset_ms=k*40, delta=d))
    seq = decode_phoxseq(encode_phoxseq(base_bytes, sb.n, fs, fps=25.0))
    from crypsorender.io.phoxdelta_codec import compose_phoxdeltas, apply_phoxdelta
    from crypsorender.io.phoxseq_codec import apply_phoxseq_at_time
    # Cumulative via compose
    composed = compose_phoxdeltas([f.delta for f in seq.frames])
    sb_compose = apply_phoxdelta(sb, composed, copy=True)
    # Cumulative via apply_phoxseq_at_time after last time
    sb_seq = apply_phoxseq_at_time(sb, seq, seq.time_end_ms)
    assert np.allclose(sb_compose.opacities, sb_seq.opacities), "compose != cumulative apply"
    print(f"  PASS gate 5 (compose and cumulative apply agree on opacities)")


def test_compression_actually_helps():
    """Compressed payload should be smaller for a structured frame."""
    sb, base_bytes = _make_base(2000)
    bcrc = compute_base_crc(base_bytes)
    ids = np.arange(800, dtype=np.uint32)
    # All-zero opacity attribute -> highly compressible
    opa = np.zeros((800, 1), dtype=np.float32)
    raw_d = encode_phoxdelta(bcrc, sb.n, ids, {'opacity': opa})
    delta = decode_phoxdelta(raw_d)
    f = PhoxSeqFrame(time_offset_ms=0, delta=delta)
    raw_uncompressed = encode_phoxseq(base_bytes, sb.n, [f], compress_payload=False)
    raw_compressed   = encode_phoxseq(base_bytes, sb.n, [f], compress_payload=True)
    print(f"[compression] uncompressed {len(raw_uncompressed):,} B  "
          f"compressed {len(raw_compressed):,} B  "
          f"ratio {len(raw_uncompressed)/max(len(raw_compressed),1):.2f}x")
    assert len(raw_compressed) < len(raw_uncompressed), "compression made things bigger"
    print("  PASS gate 6 (compression reduces size for structured payloads)")


def main():
    print("=" * 70)
    print("v34 .phoxseq sequence codec acceptance test")
    print("=" * 70)
    print()
    test_round_trip_byte_identical(); print()
    test_frame_index_offsets(); print()
    test_timeline_monotone(); print()
    test_apply_one_frame(); print()
    test_compose_equivalence(); print()
    test_compression_actually_helps(); print()
    print("=" * 70)
    print("ALL GATES PASS -- v34 .phoxseq codec implementation accepted")
    print("=" * 70)


if __name__ == '__main__':
    main()
