"""v40 germ codecs acceptance test."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
import numpy as np
from crypsorender.io.germ_codec import (
    write_kappa_chunk, read_kappa_chunk,
    write_cusp_chunk, read_cusp_chunk,
    write_pearcey_chunk, read_pearcey_chunk,
)

def test_kappa_round_trip():
    rng = np.random.default_rng(0)
    n = 5000
    kappa = rng.uniform(0, 0.33, size=n).astype(np.float32)
    chunk = write_kappa_chunk(kappa)
    decoded = read_kappa_chunk(chunk)
    chunk2 = write_kappa_chunk(decoded)
    assert chunk == chunk2, "κ chunk not byte-identical re-encode"
    err = np.abs(decoded - kappa)
    print(f"[κ q8] N={n} -> {len(chunk):,} bytes  max err {err.max()*1000:.3f} mrad  mean {err.mean()*1000:.3f}")
    assert err.max() < 0.003, f"κ decode error {err.max()} > 0.003 spec"
    print("  PASS gate 1+2 (round-trip + precision)")

def test_cusp_round_trip():
    rng = np.random.default_rng(1)
    n = 5000
    cusp = rng.uniform(0, 1, size=n).astype(np.float32)
    chunk = write_cusp_chunk(cusp)
    decoded = read_cusp_chunk(chunk)
    chunk2 = write_cusp_chunk(decoded)
    assert chunk == chunk2
    err = np.abs(decoded - cusp)
    print(f"[cusp q8] N={n} -> {len(chunk):,} bytes  max err {err.max():.4f}  mean {err.mean():.4f}")
    assert err.max() < 0.005
    print("  PASS gate 3+4 (cusp round-trip + precision)")

def test_pearcey_round_trip():
    rng = np.random.default_rng(7)
    n = 1000
    k1 = rng.standard_normal(n).astype(np.float32) * 0.1
    k2 = rng.standard_normal(n).astype(np.float32) * 0.1
    chi = rng.standard_normal(n).astype(np.float32) * 0.05
    omega = rng.standard_normal(n).astype(np.float32) * 0.05
    zeta = rng.standard_normal(n).astype(np.float32) * 0.02
    chunk = write_pearcey_chunk(k1, k2, chi, omega, zeta)
    d1, d2, dc, do, dz = read_pearcey_chunk(chunk)
    chunk2 = write_pearcey_chunk(d1, d2, dc, do, dz)
    assert chunk == chunk2
    err = max(np.abs(d1-k1).max(), np.abs(d2-k2).max(), np.abs(dc-chi).max(),
              np.abs(do-omega).max(), np.abs(dz-zeta).max())
    print(f"[pearcey f16] N={n} -> {len(chunk):,} bytes (10 B/blob); max coef err {err:.5f}")
    assert err < 1e-3, f"f16 precision exceeded: {err}"
    print("  PASS gate 5+6 (pearcey round-trip + f16 precision)")

def test_crc_corruption():
    rng = np.random.default_rng(2)
    cusp = rng.uniform(0, 1, size=100).astype(np.float32)
    chunk = bytearray(write_cusp_chunk(cusp))
    chunk[10] ^= 0x01
    try:
        read_cusp_chunk(bytes(chunk))
        raise AssertionError("CRC corruption not detected")
    except ValueError as e:
        assert 'CRC' in str(e)
        print(f"[CRC] detected: {e}")
        print("  PASS gate 7 (CRC integrity)")

def main():
    print("=" * 70)
    print("v40 germ codecs acceptance test")
    print("=" * 70)
    print()
    test_kappa_round_trip(); print()
    test_cusp_round_trip(); print()
    test_pearcey_round_trip(); print()
    test_crc_corruption(); print()
    print("=" * 70)
    print("ALL GATES PASS — v40 germ codecs accepted")
    print("=" * 70)

if __name__ == '__main__':
    main()
