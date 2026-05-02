"""Acceptance test for v33 material_hints chunk."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np
from crypsorender.io.material_codec import (
    write_material_chunk, read_material_chunk,
    derive_material_hints, derive_view_dependence_score,
    MATERIAL_HINT_NAMES,
    MATERIAL_HINT_DIFFUSE, MATERIAL_HINT_GLOSSY, MATERIAL_HINT_MIRROR,
    MATERIAL_HINT_FLOATER, MATERIAL_HINT_EMISSIVE,
)


def test_round_trip():
    rng = np.random.default_rng(0)
    n = 5000
    hint = rng.integers(0, 7, size=n, dtype=np.uint8)
    conf = rng.integers(0, 256, size=n, dtype=np.uint8)
    vdep = rng.integers(0, 256, size=n, dtype=np.uint8)
    mip  = rng.integers(0, 256, size=n, dtype=np.uint8)
    chunk1 = write_material_chunk(hint, conf, vdep, mip)
    h2, c2, v2, m2 = read_material_chunk(chunk1)
    assert (h2 == hint).all() and (c2 == conf).all() and (v2 == vdep).all() and (m2 == mip).all()
    chunk2 = write_material_chunk(h2, c2, v2, m2)
    assert chunk1 == chunk2
    print(f"[round-trip] N={n} -> {len(chunk1):,} bytes (= 6 + N*4 + 4 = {6+n*4+4})")
    print("  PASS gate 1 (round-trip byte-identical)")


def test_crc_corruption():
    n = 100
    hint = np.zeros(n, dtype=np.uint8)
    chunk = bytearray(write_material_chunk(hint, hint, hint, hint))
    chunk[10] ^= 0x01
    try:
        read_material_chunk(bytes(chunk))
        raise AssertionError("CRC corruption not detected")
    except ValueError as e:
        assert 'CRC' in str(e)
        print(f"[CRC] corruption detected: {e}")
        print("  PASS gate 2 (CRC integrity)")


def test_synthetic_classifier():
    """Build synthetic splats with known material types; verify classifier picks them up."""
    rng = np.random.default_rng(7)
    n = 1000
    # 200 diffuse: low SH bands 1-3, high opacity
    # 200 glossy: high band-1, lower band-3, high opacity
    # 200 mirror: high band-3, high opacity
    # 200 floater: low opacity, low κ, sparse neighbor distance
    # 200 unknown: random everything
    sh_dc = rng.standard_normal((n, 3)).astype(np.float32) * 0.5
    sh_rest = np.zeros((n, 45), dtype=np.float32)
    opa = np.zeros(n, dtype=np.float32)
    kappa = np.zeros(n, dtype=np.float32)
    nd = np.zeros(n, dtype=np.float32)

    def slice_(start, end): return slice(start, end)

    # Diffuse [0:200]
    sh_rest[0:200, :] = rng.standard_normal((200, 45)).astype(np.float32) * 0.001  # tiny
    opa[0:200] = 2.0   # sigmoid > 0.88
    kappa[0:200] = 0.10
    nd[0:200] = 0.005

    # Glossy [200:400]
    sh_rest[200:400, :9] = rng.standard_normal((200, 9)).astype(np.float32) * 0.5   # band 1
    sh_rest[200:400, 24:] = rng.standard_normal((200, 21)).astype(np.float32) * 0.05  # tiny band 3
    opa[200:400] = 2.0
    kappa[200:400] = 0.10
    nd[200:400] = 0.005

    # Mirror [400:600]
    sh_rest[400:600, 24:] = rng.standard_normal((200, 21)).astype(np.float32) * 1.0  # big band 3
    opa[400:600] = 2.0
    kappa[400:600] = 0.10
    nd[400:600] = 0.005

    # Floater [600:800]
    sh_rest[600:800, :] = rng.standard_normal((200, 45)).astype(np.float32) * 0.05
    opa[600:800] = -3.0    # sigmoid ≈ 0.05 — very low opacity
    kappa[600:800] = 0.0001  # very flat
    nd[600:800] = 0.10       # sparse

    # Unknown [800:1000]
    sh_rest[800:1000, :] = rng.standard_normal((200, 45)).astype(np.float32) * 0.3
    opa[800:1000] = rng.uniform(-1, 1, size=200)
    kappa[800:1000] = rng.uniform(0.01, 0.2, size=200)
    nd[800:1000] = rng.uniform(0.005, 0.05, size=200)

    hint, conf = derive_material_hints(sh_dc, sh_rest, opa, kappa, nd)

    # Check class distribution per slice
    print(f"[synthetic classifier]")
    for label, start, end, expected in [
        ('diffuse',  0, 200,   MATERIAL_HINT_DIFFUSE),
        ('glossy',  200, 400,  MATERIAL_HINT_GLOSSY),
        ('mirror',  400, 600,  MATERIAL_HINT_MIRROR),
        ('floater', 600, 800,  MATERIAL_HINT_FLOATER),
    ]:
        seg = hint[start:end]
        match = (seg == expected).sum()
        print(f"  {label:10s} expected -> got {match}/200 correct  (most common: "
              f"{MATERIAL_HINT_NAMES[int(np.bincount(seg, minlength=7).argmax())]})")
        assert match >= 100, f"only {match}/200 {label} classified correctly"
    print("  PASS gate 3 (classifier picks up at least 50% of each known class)")


def test_view_dependence_score():
    rng = np.random.default_rng(11)
    n = 100
    sh_dc = np.full((n, 3), 0.5, dtype=np.float32)
    sh_rest_low = np.zeros((n, 45), dtype=np.float32) + 0.001
    sh_rest_high = rng.standard_normal((n, 45)).astype(np.float32) * 1.0
    score_low = derive_view_dependence_score(sh_dc, sh_rest_low)
    score_high = derive_view_dependence_score(sh_dc, sh_rest_high)
    print(f"[view-dep score] low SH-rest -> mean {score_low.mean():.1f}, "
          f"high SH-rest -> mean {score_high.mean():.1f}")
    assert score_low.mean() < 30, "low-variation SH gave high score"
    assert score_high.mean() > 100, "high-variation SH gave low score"
    print("  PASS gate 4 (view-dep score discriminates)")


def main():
    print("=" * 70)
    print("v33 material_hints chunk acceptance test")
    print("=" * 70)
    print()
    test_round_trip(); print()
    test_crc_corruption(); print()
    test_synthetic_classifier(); print()
    test_view_dependence_score(); print()
    print("=" * 70)
    print("ALL GATES PASS -- v33 material codec implementation accepted")
    print("=" * 70)


if __name__ == '__main__':
    main()
