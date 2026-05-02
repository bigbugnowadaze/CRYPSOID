"""Acceptance test for v31 Addition 2 -- kNN edges chunk."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np
from crypsorender.io.edges_codec import (
    write_edges_chunk, read_edges_chunk,
    derive_knn_edges, validate_edges,
    EDGES_CHUNK_ID, EDGES_CHUNK_VERSION,
)


def test_round_trip_byte_identical():
    rng = np.random.default_rng(0)
    n, k = 5000, 4
    neighbors = rng.integers(0, n, size=(n, k), dtype=np.uint32)
    chunk1 = write_edges_chunk(neighbors)
    decoded = read_edges_chunk(chunk1)
    chunk2 = write_edges_chunk(decoded)
    assert chunk1 == chunk2, "edges chunk round-trip not byte-identical"
    assert (decoded == neighbors).all(), "decoded values differ from input"
    print(f"[edges round-trip] {n} phoxoids, k={k} -> {len(chunk1):,} bytes")
    print(f"  byte-identical re-encode: OK")
    print(f"  decoded values match input exactly: OK")
    print("  PASS gate 1 (round-trip)")


def test_crc_corruption_detected():
    rng = np.random.default_rng(1)
    neighbors = rng.integers(0, 100, size=(50, 4), dtype=np.uint32)
    chunk = bytearray(write_edges_chunk(neighbors))
    chunk[10] ^= 0x01
    try:
        read_edges_chunk(bytes(chunk))
        raise AssertionError("CRC corruption not detected")
    except ValueError as e:
        assert 'CRC' in str(e)
        print(f"[CRC] corruption detected: {e}")
        print("  PASS CRC integrity gate")


def test_version_mismatch():
    rng = np.random.default_rng(2)
    neighbors = rng.integers(0, 100, size=(10, 4), dtype=np.uint32)
    chunk = bytearray(write_edges_chunk(neighbors))
    chunk[0] = 0xFF
    try:
        read_edges_chunk(bytes(chunk))
        raise AssertionError("version mismatch not detected")
    except ValueError as e:
        assert 'version' in str(e)
        print(f"[version] mismatch detected: {e}")
        print("  PASS version check")


def test_kNN_derivation():
    """Derive kNN from a point cloud, verify gates."""
    rng = np.random.default_rng(7)
    n = 5000
    # Random points in a unit cube
    xyz = rng.uniform(-1.0, 1.0, size=(n, 3))
    k = 4
    neighbors = derive_knn_edges(xyz, k=k)
    print(f"[kNN derivation] n={n} k={k} -> shape {neighbors.shape}, dtype {neighbors.dtype}")
    stats = validate_edges(neighbors, xyz)
    print(f"  validation: {stats}")
    assert stats['n_self_edges'] == 0, f"{stats['n_self_edges']} self-edges found"
    assert stats['sorted_by_distance'], "neighbors not sorted by distance"
    assert stats['all_indices_in_range'], "indices out of range"
    print("  PASS gate 3 (no self-edges)")
    print("  PASS gate 4 (sorted by distance)")
    print("  PASS index-range check")


def test_chunk_round_trip_on_real_kNN():
    """Full pipeline: derive -> write -> read -> validate."""
    rng = np.random.default_rng(11)
    n = 1000
    xyz = rng.standard_normal((n, 3))
    neighbors = derive_knn_edges(xyz, k=4)
    chunk = write_edges_chunk(neighbors)
    decoded = read_edges_chunk(chunk)
    assert (decoded == neighbors).all(), "decoded != original"
    expected_size = 6 + n * 4 * 4 + 4
    assert len(chunk) == expected_size, f"chunk size {len(chunk)} != {expected_size}"
    print(f"[real kNN round-trip] n={n} k=4 chunk={len(chunk):,} bytes (= 6 + N*k*4 + 4 = {expected_size})")
    print(f"  derived -> encoded -> decoded matches original")
    print("  PASS end-to-end pipeline")


def main():
    print("=" * 70)
    print("v31 Addition 2 -- kNN edges chunk codec acceptance test")
    print("=" * 70)
    print()
    test_round_trip_byte_identical(); print()
    test_crc_corruption_detected(); print()
    test_version_mismatch(); print()
    test_kNN_derivation(); print()
    test_chunk_round_trip_on_real_kNN(); print()
    print("=" * 70)
    print("ALL GATES PASS -- v31 edges chunk implementation accepted")
    print("=" * 70)


if __name__ == '__main__':
    main()
