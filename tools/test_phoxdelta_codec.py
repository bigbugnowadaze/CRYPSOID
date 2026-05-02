"""Acceptance test for v31 Addition 3 -- .phoxdelta patch format."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np
from crypsorender.io.phoxdelta_codec import (
    encode_phoxdelta, decode_phoxdelta,
    apply_phoxdelta, compose_phoxdeltas, base_crc,
    PhoxDelta, MAGIC, VERSION,
)


def test_round_trip_single_attr():
    """Encode a delta with just opacity changes; decode; verify equal."""
    rng = np.random.default_rng(0)
    M = 100
    base_n = 10000
    ids = rng.choice(base_n, size=M, replace=False).astype(np.uint32)
    new_opacity = rng.uniform(-3, 3, size=M).astype(np.float32)
    bcrc = 0xCAFEBABE

    bytes_ = encode_phoxdelta(bcrc, base_n, ids, {'opacity': new_opacity})
    pd = decode_phoxdelta(bytes_)
    assert pd.base_crc == bcrc
    assert pd.base_n == base_n
    assert pd.delta_count == M
    assert (pd.phoxoid_ids == ids).all()
    # Single attr → all records should have bit 3 set
    expected_mask = 1 << 3
    assert (pd.dirty_mask == expected_mask).all()
    # Decoded opacity matches
    decoded_opacity = pd.attrs['opacity'].squeeze()
    assert np.allclose(decoded_opacity, new_opacity)
    print(f"[round-trip opacity] M={M} base_n={base_n} -> {len(bytes_)} bytes")
    print(f"  id, mask, opacity all match exactly")
    print("  PASS")


def test_round_trip_multi_attr():
    """Multiple attrs: xyz + tier + opacity."""
    rng = np.random.default_rng(1)
    M = 50
    base_n = 5000
    ids = rng.choice(base_n, size=M, replace=False).astype(np.uint32)
    attrs = {
        'xyz':     rng.uniform(-1, 1, size=(M, 3)).astype(np.float32),
        'tier':    rng.integers(0, 3, size=M).astype(np.uint8),
        'opacity': rng.uniform(-2, 2, size=M).astype(np.float32),
    }
    bcrc = 0xDEADBEEF
    bytes_ = encode_phoxdelta(bcrc, base_n, ids, attrs)
    pd = decode_phoxdelta(bytes_)
    assert (pd.phoxoid_ids == ids).all()
    expected = (1 << 0) | (1 << 6) | (1 << 3)   # xyz + tier + opacity
    assert (pd.dirty_mask == expected).all()
    assert np.allclose(pd.attrs['xyz'], attrs['xyz'])
    assert (pd.attrs['tier'].squeeze() == attrs['tier']).all()
    assert np.allclose(pd.attrs['opacity'].squeeze(), attrs['opacity'])
    print(f"[round-trip multi-attr] M={M}, attrs=(xyz, tier, opacity) -> {len(bytes_)} bytes")
    print("  PASS")


def test_apply_to_splat_buffer():
    """Build a fake splat buffer, build a delta, apply it, verify in-place."""
    from crypsorender.io.splat_buffer import SplatBuffer
    rng = np.random.default_rng(7)
    n = 1000
    sb = SplatBuffer(
        n=n,
        xyz=rng.standard_normal((n, 3)).astype(np.float32),
        scales=rng.uniform(-3, 0, size=(n, 3)).astype(np.float32),
        quats=rng.standard_normal((n, 4)).astype(np.float32),
        opacities=rng.uniform(-2, 2, size=n).astype(np.float32),
        sh_dc=rng.standard_normal((n, 3)).astype(np.float32),
        scene_format='test',
    )

    # Modify 5 phoxoids: set opacity to -10 (sigmoid → near 0)
    M = 5
    ids = np.array([3, 17, 99, 256, 800], dtype=np.uint32)
    new_opacity = np.full(M, -10.0, dtype=np.float32)
    delta = decode_phoxdelta(encode_phoxdelta(0, n, ids, {'opacity': new_opacity}))

    sb_after = apply_phoxdelta(sb, delta, copy=True)
    # Targeted ids changed
    assert (sb_after.opacities[ids] == -10.0).all()
    # Other ids unchanged
    other = np.setdiff1d(np.arange(n), ids)
    assert np.allclose(sb_after.opacities[other], sb.opacities[other])
    # Source unchanged (copy=True)
    assert not (sb.opacities[ids] == -10.0).all()
    print(f"[apply] n={n}, modified {M} phoxoids' opacity")
    print(f"  targeted ids changed; others untouched; source unchanged (copy)")
    print("  PASS")


def test_compose_two_deltas():
    """Two deltas; later wins per (id, attr)."""
    base_n = 100
    bcrc = 0x12345678
    # Delta A: change opacity on ids [10, 20]
    a_bytes = encode_phoxdelta(bcrc, base_n,
                               np.array([10, 20], dtype=np.uint32),
                               {'opacity': np.array([0.5, 0.5], dtype=np.float32)})
    a = decode_phoxdelta(a_bytes)
    # Delta B: change opacity on id [20] (overrides A) + tier on id [30] (new)
    b_bytes = encode_phoxdelta(bcrc, base_n,
                               np.array([20, 30], dtype=np.uint32),
                               {'opacity': np.array([0.9, 0.1], dtype=np.float32)})
    b = decode_phoxdelta(b_bytes)

    composed = compose_phoxdeltas([a, b])
    # Should have ids {10, 20, 30}
    assert sorted(composed.phoxoid_ids.tolist()) == [10, 20, 30]
    # id 10 -> opacity 0.5 (only in A)
    # id 20 -> opacity 0.9 (B wins)
    # id 30 -> opacity 0.1 (only in B)
    record_idxs = composed.attr_record_idx['opacity']
    ids_with_opacity = composed.phoxoid_ids[record_idxs]
    opacity_vals = composed.attrs['opacity'].squeeze()
    lookup = dict(zip(ids_with_opacity.tolist(), opacity_vals.tolist()))
    assert np.isclose(lookup[10], 0.5)
    assert np.isclose(lookup[20], 0.9), f"expected ~0.9 (B wins), got {lookup[20]}"
    assert np.isclose(lookup[30], 0.1)
    print(f"[compose] 2 deltas merged; later wins per (id, attr)")
    print(f"  id 10 -> {lookup[10]} (A only)")
    print(f"  id 20 -> {lookup[20]} (B overrides A)")
    print(f"  id 30 -> {lookup[30]} (B only, new)")
    print("  PASS")


def test_apply_then_re_encode_round_trip():
    """Build fake base; apply delta; re-build delta from diff; verify equivalent."""
    from crypsorender.io.splat_buffer import SplatBuffer
    rng = np.random.default_rng(11)
    n = 500
    sb_base = SplatBuffer(
        n=n,
        xyz=rng.standard_normal((n, 3)).astype(np.float32),
        scales=rng.uniform(-3, 0, size=(n, 3)).astype(np.float32),
        quats=rng.standard_normal((n, 4)).astype(np.float32),
        opacities=rng.uniform(-2, 2, size=n).astype(np.float32),
        sh_dc=rng.standard_normal((n, 3)).astype(np.float32),
        scene_format='test',
    )
    ids = np.array([5, 17, 100, 200], dtype=np.uint32)
    new_opa = np.array([1.5, -2.0, 0.0, 3.3], dtype=np.float32)
    delta1 = decode_phoxdelta(encode_phoxdelta(0xABCD, n, ids, {'opacity': new_opa}))
    sb_modified = apply_phoxdelta(sb_base, delta1, copy=True)

    # Now re-derive a delta from the diff
    diff_mask = sb_modified.opacities != sb_base.opacities
    diff_ids = np.where(diff_mask)[0].astype(np.uint32)
    diff_vals = sb_modified.opacities[diff_ids]
    delta2 = decode_phoxdelta(encode_phoxdelta(0xABCD, n, diff_ids, {'opacity': diff_vals}))

    # Apply delta2 to sb_base; should match sb_modified exactly
    sb_check = apply_phoxdelta(sb_base, delta2, copy=True)
    assert np.allclose(sb_check.opacities, sb_modified.opacities)
    print(f"[apply+derive+reapply] {len(diff_ids)} diffs round-trip exactly")
    print("  PASS")


def main():
    print("=" * 70)
    print("v31 Addition 3 -- .phoxdelta patch format acceptance test")
    print("=" * 70)
    print()
    test_round_trip_single_attr(); print()
    test_round_trip_multi_attr(); print()
    test_apply_to_splat_buffer(); print()
    test_compose_two_deltas(); print()
    test_apply_then_re_encode_round_trip(); print()
    print("=" * 70)
    print("ALL GATES PASS -- .phoxdelta codec implementation accepted")
    print("=" * 70)


if __name__ == '__main__':
    main()
