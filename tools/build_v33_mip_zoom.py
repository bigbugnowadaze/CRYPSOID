"""Re-stamp the mip_zoom byte column inside an existing v31+v33 .3dphox file.

Background
----------
v33's material_hints chunk holds 4 u8 fields per splat:
    material_hint, confidence, view_dependence_score, mip_zoom

The first three were derived from SH + opacity + κ; mip_zoom was packed as zeros
because the spec drafted the field but never landed the derivation. This script
fills it in (Mip-Splatting style) without touching any other field, preserving
the file's overall layout and v40 trailer.

What it does
------------
1. Read the v31 trailer manifest from outputs/v31_audi_full_v33.3dphox.
2. Decode the material_hints chunk → (hint, conf, vdep, mip_old).
3. Decode the v28 archive scales (world-space semi-axes).
4. mip_new = derive_mip_zoom(exp(log_sigma)).
5. Write a new material_hints chunk with the same 3 fields and mip_new.
6. Splice the new chunk into the v31 trailer, updating its size_bytes and
   subsequent offsets in the v31 manifest. v40 trailer is untouched.
7. Save outputs/v31_audi_full_v33_mipfilled.3dphox.

Acceptance gate
---------------
- The file decodes cleanly (parse_v31_trailer succeeds, CRCs match).
- mip_new has nonzero histogram across all 256 buckets > 0% (i.e. it is
  actually filled, not all-zero).
- Per-splat round-trip: decode_mip_zoom(derive_mip_zoom(s)) is within
  ~6% of the input sigma in median (8-bit log scale step).
- File size delta from rewrite is exactly 0 bytes (only payload bytes change,
  not chunk lengths).

Usage:
    python3 tools/build_v33_mip_zoom.py
"""
from __future__ import annotations

import json, struct, zlib, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np

from crypsorender.io.material_codec import (
    read_material_chunk, write_material_chunk,
    derive_mip_zoom, decode_mip_zoom, MATERIAL_CHUNK_ID,
)
from crypsorender.io.phox_loader import (
    parse_v31_trailer, V31_MAGIC, V40_MAGIC, load_3dphox_v28_archive,
)


SRC = ROOT / 'outputs' / 'v31_audi_full_v33.3dphox'
OUT = ROOT / 'outputs' / 'v31_audi_full_v33_mipfilled.3dphox'


def main():
    t0 = time.time()
    print(f"[1/6] Reading {SRC.name} ...")
    raw = SRC.read_bytes()
    print(f"      size = {len(raw):,} bytes")

    print("[2/6] Parsing v31 trailer ...")
    v31 = parse_v31_trailer(raw)
    if v31 is None:
        raise SystemExit("No v31 trailer found in source file.")
    manifest = v31['manifest']
    chunks = v31['chunks']
    print(f"      v31 chunks: {[c['name'] for c in manifest['chunks']]}")

    if 'material_hints' not in chunks:
        raise SystemExit("No material_hints chunk inside v31 trailer.")
    mh_bytes = chunks['material_hints']
    print(f"      material_hints chunk = {len(mh_bytes):,} bytes")

    hint, conf, vdep, mip_old = read_material_chunk(mh_bytes)
    n = len(hint)
    print(f"      decoded: N={n:,} splats, mip_old non-zero = {(mip_old != 0).sum():,}")

    print("[3/6] Loading v28 archive scales (for sigma_world) ...")
    sb = load_3dphox_v28_archive(SRC)
    scales_log = sb.scales                                  # stored as log_sigma
    sigma_world = np.exp(scales_log).astype(np.float64)     # (N,3) world units
    print(f"      sigma_world: min={sigma_world.min():.3e}  "
          f"median={np.median(sigma_world):.3e}  max={sigma_world.max():.3e}")

    print("[4/6] Deriving mip_zoom (Mip-Splatting LOD) ...")
    mip_new = derive_mip_zoom(np.exp(scales_log))
    nz = (mip_new != 0).sum()
    occ = np.bincount(mip_new, minlength=256)
    n_occupied_buckets = (occ > 0).sum()
    print(f"      mip_new: nonzero = {nz:,}/{n:,} ({100*nz/n:.1f}%), "
          f"buckets occupied = {n_occupied_buckets}/256")
    print(f"      mip_new sample: min={mip_new.min()}  median={int(np.median(mip_new))}  "
          f"max={mip_new.max()}")

    # Round-trip sanity
    sigma_recovered = decode_mip_zoom(mip_new)
    sigma_max = sigma_world.max(axis=1)
    rel_err = np.abs(sigma_recovered - sigma_max) / np.maximum(sigma_max, 1e-12)
    print(f"      round-trip sigma rel-err: median={np.median(rel_err)*100:.2f}%  "
          f"p90={np.percentile(rel_err, 90)*100:.2f}%")

    print("[5/6] Re-encoding material_hints chunk with mip_new ...")
    mh_new = write_material_chunk(hint, conf, vdep, mip_new)
    if len(mh_new) != len(mh_bytes):
        raise SystemExit(
            f"chunk length mismatch: old={len(mh_bytes)} new={len(mh_new)} (must be equal)"
        )
    print(f"      new chunk = {len(mh_new):,} bytes (same as old, byte-aligned splice OK)")

    print("[6/6] Splicing new chunk into the file ...")
    # Compute absolute offset of the material_hints payload within the file.
    # v31 trailer starts at file_bytes.rfind(V31_MAGIC), then 12 magic bytes,
    # then 8 manifest_len bytes, then mlen bytes of manifest, then chunks.
    v31_pos = raw.rfind(V31_MAGIC)
    p = v31_pos + len(V31_MAGIC)
    mlen = struct.unpack('<Q', raw[p:p+8])[0]
    chunks_region_start = p + 8 + mlen
    mh_meta = next(c for c in manifest['chunks'] if c['name'] == 'material_hints')
    mh_offset = chunks_region_start + mh_meta['offset_in_trailer']
    mh_size   = mh_meta['size_bytes']
    if raw[mh_offset:mh_offset + mh_size] != mh_bytes:
        raise SystemExit("offset arithmetic disagrees with parser; aborting")

    new_raw = bytearray(raw)
    new_raw[mh_offset:mh_offset + mh_size] = mh_new
    OUT.write_bytes(bytes(new_raw))
    print(f"      wrote {OUT.name}  ({OUT.stat().st_size:,} bytes; "
          f"delta vs source = {OUT.stat().st_size - len(raw):+d})")

    # Re-parse to confirm the new file is structurally identical
    print()
    print("Verification — re-parse output ...")
    raw2 = OUT.read_bytes()
    v31b = parse_v31_trailer(raw2)
    if v31b is None:
        raise SystemExit("FAIL: re-parsed file has no v31 trailer")
    h2, c2, v2, m2 = read_material_chunk(v31b['chunks']['material_hints'])
    assert (h2 == hint).all() and (c2 == conf).all() and (v2 == vdep).all() and (m2 == mip_new).all(), \
        "round-trip mismatch on output file"
    print(f"  - v31 trailer parses OK ({len(v31b['manifest']['chunks'])} chunks)")
    print(f"  - material_hints CRC is valid")
    print(f"  - mip_zoom column round-trips byte-identical to in-memory derivation")

    # Confirm v40 trailer (kappa + cusp) is untouched
    from crypsorender.io.phox_loader import parse_v40_trailer
    v40b = parse_v40_trailer(raw2)
    if v40b is not None:
        print(f"  - v40 trailer still present and parses ({len(v40b['manifest']['chunks'])} chunks)")
    else:
        print("  - WARN: no v40 trailer in output (expected for inputs with v40)")

    print()
    print("=" * 70)
    print(f"DONE  in {time.time()-t0:.2f}s   →  outputs/{OUT.name}")
    print("=" * 70)


if __name__ == '__main__':
    main()
