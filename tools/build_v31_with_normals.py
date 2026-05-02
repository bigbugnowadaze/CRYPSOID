"""Build a v31-versioned .3dphox by appending a normals chunk to a v28 archive.

Layout:
    [v28 archive bytes ............... unchanged ........]
    [v31 trailer magic: CRYPSOID31\\0  (11 bytes)        ]
    [trailer manifest length: uint64 LE                  ]
    [trailer manifest JSON (lists trailer chunks + offsets)]
    [chunk 0x12 normals (per docs/v31_graph_extension_spec.md)]

Backward compatible: v28 readers stop at end-of-v28 chunks; v31 readers see
the trailer marker and parse the additional chunks.

Usage:
    python3 tools/build_v31_with_normals.py
"""
from __future__ import annotations
import json, struct, time, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np
from crypsorender.io.normals_codec import (
    derive_normals_mls, write_normals_chunk, read_normals_chunk, NORMALS_CHUNK_ID,
)
from crypsorender.io.phox_loader import load_3dphox

V28_ARCHIVE = ROOT / 'outputs/v28_sh_vq_exact_archive_container.3dphox'
V31_OUT     = ROOT / 'outputs/v31_audi_with_normals.3dphox'

V31_MAGIC = b'CRYPSOID31\x00'


def main():
    print(f"Loading v28 archive ({V28_ARCHIVE.stat().st_size:,} bytes)...", flush=True)
    v28_bytes = V28_ARCHIVE.read_bytes()
    sb = load_3dphox(V28_ARCHIVE)
    print(f"  {sb.n} splats", flush=True)

    print(f"Deriving normals via MLS+quadric (k=24)...", flush=True)
    t = time.time()
    # Use float32 for speed; result still good enough
    normals, tangent_angles = derive_normals_mls(
        sb.xyz.astype(np.float64), k=24, refine_quadric=True
    )
    print(f"  done in {time.time()-t:.1f}s", flush=True)

    # Build the normals chunk
    normals_chunk = write_normals_chunk(normals, tangent_angles)
    print(f"  normals chunk: {len(normals_chunk):,} bytes "
          f"(= 6 header + {sb.n}*4 payload + 4 CRC = {6 + sb.n*4 + 4})", flush=True)

    # Build trailer manifest
    trailer_manifest = {
        "format": "CRYPSOID_3DPHOX_V31_TRAILER",
        "version": "v0.31",
        "base_format": "v28_sh_vq_exact_archive",
        "base_size_bytes": len(v28_bytes),
        "n_phoxoids": sb.n,
        "chunks": [
            {
                "chunk_id": NORMALS_CHUNK_ID,
                "name": "normals_oct24_tangent8",
                "offset_in_trailer": 0,   # filled below
                "size_bytes": len(normals_chunk),
                "encoding": "octahedral 24-bit normal + 8-bit tangent angle",
                "spec": "docs/v31_graph_extension_spec.md Addition 1",
            },
        ],
    }
    manifest_json = json.dumps(trailer_manifest, indent=2).encode('utf-8')

    # Compute final layout. Trailer chunk offsets are relative to start of payload section.
    # payload section starts at: trailer_magic + uint64 + manifest = 11 + 8 + len(manifest_json)
    # So normals_chunk offset = 0 (first/only chunk)
    trailer_manifest['chunks'][0]['offset_in_trailer'] = 0
    manifest_json = json.dumps(trailer_manifest, indent=2).encode('utf-8')

    # Assemble v31 file
    out = (v28_bytes
           + V31_MAGIC
           + struct.pack('<Q', len(manifest_json))
           + manifest_json
           + normals_chunk)
    V31_OUT.write_bytes(out)
    print(f"\nwrote {V31_OUT}")
    print(f"  total size: {len(out):,} bytes "
          f"({len(out)/V28_ARCHIVE.stat().st_size:.3f}x v28 archive)")
    print(f"  v28 region: {len(v28_bytes):,} bytes (verbatim)")
    print(f"  v31 trailer: {len(out) - len(v28_bytes):,} bytes "
          f"(magic + manifest + normals chunk)")

    # ---- ROUND-TRIP VERIFICATION ----
    print(f"\nRound-trip verification:", flush=True)
    raw = V31_OUT.read_bytes()
    # Find trailer marker
    pos = raw.rfind(V31_MAGIC)
    assert pos == len(v28_bytes), f"trailer at {pos}, expected {len(v28_bytes)}"
    # Parse trailer
    p = pos + len(V31_MAGIC)
    mlen = struct.unpack('<Q', raw[p:p+8])[0]; p += 8
    parsed_manifest = json.loads(raw[p:p+mlen].decode('utf-8')); p += mlen
    print(f"  trailer manifest parsed: {parsed_manifest['format']}")
    assert parsed_manifest['n_phoxoids'] == sb.n
    # Read normals chunk per the manifest
    cinfo = parsed_manifest['chunks'][0]
    chunk_bytes = raw[p + cinfo['offset_in_trailer'] : p + cinfo['offset_in_trailer'] + cinfo['size_bytes']]
    decoded_normals, decoded_tangents = read_normals_chunk(chunk_bytes)
    # Decoded should match what we derived (after lattice quantization)
    re_chunk = write_normals_chunk(decoded_normals, decoded_tangents)
    assert re_chunk == chunk_bytes, "round-trip not byte-identical"
    # And the bytes match what we just wrote
    assert chunk_bytes == normals_chunk
    print(f"  ✓ chunk bytes byte-identical to source")
    print(f"  ✓ decode(encode(decode(...))) is fixed point")
    print(f"  ✓ {sb.n} normals decoded with mean |N|={np.linalg.norm(decoded_normals, axis=1).mean():.6f}")

    # Verify the v28 region is byte-identical to the original v28 archive
    assert raw[:len(v28_bytes)] == v28_bytes
    print(f"  ✓ v28 region is byte-identical to original v28 archive (backward compatible)")

    print(f"\nv31 file ready: {V31_OUT}")


if __name__ == '__main__':
    main()
