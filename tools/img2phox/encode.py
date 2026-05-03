"""F.4 — BlobBundle -> .3dphox encoder.

For Phase F we ship a minimal v25-style attribute-group container directly,
without going through the full v25 build script (which expects a whole
training PLY structure). The output is renderable by the existing
crypsorender pipeline.

Container layout (compatible with phox_loader.load_3dphox_v25_render):
    magic         11 bytes  "CRYPSOID25\\0"
    manifest_len  8 bytes   uint64 little-endian
    manifest      JSON describing chunks
    chunks (zlib-compressed):
        xyz_u24_fixed
        scale_f16
        quat_i16_norm4
        dc_rgb_opacity_u8
        tier_labels_u8
        sh_rest_f32   (zeros if blobs.sh_rest is None)
"""
from __future__ import annotations
import json
import struct
import zlib
from pathlib import Path

import numpy as np

from .data_classes import BlobBundle


MAGIC = b"CRYPSOID25\x00"


def _encode_xyz_u24_fixed(xyz: np.ndarray, bounds_min, bounds_max) -> bytes:
    """Quantize XYZ to 24-bit per-axis fixed point in the given bounds."""
    mn = np.asarray(bounds_min, dtype=np.float64)
    mx = np.asarray(bounds_max, dtype=np.float64)
    span = (mx - mn)
    span = np.where(span < 1e-9, 1e-9, span)
    q = np.clip(((xyz.astype(np.float64) - mn) / span) * float((1 << 24) - 1),
                0, (1 << 24) - 1).astype(np.uint32)
    n = q.shape[0]
    out = np.zeros((n, 9), dtype=np.uint8)
    for j in range(3):
        out[:, 3*j]   = (q[:, j]      ) & 0xFF
        out[:, 3*j+1] = (q[:, j] >> 8 ) & 0xFF
        out[:, 3*j+2] = (q[:, j] >> 16) & 0xFF
    return out.tobytes()


def _encode_scale_f16(scales: np.ndarray) -> bytes:
    return scales.astype(np.float16).tobytes()


def _encode_quat_i16_norm4(quats: np.ndarray) -> bytes:
    return (quats * 32767.0).clip(-32767, 32767).astype(np.int16).tobytes()


def _encode_dc_rgb_opacity_u8(sh_dc: np.ndarray, opacity: np.ndarray) -> bytes:
    n = sh_dc.shape[0]
    arr = np.zeros((n, 4), dtype=np.uint8)
    arr[:, :3] = (sh_dc.clip(0, 1) * 255).astype(np.uint8)
    arr[:, 3]  = (opacity.clip(0, 1) * 255).astype(np.uint8)
    return arr.tobytes()


def encode_blobbundle_to_3dphox(blobs: BlobBundle, out_path: Path) -> int:
    """Write blobs to a v25-style .3dphox. Returns the byte size of the file."""
    n = len(blobs)
    if n == 0:
        raise ValueError("BlobBundle is empty")

    bounds_min = blobs.xyz.min(axis=0).tolist()
    bounds_max = blobs.xyz.max(axis=0).tolist()

    # Encode each chunk + zlib-compress
    chunks = {}

    raw_xyz   = _encode_xyz_u24_fixed(blobs.xyz, bounds_min, bounds_max)
    raw_scale = _encode_scale_f16(blobs.scales)
    raw_quat  = _encode_quat_i16_norm4(blobs.quats)
    raw_dc    = _encode_dc_rgb_opacity_u8(blobs.sh_dc, blobs.opacity)
    tier = blobs.tier if blobs.tier is not None else np.full(n, 2, dtype=np.uint8)
    raw_tier  = tier.astype(np.uint8).tobytes()
    raw_shr   = (blobs.sh_rest.astype(np.float32).tobytes()
                 if blobs.sh_rest is not None
                 else np.zeros((n, 45), dtype=np.float32).tobytes())

    # Build manifest with offsets
    manifest_chunks = []
    blob_buf = b''
    cursor = 0
    for name, raw, shape in [
        ('xyz_u24_fixed',     raw_xyz,   [n, 9]),
        ('scale_f16',         raw_scale, [n, 3]),
        ('quat_i16_norm4',    raw_quat,  [n, 4]),
        ('dc_rgb_opacity_u8', raw_dc,    [n, 4]),
        ('tier_labels_u8',    raw_tier,  [n]),
        ('sh_rest_f32',       raw_shr,   [n, 45]),
    ]:
        comp = zlib.compress(raw, level=6)
        meta = {
            'name': name, 'offset': cursor,
            'compressed_bytes': len(comp), 'uncompressed_bytes': len(raw),
            'crc32': zlib.crc32(raw) & 0xFFFFFFFF, 'shape': shape,
        }
        if name == 'xyz_u24_fixed':
            meta['bounds_min'] = bounds_min
            meta['bounds_max'] = bounds_max
        manifest_chunks.append(meta)
        blob_buf += comp
        cursor += len(comp)

    manifest = {
        'format': 'CRYPSOID v25 attribute-group (img2phox-emitted)',
        'n': n,
        'chunks': manifest_chunks,
    }
    mjson = json.dumps(manifest, separators=(',', ':')).encode('utf-8')
    out = MAGIC + struct.pack('<Q', len(mjson)) + mjson + blob_buf
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(out)
    return len(out)
