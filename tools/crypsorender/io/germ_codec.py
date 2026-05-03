"""v40 native germ chunks: persist MLS-derived κ + cusp magnitude + (optional) 5-coef Pearcey germ.

Eliminates the load-time MLS pass (currently 3-80s depending on splat count).
Per docs/v40_native_germ_chunks_spec.md.

Chunks:
- 0x15 kappa_q8           : 1 byte/blob   (Pauly surface variation, 0-0.5 mapped to u8 0-255)
- 0x16 cusp_q8            : 1 byte/blob   (cusp_norm 0-1 mapped to u8 0-255)
- 0x17 pearcey_germ_f16   : 10 bytes/blob (κ₁, κ₂, χ, ω, ζ as f16) — optional

Each chunk header:
    1 byte version (0x01)
    1 byte reserved (0)
    4 bytes count (= N, little-endian uint32)
    N * stride bytes payload
    4 bytes CRC32

Acceptance gates:
1. Round-trip byte-identical (codecs are deterministic quantization).
2. CRC corruption detected.
3. Decoded values within q8 / f16 precision of source.
"""
from __future__ import annotations
import struct, zlib
import numpy as np

KAPPA_CHUNK_ID   = 0x15
CUSP_CHUNK_ID    = 0x16
PEARCEY_CHUNK_ID = 0x17
VERSION = 0x01

# κ range: surface-variation index ∈ [0, 1/3]; we use [0, 0.5] for noise headroom
KAPPA_RANGE = 0.5
# cusp_norm: already normalized to [0, 1] in derivation


# ---------- κ (Pauly surface-variation) ----------

def encode_kappa_q8(kappa: np.ndarray) -> bytes:
    """κ in [0, ~0.33] → u8 0..255 (mapped from [0, 0.5])."""
    q = np.clip(kappa / KAPPA_RANGE * 255.0, 0, 255).astype(np.uint8)
    return q.tobytes()


def decode_kappa_q8(payload: bytes, n: int) -> np.ndarray:
    assert len(payload) == n
    q = np.frombuffer(payload, dtype=np.uint8, count=n)
    return (q.astype(np.float32) / 255.0) * KAPPA_RANGE


def write_kappa_chunk(kappa: np.ndarray) -> bytes:
    n = len(kappa)
    payload = encode_kappa_q8(kappa)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return bytes([VERSION, 0]) + struct.pack('<I', n) + payload + struct.pack('<I', crc)


def read_kappa_chunk(chunk_bytes: bytes) -> np.ndarray:
    if len(chunk_bytes) < 10:
        raise ValueError(f"kappa chunk too short: {len(chunk_bytes)}")
    v = chunk_bytes[0]
    if v != VERSION:
        raise ValueError(f"kappa chunk version 0x{v:02x}")
    n = struct.unpack('<I', chunk_bytes[2:6])[0]
    expected = 6 + n + 4
    if len(chunk_bytes) != expected:
        raise ValueError(f"kappa chunk length {len(chunk_bytes)} != {expected}")
    payload = chunk_bytes[6:6+n]
    stored_crc = struct.unpack('<I', chunk_bytes[6+n:])[0]
    if stored_crc != (zlib.crc32(payload) & 0xFFFFFFFF):
        raise ValueError(f"kappa chunk CRC mismatch")
    return decode_kappa_q8(payload, n)


# ---------- cusp magnitude ----------

def encode_cusp_q8(cusp_norm: np.ndarray) -> bytes:
    q = np.clip(cusp_norm * 255.0, 0, 255).astype(np.uint8)
    return q.tobytes()


def decode_cusp_q8(payload: bytes, n: int) -> np.ndarray:
    assert len(payload) == n
    return np.frombuffer(payload, dtype=np.uint8, count=n).astype(np.float32) / 255.0


def write_cusp_chunk(cusp_norm: np.ndarray) -> bytes:
    n = len(cusp_norm)
    payload = encode_cusp_q8(cusp_norm)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return bytes([VERSION, 0]) + struct.pack('<I', n) + payload + struct.pack('<I', crc)


def read_cusp_chunk(chunk_bytes: bytes) -> np.ndarray:
    if len(chunk_bytes) < 10:
        raise ValueError(f"cusp chunk too short: {len(chunk_bytes)}")
    v = chunk_bytes[0]
    if v != VERSION:
        raise ValueError(f"cusp chunk version 0x{v:02x}")
    n = struct.unpack('<I', chunk_bytes[2:6])[0]
    expected = 6 + n + 4
    if len(chunk_bytes) != expected:
        raise ValueError(f"cusp chunk length {len(chunk_bytes)} != {expected}")
    payload = chunk_bytes[6:6+n]
    stored_crc = struct.unpack('<I', chunk_bytes[6+n:])[0]
    if stored_crc != (zlib.crc32(payload) & 0xFFFFFFFF):
        raise ValueError(f"cusp chunk CRC mismatch")
    return decode_cusp_q8(payload, n)


# ---------- Optional full 5-coef Pearcey germ (f16) ----------

def encode_pearcey_f16(kappa1, kappa2, chi, omega, zeta) -> bytes:
    arr = np.stack([kappa1, kappa2, chi, omega, zeta], axis=1).astype('<f2')
    return arr.tobytes()


def decode_pearcey_f16(payload: bytes, n: int):
    assert len(payload) == n * 10
    arr = np.frombuffer(payload, dtype='<f2', count=n*5).reshape(n, 5).astype(np.float32)
    return arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4]


def write_pearcey_chunk(kappa1, kappa2, chi, omega, zeta) -> bytes:
    n = len(kappa1)
    payload = encode_pearcey_f16(kappa1, kappa2, chi, omega, zeta)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return bytes([VERSION, 0]) + struct.pack('<I', n) + payload + struct.pack('<I', crc)


def read_pearcey_chunk(chunk_bytes: bytes):
    if len(chunk_bytes) < 10:
        raise ValueError(f"pearcey chunk too short: {len(chunk_bytes)}")
    v = chunk_bytes[0]
    if v != VERSION:
        raise ValueError(f"pearcey chunk version 0x{v:02x}")
    n = struct.unpack('<I', chunk_bytes[2:6])[0]
    expected = 6 + n*10 + 4
    if len(chunk_bytes) != expected:
        raise ValueError(f"pearcey chunk length {len(chunk_bytes)} != {expected}")
    payload = chunk_bytes[6:6+n*10]
    stored_crc = struct.unpack('<I', chunk_bytes[6+n*10:])[0]
    if stored_crc != (zlib.crc32(payload) & 0xFFFFFFFF):
        raise ValueError(f"pearcey chunk CRC mismatch")
    return decode_pearcey_f16(payload, n)
