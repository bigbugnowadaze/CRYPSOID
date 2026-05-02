"""v31 Addition 2 -- kNN edges chunk codec.
chunk_id 0x13. 16 B/blob (k=4 u32). Per docs/v31_graph_extension_spec.md.
"""
from __future__ import annotations
import struct, zlib
import numpy as np

EDGES_CHUNK_ID = 0x13
EDGES_CHUNK_VERSION = 0x01
DEFAULT_K = 4


def encode_edges_payload(neighbors: np.ndarray) -> bytes:
    assert neighbors.dtype == np.uint32
    assert neighbors.ndim == 2
    return neighbors.astype('<u4').tobytes()


def decode_edges_payload(payload: bytes, n: int, k: int) -> np.ndarray:
    assert len(payload) == n * k * 4
    return np.frombuffer(payload, dtype='<u4', count=n*k).reshape(n, k).astype(np.uint32)


def write_edges_chunk(neighbors: np.ndarray) -> bytes:
    n, k = neighbors.shape
    assert 1 <= k <= 255
    payload = encode_edges_payload(neighbors)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return (bytes([EDGES_CHUNK_VERSION, k & 0xFF])
            + struct.pack('<I', n) + payload + struct.pack('<I', crc))


def read_edges_chunk(chunk_bytes: bytes) -> np.ndarray:
    if len(chunk_bytes) < 10:
        raise ValueError(f"edges chunk too short: {len(chunk_bytes)}")
    version = chunk_bytes[0]
    if version != EDGES_CHUNK_VERSION:
        raise ValueError(f"unsupported edges chunk version 0x{version:02x}")
    k = chunk_bytes[1]
    if k < 1 or k > 255:
        raise ValueError(f"invalid k={k}")
    n = struct.unpack('<I', chunk_bytes[2:6])[0]
    expected_len = 6 + n*k*4 + 4
    if len(chunk_bytes) != expected_len:
        raise ValueError(f"edges chunk length mismatch: {len(chunk_bytes)} != {expected_len}")
    payload = chunk_bytes[6:6 + n*k*4]
    stored_crc = struct.unpack('<I', chunk_bytes[6 + n*k*4:])[0]
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    if stored_crc != actual_crc:
        raise ValueError(f"edges chunk CRC mismatch: stored 0x{stored_crc:08x}, computed 0x{actual_crc:08x}")
    return decode_edges_payload(payload, n, k)


def derive_knn_edges(xyz: np.ndarray, k: int = DEFAULT_K) -> np.ndarray:
    """Compute k nearest-neighbor indices per point. Robust to duplicate xyz."""
    from sklearn.neighbors import BallTree
    n_pts = xyz.shape[0]
    if k >= n_pts:
        raise ValueError(f"k={k} too large for {n_pts} points")
    tree = BallTree(xyz)
    extra = max(8, k)
    _, idx = tree.query(xyz, k=k + extra)
    n = idx.shape[0]
    self_arr = np.arange(n, dtype=idx.dtype)[:, None]
    is_not_self = idx != self_arr
    pos = np.cumsum(is_not_self, axis=1)
    keep = is_not_self & (pos <= k)
    counts = keep.sum(axis=1)
    if not (counts == k).all():
        bad = int((counts != k).sum())
        raise RuntimeError(
            f"derive_knn_edges: {bad} points have <{k} non-self neighbors among "
            f"k+extra={k+extra}; check for massive duplicates")
    return idx[keep].reshape(n, k).astype(np.uint32)


def validate_edges(neighbors: np.ndarray, xyz: np.ndarray) -> dict:
    n, k = neighbors.shape
    assert n == xyz.shape[0]
    self_idx = np.arange(n, dtype=np.uint32)
    has_self = (neighbors == self_idx[:, None]).any(axis=1)
    n_self = int(has_self.sum())
    sample = min(1000, n)
    sample_dists = np.linalg.norm(xyz[neighbors[:sample]] - xyz[:sample, None, :], axis=2)
    sorted_check = bool((np.diff(sample_dists, axis=1) >= -1e-9).all())
    in_range = bool((neighbors < n).all() and (neighbors >= 0).all())
    return dict(n=n, k=k, n_self_edges=n_self,
                sorted_by_distance=sorted_check, all_indices_in_range=in_range)
