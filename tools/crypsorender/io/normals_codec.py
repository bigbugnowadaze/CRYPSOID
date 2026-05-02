"""v31 Addition 1 -- octahedral normal codec + .3dphox chunk reader/writer
+ MLS normal derivation (with quadric refinement).
4 bytes/phoxoid: 24-bit oct normal + 8-bit tangent angle.
Per docs/v31_graph_extension_spec.md.
"""
from __future__ import annotations
import struct, zlib
import numpy as np


def _sign_not_zero(v):
    return np.where(v >= 0.0, 1.0, -1.0)


def normal_to_oct(normals: np.ndarray) -> np.ndarray:
    n = normals / np.linalg.norm(normals, axis=-1, keepdims=True).clip(min=1e-12)
    p = n[..., :2] / np.abs(n).sum(axis=-1, keepdims=True).clip(min=1e-12)
    z = n[..., 2:3]
    folded = (1.0 - np.abs(p[..., 1:2])) * _sign_not_zero(p[..., 0:1])
    folded2 = (1.0 - np.abs(p[..., 0:1])) * _sign_not_zero(p[..., 1:2])
    p = np.where(z < 0.0, np.concatenate([folded, folded2], axis=-1), p)
    return p


def oct_to_normal(oct_xy: np.ndarray) -> np.ndarray:
    x = oct_xy[..., 0]; y = oct_xy[..., 1]
    z = 1.0 - np.abs(x) - np.abs(y)
    fold_mask = z < 0.0
    x_f = (1.0 - np.abs(y)) * _sign_not_zero(x)
    y_f = (1.0 - np.abs(x)) * _sign_not_zero(y)
    x = np.where(fold_mask, x_f, x)
    y = np.where(fold_mask, y_f, y)
    n = np.stack([x, y, z], axis=-1)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True).clip(min=1e-12)
    return n


def quantize_oct_24bit(oct_xy: np.ndarray) -> np.ndarray:
    q = np.clip(((oct_xy + 1.0) * 0.5 * 4095.0).round(), 0, 4095).astype(np.uint32)
    qx = q[..., 0]; qy = q[..., 1]
    b0 = (qx & 0xFF).astype(np.uint8)
    b1 = (((qy & 0x00F) << 4) | ((qx >> 8) & 0x0F)).astype(np.uint8)
    b2 = ((qy >> 4) & 0xFF).astype(np.uint8)
    return np.stack([b0, b1, b2], axis=-1)


def dequantize_oct_24bit(packed: np.ndarray) -> np.ndarray:
    b0 = packed[..., 0].astype(np.uint32)
    b1 = packed[..., 1].astype(np.uint32)
    b2 = packed[..., 2].astype(np.uint32)
    qx = ((b1 & 0x0F) << 8) | b0
    qy = (b2 << 4) | ((b1 >> 4) & 0x0F)
    x = qx.astype(np.float64) / 4095.0 * 2.0 - 1.0
    y = qy.astype(np.float64) / 4095.0 * 2.0 - 1.0
    return np.stack([x, y], axis=-1)


def tangent_angle_to_byte(angle_rad: np.ndarray) -> np.ndarray:
    a = np.mod(angle_rad, 2.0 * np.pi)
    raw = a / (2.0 * np.pi) * 256.0 + 1e-9
    return np.clip(np.floor(raw).astype(np.int32), 0, 255).astype(np.uint8)


def byte_to_tangent_angle(b: np.ndarray) -> np.ndarray:
    return b.astype(np.float64) / 256.0 * 2.0 * np.pi


def encode_normals_payload(normals, tangent_angles):
    assert normals.shape == (len(tangent_angles), 3)
    oct_xy = normal_to_oct(normals)
    packed_n = quantize_oct_24bit(oct_xy)
    packed_t = tangent_angle_to_byte(tangent_angles)
    return np.concatenate([packed_n, packed_t[:, None]], axis=1).astype(np.uint8).tobytes()


def decode_normals_payload(payload, n):
    assert len(payload) == n * 4
    arr = np.frombuffer(payload, dtype=np.uint8).reshape(n, 4)
    normals = oct_to_normal(dequantize_oct_24bit(arr[:, :3]))
    return normals, byte_to_tangent_angle(arr[:, 3])


NORMALS_CHUNK_ID = 0x12
NORMALS_CHUNK_VERSION = 0x01


def write_normals_chunk(normals, tangent_angles):
    n = len(normals)
    payload = encode_normals_payload(normals, tangent_angles)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return (bytes([NORMALS_CHUNK_VERSION, 0x00]) + struct.pack('<I', n)
            + payload + struct.pack('<I', crc))


def read_normals_chunk(chunk_bytes):
    if len(chunk_bytes) < 10:
        raise ValueError(f"normals chunk too short: {len(chunk_bytes)}")
    version = chunk_bytes[0]
    if version != NORMALS_CHUNK_VERSION:
        raise ValueError(f"unsupported normals chunk version 0x{version:02x}")
    n = struct.unpack('<I', chunk_bytes[2:6])[0]
    expected_len = 6 + n * 4 + 4
    if len(chunk_bytes) != expected_len:
        raise ValueError(f"normals chunk length mismatch: {len(chunk_bytes)} != {expected_len}")
    payload = chunk_bytes[6:6 + n * 4]
    stored_crc = struct.unpack('<I', chunk_bytes[6 + n * 4:])[0]
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    if stored_crc != actual_crc:
        raise ValueError(f"normals chunk CRC mismatch: stored 0x{stored_crc:08x}, computed 0x{actual_crc:08x}")
    return decode_normals_payload(payload, n)


def derive_normals_mls(xyz, k=24, world_up=(0.0, 1.0, 0.0), refine_quadric=True):
    """MLS normal estimation with optional quadric refinement to remove the
    plane-fit-on-curved-surface bias. Returns (normals (N,3), tangent_angles (N,))."""
    from sklearn.neighbors import BallTree
    n_pts = xyz.shape[0]
    tree = BallTree(xyz)
    _, idx = tree.query(xyz, k=k + 1)
    neighbors = xyz[idx[:, 1:]]
    centroids = neighbors.mean(axis=1, keepdims=True)
    centered = neighbors - centroids
    cov = np.einsum('nki,nkj->nij', centered, centered) / k
    _, eigvecs = np.linalg.eigh(cov)
    n0 = eigvecs[:, :, 0]
    up = np.asarray(world_up, dtype=np.float64)
    n0[(n0 @ up) < 0.0] *= -1.0
    n0 /= np.linalg.norm(n0, axis=1, keepdims=True).clip(min=1e-12)
    if refine_quadric:
        ref = np.where(np.abs(n0[:, 0:1]) > 0.9,
                       np.array([0.0, 0.0, 1.0]),
                       np.array([1.0, 0.0, 0.0]))
        t1 = np.cross(n0, ref)
        t1n = np.linalg.norm(t1, axis=1, keepdims=True).clip(min=1e-12)
        t1 = t1 / t1n
        t2 = np.cross(n0, t1)
        rel = neighbors - xyz[:, None, :]
        u_l = np.einsum('nki,ni->nk', rel, t1)
        v_l = np.einsum('nki,ni->nk', rel, t2)
        w_l = np.einsum('nki,ni->nk', rel, n0)
        M = np.stack([np.ones_like(u_l), u_l, v_l, u_l**2, u_l*v_l, v_l**2], axis=2)
        MtM = np.einsum('nki,nkj->nij', M, M) + 1e-9 * np.eye(6)[None]
        Mtw = np.einsum('nki,nk->ni', M, w_l)[..., None]
        coef = np.linalg.solve(MtM, Mtw)[..., 0]
        b_, c_ = coef[:, 1], coef[:, 2]
        local_n = np.stack([-b_, -c_, np.ones_like(b_)], axis=1)
        local_n /= np.linalg.norm(local_n, axis=1, keepdims=True)
        normals = local_n[:, 0:1]*t1 + local_n[:, 1:2]*t2 + local_n[:, 2:3]*n0
        normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)
        normals[(normals @ up) < 0.0] *= -1.0
    else:
        normals = n0
    world_x = np.array([1.0, 0.0, 0.0])
    parallel = np.abs(normals @ world_x) > 0.95
    ref2 = np.where(parallel[:, None], np.array([0.0, 1.0, 0.0]), world_x)
    proj = ref2 - (np.einsum('ni,ni->n', ref2, normals)[:, None]) * normals
    proj /= np.linalg.norm(proj, axis=1, keepdims=True).clip(min=1e-12)
    up_proj = up - (np.einsum('i,ni->n', up, normals)[:, None]) * normals
    up_proj /= np.linalg.norm(up_proj, axis=1, keepdims=True).clip(min=1e-12)
    cos_a = np.einsum('ni,ni->n', proj, up_proj).clip(-1.0, 1.0)
    cross = np.cross(proj, up_proj)
    sin_a = np.einsum('ni,ni->n', cross, normals)
    tangent_angles = np.mod(np.arctan2(sin_a, cos_a), 2.0 * np.pi)
    return normals.astype(np.float64), tangent_angles.astype(np.float64)
