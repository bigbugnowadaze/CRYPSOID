"""Load .3dphox v25/v27/v28 containers into SplatBuffer."""

import json
import struct
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .splat_buffer import SplatBuffer


def read_container(path: Path) -> Tuple[bytes, Dict[str, Any], bytes, Dict, Callable, Callable]:
    """Read .3dphox container and return (magic, manifest, blob, chunks, comp_fn, dec_fn)."""
    with path.open("rb") as f:
        magic = f.read(11)
        manifest_len = struct.unpack("<Q", f.read(8))[0]
        manifest = json.loads(f.read(manifest_len))
        blob = f.read()

    chunks = {c["name"]: c for c in manifest["chunks"]}

    def comp(name: str) -> bytes:
        c = chunks[name]
        return blob[c["offset"] : c["offset"] + c["compressed_bytes"]]

    def dec(name: str) -> bytes:
        return zlib.decompress(comp(name))

    return magic, manifest, blob, chunks, comp, dec


def decode_u24_xyz(
    raw: bytes, n: int, bounds_min: List[float], bounds_max: List[float]
) -> np.ndarray:
    """Decode 24-bit quantized XYZ coordinates."""
    a = np.frombuffer(raw, dtype=np.uint8).reshape(n, 9)
    q = np.empty((n, 3), dtype=np.uint32)
    for j in range(3):
        q[:, j] = (
            a[:, 3 * j].astype(np.uint32)
            | (a[:, 3 * j + 1].astype(np.uint32) << 8)
            | (a[:, 3 * j + 2].astype(np.uint32) << 16)
        )
    mn = np.asarray(bounds_min, dtype=np.float32)
    mx = np.asarray(bounds_max, dtype=np.float32)
    return (q.astype(np.float32) / float((1 << 24) - 1)) * (mx - mn) + mn


def decode_f16_scales(raw: bytes, n: int) -> np.ndarray:
    """Decode float16 scales to float32."""
    f16_arr = np.frombuffer(raw, dtype=np.float16).reshape(n, 3)
    return f16_arr.astype(np.float32)


def decode_i16_quats(raw: bytes, n: int) -> np.ndarray:
    """Decode int16 normalized quaternions (wxyz, norm=4)."""
    i16_arr = np.frombuffer(raw, dtype=np.int16).reshape(n, 4)
    return i16_arr.astype(np.float32) / 32767.0


def decode_dc_rgb_opacity_u8(raw: bytes, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Decode uint8 DC RGB and opacity."""
    dc = np.frombuffer(raw, dtype=np.uint8).reshape(n, 4)
    rgb = dc[:, :3].astype(np.float32) / 255.0
    opacity = dc[:, 3].astype(np.float32) / 255.0
    return rgb, opacity


def load_3dphox_v28_render(path: Path) -> SplatBuffer:
    """Load .3dphox v28 render container (VQ-encoded SH).

    This is a render container with compressed SH via VQ codebook.
    It has tier labels and the five passthrough chunks from v25.
    """
    magic, manifest, blob, chunks, comp, dec = read_container(path)

    if not magic.startswith(b"CRYPSOID28"):
        raise ValueError(f"Expected CRYPSOID28 magic, got {magic}")

    # Get splat count
    xyz_chunk = chunks["xyz_u24_fixed"]
    n = int(xyz_chunk["shape"][0])

    # Decompress chunks
    xyz_raw = dec("xyz_u24_fixed")
    scales_raw = dec("scale_f16")
    quats_raw = dec("quat_i16_norm4")
    dc_raw = dec("dc_rgb_opacity_u8")
    tier_raw = dec("tier_labels_u8")

    # For v28 render container, SH is VQ-encoded: we need the codebook and indices
    # For v0.1 we'll just decode DC (degree-0 only) and use sh_rest = None
    # In v0.2+ we'd reconstruct from VQ codebook + indices

    # Decode all fields
    xyz = decode_u24_xyz(xyz_raw, n, xyz_chunk["bounds_min"], xyz_chunk["bounds_max"])
    scales = decode_f16_scales(scales_raw, n)
    quats = decode_i16_quats(quats_raw, n)
    sh_dc, opacity = decode_dc_rgb_opacity_u8(dc_raw, n)
    tier_labels = np.frombuffer(tier_raw, dtype=np.uint8)

    # SH rest: decode from VQ codebook + indices (per build_v27_fast.py).
    # Group g of the 45-dim SH vector lives at columns [g*15 : (g+1)*15].
    # sh_q8 = codebook[g, idx[:, g]]  for g in {0, 1, 2}, then concatenated.
    if "sh_vq128_idx_u8" in chunks and "sh_vq128_codebook_i8" in chunks:
        idx_raw = dec("sh_vq128_idx_u8")
        cb_raw = dec("sh_vq128_codebook_i8")
        idx = np.frombuffer(idx_raw, dtype=np.uint8).reshape(n, 3)              # (n, 3)
        cb = np.frombuffer(cb_raw, dtype=np.int8).reshape(3, 128, 15)           # (3, 128, 15)
        sh_q8 = np.empty((n, 45), dtype=np.int8)
        for g in range(3):
            sh_q8[:, g*15:(g+1)*15] = cb[g][idx[:, g]]                          # (n, 15)
        # Recover float SH coefficients with the global scale recorded in v25.
        global_scale = 0.006946287755891094
        sh_rest = (sh_q8.astype(np.float32) * np.float32(global_scale))
    else:
        sh_rest = None

    return SplatBuffer(
        n=n,
        xyz=xyz,
        scales=scales,
        quats=quats,
        opacities=opacity,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        tier=tier_labels,
        germ=None,
        correction=None,
        source=str(path),
        scene_format="3dphox_v28_render",
    )


def load_3dphox_v25_render(path: Path) -> SplatBuffer:
    """Load .3dphox v25 render container (float32 SH).

    v25 is the attribute-group container with full float32 SH coefficients.
    """
    magic, manifest, blob, chunks, comp, dec = read_container(path)

    if not magic.startswith(b"CRYPSOID25"):
        raise ValueError(f"Expected CRYPSOID25 magic, got {magic}")

    # Get splat count
    xyz_chunk = chunks["xyz_u24_fixed"]
    n = int(xyz_chunk["shape"][0])

    # Decompress chunks
    xyz_raw = dec("xyz_u24_fixed")
    scales_raw = dec("scale_f16")
    quats_raw = dec("quat_i16_norm4")
    dc_raw = dec("dc_rgb_opacity_u8")
    tier_raw = dec("tier_labels_u8")
    sh_rest_raw = dec("sh_rest_f32")  # Full SH in v25

    # Decode fields
    xyz = decode_u24_xyz(xyz_raw, n, xyz_chunk["bounds_min"], xyz_chunk["bounds_max"])
    scales = decode_f16_scales(scales_raw, n)
    quats = decode_i16_quats(quats_raw, n)
    sh_dc, opacity = decode_dc_rgb_opacity_u8(dc_raw, n)
    tier_labels = np.frombuffer(tier_raw, dtype=np.uint8)

    # Decode SH rest (45 float32 coefficients per splat, degrees 1-3)
    sh_rest = np.frombuffer(sh_rest_raw, dtype=np.float32).reshape(n, 45)

    return SplatBuffer(
        n=n,
        xyz=xyz,
        scales=scales,
        quats=quats,
        opacities=opacity,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        tier=tier_labels,
        germ=None,
        correction=None,
        source=str(path),
        scene_format="3dphox_v25",
    )


def load_3dphox_v27_render(path: Path) -> SplatBuffer:
    """Load .3dphox v27 render container (VQ-encoded SH, verified anchor).

    v27 is like v28 but is the verified reference point.
    """
    # For v0.1, v27 is loaded the same way as v28 (DC-only)
    return load_3dphox_v28_render(path)


def load_3dphox_v28_archive(path: Path) -> SplatBuffer:
    """Load .3dphox v28 EXACT-archive container (VQ + per-tier-group residuals).

    The archive container reproduces the original v25 q8 SH stream byte-exact:
        sh_q8[i, g*15:(g+1)*15] = codebook[g, idx[i, g]] + residual[tier(i), rank_in_tier(i), g, :]
    Then sh_float = sh_q8 * global_scale.
    """
    magic, manifest, blob, chunks, comp, dec = read_container(path)
    if not magic.startswith(b"CRYPSOID28"):
        raise ValueError(f"Expected CRYPSOID28 magic, got {magic!r}")
    if "EXACT_ARCHIVE" not in manifest.get("format", ""):
        raise ValueError(f"Not a v28 EXACT archive container: format={manifest.get('format')}")

    n = int(chunks["xyz_u24_fixed"]["shape"][0])
    xyz_chunk = chunks["xyz_u24_fixed"]

    xyz = decode_u24_xyz(dec("xyz_u24_fixed"), n, xyz_chunk["bounds_min"], xyz_chunk["bounds_max"])
    scales = decode_f16_scales(dec("scale_f16"), n)
    quats = decode_i16_quats(dec("quat_i16_norm4"), n)
    sh_dc, opacity = decode_dc_rgb_opacity_u8(dec("dc_rgb_opacity_u8"), n)
    tier_labels = np.frombuffer(dec("tier_labels_u8"), dtype=np.uint8)

    # VQ centroids
    idx = np.frombuffer(dec("sh_vq128_idx_u8"), dtype=np.uint8).reshape(n, 3)
    cb = np.frombuffer(dec("sh_vq128_codebook_i8"), dtype=np.int8).reshape(3, 128, 15)
    sh_q8 = np.empty((n, 45), dtype=np.int16)  # int16 to hold cb + residual
    for g in range(3):
        sh_q8[:, g*15:(g+1)*15] = cb[g][idx[:, g]].astype(np.int16)

    # Per-tier-group exact residuals (these add to VQ to recover the original q8)
    for t in range(3):
        # Splat indices belonging to tier t, in their original order
        tier_idx = np.where(tier_labels == t)[0]
        for g in range(3):
            chunk_name = f"sh_exact_residual_t{t}_g{g}_int8"
            if chunk_name not in chunks:
                continue
            res_raw = dec(chunk_name)
            res = np.frombuffer(res_raw, dtype=np.int8).reshape(-1, 15).astype(np.int16)
            assert res.shape[0] == len(tier_idx),                 f"residual {chunk_name}: rows={res.shape[0]} expected {len(tier_idx)}"
            sh_q8[tier_idx, g*15:(g+1)*15] += res

    # Clip back to int8 range and convert to float
    sh_q8 = np.clip(sh_q8, -128, 127).astype(np.int8)
    global_scale = 0.006946287755891094
    sh_rest = sh_q8.astype(np.float32) * np.float32(global_scale)

    return SplatBuffer(
        n=n, xyz=xyz, scales=scales, quats=quats, opacities=opacity,
        sh_dc=sh_dc, sh_rest=sh_rest, tier=tier_labels,
        germ=None, correction=None,
        source=str(path), scene_format="3dphox_v28_archive",
    )


def load_3dphox(path: Path) -> SplatBuffer:
    """Auto-dispatch by magic + format string."""
    with path.open("rb") as f:
        magic = f.read(11)
        ml = struct.unpack("<Q", f.read(8))[0]
        manifest = json.loads(f.read(ml))
    fmt = manifest.get("format", "")
    if magic.startswith(b"CRYPSOID25"):
        return load_3dphox_v25_render(path)
    if magic.startswith(b"CRYPSOID27"):
        return load_3dphox_v27_render(path)
    if magic.startswith(b"CRYPSOID28"):
        if "EXACT_ARCHIVE" in fmt:
            return load_3dphox_v28_archive(path)
        return load_3dphox_v28_render(path)
    raise ValueError(f"Unknown .3dphox magic: {magic!r}")
