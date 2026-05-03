"""v33 material_hints chunk codec for .3dphox.

Per docs/v32_v33_lighting_materials_spec.md.

Per-phoxoid (4 bytes total):
    material_hint        (u8 enum)  see MATERIAL_HINT_*
    confidence           (u8 0-255) how cleanly the splat fits its class
    view_dependence_score (u8 0-255) magnitude of SH bands 1-3 vs band 0
    mip_zoom             (u8) log2(max-frequency in screen pixels), Mip-Splatting style

Chunk format (chunk_id 0x14, "material_hints"):
    1 byte version       (currently 0x01)
    1 byte fields_per_blob (currently 0x04)
    4 bytes count        (= N, little-endian uint32)
    N * 4 bytes payload
    4 bytes CRC32
"""

from __future__ import annotations
import struct, zlib
import numpy as np


MATERIAL_CHUNK_ID = 0x14
MATERIAL_CHUNK_VERSION = 0x01
FIELDS_PER_BLOB = 4

MATERIAL_HINT_UNKNOWN     = 0
MATERIAL_HINT_DIFFUSE     = 1
MATERIAL_HINT_GLOSSY      = 2
MATERIAL_HINT_MIRROR      = 3
MATERIAL_HINT_TRANSPARENT = 4
MATERIAL_HINT_EMISSIVE    = 5
MATERIAL_HINT_FLOATER     = 6

MATERIAL_HINT_NAMES = {
    0: 'unknown', 1: 'diffuse', 2: 'glossy', 3: 'mirror',
    4: 'transparent', 5: 'emissive', 6: 'floater',
}


def encode_material_payload(material_hint, confidence, view_dependence, mip_zoom):
    n = len(material_hint)
    assert all(len(a) == n for a in [confidence, view_dependence, mip_zoom])
    arr = np.stack([material_hint, confidence, view_dependence, mip_zoom],
                   axis=1).astype(np.uint8)
    return arr.tobytes()


def decode_material_payload(payload, n):
    assert len(payload) == n * FIELDS_PER_BLOB
    arr = np.frombuffer(payload, dtype=np.uint8).reshape(n, FIELDS_PER_BLOB)
    return arr[:, 0].copy(), arr[:, 1].copy(), arr[:, 2].copy(), arr[:, 3].copy()


def write_material_chunk(material_hint, confidence, view_dependence, mip_zoom):
    n = len(material_hint)
    payload = encode_material_payload(material_hint, confidence, view_dependence, mip_zoom)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return (bytes([MATERIAL_CHUNK_VERSION, FIELDS_PER_BLOB])
            + struct.pack('<I', n) + payload + struct.pack('<I', crc))


def read_material_chunk(chunk_bytes):
    if len(chunk_bytes) < 10:
        raise ValueError(f"material chunk too short: {len(chunk_bytes)}")
    version = chunk_bytes[0]
    if version != MATERIAL_CHUNK_VERSION:
        raise ValueError(f"unsupported material chunk version 0x{version:02x}")
    fields = chunk_bytes[1]
    if fields != FIELDS_PER_BLOB:
        raise ValueError(f"unexpected fields_per_blob {fields}, expected {FIELDS_PER_BLOB}")
    n = struct.unpack('<I', chunk_bytes[2:6])[0]
    expected_len = 6 + n * FIELDS_PER_BLOB + 4
    if len(chunk_bytes) != expected_len:
        raise ValueError(f"material chunk length mismatch: {len(chunk_bytes)} != {expected_len}")
    payload = chunk_bytes[6:6 + n * FIELDS_PER_BLOB]
    stored_crc = struct.unpack('<I', chunk_bytes[6 + n * FIELDS_PER_BLOB:])[0]
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    if stored_crc != actual_crc:
        raise ValueError(f"material chunk CRC mismatch")
    return decode_material_payload(payload, n)


def derive_view_dependence_score(sh_dc, sh_rest):
    n = sh_dc.shape[0]
    if sh_rest is None:
        return np.zeros(n, dtype=np.uint8)
    dc_mag = np.linalg.norm(sh_dc, axis=1)
    rest_mag = np.linalg.norm(sh_rest, axis=1)
    ratio = rest_mag / (dc_mag + 1e-3)
    score = np.clip(ratio / 4.0 * 255.0, 0, 255).astype(np.uint8)
    return score


def derive_material_hints(sh_dc, sh_rest, opacities, kappa=None, neighbor_distances=None):
    n = sh_dc.shape[0]
    dc_mag = np.linalg.norm(sh_dc, axis=1)
    SH_C0 = 0.28209479177387814
    base_color_rgb = sh_dc * SH_C0 + 0.5
    base_brightness = base_color_rgb.mean(axis=1).clip(-1, 2)

    if sh_rest is not None:
        rest_mag = np.linalg.norm(sh_rest, axis=1)
        view_dep_ratio = rest_mag / (dc_mag + 1e-3)
        band1 = np.linalg.norm(sh_rest[:, :9], axis=1)
        band3 = np.linalg.norm(sh_rest[:, 24:], axis=1)
    else:
        view_dep_ratio = np.zeros(n)
        band1 = band3 = np.zeros(n)

    sigmoid_opacity = 1.0 / (1.0 + np.exp(-opacities))

    if neighbor_distances is not None:
        nd_p90 = np.percentile(neighbor_distances, 90)
        sparse = neighbor_distances > nd_p90
    else:
        sparse = np.zeros(n, dtype=bool)
    if kappa is not None:
        flat = kappa < np.percentile(kappa, 25)
    else:
        flat = np.zeros(n, dtype=bool)
    low_opa = sigmoid_opacity < 0.3

    hint = np.zeros(n, dtype=np.uint8)
    conf = np.zeros(n, dtype=np.uint8)

    floater_score = sparse.astype(int) + flat.astype(int) + low_opa.astype(int)
    is_floater = floater_score >= 2
    hint[is_floater] = MATERIAL_HINT_FLOATER
    conf[is_floater] = np.clip(floater_score[is_floater] * 80, 0, 255).astype(np.uint8)

    is_emissive = (base_brightness > 1.0) & (band3 > 0.3)
    hint[is_emissive & ~is_floater] = MATERIAL_HINT_EMISSIVE
    conf[is_emissive & ~is_floater] = 200

    is_mirror = (band3 / (dc_mag + 1e-3) > 1.5) & (sigmoid_opacity > 0.7)
    mirror_set = is_mirror & ~is_floater & ~is_emissive
    hint[mirror_set] = MATERIAL_HINT_MIRROR
    conf[mirror_set] = np.clip(band3[mirror_set] * 50, 0, 255).astype(np.uint8)

    is_glossy = (view_dep_ratio > 0.5) & (band1 > band3 * 1.5)
    glossy_set = is_glossy & ~is_floater & ~is_emissive & ~is_mirror
    hint[glossy_set] = MATERIAL_HINT_GLOSSY
    conf[glossy_set] = np.clip(view_dep_ratio[glossy_set] * 80, 0, 255).astype(np.uint8)

    is_diffuse = (view_dep_ratio < 0.2) & (sigmoid_opacity > 0.5)
    diffuse_set = is_diffuse & ~is_floater & ~is_emissive & ~is_mirror & ~is_glossy
    hint[diffuse_set] = MATERIAL_HINT_DIFFUSE
    conf[diffuse_set] = np.clip(255 - (view_dep_ratio[diffuse_set] * 1000).astype(int), 0, 255).astype(np.uint8)

    return hint, conf


# ------- Derivation: Mip-Splatting mip_zoom (added 2026-05-02) -------

MIP_ZOOM_FOCAL_REF  = 1024.0
MIP_ZOOM_LOD_BIAS   = 8.0
MIP_ZOOM_LOD_SCALE  = 8.0


def derive_mip_zoom(scales, focal_ref=MIP_ZOOM_FOCAL_REF):
    """Per-splat Mip-Splatting LOD byte from world-space scales.

    scales: (N, 3) world-space semi-axis lengths (linear, not log).
    focal_ref: pixels per unit length at unit distance.
    Returns (N,) uint8.
    """
    if scales.ndim != 2 or scales.shape[1] != 3:
        raise ValueError(f"derive_mip_zoom: scales must be (N,3), got {scales.shape}")
    sigma_world = np.maximum.reduce([scales[:, 0], scales[:, 1], scales[:, 2]]).astype(np.float64)
    sigma_world = np.maximum(sigma_world, 1e-12)
    pix_at_d1   = focal_ref * sigma_world
    lod         = np.log2(pix_at_d1)
    q           = (lod + MIP_ZOOM_LOD_BIAS) * MIP_ZOOM_LOD_SCALE
    return np.clip(np.round(q), 0, 255).astype(np.uint8)


def decode_mip_zoom(mip_zoom_u8, focal_ref=MIP_ZOOM_FOCAL_REF):
    """Recover sigma_world from the mip_zoom byte (lossy round-trip)."""
    pix_at_d1 = np.power(
        2.0, mip_zoom_u8.astype(np.float64) / MIP_ZOOM_LOD_SCALE - MIP_ZOOM_LOD_BIAS
    )
    return (pix_at_d1 / focal_ref).astype(np.float32)
