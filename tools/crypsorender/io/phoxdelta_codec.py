"""v31 Addition 3 -- .phoxdelta patch format.

Sparse modify-only patches over a base .3dphox. Per docs/v31_graph_extension_spec.md.

Header (24 bytes):
    Magic          8 bytes   b"PHOXDLT\\0"
    Version        1 byte    0x01
    Reserved       3 bytes   zero
    Base CRC32     4 bytes   little-endian uint32 (CRC32 of base file bytes)
    Base N         4 bytes   little-endian uint32 (expected phoxoid count in base)
    Delta count    4 bytes   little-endian uint32 (number of records following)

Per-record:
    phoxoid_id    4 bytes   uint32 little-endian (index in base)
    dirty_mask    2 bytes   uint16 little-endian (which attributes are present)
    payload       variable  changed attributes in fixed order, low-bit first

Dirty mask bits:
    bit 0 = xyz                  (3 * float32 = 12 B)
    bit 1 = scale                (3 * float32 = 12 B)
    bit 2 = quat                 (4 * float32 = 16 B)
    bit 3 = opacity              (1 * float32 =  4 B)
    bit 4 = f_dc                 (3 * float32 = 12 B)
    bit 5 = f_rest               (45 * float32 = 180 B)
    bit 6 = tier_label           (1 * uint8   =  1 B)
    bit 7 = germ coefficients    (5 * float32 = 20 B)
    bit 8 = normal + tangent     (4 * uint8   =  4 B, oct24+tan8)
    bit 9 = kNN neighbors        (k * uint32  variable, k from companion edges chunk)
    bits 10-15: reserved

Composition: later wins per (phoxoid_id, attribute).

Insert / delete are RESERVED for v32 (delta is modify-only).
"""

from __future__ import annotations
import struct, zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

MAGIC = b"PHOXDLT\x00"
VERSION = 0x01

# Bit definitions
BIT_XYZ      = 0
BIT_SCALE    = 1
BIT_QUAT     = 2
BIT_OPACITY  = 3
BIT_F_DC     = 4
BIT_F_REST   = 5
BIT_TIER     = 6
BIT_GERM     = 7
BIT_NORMAL   = 8
BIT_NEIGHBORS = 9

# Bit -> (name, dtype, n_elements per record)
BIT_LAYOUT = [
    (BIT_XYZ,       'xyz',       'float32', 3),
    (BIT_SCALE,     'scale',     'float32', 3),
    (BIT_QUAT,      'quat',      'float32', 4),
    (BIT_OPACITY,   'opacity',   'float32', 1),
    (BIT_F_DC,      'f_dc',      'float32', 3),
    (BIT_F_REST,    'f_rest',    'float32', 45),
    (BIT_TIER,      'tier',      'uint8',   1),
    (BIT_GERM,      'germ',      'float32', 5),
    (BIT_NORMAL,    'normal',    'uint8',   4),    # 3-byte oct + 1-byte tangent
    # bit 9 (neighbors) is variable-length; handled separately if used.
]

DTYPE_BYTES = {'float32': 4, 'uint8': 1, 'uint32': 4}


def _attr_size(bit: int, k_neighbors: int = 4) -> int:
    """Bytes per record for a given bit. k_neighbors only used for bit 9."""
    if bit == BIT_NEIGHBORS:
        return k_neighbors * 4
    for b, _, dt, n in BIT_LAYOUT:
        if b == bit:
            return DTYPE_BYTES[dt] * n
    raise ValueError(f"unknown bit {bit}")


@dataclass
class PhoxDelta:
    """A parsed .phoxdelta file in memory."""
    base_crc: int
    base_n: int
    phoxoid_ids: np.ndarray              # (M,) uint32
    dirty_mask: np.ndarray               # (M,) uint16
    attrs: Dict[str, np.ndarray] = field(default_factory=dict)
    # For sparse attrs, attrs[name] is a (M_with_bit, ...) array; we also
    # need an index into phoxoid_ids that says which records had this bit set.
    attr_record_idx: Dict[str, np.ndarray] = field(default_factory=dict)
    # bit 9 (neighbors) is variable per record; for v0.1 we only support
    # uniform k via header tag.
    k_neighbors: int = 4

    @property
    def delta_count(self) -> int:
        return len(self.phoxoid_ids)


# ---------- Encode ----------

def encode_phoxdelta(base_crc: int, base_n: int,
                    phoxoid_ids: np.ndarray,
                    attrs: Dict[str, np.ndarray]) -> bytes:
    """Encode a .phoxdelta file from a list of phoxoid changes.

    Args:
        base_crc: CRC32 of the base .3dphox file.
        base_n: Number of phoxoids in base.
        phoxoid_ids: (M,) uint32 ids of changed phoxoids.
        attrs: dict mapping attribute name (e.g. 'opacity', 'xyz', 'tier') to
               (M, ...) array. Only listed attrs are considered "dirty"; other
               bits stay 0 in the mask.

    All listed attrs must have shape (M, ...) where M = len(phoxoid_ids); a
    given record's dirty_mask bits are computed from which attrs are present
    (currently uniform: every record has the same set of dirty bits).
    """
    M = len(phoxoid_ids)
    name_to_bit = {name: bit for bit, name, _, _ in BIT_LAYOUT}
    dirty_mask = 0
    for name in attrs:
        if name == 'neighbors':
            dirty_mask |= (1 << BIT_NEIGHBORS)
        elif name in name_to_bit:
            dirty_mask |= (1 << name_to_bit[name])
        else:
            raise ValueError(f"unknown attr {name}")

    # Per-record output: id (4) + mask (2) + payload (sum of attr sizes)
    record_payload_size = 0
    for bit, name, dt, n in BIT_LAYOUT:
        if dirty_mask & (1 << bit) and name in attrs:
            record_payload_size += DTYPE_BYTES[dt] * n
    if dirty_mask & (1 << BIT_NEIGHBORS):
        nb_arr = attrs['neighbors']
        k = nb_arr.shape[1]
        record_payload_size += k * 4
    else:
        k = 4

    # Build header
    parts = [
        MAGIC,
        bytes([VERSION, 0, 0, 0]),
        struct.pack('<I', base_crc & 0xFFFFFFFF),
        struct.pack('<I', base_n),
        struct.pack('<I', M),
    ]

    # Build records in order
    body_bufs = []
    ids_arr = np.asarray(phoxoid_ids, dtype=np.uint32)
    mask_bytes = struct.pack('<H', dirty_mask)
    for i in range(M):
        body_bufs.append(struct.pack('<I', int(ids_arr[i])))
        body_bufs.append(mask_bytes)
        # append each attr in canonical bit order
        for bit, name, dt, n in BIT_LAYOUT:
            if dirty_mask & (1 << bit) and name in attrs:
                arr = np.asarray(attrs[name][i], dtype=dt)
                body_bufs.append(arr.tobytes())
        if dirty_mask & (1 << BIT_NEIGHBORS):
            nb = np.asarray(attrs['neighbors'][i], dtype=np.uint32)
            body_bufs.append(nb.tobytes())

    return b"".join(parts) + b"".join(body_bufs)


# ---------- Decode ----------

def decode_phoxdelta(data: bytes, k_neighbors: int = 4) -> PhoxDelta:
    """Parse a .phoxdelta file."""
    if len(data) < 24:
        raise ValueError(f"phoxdelta too short: {len(data)} bytes")
    if data[:8] != MAGIC:
        raise ValueError(f"phoxdelta bad magic: {data[:8]!r}")
    version = data[8]
    if version != VERSION:
        raise ValueError(f"unsupported phoxdelta version 0x{version:02x}")
    base_crc = struct.unpack('<I', data[12:16])[0]
    base_n = struct.unpack('<I', data[16:20])[0]
    M = struct.unpack('<I', data[20:24])[0]

    p = 24
    name_to_bit = {name: bit for bit, name, _, _ in BIT_LAYOUT}
    bit_to_layout = {bit: (name, dt, n) for bit, name, dt, n in BIT_LAYOUT}
    ids = np.zeros(M, dtype=np.uint32)
    masks = np.zeros(M, dtype=np.uint16)
    # We'll collect per-attribute arrays. Since dirty_mask can vary per
    # record (v0.1 doesn't enforce uniform), we accumulate dynamic lists.
    per_attr_records: Dict[str, list] = {}      # name -> list of (record_idx, value)
    per_attr_records['neighbors'] = []

    for i in range(M):
        ids[i] = struct.unpack('<I', data[p:p+4])[0]; p += 4
        masks[i] = struct.unpack('<H', data[p:p+2])[0]; p += 2
        m = int(masks[i])
        for bit, (name, dt, n) in sorted(bit_to_layout.items()):
            if m & (1 << bit):
                sz = DTYPE_BYTES[dt] * n
                arr = np.frombuffer(data[p:p+sz], dtype=dt, count=n)
                if n == 1:
                    arr = arr[0]
                per_attr_records.setdefault(name, []).append((i, np.array(arr)))
                p += sz
        if m & (1 << BIT_NEIGHBORS):
            sz = k_neighbors * 4
            nb = np.frombuffer(data[p:p+sz], dtype='<u4', count=k_neighbors).copy()
            per_attr_records['neighbors'].append((i, nb))
            p += sz

    # Pack into PhoxDelta
    pd = PhoxDelta(
        base_crc=base_crc, base_n=base_n,
        phoxoid_ids=ids, dirty_mask=masks,
        k_neighbors=k_neighbors,
    )
    for name, records in per_attr_records.items():
        if not records:
            continue
        idxs = np.array([r[0] for r in records], dtype=np.int64)
        vals = np.stack([np.atleast_1d(r[1]) for r in records])
        pd.attrs[name] = vals
        pd.attr_record_idx[name] = idxs
    return pd


# ---------- Apply / Compose ----------

def apply_phoxdelta(splat_buffer, delta: PhoxDelta, *, copy: bool = True):
    """Apply a .phoxdelta to an in-memory SplatBuffer.

    Args:
        splat_buffer: a crypsorender.io.splat_buffer.SplatBuffer
        delta: parsed PhoxDelta
        copy: if True, return a modified copy; if False, mutate in place

    Mapping from delta attr names to SplatBuffer fields:
        xyz -> xyz
        scale -> scales
        quat -> quats
        opacity -> opacities
        f_dc -> sh_dc
        f_rest -> sh_rest
        tier -> tier
        germ -> germ.coefs (if present)
        normal -> not currently a SplatBuffer field; ignored (will be when v31 chunks land)
        neighbors -> not currently a SplatBuffer field; ignored
    """
    from copy import copy as shallow_copy
    if copy:
        sb = shallow_copy(splat_buffer)
        # numpy arrays — copy the ones we'll modify
        for f in ('xyz', 'scales', 'quats', 'opacities', 'sh_dc', 'sh_rest', 'tier'):
            v = getattr(sb, f, None)
            if v is not None:
                setattr(sb, f, v.copy())
    else:
        sb = splat_buffer

    if delta.base_n != sb.n:
        raise ValueError(f"phoxdelta base_n {delta.base_n} != splat_buffer.n {sb.n}")

    name_map = {
        'xyz':     'xyz',
        'scale':   'scales',
        'quat':    'quats',
        'opacity': 'opacities',
        'f_dc':    'sh_dc',
        'f_rest':  'sh_rest',
        'tier':    'tier',
    }
    for name, field_name in name_map.items():
        if name not in delta.attrs:
            continue
        target = getattr(sb, field_name, None)
        if target is None:
            continue
        record_idxs = delta.attr_record_idx[name]
        # phoxoid ids whose records carry this attribute
        ids = delta.phoxoid_ids[record_idxs]
        vals = delta.attrs[name]
        if vals.ndim == 2 and vals.shape[1] == 1:
            vals = vals[:, 0]
        target[ids] = vals.astype(target.dtype)
    return sb


def compose_phoxdeltas(deltas: List[PhoxDelta]) -> PhoxDelta:
    """Compose a list of deltas; later deltas win per (phoxoid_id, attribute).

    Returns a single PhoxDelta whose phoxoid_id list is the union of all input
    ids, with attributes merged so the rightmost delta's value wins.
    """
    if not deltas:
        raise ValueError("compose_phoxdeltas: empty list")
    base_crc = deltas[0].base_crc
    base_n = deltas[0].base_n
    for d in deltas[1:]:
        if d.base_crc != base_crc:
            raise ValueError("compose: base_crc mismatch")
        if d.base_n != base_n:
            raise ValueError("compose: base_n mismatch")

    # Per-id, per-attr → value
    per_id_attr: Dict[int, Dict[str, np.ndarray]] = {}
    for d in deltas:
        for name, vals in d.attrs.items():
            record_idxs = d.attr_record_idx[name]
            ids = d.phoxoid_ids[record_idxs]
            for i, pid in enumerate(ids):
                pid_int = int(pid)
                per_id_attr.setdefault(pid_int, {})[name] = vals[i]

    # Rebuild
    out_ids = sorted(per_id_attr.keys())
    M = len(out_ids)
    # Determine attribute set actually used
    attr_names = set()
    for d in per_id_attr.values():
        attr_names.update(d.keys())

    out_attrs = {}
    out_idx = {}
    for name in attr_names:
        rows = []
        idxs = []
        for i, pid in enumerate(out_ids):
            if name in per_id_attr[pid]:
                rows.append(per_id_attr[pid][name])
                idxs.append(i)
        out_attrs[name] = np.stack([np.atleast_1d(r) for r in rows])
        out_idx[name] = np.array(idxs, dtype=np.int64)

    pd = PhoxDelta(
        base_crc=base_crc, base_n=base_n,
        phoxoid_ids=np.array(out_ids, dtype=np.uint32),
        dirty_mask=np.array([0]*M, dtype=np.uint16),  # rebuilt below
        k_neighbors=deltas[0].k_neighbors,
    )
    name_to_bit = {name: bit for bit, name, _, _ in BIT_LAYOUT}
    name_to_bit['neighbors'] = BIT_NEIGHBORS
    # Per-record dirty mask is the OR of bits whose attrs this record has
    for name, idxs in out_idx.items():
        bit = name_to_bit[name]
        for ri in idxs:
            pd.dirty_mask[ri] |= (1 << bit)
    pd.attrs = out_attrs
    pd.attr_record_idx = out_idx
    return pd


def base_crc(file_bytes: bytes) -> int:
    """Compute CRC32 of a base .3dphox file (used in delta header)."""
    return zlib.crc32(file_bytes) & 0xFFFFFFFF
