"""v34 .phoxseq -- temporal sequence container over .phoxdelta frames.

Concept
-------
A v34 `.phoxseq` is a *bag of v31 phoxdelta frames* with shared metadata so a
renderer can play back a time-varying scene (volumetric video) on top of a
single base `.3dphox`. Each frame in the sequence is a complete v31 phoxdelta
(modify-only) record, plus optional `births[]` and `deaths[]` lists for
phoxoids that pop in or out (deferred to v34.1: see header reserved bits).

Structure
---------
File layout:

    header (40 bytes)
        Magic         8 bytes   b"PHOXSEQ\\0"
        Version       1 byte    0x01
        Reserved      3 bytes   zero
        Base CRC32    4 bytes   little-endian uint32 (CRC32 of base file)
        Base N        4 bytes   little-endian uint32
        Frame count   4 bytes   little-endian uint32
        FPS milli     4 bytes   little-endian uint32 (frames per kilosecond, i.e. fps * 1000)
        Time start ms 4 bytes   little-endian int32 (signed, ms)
        Time end ms   4 bytes   little-endian int32
        Reserved      4 bytes   zero (room for births/deaths chunk offsets in v34.1)

    frame index (16 bytes per frame):
        time_offset_ms  4 bytes   int32  (relative to time_start_ms)
        flags           2 bytes   uint16 (bit 0 = compressed phoxdelta payload)
        reserved        2 bytes   zero
        offset          4 bytes   uint32 (absolute file offset of phoxdelta bytes)
        size            4 bytes   uint32 (length in bytes)

    payload region:
        Frame 0 phoxdelta bytes (uncompressed; or zlib-compressed if flag set)
        Frame 1 phoxdelta bytes
        ...

Births and deaths
-----------------
v34.0 carries them implicitly: a frame whose phoxdelta sets `opacity = 0`
effectively kills a phoxoid for that frame. v34.1 will add a separate
births[] table and a deaths[] table outside per-frame payloads — the reserved
4 bytes in the header are placeholder offsets for those.

Composition with the v31 trailer
--------------------------------
A `.phoxseq` is an *external* file -- it lives next to the base `.3dphox`
rather than as a trailer. This keeps base files free of timeline cruft and
lets multiple sequences (e.g. left-walk.phoxseq, right-walk.phoxseq) share
one base. A "v34 trailer mode" (sequence appended to the base file with
its own magic) is reserved for v34.2 if it turns out to be useful.

Acceptance gates (v34 build sign-off)
-------------------------------------
1. Round-trip: encode -> decode -> re-encode is byte-identical.
2. Frame index: each frame's `offset + size` lands inside the payload region
   and sums to file_size - header_size - frame_index_size.
3. CRC: each frame's phoxdelta CRC matches its in-frame CRC field (the
   phoxdelta itself doesn't carry one; the header carries `base_crc` only).
4. Timeline integrity: time_offset_ms strictly monotone non-decreasing.
5. Apply: applying frame K of a sequence to its base produces the same
   SplatBuffer as parsing frame K's phoxdelta directly.
6. Compose: applying frames [0..K] in order is equivalent to applying
   compose_phoxdeltas([f0, ..., fK]).
"""

from __future__ import annotations
import struct, zlib
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .phoxdelta_codec import (
    PhoxDelta, encode_phoxdelta, decode_phoxdelta, apply_phoxdelta,
    compose_phoxdeltas, base_crc as compute_base_crc,
)


SEQ_MAGIC   = b"PHOXSEQ\x00"
SEQ_VERSION = 0x01

HEADER_SIZE        = 40
FRAME_INDEX_SIZE   = 16

FLAG_COMPRESSED = 0x01


@dataclass
class PhoxSeqFrame:
    """One frame's metadata + decoded phoxdelta."""
    time_offset_ms: int
    delta: PhoxDelta
    flags: int = 0


@dataclass
class PhoxSeq:
    """A parsed .phoxseq sequence in memory."""
    base_crc: int
    base_n: int
    fps_milli: int                         # fps * 1000 (so 24.000 fps stores as 24000)
    time_start_ms: int
    time_end_ms: int
    frames: List[PhoxSeqFrame] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def fps(self) -> float:
        return self.fps_milli / 1000.0

    @property
    def duration_ms(self) -> int:
        return self.time_end_ms - self.time_start_ms


# ---------- Encode ----------

def encode_phoxseq(base_file_bytes: bytes,
                   base_n: int,
                   frames: List[PhoxSeqFrame],
                   fps: float = 24.0,
                   compress_payload: bool = True) -> bytes:
    """Encode a list of (time, phoxdelta) frames into one .phoxseq blob.

    Args:
        base_file_bytes: bytes of the base .3dphox (used to compute base_crc).
        base_n: phoxoid count of the base.
        frames: list of PhoxSeqFrame in **strictly non-decreasing time order**.
        fps: frames-per-second metadata (informational).
        compress_payload: if True, each phoxdelta payload is zlib-compressed
                          and the FLAG_COMPRESSED bit is set per frame.

    Returns:
        Encoded .phoxseq bytes.
    """
    base_crc = compute_base_crc(base_file_bytes)
    if not frames:
        raise ValueError("phoxseq must have at least one frame")
    times = [f.time_offset_ms for f in frames]
    for i in range(1, len(times)):
        if times[i] < times[i-1]:
            raise ValueError(f"phoxseq frames not monotone in time: frame[{i}].t={times[i]} < frame[{i-1}].t={times[i-1]}")
    time_start_ms = int(times[0])
    time_end_ms   = int(times[-1])
    fps_milli     = int(round(fps * 1000.0))

    # Encode each frame's payload
    payloads = []
    flags_list = []
    for f in frames:
        if f.delta.base_crc != base_crc or f.delta.base_n != base_n:
            raise ValueError(
                f"frame at t={f.time_offset_ms} has base_crc/base_n that doesn't match container"
            )
        raw = encode_phoxdelta(f.delta.base_crc, f.delta.base_n,
                               f.delta.phoxoid_ids, f.delta.attrs)
        if compress_payload:
            raw = zlib.compress(raw, level=6)
            flags_list.append(FLAG_COMPRESSED)
        else:
            flags_list.append(0)
        payloads.append(raw)

    # Build header
    header = bytearray()
    header += SEQ_MAGIC
    header += bytes([SEQ_VERSION, 0, 0, 0])
    header += struct.pack('<I', base_crc)
    header += struct.pack('<I', base_n)
    header += struct.pack('<I', len(frames))
    header += struct.pack('<I', fps_milli)
    header += struct.pack('<i', time_start_ms)
    header += struct.pack('<i', time_end_ms)
    header += b'\x00\x00\x00\x00'                # reserved
    assert len(header) == HEADER_SIZE

    # Frame index — payloads start right after header + frame index
    payload_start = HEADER_SIZE + FRAME_INDEX_SIZE * len(frames)
    cursor = payload_start
    index = bytearray()
    for f, p, flags in zip(frames, payloads, flags_list):
        index += struct.pack('<i', f.time_offset_ms)
        index += struct.pack('<H', flags)
        index += b'\x00\x00'                     # reserved
        index += struct.pack('<I', cursor)
        index += struct.pack('<I', len(p))
        cursor += len(p)
    assert len(index) == FRAME_INDEX_SIZE * len(frames)

    return bytes(header) + bytes(index) + b''.join(payloads)


# ---------- Decode ----------

def decode_phoxseq(data: bytes) -> PhoxSeq:
    """Parse a .phoxseq blob."""
    if len(data) < HEADER_SIZE:
        raise ValueError(f"phoxseq too short: {len(data)} bytes")
    if data[:8] != SEQ_MAGIC:
        raise ValueError(f"phoxseq bad magic: {data[:8]!r}")
    version = data[8]
    if version != SEQ_VERSION:
        raise ValueError(f"unsupported phoxseq version 0x{version:02x}")
    base_crc      = struct.unpack('<I', data[12:16])[0]
    base_n        = struct.unpack('<I', data[16:20])[0]
    frame_count   = struct.unpack('<I', data[20:24])[0]
    fps_milli     = struct.unpack('<I', data[24:28])[0]
    time_start_ms = struct.unpack('<i', data[28:32])[0]
    time_end_ms   = struct.unpack('<i', data[32:36])[0]

    seq = PhoxSeq(
        base_crc=base_crc, base_n=base_n,
        fps_milli=fps_milli,
        time_start_ms=time_start_ms, time_end_ms=time_end_ms,
        frames=[],
    )

    p = HEADER_SIZE
    for i in range(frame_count):
        t      = struct.unpack('<i', data[p:p+4])[0]
        flags  = struct.unpack('<H', data[p+4:p+6])[0]
        offset = struct.unpack('<I', data[p+8:p+12])[0]
        size   = struct.unpack('<I', data[p+12:p+16])[0]
        if offset + size > len(data):
            raise ValueError(f"frame {i} payload out of range: offset={offset} size={size} len={len(data)}")
        payload = data[offset:offset+size]
        if flags & FLAG_COMPRESSED:
            payload = zlib.decompress(payload)
        delta = decode_phoxdelta(payload)
        seq.frames.append(PhoxSeqFrame(time_offset_ms=t, delta=delta, flags=flags))
        p += FRAME_INDEX_SIZE

    return seq


# ---------- Apply ----------

def apply_phoxseq_at_time(splat_buffer, seq: PhoxSeq, t_ms: int):
    """Apply all frames with time_offset_ms <= t_ms to the base, in order.

    Returns a new SplatBuffer (does not mutate input).
    Effectively: compose all frames up to t, then apply once.
    """
    relevant = [f for f in seq.frames if f.time_offset_ms <= t_ms]
    if not relevant:
        from copy import copy as shallow_copy
        return shallow_copy(splat_buffer)
    composed = compose_phoxdeltas([f.delta for f in relevant])
    return apply_phoxdelta(splat_buffer, composed, copy=True)


def apply_phoxseq_frame(splat_buffer, seq: PhoxSeq, frame_idx: int):
    """Apply a single frame's phoxdelta (not cumulative)."""
    f = seq.frames[frame_idx]
    return apply_phoxdelta(splat_buffer, f.delta, copy=True)
