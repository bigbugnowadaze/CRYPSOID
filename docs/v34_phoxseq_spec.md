# v34 — `.phoxseq` Temporal Sequence Spec (one-pager)

**Status:** implemented + acceptance-tested, 2026-05-02.
**Goal:** time-varying scenes (volumetric video, animated bloom, particle bursts) on top of a single static `.3dphox` base.
**Depends on:** v31 Addition 3 — `.phoxdelta` patch format.

## Why v34

A single `.phoxdelta` is one-shot: "given base B, here's diff D → state B+D". For animation we need to play a *sequence* of states without re-encoding the whole base each frame. v34 wraps a list of `.phoxdelta` frames in a container with shared metadata and an O(1) seek index.

This unlocks:
- **Volumetric video** played at native fps over a CRYPSOID base.
- **Animated material flips** (e.g. dim halo for 100 ms, then restore).
- **Particle bursts** modeled as opacity-modulating frames.
- **Live editing** — viewer can scrub a timeline, jump to a frame.

## Non-goals (explicitly deferred to v34.1+)

- Births and deaths (phoxoids that pop in/out). v34.0 carries deaths *implicitly* (a frame setting opacity = 0 is "dead for this frame"); explicit births/deaths chunks land in v34.1.
- Inter-frame compression (delta-of-delta). Each frame is a self-contained `.phoxdelta`. v34.2 may add temporal prediction.
- Audio. Sequence is geometry-only; pair with separate WAV/OGG by convention.
- A "v34 trailer" mode (sequence appended to base file). v34 ships as an external `.phoxseq` file. v34.2 may add the trailer mode.

## File structure

```
header (40 bytes)
    Magic         8 bytes   b"PHOXSEQ\0"
    Version       1 byte    0x01
    Reserved      3 bytes   zero
    Base CRC32    4 bytes   little-endian uint32 (CRC32 of base file)
    Base N        4 bytes   little-endian uint32
    Frame count   4 bytes   little-endian uint32
    FPS milli     4 bytes   little-endian uint32 (fps * 1000)
    Time start ms 4 bytes   little-endian int32
    Time end ms   4 bytes   little-endian int32
    Reserved      4 bytes   zero (placeholder for v34.1 births/deaths offsets)

frame index (16 bytes per frame):
    time_offset_ms   4 bytes   int32
    flags            2 bytes   uint16   bit 0 = compressed phoxdelta payload
    reserved         2 bytes   zero
    offset           4 bytes   uint32   absolute file offset of payload
    size             4 bytes   uint32

payload region:
    Frame 0 phoxdelta bytes (uncompressed; or zlib-compressed if flag set)
    Frame 1 phoxdelta bytes
    ...
```

## Cost on the Audi (763,800 phoxoids, 24-frame 1-second halo bloom)

| File | Bytes | Per-frame avg | vs base v40 |
|---|---:|---:|---:|
| Base (`v40_audi_full_mipfilled.3dphox`) | 52,023,157 | — | 100% |
| `v34_audi_halo_bloom.phoxseq` (24f, 10k splats/frame, opacity-only) | 904,024 | 37,668 B | +1.7% |

**Frame compression:** opacity-only deltas with zlib achieve 5–6× compression on structured payloads (per acceptance test gate 6).

## API surface

```python
from crypsorender.io.phoxseq_codec import (
    PhoxSeq, PhoxSeqFrame,
    encode_phoxseq, decode_phoxseq,
    apply_phoxseq_frame,        # apply just one frame's phoxdelta
    apply_phoxseq_at_time,      # cumulative apply up to t_ms
)

# Build:
frames = [PhoxSeqFrame(time_offset_ms=t, delta=phoxdelta_for_t) for t in timestamps]
raw = encode_phoxseq(base_bytes, base_n, frames, fps=24.0)

# Read:
seq = decode_phoxseq(raw)
sb_at_t = apply_phoxseq_at_time(base_splat_buffer, seq, 500)   # state at t=500ms
```

## Acceptance gates (all PASS, 2026-05-02)

1. **Round-trip byte-identical** — encode → decode → re-encode produces same bytes.
2. **Frame index integrity** — offsets are monotone, payloads non-overlapping, last frame ends at file end.
3. **Timeline monotone** — encoder rejects frame lists with decreasing time.
4. **Single-frame apply** — `apply_phoxseq_frame(sb, seq, k)` matches manually applying that frame's phoxdelta.
5. **Compose equivalence** — cumulative `apply_phoxseq_at_time(sb, seq, t_end)` matches `compose_phoxdeltas([all frames]).apply(sb)`.
6. **Compression payoff** — zlib-compressed payload < uncompressed for structured (e.g. opacity-only) frames.

## Backward compatibility

- `.phoxseq` is an *external* file — base `.3dphox` is unchanged. Any v25/v28/v31/v33/v40 reader ignores it (it's a sibling file, not a trailer).
- A renderer that doesn't know v34 simply doesn't load the sequence; the base scene plays as a still.
- v31 `.phoxdelta` codec is unchanged; v34 wraps existing v31 phoxdelta bytes verbatim.

## Files added by v34

```
tools/crypsorender/io/phoxseq_codec.py   — codec + apply functions
tools/test_phoxseq_codec.py              — 6-gate acceptance test (all PASS)
tools/build_v34_audi_demo.py             — sample halo-bloom .phoxseq builder
outputs/v34_audi_halo_bloom.phoxseq      — 904 KB demo over Audi
```

## Honest scope summary

| Claim | True / partial / aspirational |
|---|---|
| Plays a sequence of phoxdeltas with O(1) seek | **True** — frame index gives offset+size for each frame |
| External file — base `.3dphox` not modified | **True** — sibling-file convention |
| Births/deaths supported | **Partial** — implicit (opacity=0); explicit lists deferred to v34.1 |
| Inter-frame compression | **Partial** — per-frame zlib only; temporal prediction is v34.2 |
| Renderer integration in viewer | **Aspirational** — codec works; viewer wire-up is a separate task |
| Required for any compelling future work | **No** — it's an enabling feature, not a critical path |
