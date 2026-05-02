# v31 — Graph Extension Spec (one-pager)

**Status:** draft for sign-off, 2026-05-01.
**Goal:** ship the smallest set of additions that turns CRYPSOID from "a splat codec" into "the explicit-math universal scene representation," per the strategy in `questions for claude.md`.

## Why v31 (not v30 or v40)

Three additions move the project the furthest with the least cost:

1. **Explicit normal + tangent frame per phoxoid** — gives every blob a real surface orientation, not just an implicit one inside its covariance. Unlocks normal residuals, relighting, and the bridge from "Gaussian splat" to "oriented surface element." ~4 bytes/phoxoid.
2. **kNN edges as a first-class chunk** — turns the implicit neighbor graph (already used at fit-time and discarded) into a stored, addressable graph. Unlocks LOD selection, neighborhood-aware deltas, and the "geometric attention" framing. ~12 bytes/phoxoid.
3. **`.phoxdelta` patch format** — small, low-rank updates that compose against a base `.3dphox`. Differentiates us from every static-format competitor; enables continual capture, scene edits, and A/B variants without re-encoding. Sparse: cost scales with what actually changed.

Together: ~16 bytes/phoxoid added to the `.3dphox` (≈12 MB on the Audi at 763k blobs, well under the v28-archive's 30.7 MB), plus a brand-new sidecar file format for deltas.

## Non-goals (explicitly out of scope)

- TSDF fusion, Poisson reconstruction, voxel grids — discussed in the strategy doc, deferred.
- Image/video → `.3dphox` compiler (the COLMAP-equivalent step) — needs v31 to land first; comes in v32+.
- Confidence/provenance per phoxoid (continuous scalar, source-view back-pointers) — useful for the eventual compiler, premature now.
- Scene-class template libraries — premature without more PhoxBench scene diversity.

## Addition 1 — Explicit normal + tangent frame

### What's stored
For each of the N phoxoids, append:

| Field | Bytes | Encoding | Notes |
|---|---:|---|---|
| `normal_oct`     | 3 | octahedral, 24-bit | unit normal direction; ~0.005 rad precision |
| `tangent_angle`  | 1 | u8, 0–255 → 0–2π   | rotation of tangent frame around the normal; ~1.4° precision |

**Total: 4 bytes/phoxoid.** On the Audi (763,800 blobs): 2.99 MB.

### How it's derived
- For PLY → `.3dphox` ingest: per-blob, take the 8-NN neighborhood, fit a local quadric (MLS-style, 5-coef Pearcey basis we already have), normal is the unit vector ⊥ the fit plane.
- For trained 3DGS PLYs (no normals): same MLS fit. The germ fitter already does this; we just save the normal it computes instead of throwing it away.
- Tangent angle: pick the principal-curvature direction in the tangent plane; encode the angle from a deterministic reference (e.g., the projection of world-up).

### Chunk layout
New chunk type in `.3dphox`: `normals` (chunk_id `0x12`).
- 1 byte version (currently `0x01`)
- 4 bytes `count` (must equal N)
- N × 4 bytes payload
- CRC32 over payload

### Why this matters
- Normal residuals can now be encoded as deltas against the per-phoxoid normal, not against world-coordinate normals. Much smaller residuals.
- Phoxoid relighting is suddenly possible (re-shade with new lighting; today the SH coefficients bake in a fixed radiance).
- Bridges the surfel ancestry: a v31 phoxoid is a *strict* superset of a surfel, plus the Pearcey germ.

## Addition 2 — kNN edges chunk

### What's stored
For each phoxoid, store its k=4 nearest-neighbor indices. Indices are u32 (we have splat counts up to ~10⁷; u32 is comfortable).

| Field | Bytes | Encoding | Notes |
|---|---:|---|---|
| `neighbors[4]` | 16 | 4 × u32 | indices into the same `.3dphox`'s phoxoid array |

Optional companion fields (future, behind a feature flag):
- `weights[4]`: 4 × u8, edge similarity (0–255)

**Total: 16 bytes/phoxoid** (no weights). On the Audi: 11.96 MB.

### How it's derived
- Compute kNN over xyz at ingest time using sklearn's BallTree (already a dep). Distance metric: euclidean.
- Self-edges excluded.
- Sort each blob's neighbor list by distance ascending; ties broken by index.

### Chunk layout
New chunk type: `graph_knn_edges` (chunk_id `0x13`).
- 1 byte version
- 1 byte `k` (currently `0x04`)
- 4 bytes `count`
- N × k × 4 bytes payload
- CRC32

### Why this matters
- LOD selection can prune by neighbor agreement (drop blobs whose neighbors carry equivalent radiance).
- Delta patches can be expressed *over edges* — "this group of 12 connected phoxoids shifted together" — which is much smaller than naming each one.
- The "geometric attention" framing becomes literal: the renderer or a downstream consumer can ask each phoxoid "who do you listen to?" and walk the graph.

## Addition 3 — `.phoxdelta` patch format

### What it is
A standalone sidecar file that references a base `.3dphox` by hash, and encodes sparse per-phoxoid changes. Composition is just "load base, apply delta in order."

### File header
| Field | Bytes | Notes |
|---|---:|---|
| Magic   | 8 | `b"PHOXDLT\0"` |
| Version | 1 | `0x01` |
| Base CRC32 | 4 | CRC32 of the base `.3dphox` file (sanity check) |
| Base N | 4 | u32, expected phoxoid count in base |
| Delta count | 4 | u32, number of changed phoxoids in this patch |
| Reserved | 3 | zero |

### Per-changed-phoxoid record
| Field | Bytes | Notes |
|---|---:|---|
| `phoxoid_id` | 4 | u32, index into base file |
| `dirty_mask` | 2 | u16, bitfield: which attributes are changed |
| Payload | variable | only the changed attributes, in fixed order |

`dirty_mask` bits (low to high):
- bit 0: xyz
- bit 1: scale
- bit 2: quaternion
- bit 3: opacity
- bit 4: f_dc (DC SH)
- bit 5: f_rest (higher SH)
- bit 6: tier_label
- bit 7: germ coefficients (5-coef Pearcey)
- bit 8: normal + tangent
- bit 9: kNN neighbors
- bits 10–15: reserved

If a bit is set, the corresponding attribute (with its standard v31 encoding) appears in the payload, in bit order.

### Operations
- **Apply:** stream the delta records, look up each `phoxoid_id` in the base, overwrite the named attributes.
- **Compose two deltas:** later wins per (phoxoid_id, attribute).
- **Insert** (new phoxoid not in base): reserved for v32; v31 deltas are *modify-only*. Adding/removing blobs requires a re-encode.

### Why this matters
- Continual capture: scan a room, ship a 50 MB `.3dphox`. Re-scan after furniture moves, ship a 200 KB `.phoxdelta` instead of 50 MB.
- Scene edits: artist relights a region → one delta. CRYPSOID viewer can compose `base + lighting_v1.phoxdelta + furniture_moved.phoxdelta` live.
- A/B variants for benchmarking: same base, different deltas, identical pixel paths — perfect for ablations.
- This is the explicit-math answer to LoRA-for-NeRF. We get the same "small patch over a heavy base" property, and ours is *readable* (you can `xxd` a `.phoxdelta` and understand what changed).

## Acceptance gates (v31 build sign-off)

1. `.3dphox` v31 reader/writer round-trips: write → read → byte-identical re-encode.
2. Normal chunk: per-blob unit-norm assertion; angular error vs. true normal on synthetic sphere ≤ 0.01 rad.
3. kNN chunk: each blob's neighbor list is sorted by distance ascending; no self-edges; queryable in O(1).
4. `.phoxdelta` apply: delta + base ⟹ same bytes as a fresh `.3dphox` rebuilt from the deltaed state.
5. Bit-exactness with v28: every existing v28 chunk in v31 is byte-identical (v31 only adds chunks; doesn't modify any v28 chunk).
6. CI: a v31 smoke test in `.github/workflows/test.yml` builds a v31 file, applies a 3-blob `.phoxdelta`, asserts result.

## Estimated cost on the Audi A5 (763,800 phoxoids)

| Addition | Per-blob bytes | Total bytes | vs v28 archive (32,162,548 B) |
|---|---:|---:|---:|
| Normal + tangent frame | 4 | 3,055,200 | +9.5% |
| kNN edges (k=4) | 16 | 12,220,800 | +38.0% |
| Both | 20 | 15,276,000 | +47.5% |
| **v31 archive estimate** |  | **~47,438,548** | (vs v28 archive) |

Bits-per-Gaussian estimate: ~497 bpg (vs v28's 337 bpg). That's a regression on bitrate alone — but it buys the structural extensions that justify the project's existence. The honest framing in the README would shift from "1.40× smaller than zstd-12 PLY" to "1.40× smaller AND ships the graph + normal infrastructure that nothing else in the splat ecosystem has."

## Phasing

This spec is for sign-off, not implementation. After sign-off, suggested order:
1. Add normal chunk (Addition 1) — smallest, lowest risk, immediate value.
2. Wire normal-aware shading into the renderer (optional flag) — proves the chunk pays for itself.
3. Add kNN chunk (Addition 2).
4. Build `.phoxdelta` reader/writer + acceptance test 4.
5. Bump CI gates; re-run PhoxBench Tier 0/1 (no regression expected — all existing chunks unchanged).

Each step is a separate phased reviewable artifact, per project convention.
