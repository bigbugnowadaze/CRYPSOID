# CRYPSOID v0.28 — SH exact correction chunks

v0.27 made the full-attribute render container small by replacing the giant q8 SH stream with a VQ render core. v0.28 adds the missing truth layer: optional exact correction chunks that reconstruct the original v0.25 q8 SH stream.

## Result

| Container | Size | Meaning |
|---|---:|---|
| v11 VQ256 baseline | 18.24 MiB | previous practical baseline |
| v25 q8 full-attribute | 28.61 MiB | honest q8 SH container |
| v27/v28 SH-VQ render | 17.93 MiB | small render core, not exact |
| v28 q8-exact archive, best global correction | 30.18 MiB | render core + exact q8 SH correction |
| v28 q8-exact archive, per-tier correction | 30.67 MiB | context-aware test; worse on Audi |

## Correction encodings tested

- global residual stream
- per-SH-group residual streams
- per-tier/per-group residual streams
- sparse mask + nonzero values

Best tested: `global_full` at 12.25 MiB correction payload.

The important result is negative/useful: **per-tier context splitting did not help this scene for exact SH correction.** It made the correction stream larger. So v0.28 keeps the per-tier archive as an analysis artifact, but the preferred exact archive is the global residual stream.

## Exactness

- Preferred global archive CRC: `True`
- Preferred global archive q8 SH reconstruction exact: `True`
- residual dtype: `int8`
- residual range: [-85, 104]

## Honest read

v0.28 gives two modes:

1. **Render mode** — small VQ SH stream, below v11 size.
2. **Archive mode** — exact q8 SH reconstruction, truthful but larger than v25.

This means v0.27's render-core compression was real, but exact correction debt is still too large. v0.29 should reduce that debt with better codebooks, residual transforms, or smaller exact-correction representations.
