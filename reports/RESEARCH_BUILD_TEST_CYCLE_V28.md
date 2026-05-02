# CRYPSOID v0.28 — context-aware SH exact correction chunks

v0.27 made the full-attribute render container small by replacing the giant q8 SH stream with a VQ render core. v0.28 adds the missing truth layer: optional exact correction chunks that reconstruct the original v25 q8 SH stream.

## Result

| Container | Size | Meaning |
|---|---:|---|
| v11 VQ256 baseline | 18.24 MiB | previous practical baseline |
| v25 q8 full-attribute | 28.61 MiB | honest q8 SH container |
| v27 SH-VQ render | 17.93 MiB | small render core, not exact |
| v28 SH-VQ render | 17.93 MiB | same render core, v28 manifest |
| v28 q8-exact archive | 30.67 MiB | render core + q8-exact correction chunks |

## Correction encoding tested

- global residual stream
- per-SH-group residual streams
- per-tier/per-group residual streams
- sparse mask + nonzero values

Chosen: `per_tier_group`, because it is context-aware and keeps the correction chunks aligned with CRYPSOID tiers.

## Exactness

- CRC readback: `True`
- q8 SH reconstruction exact: `True`
- residual dtype: `int8`
- residual range: [-85, 104]

## Honest read

v0.28 gives two modes:

1. **Render mode** — small VQ SH stream, below v11 size.
2. **Archive mode** — exact q8 SH reconstruction, larger but truthful.

This is not yet the final win. The next compression target is reducing exact-correction debt with better context-conditioned codebooks or residual transforms.
