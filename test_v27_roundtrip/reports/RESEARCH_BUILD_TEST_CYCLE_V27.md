# CRYPSOID v0.27 — SH attribute debt breaker

This cycle attacks the real v0.25 loop: the honest full-attribute container got large because SH residuals were 41.89% of the file.

## Result

| Container | Size | Change |
|---|---:|---:|
| v11 VQ128 baseline | 18.24 MiB | baseline |
| v25 q8 full attribute container | 28.61 MiB | +56.86% vs v11 |
| v27 SH-VQ full attribute render container | 17.92 MiB | -37.35% vs v25 |

## What is preserved

Copied exactly from v25: tier labels, XYZ u24, DC/opacity u8, scale f16, quaternion i16.

Changed: `sh_rest_q8_global` becomes `sh_vq128_idx_u8` + `sh_vq128_codebook_i8`.

## Caveat

This is a render container, not an exact SH archive. V0.28 needs exact correction chunks and context-aware codebooks.

CRC readback: `True`.
