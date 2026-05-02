# CRYPSOID Research/Build/Test Cycle v0.21

**v0.21 = actual hash-context residual container.**

This cycle converts the v0.20 hash-context audit into written CRYPSOID sidecar containers with context IDs, precision tiers, per-chunk headers, residual payload bytes, and optional exact-correction accounting. It does not replace the renderer yet.

## Audi A5 result

| Metric | v0.20 audit | v0.21 actual container |
|---|---:|---:|
| Accepted chunks | 247 | 247 |
| Covered splats | 94,006 | 94006 |
| Covered percent | 12.31% | 12.31% |
| Residual payload bytes | 880,815 estimated | 880815 written |
| Render sidecar container | estimated overhead only | 938750 bytes |
| Hybrid size including fallback | 16.85 MiB estimated | 16.89 MiB actual-accounted |
| Delta vs v11 VQ256 | -7.63% | -7.40% |
| Ratio vs source PLY | 10.21x | 10.18x |
| Reduction vs source PLY | 90.20% | 90.18% |
| Optional archive/exact total | 17.20 MiB estimated | 16.89 MiB actual-accounted |

## What changed

- Added `CRYPSOID_3DPHOX_HASH_CONTEXT_RESIDUAL_V21` sidecar container.
- Added per-chunk context IDs, context class, precision tier, splat count, payload length, and exact-correction byte accounting.
- Added companion `CRYPSOID_3DPHOX_HASH_CONTEXT_ARCHIVE_EXACT_V21` exact-correction container.
- Added readback sanity: magic, manifest length, chunk count, and payload count verified during build.

## Honest status

This is stronger than v0.20 because bytes are actually written into a CRYPSOID container. It is still not a production arithmetic/rANS implementation; payloads are range-coded proxy streams sized from v0.20 entropy estimates. The next cycle should replace the proxy stream with a simple real bitpacker/rANS-like coder or move to the hybrid decoder/renderer.

## Next

**v0.22 = hybrid decoder: phoxoid residual sidecar + fallback VQ splats -> splat-compatible preview scene.**
