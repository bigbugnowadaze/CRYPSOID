# Tier 1.5 — compression baselines

**Test corpus:** Audi A5 PLY, 763,800 splats, 180,258,277 bytes raw.

## Results obtained before sandbox disk exhaustion

All numbers are real; runs that hit timeouts or disk-full are marked.

| Format | Size (bytes) | MiB | Ratio vs raw PLY | % of raw PLY | Lossy? | Notes |
|---|---:|---:|---:|---:|---|---|
| **raw PLY** | 180,258,277 | 171.91 | 1.00× | 100.0% | n/a | baseline |
| gzip -1 | 71,445,344 | 68.14 | 2.52× | 39.6% | no | fast |
| gzip -6 | 66,407,423 | 63.33 | 2.71× | 36.8% | no | default |
| gzip -9 | 66,348,062 | 63.27 | 2.72× | 36.8% | no | max |
| zstd -1 | 96,237,893 | 91.78 | 1.87× | 53.4% | no | (oddly worse than gzip at -1; deflate beats lz4-style at low levels) |
| zstd -3 | 60,249,067 | 57.46 | 2.99× | 33.4% | no | default |
| zstd -6 | 51,454,027 | 49.07 | 3.50× | 28.5% | no | |
| zstd -9 | 49,432,999 | 47.14 | 3.65× | 27.4% | no | |
| zstd -12 | 44,917,382 | 42.84 | **4.01×** | 24.9% | no | **best lossless baseline measured** |
| zstd -15 | timeout | — | — | — | no | killed at 35s wall-clock; disk-limited |
| zstd -19, -22 | timeout | — | — | — | no | same |
| xz -1, -6 | timeout | — | — | — | no | (sandbox per-call cap) |
| .npz (compressed) | not measured | — | — | — | no | disk-full before write |
| .npz (raw) | not measured | — | — | — | no | same |
| Draco | not installed | — | — | — | yes | (would need install + Python bindings) |
| | | | | | | |
| **CRYPSOID v25 attribute-group** | 29,998,397 | 28.61 | **6.01×** | 16.6% | yes (q8) | |
| **CRYPSOID v27 SH-VQ render** | 18,796,089 | 17.93 | **9.59×** | 10.4% | yes (VQ + q8) | |
| **CRYPSOID v28 SH-VQ render** | 18,795,838 | 17.93 | **9.59×** | 10.4% | yes (VQ + q8) | |
| **CRYPSOID v28 q8-EXACT archive** | 32,162,548 | 30.67 | **5.60×** | 17.8% | exact-at-q8 grid | |
| **CRYPSOID v29 best residual archive** | 31,615,009 | 30.15 | **5.70×** | 17.5% | exact-at-q8 grid | |

## Honest read

**Against the most aggressive lossless baseline we measured (zstd -12 at 4.01×):**
- v28 EXACT archive: 32.16 MB vs 44.92 MB → **1.40× smaller than zstd-12, at the same q8 fidelity.**
- v28 VQ render: 18.80 MB vs 44.92 MB → **2.39× smaller than zstd-12, with measurable VQ quality loss (PSNR 38.76 dB vs PLY).**
- v29 best residual: 31.62 MB vs 44.92 MB → **1.42× smaller than zstd-12.**

**Against the original "5.6× / 9.6× vs raw PLY" headline:**
- Those ratios are real but vs the most bloated baseline. Against a tuned zstd, the EXACT archive's win shrinks from 5.60× to **1.40×**.
- The lossy VQ render's 9.59× vs raw PLY shrinks to **2.39× vs zstd-12**, and that's not even a fair comparison because zstd-12 is lossless and VQ render is not.

## What this means for project messaging

The honest claim is:
> "CRYPSOID's v28 EXACT archive is ~1.4× smaller than tuned zstd PLY at the same q8 quantization grid. The v28 VQ render is ~2.4× smaller than zstd PLY but with lossy SH approximation (~5% per-coefficient RMSE; PSNR vs PLY 38.76 dB)."

This is still a real, meaningful win — but it is NOT 5.6×/9.6×. The headline number depends on which baseline you compare against, and the bloat of raw PLY dominates the cheaper claim.

The architectural claims (tier-aware dispatch, phoxoidal density, no GPU dependency, random access, decoder simplicity) are orthogonal to the size win and stand on their own.

## Items not yet measured

1. **zstd -15, -19, -22.** Each timed out in the sandbox (>45s). Likely results based on extrapolation from -9 → -12 trend: zstd -19 might reach ~38–40 MiB (4.5×). Would shrink CRYPSOID's win further.
2. **xz / lzma.** Would likely beat zstd at high levels but slower. Not measured.
3. **`.npz` (numpy savez_compressed).** Almost certainly very close to gzip-6 since it's just deflate inside a zip.
4. **Draco.** Specifically designed for 3D meshes; unclear how well it does on Gaussian splat point clouds (no triangle topology). Would need install + bindings.
5. **SOG / SOGS / Self-Organising Gaussians.** Reference results from the actual splat-compression literature. Would be the most rigorous comparison.

## Sandbox-state note

This run was interrupted by the sandbox's `/etc/srt-settings` filling up — bash subsequent calls fail with `ENOSPC` until the sandbox is reclaimed. The above numbers are from runs that completed before that point. Re-running with extra time and fresh disk would unlock zstd -19, xz -9, and the `.npz` variants, but I do not expect any of those to dramatically change the headline (they'd shrink CRYPSOID's compression edge further, not flip the qualitative story).
