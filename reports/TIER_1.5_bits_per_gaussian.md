# Tier 1.5 — bits per Gaussian (bpg) normalization

**N = 763,800 Gaussians.** bpg = 8 × file_size_bytes / N.

Standard 3DGS-paper normalization. Lets compression be compared across scenes of different splat counts.

## Reference numbers

The raw cost per Gaussian if you store every attribute as float32 (no compression):
- xyz (3) + scale (3) + rot (4) + opa (1) + f_dc (3) + f_rest (45) = 59 floats = 236 bytes = **1,888 bits/Gaussian** uncompressed.

The PLY file stores exactly that plus a small ASCII header, so PLY is essentially the float32 ceiling.

## Measured bits/Gaussian

| Format | Size (bytes) | bits/Gaussian | Notes |
|---|---:|---:|---|
| Theoretical max (59 × float32) | — | 1,888 | upper bound |
| **raw PLY** | 180,258,277 | **1,888.4** | matches the ceiling (header overhead negligible) |
| gzip -9 | 66,348,062 | **694.8** | 36.8× of raw |
| zstd -12 | 44,917,382 | **470.4** | best lossless baseline |
| zstd -19 (extrapolated) | ~38–40 MiB | ~410–425 | not measured (sandbox timeout) |
| | | | |
| **CRYPSOID v25 attribute-group** | 29,998,397 | **314.2** | q8 SH + u24 xyz + f16 scale etc. |
| **CRYPSOID v27 SH-VQ render** | 18,796,089 | **196.9** | SH replaced by 128-codebook VQ |
| **CRYPSOID v28 SH-VQ render** | 18,795,838 | **196.9** | identical to v27 in size |
| **CRYPSOID v28 q8-EXACT archive** | 32,162,548 | **336.9** | bit-exact to v25 q8 grid; ~1.40× better than zstd-12 |
| **CRYPSOID v29 best residual archive** | 31,615,009 | **331.2** | morton-splat-major + zlib9 |

## Reading the table

The fair comparison rows are:

| Comparison | bpg | Result |
|---|---:|---|
| zstd-12 PLY (lossless) | 470.4 | baseline |
| **CRYPSOID v28 EXACT archive** (bit-exact at q8 grid) | **336.9** | **1.40× smaller** |
| **CRYPSOID v28 VQ render** (lossy SH, 5% RMSE per coeff) | **196.9** | **2.39× smaller**, but at lower fidelity |

For context, recent published splat-compression work hits roughly:
- Self-Organizing Gaussians (SOG) ≈ 50–80 bpg
- Compact 3D Gaussian Splatting ≈ 60–100 bpg
- HAC ≈ 30–60 bpg

So **CRYPSOID at 197–337 bpg is meaningful versus zstd PLY but is not yet competitive with state-of-the-art splat-specific compression.** The win-vs-zstd is real; the loss-vs-SOTA is also real. The honest framing: CRYPSOID is a tier-aware structural alternative, not a SOTA bitrate compressor.

## Where the bits go (v28 EXACT archive)

| Chunk | Compressed bytes | bits/Gaussian (this chunk) | % of archive |
|---|---:|---:|---:|
| tier_labels_u8 | 3,754 | 0.04 | 0.01% |
| xyz_u24_fixed | 6,242,122 | 65.4 | 19.4% |
| dc_rgb_opacity_u8 | 2,547,608 | 26.7 | 7.9% |
| scale_f16 | 2,882,949 | 30.2 | 9.0% |
| quat_i16_norm4 | 5,755,963 | 60.3 | 17.9% |
| sh_vq128_idx_u8 | 1,356,120 | 14.2 | 4.2% |
| sh_vq128_codebook_i8 | 3,920 | 0.04 | 0.01% |
| sh_exact_residual_t*_g* (9 chunks) | 13,360,400 | 139.9 | 41.5% |
| (manifest + framing overhead) | ~10,000 | 0.10 | <0.1% |
| **Total** | 32,162,548 | **336.9** | 100% |

**The exact-correction residuals are 41.5% of the archive's bits.** That's the price of bit-exactness over the VQ render core. v29's residual sweep won 0.24% on this; HAC-style learned correction codecs would likely win significantly more.

## Honest takeaway

- **CRYPSOID v28 is not yet SOTA for splat compression** — it's 3-7× larger than published splat-specific codecs.
- **CRYPSOID v28 IS meaningfully better than zstd PLY** at the same q8 fidelity (~1.4× smaller at the EXACT level, ~2.4× smaller at the VQ-render level).
- **The bit-exactness is verified** (item 1.5.1) and the architecture is novel; the bitrate just hasn't been the focus of project work yet.
- The v29 residual-codec sweep already showed marginal returns (0.24% improvement). The big remaining bitrate wins are in (a) cusp-aware germ basis from Tier 2 (might let SH coefficients be sparser), (b) entropy-coded VQ indices instead of raw u8, (c) a learned arithmetic coder over the residuals.
