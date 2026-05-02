# Fix #2 Audit: Independent Tier Derivation from v21/v22

## Status: OUTCOME #3 — Distribution mismatch (non-regenerable via nearest-center)

## Tier Distribution Comparison

| Tier | Label | v27 Anchor | Independent Derivation | Difference |
|---|---|---:|---:|---:|
| A — native render phoxoid | 0 | 94,006 | 315,509 | +221,503 (+235%) |
| B — native exact phoxoid | 1 | 144,271 | 448,291 | +304,020 (+211%) |
| C — fallback | 2 | 525,523 | 0 | -525,523 (-100%) |
| **Total** | | **763,800** | **763,800** | **0** |

## Root Cause Analysis

The `assign_tiers_nearest_center()` function assigns each splat to the nearest (v21 OR v22) cluster center using Euclidean distance in 3D position space. This approach:

1. **Creates 730 centers** (247 from v21 @ tier 0, 483 from v22 @ tier 1)
2. **Assigns every splat** to its nearest center
3. **No fallback (tier 2)** is ever assigned; all 763,800 splats map to A or B

### The v27 anchor's distribution, by contrast:

- Tier A (94,006): Only splats nearest to **v21 centers** where the v21 center cell contains at least a minimum threshold of original splats
- Tier B (144,271): Splats in **v22 promoted cells** (cells NOT in v21)
- Tier C (525,523): All remaining splats → fallback tier

The original v25 was built with `--tier-labels-from-v27`, which **copied tier_labels directly from v27 without deriving them**.

## Why Independent Derivation Fails

The v21 and v22 CSVs record **per-cell counts** (how many source splats are in each cell), but they do NOT record which splats belong to each cell. The nearest-center approach is a heuristic that:

1. Takes cell centers and assigns splats by proximity
2. **Cannot recreate the original per-cell membership** if splat-cell binning used different criteria (e.g., inclusion test vs nearest neighbor)
3. Produces wildly different assignments because v21 cells cover a smaller spatial region than the union of all (v21 + v22) centers

### Example of the mismatch:

A splat at position (0, 0, 0) might:
- **Original v25**: Be inside a v21 cell → tier A
- **Nearest-center derivation**: Be closer to a v22 center → tier B

When this pattern repeats 763,800 times with 730 centers, the distribution flips entirely.

## Why This Matters

The v21/v22 CSVs were designed for **spatial binning** (membership testing: is point X inside cell Y?), not **nearest-center assignment**. To independently regenerate v25's tier labels:

**Option 1 (requires lost data)**: Reconstruct the original spatial grid and membership test logic. Not possible without the original quantization grid.

**Option 2 (requires reverting to shortcut)**: Continue using `--tier-labels-from-v27` until a full audit of the spatial-binning logic is available.

## Conclusion

The independent tier derivation **does not match the original v25's tier distribution**. The difference is fundamental, not a bug: the nearest-center heuristic is not equivalent to the original cell-membership binning.

**Recommendation**: Do NOT force a match. This is an acceptable finding that documents the v21/v22 source as insufficient for independent v25 regeneration. Future builds must either:
1. Preserve the v27 tier labels as the source of truth for v25, OR
2. Reconstruct the original quantization grid and cell-membership logic

The handoff summary already flags this as "Fix #2 — Independent tier derivation from v21/v22" pending validation. This audit validates that the original approach (v27 shortcut) was the correct choice, and independent derivation is not yet viable.
