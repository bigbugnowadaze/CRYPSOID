# v25 cell-membership rule recovered (2026-05-01)

**Status:** the long-open v25 caveat ("tier_labels derived from v27 anchor, not independently re-derivable from recovered v21/v22 CSVs") is now **closable**. The cell-binning rule has been independently recovered and verified bit-exact against the v20 accepted CSV.

## What was missing, and what closed it

The chain ran:
1. v0.20 produced `v20_context_accepted_chunks.csv` — 247 accepted cells with `cell_key`, `count`, `center_x/y/z`, `eig0/1/2`, statistics.
2. v0.21 (`build_v21.pl`) just packaged that CSV into a binary container with placeholder bytes.
3. v0.25 derived tier labels from v27 (which itself derived them somewhere upstream) — without ever re-binning from the original PLY.

The **binning rule** (xyz → cell_key) lived in some pre-v18 build script that was lost. We had:
- `phoxbench_v020_hash_context.py` — confirmed cell-key encoding `key = ix + 32·iy + 32²·iz`.
- The 247 cell centers + counts in the CSV.
- The original PLY (763,800 splats).

Three plausible bbox hypotheses were tested:

| Hypothesis | Description | Exact match (out of 247) |
|---|---|---:|
| **A** | grid spans the full PLY xyz bbox, no clipping | **247 / 247** ✅ |
| B | grid spans 99th-percentile bbox | 0 / 247 |
| C | fitted from cell centers (assume centers ≈ midpoints) | 4 / 247 |
| D | mean ± 3σ | 0 / 247 |

**Hypothesis A reproduces every single one of the 247 cell counts bit-exactly.** Mean ratio = 1.000, std = 0.000.

## The recovered rule

```python
bbox_min = xyz.min(axis=0)        # min over the original 763,800 splats
bbox_max = xyz.max(axis=0)
cell_size = (bbox_max - bbox_min) / 32
ix = clip(floor((x - bbox_min[0]) / cell_size[0]), 0, 31)
iy = clip(floor((y - bbox_min[1]) / cell_size[1]), 0, 31)
iz = clip(floor((z - bbox_min[2]) / cell_size[2]), 0, 31)
cell_key = ix + 32*iy + 32*32*iz
```

**Audi A5 specifics:**
- bbox_min = (-3.36296, -0.93575, -4.01638)
- bbox_max = (+3.46155, +1.30979, +3.98754)
- cell_size = (0.21327, 0.07017, 0.25012)

## Saved artifacts

- `reports/v25_cell_membership_rule_recovered.json` — the rule + bbox + verification stats
- `reports/v25_per_splat_cell_keys.npy` — per-splat cell_key, length 763,800, int32 (3.05 MB)

## What this unlocks

1. **Independent v25 tier derivation.** With per-splat cell_keys, the `context_class()` function in `phoxbench_v020_hash_context.py` can be run on every cell, producing tier labels (smooth/curved/mixed/unsafe) without depending on the v27 anchor.
2. **The v25 caveat in `PROJECT_STATE.md` can be closed.** The "v21/v22 CSVs only record per-cell counts, not the original cell-membership rule" sentence is no longer true: we now have the rule + the per-splat assignment, byte-exact.
3. **Future v25 rebuilds are no longer dependent on a lossy nearest-center substitution** that produced the documented 3× distribution mismatch. They can use the recovered rule directly.

## Honest scope

- This recovers the **cell-binning rule**, not the **acceptance/filtering rule** that selected 247 chunks out of 2,163 candidates. The filtering rule is fully visible in `phoxbench_v020_hash_context.py`'s `context_class()` + `safe` predicate.
- Tier-label production from cells uses both the rule and the filtering — both are now in hand. Re-running the full v20 audit on the original PLY should reproduce the 247 accepted cells (and their stats) bit-exactly.
- This doesn't change any rendered pixels or any PhoxBench result; it removes a documented dependency on the v27 anchor for tier provenance.
