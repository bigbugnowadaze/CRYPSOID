#!/usr/bin/env python3
"""
Incremental driver around build_v29_residual_transform_sweep.py.

The full v29 sweep doesn't fit in a single bash sandbox window. This driver
runs N candidates per invocation across all requested codecs, appending results
to v29_sweep/reports/sweep_progress.json so the next invocation continues.

Usage:
  python3 tools/run_v29_incremental.py --n 4
  python3 tools/run_v29_incremental.py --finalize
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import build_v29_residual_transform_sweep as v29  # type: ignore

V25 = ROOT / "outputs" / "v25_attribute_group_render_container.3dphox"
V27 = ROOT / "recovery_v2" / "v27_attribute_group_sh_vq_render_container.3dphox"
V25_REPORT = ROOT / "reports" / "PHOXBENCH_V25_ATTRIBUTE_GROUP_REPORT.json"
V27_REPORT = ROOT / "recovery_v2" / "reports" / "PHOXBENCH_V27_SH_DEBT_REPORT.json"
OUT = ROOT / "v29_sweep"
PROGRESS_FILE = OUT / "reports" / "sweep_progress.json"

CANDIDATES = [
    "splat_major_raw",
    "coefficient_major_transpose",
    "group_major_3x15",
    "band_split_low_mid_high",
    "morton_splat_major",
    "morton_delta_i16",
    "tier_then_coefficient_major",
    "zigzag_splat_major_u16",
    "sign_magnitude_planes",
    "zero_mask_values",
    "bitplane_zigzag_u8",
]
CODECS = ["zlib6", "zlib9"]


def load_progress():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"results": [], "completed_candidates": [], "codecs": CODECS}


def save_progress(p):
    PROGRESS_FILE.write_text(json.dumps(p, indent=2, default=str))


def run_one_candidate(state, cname, codecs):
    print(f"[candidate {cname}] encoding ...", flush=True)
    t0 = time.perf_counter()
    raw, meta, decoder = v29.encode_candidate_raw(cname, state)
    print(f"  encode time: {time.perf_counter()-t0:.2f}s, raw bytes: {len(raw):,}", flush=True)
    exact_layout = bool((decoder(raw, meta, state) == state.residual).all())
    out = []
    for codec in codecs:
        t1 = time.perf_counter()
        try:
            payload = v29.codec_compress(codec, raw)
            elapsed = time.perf_counter() - t1
            rt = v29.codec_decompress(codec, payload)
            exact = bool(exact_layout and rt == raw)
            print(f"  {codec:<8} comp={len(payload):,}  ratio={len(raw)/max(len(payload),1):.3f}  t={elapsed:.2f}s  exact={exact}", flush=True)
            out.append({
                "candidate": cname, "codec": codec,
                "compressed_bytes": len(payload), "raw_bytes": len(raw),
                "ratio_raw_to_compressed": len(raw) / max(len(payload), 1),
                "seconds": elapsed,
                "exact_reversible_layout": exact_layout,
                "codec_roundtrip_exact": exact,
                "meta": meta, "ok": exact,
            })
        except Exception as e:
            out.append({"candidate": cname, "codec": codec, "error": str(e), "ok": False})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--all-codecs", action="store_true")
    ap.add_argument("--n", type=int, default=1, help="how many pending candidates to process this run")
    args = ap.parse_args()

    codecs = ["zlib6", "zlib9", "bz2", "lzma6"] if args.all_codecs else CODECS

    progress = load_progress()
    if progress.get("codecs") != codecs:
        print(f"Codec list changed; resetting progress.", flush=True)
        progress = {"results": [], "completed_candidates": [], "codecs": codecs}

    if args.finalize:
        results = progress["results"]
        ok = [r for r in results if r.get("ok") and "compressed_bytes" in r]
        ok.sort(key=lambda r: r["compressed_bytes"])
        if not ok:
            print("No exactly-reversible results to rank.")
            return
        best = ok[0]
        print(f"\n=== Best: {best['candidate']} / {best['codec']} ===")
        print(f"  compressed: {best['compressed_bytes']:,} bytes ({best['compressed_bytes']/1024/1024:.2f} MiB)")
        print(f"  ratio: {best['ratio_raw_to_compressed']:.3f}")
        print("\n=== Top 5 ===")
        for r in ok[:5]:
            print(f"  {r['candidate']:<30} {r['codec']:<8} {r['compressed_bytes']:>12,}")
        print("\nReloading state for archive write ...", flush=True)
        state = v29.load_real_state(V25, V27, V25_REPORT, V27_REPORT)
        archive_path = OUT / "outputs" / "v29_residual_transform_archive.3dphox"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_size = v29.write_best_archive_if_real(state, best, archive_path)
        if archive_size:
            print(f"\nArchive: {archive_path} ({archive_size:,} bytes / {archive_size/1024/1024:.2f} MiB)")
        summary = {
            "mode": "real_v25_v27",
            "v25_size_bytes": V25.stat().st_size,
            "v27_size_bytes": V27.stat().st_size,
            "best": best,
            "all_results_ranked": ok,
            "errors_or_skipped": [r for r in results if not (r.get("ok") and "compressed_bytes" in r)],
            "archive_path": str(archive_path) if archive_size else None,
            "archive_bytes": archive_size,
        }
        (OUT / "reports" / "PHOXBENCH_V29_RESIDUAL_TRANSFORM_REPORT.json").write_text(json.dumps(summary, indent=2, default=str))
        print("\nSummary: v29_sweep/reports/PHOXBENCH_V29_RESIDUAL_TRANSFORM_REPORT.json")
        return

    pending = [c for c in CANDIDATES if c not in progress["completed_candidates"]]
    if not pending:
        print("All done. Run with --finalize.")
        return
    print(f"Loading state once (will run up to {args.n} candidates) ...", flush=True)
    t0 = time.perf_counter()
    state = v29.load_real_state(V25, V27, V25_REPORT, V27_REPORT)
    print(f"State loaded in {time.perf_counter()-t0:.2f}s", flush=True)
    todo = pending[:args.n]
    for cname in todo:
        print(f"\n--- Candidate {cname} ({len(progress['completed_candidates'])+1}/{len(CANDIDATES)}) ---")
        out = run_one_candidate(state, cname, codecs)
        progress["results"].extend(out)
        progress["completed_candidates"].append(cname)
        save_progress(progress)
    remaining = len(CANDIDATES) - len(progress["completed_candidates"])
    print(f"\n{len(progress['completed_candidates'])} candidates done, {remaining} pending.")


if __name__ == "__main__":
    main()
