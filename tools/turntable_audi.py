"""Render an orbiting turntable of the Audi from the v28 EXACT archive.

For each yaw angle in [0, 360), do a chunked Audi render at moderate
resolution + subsample, vertical-flip the saved PNG to fix orientation,
then stitch with ffmpeg into an MP4 loop.

Designed for the bash sandbox: each frame is its own state dir + chunked
render so per-call time stays under 45 seconds.

Usage:
    python3 tools/turntable_audi.py --frames 36 --size 384 --max-points 80000
    python3 tools/turntable_audi.py --finalize          # stitch existing frames
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path("/sessions/ecstatic-sleepy-curie/mnt/Crypsoid")
SCENE = ROOT / "outputs" / "v28_sh_vq_exact_archive_container.3dphox"
OUT_DIR = ROOT / "renders" / "crypsorender_v01" / "turntable"


def run_one(yaw_deg, pitch_deg, distance, fov, size, max_points, frame_idx, mode="gaussian"):
    """Render a single frame, save flipped PNG."""
    state_dir = Path(f"/tmp/state_tt_{frame_idx:03d}")
    state_dir.mkdir(parents=True, exist_ok=True)
    init_args = [
        "python3", "-B", "tools/render_phox_chunked.py",
        "--scene", str(SCENE), "--is-phox",
        "--size", str(size), "--max-points", str(max_points),
        "--use-sh",
        "--yaw", str(yaw_deg), "--pitch", str(pitch_deg),
        "--distance", str(distance), "--fov", str(fov),
        "--state-dir", str(state_dir), "--init",
    ]
    subprocess.run(init_args, cwd=str(ROOT), check=True, capture_output=True)
    # Render in 200k batches until done (smaller scenes finish in 1)
    for _ in range(8):
        r = subprocess.run([
            "python3", "-B", "tools/render_phox_chunked.py",
            "--state-dir", str(state_dir),
            "--batch", "200000", "--mode", mode,
        ], cwd=str(ROOT), check=True, capture_output=True, text=True)
        if "already done" in r.stdout or "already done" in r.stderr:
            break
    # Finalize to a temp file, then flip Y and save final frame
    tmp_png = state_dir / "frame_raw.png"
    subprocess.run([
        "python3", "-B", "tools/render_phox_chunked.py",
        "--state-dir", str(state_dir),
        "--finalize", "--out", str(tmp_png),
    ], cwd=str(ROOT), check=True, capture_output=True)
    from PIL import Image
    img = Image.open(tmp_png).transpose(Image.FLIP_TOP_BOTTOM)
    out_path = OUT_DIR / f"frame_{frame_idx:03d}.png"
    img.save(out_path)
    # Clean state dir to save disk
    shutil.rmtree(state_dir, ignore_errors=True)
    return out_path


def stitch_video(fps=18, out=None):
    out = out or (OUT_DIR / "audi_turntable.mp4")
    pattern = str(OUT_DIR / "frame_%03d.png")
    # Standard-issue MP4 with H.264, baseline-profile, broadly playable
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",   # ensure even dims for libx264
        "-loop", "0",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"saved {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=36)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--max-points", type=int, default=80000)
    ap.add_argument("--pitch", type=float, default=8.0)
    ap.add_argument("--distance", type=float, default=1.4)
    ap.add_argument("--fov", type=float, default=45.0)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=999)
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--finalize-only", action="store_true",
                    help="skip rendering; just ffmpeg-stitch existing frame_*.png")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.finalize_only:
        stitch_video(fps=args.fps)
        return

    end_frame = min(args.start_frame + args.max_frames, args.frames)
    for i in range(args.start_frame, end_frame):
        yaw = 360.0 * i / args.frames
        out_path = OUT_DIR / f"frame_{i:03d}.png"
        if out_path.exists():
            print(f"  frame {i:03d} ({yaw:.1f} deg) -- skip (exists)")
            continue
        t0 = time.perf_counter()
        run_one(yaw, args.pitch, args.distance, args.fov, args.size, args.max_points, i)
        dt = time.perf_counter() - t0
        print(f"  frame {i:03d} ({yaw:.1f} deg) saved in {dt:.1f}s")

    if end_frame >= args.frames:
        stitch_video(fps=args.fps)


if __name__ == "__main__":
    main()
