"""Build the final Tier 2 showcase contact sheet.

Bundles together (when sandbox runs everything):
  - Audi @ faithful Newton (tight side view)
  - Audi @ Gaussian (same view) for direct A/B comparison
  - PhoxBench cusp scene side-by-side (Gaussian vs Phoxoid)
  - PhoxBench saddle scene side-by-side
  - Killer-ratio summary table (rendered as image)
  - Multi-view distribution table (mean/median/worst PSNR/SSIM)
  - File-size + bits-per-Gaussian chart from Tier 1.5

Inputs are read from their canonical paths in renders/crypsorender_v01/ and
phoxbench/runs/. Missing inputs are left blank with a placeholder note.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

R = Path("/sessions/ecstatic-sleepy-curie/mnt/Crypsoid")
OUT_DIR = R / "renders" / "crypsorender_v01"
PHOX = R / "phoxbench" / "runs"


def load_or_placeholder(path: Path, w: int, h: int, label: str) -> Image.Image:
    if path.exists():
        try:
            return Image.open(path).convert("RGB").resize((w, h), Image.LANCZOS)
        except Exception as e:
            pass
    img = Image.new("RGB", (w, h), (40, 40, 50))
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        f = ImageFont.load_default()
    d.text((10, h // 2 - 8), f"missing: {label}\n{path.name}", fill=(160, 160, 170), font=f)
    return img


def render_killer_table(rows: list, w: int = 720, row_h: int = 32) -> Image.Image:
    """rows: list of dicts with keys 'scene', 'budget', 'gauss_rmse', 'phox_rmse', 'killer'."""
    h = 60 + row_h * (len(rows) + 1)
    img = Image.new("RGB", (w, h), (24, 24, 30))
    d = ImageDraw.Draw(img)
    try:
        f_t = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        f_h = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        f_r = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        f_t = f_h = f_r = ImageFont.load_default()
    d.text((16, 12), "PhoxBench Tier 0 — killer ratio (Gaussians needed to match phoxoid quality)",
           fill=(240, 240, 240), font=f_t)
    cols = ["scene", "budget", "gauss RMSE", "phox RMSE", "killer Gauss-blob count", "ratio"]
    xs = [16, 160, 240, 360, 480, 640]
    y = 50
    for c, x in zip(cols, xs):
        d.text((x, y), c, fill=(180, 200, 220), font=f_h)
    y += row_h
    for r in rows:
        d.text((xs[0], y), str(r.get("scene", "")), fill=(220, 220, 220), font=f_r)
        d.text((xs[1], y), str(r.get("budget", "")), fill=(220, 220, 220), font=f_r)
        d.text((xs[2], y), f"{r.get('gauss_rmse', 0):.4f}", fill=(220, 220, 220), font=f_r)
        d.text((xs[3], y), f"{r.get('phox_rmse', 0):.4f}", fill=(220, 220, 220), font=f_r)
        k = r.get("killer", -1)
        d.text((xs[4], y), str(k) if k > 0 else "no match within 4096", fill=(220, 220, 220), font=f_r)
        d.text((xs[5], y), f"{(k/r['budget']):.2f}x" if k > 0 else "—", fill=(220, 220, 220), font=f_r)
        y += row_h
    return img


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Top row: Audi at faithful + at gaussian + tier overlay
    audi_faith = load_or_placeholder(OUT_DIR / "T2_audi_faithful_512.png", 512, 512,
                                      "Audi @ faithful Newton")
    audi_gauss = load_or_placeholder(OUT_DIR / "T2_audi_gaussian_512.png", 512, 512,
                                      "Audi @ Gaussian baseline")
    tier_ov    = load_or_placeholder(OUT_DIR / "v28_tier_overlay_200k.png", 512, 512,
                                      "Tier overlay")

    # Middle row: PhoxBench cusp + saddle side-by-sides
    cusp_sbs   = load_or_placeholder(PHOX / "cusp_b128" / "side_by_side.png", 768, 256,
                                      "cusp B=128 side-by-side")
    saddle_sbs = load_or_placeholder(PHOX / "saddle_b128" / "side_by_side.png", 768, 256,
                                      "saddle B=128 side-by-side")

    # Killer table from summary.json
    rows = []
    summary_path = PHOX / "summary.json"
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text())
            for item in data:
                rows.append(dict(
                    scene=item["scene"], budget=item["blob_budget"],
                    gauss_rmse=item["fit_rmse"]["gaussian"],
                    phox_rmse=item["fit_rmse"]["phoxoid"],
                    killer=item.get("killer_ratio_gaussian_blobs_to_match_phoxoid", -1),
                ))
        except Exception:
            pass
    killer_img = render_killer_table(rows)

    # Multi-view summary
    mv_path = OUT_DIR / "multiview" / "multiview_summary.json"
    multiview_img = load_or_placeholder(mv_path.with_suffix(".png"), 720, 200, "multi-view chart (run multiview first)")

    # File size chart
    file_size = load_or_placeholder(OUT_DIR / "file_sizes.png", 720, 280, "file size chart")

    # Compose
    PAD = 18
    W = 512 * 3 + PAD * 4
    title_h = 96
    rh1 = 512  # top
    rh2 = 256  # phoxbench scenes
    rh3 = killer_img.height
    rh4 = file_size.height
    H = title_h + PAD + rh1 + PAD + rh2 + PAD + rh2 + PAD + rh3 + PAD + rh4 + PAD * 2

    sheet = Image.new("RGB", (W, H), (18, 18, 24))
    d = ImageDraw.Draw(sheet)
    try:
        f_t = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
        f_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        f_l = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        f_t = f_s = f_l = ImageFont.load_default()
    d.text((PAD, PAD), "CRYPSOID  v0.3   Tier 2 deliverable", fill=(255, 255, 255), font=f_t)
    d.text((PAD, PAD + 38),
           "Faithful per-pixel phoxoidal density (5-coef Pearcey basis) + PhoxBench Tier 0  "
           "—  pure CPU, no GPU stack, no torch, no gsplat",
           fill=(200, 210, 220), font=f_s)
    d.text((PAD, PAD + 60),
           "Tier 2 = the project's reason to exist made measurable.",
           fill=(180, 200, 220), font=f_s)

    y = title_h + PAD
    sheet.paste(audi_gauss, (PAD, y))
    sheet.paste(audi_faith, (PAD * 2 + 512, y))
    sheet.paste(tier_ov, (PAD * 3 + 1024, y))
    d.text((PAD, y + 512 + 4),       "Audi @ Gaussian (Tier 1 baseline)", fill=(220,220,220), font=f_l)
    d.text((PAD * 2 + 512, y + 512 + 4), "Audi @ faithful Newton (Tier 2)", fill=(220,220,220), font=f_l)
    d.text((PAD * 3 + 1024, y + 512 + 4), "Tier overlay (A=red B=green C=blue)", fill=(220,220,220), font=f_l)

    y += 512 + PAD + 18
    sheet.paste(cusp_sbs.resize((W - 2 * PAD, 256), Image.LANCZOS), (PAD, y))
    d.text((PAD, y + 256 + 4), "PhoxBench cusp scene  (B=128)  —  Gaussian | Phoxoid",
           fill=(220, 220, 220), font=f_l)

    y += 256 + PAD + 18
    sheet.paste(saddle_sbs.resize((W - 2 * PAD, 256), Image.LANCZOS), (PAD, y))
    d.text((PAD, y + 256 + 4), "PhoxBench saddle scene  (B=128)  —  Gaussian | Phoxoid",
           fill=(220, 220, 220), font=f_l)

    y += 256 + PAD + 18
    sheet.paste(killer_img, (PAD, y))

    y += killer_img.height + PAD
    sheet.paste(file_size, (PAD, y))

    out = OUT_DIR / "SHOWCASE_T2.png"
    sheet.save(out)
    print(f"saved {out}  ({sheet.size})")


if __name__ == "__main__":
    main()
