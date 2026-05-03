"""F.26 — Build the "input data, not pipeline" side-by-side panel.

Two renders, same renderer:
  LEFT:  CGI source (cgi_studio_v1.3dphox)         — clean procedural input
  RIGHT: Scan source (v40_audi_full_mipfilled.3dphox) — trained-3DGS scan

Caption: "Same renderer. Input data is the difference."
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/sessions/ecstatic-sleepy-curie/mnt/Crypsoid')
LEFT  = ROOT / 'renders' / 'crypsorender_v01' / 'SHOWCASE_CGI_STUDIO_2k.png'
RIGHT = ROOT / 'renders' / 'crypsorender_v01' / 'SHOWCASE_AUDI_PHOTOREAL_v2_2k.png'
OUT   = ROOT / 'renders' / 'crypsorender_v01' / 'SHOWCASE_CGI_VS_SCAN.png'


def label_image(img: Image.Image, title: str, subtitle: str) -> Image.Image:
    """Add a top banner with title + subtitle."""
    W, H = img.size
    BAND = 110
    out = Image.new('RGB', (W, H + BAND), (12, 12, 14))
    out.paste(img, (0, BAND))
    draw = ImageDraw.Draw(out)
    try:
        f_title = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 38)
        f_sub   = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    except Exception:
        f_title = f_sub = ImageFont.load_default()
    draw.text((28, 18), title, font=f_title, fill=(245, 245, 245))
    draw.text((28, 66), subtitle, font=f_sub, fill=(180, 180, 195))
    return out


def main():
    if not LEFT.exists() or not RIGHT.exists():
        raise SystemExit(f'Missing input(s): LEFT={LEFT.exists()} RIGHT={RIGHT.exists()}')

    a = Image.open(LEFT).convert('RGB')
    b = Image.open(RIGHT).convert('RGB')
    # Match height
    H = min(a.size[1], b.size[1])
    a = a.resize((int(a.size[0] * H / a.size[1]), H), Image.LANCZOS)
    b = b.resize((int(b.size[0] * H / b.size[1]), H), Image.LANCZOS)

    a_lbl = label_image(a, 'CGI source .3dphox',
                          'Procedurally built — clean PBR, exact normals')
    b_lbl = label_image(b, 'Trained-3DGS scan .3dphox',
                          'Real Audi scan — fuzzy at silhouette by construction')

    Hf = max(a_lbl.size[1], b_lbl.size[1])
    GAP = 24
    Wf = a_lbl.size[0] + GAP + b_lbl.size[0]
    canvas = Image.new('RGB', (Wf, Hf + 90), (12, 12, 14))
    canvas.paste(a_lbl, (0, 0))
    canvas.paste(b_lbl, (a_lbl.size[0] + GAP, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        f_foot = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 26)
        f_sub  = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    except Exception:
        f_foot = f_sub = ImageFont.load_default()
    draw.text((28, Hf + 18),
              'Same renderer. Same lit stack (Lambert + GGX + ACES + AO + shadows).',
              font=f_foot, fill=(245, 245, 245))
    draw.text((28, Hf + 50),
              'Input data is the difference.',
              font=f_sub, fill=(180, 200, 255))

    # Downscale to a sensible deliverable width
    target_w = 2400
    if canvas.size[0] > target_w:
        scale = target_w / canvas.size[0]
        canvas = canvas.resize((target_w, int(canvas.size[1] * scale)), Image.LANCZOS)
    canvas.save(OUT)
    # Also a 1600 thumb for quick view
    thumb = canvas.resize((1600, int(canvas.size[1] * 1600 / canvas.size[0])),
                          Image.LANCZOS)
    thumb.save(OUT.with_name('SHOWCASE_CGI_VS_SCAN_thumb.png'))
    print(f'wrote {OUT}  ({OUT.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
