"""F.29 — Master session deliverable contact sheet for Vince.

Bundles the four key panels from the F.23-F.28 session arc:

    Row 1: CGI 3-way        — same renderer, three .3dphox sources
    Row 2: AUDI relit A/B   — F.28 inverse-Lambert albedo recovery

With a session header and a one-line caption per panel.
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/sessions/ecstatic-sleepy-curie/mnt/Crypsoid')
ROW1 = ROOT / 'renders' / 'crypsorender_v01' / 'SHOWCASE_CGI_3WAY.png'
ROW2 = ROOT / 'renders' / 'crypsorender_v01' / 'SHOWCASE_AUDI_RELIT_AB.png'
OUT  = ROOT / 'renders' / 'crypsorender_v01' / 'SHOWCASE_SESSION_F23_F28.png'


def fonts():
    try:
        ft_huge = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 56)
        ft_big  = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 36)
        ft_med  = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 26)
        ft_sm   = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    except Exception:
        ft_huge = ft_big = ft_med = ft_sm = ImageFont.load_default()
    return ft_huge, ft_big, ft_med, ft_sm


def main():
    ft_huge, ft_big, ft_med, ft_sm = fonts()

    r1 = Image.open(ROW1).convert('RGB')
    r2 = Image.open(ROW2).convert('RGB')

    # Match widths to the wider of the two
    target_w = min(r1.size[0], 2400)
    if r1.size[0] != target_w:
        s = target_w / r1.size[0]
        r1 = r1.resize((target_w, int(r1.size[1] * s)), Image.LANCZOS)
    if r2.size[0] != target_w:
        s = target_w / r2.size[0]
        r2 = r2.resize((target_w, int(r2.size[1] * s)), Image.LANCZOS)

    HEADER = 220   # session title
    GAP    = 60    # between rows
    SECTION_H = 100  # caption strip above each row
    FOOTER = 130

    total_h = HEADER + SECTION_H + r1.size[1] + GAP + SECTION_H + r2.size[1] + FOOTER
    canvas = Image.new('RGB', (target_w, total_h), (12, 12, 14))
    d = ImageDraw.Draw(canvas)

    # Header
    d.text((40, 30), 'CRYPSOID — Session F.23 → F.28', font=ft_huge,
           fill=(245, 245, 245))
    d.text((40, 110), 'Scene contraction (negative result), CGI sources, '
           'inverse-Lambert albedo recovery',
           font=ft_med, fill=(190, 200, 220))
    d.text((40, 152), '2026-05-03  ·  Same renderer.  '
           'The aesthetic is a property of the input data.',
           font=ft_sm, fill=(155, 165, 180))
    d.line([(40, 200), (target_w - 40, 200)], fill=(60, 70, 90), width=2)

    # Row 1 caption strip
    y = HEADER
    d.text((40, y + 12), 'F.26 + F.27  —  CGI source comparison',
           font=ft_big, fill=(245, 245, 245))
    d.text((40, y + 56), 'Trained-3DGS scan  /  procedural studio  /  '
           'procedural car + HDRI environment  ·  one renderer, three .3dphox',
           font=ft_sm, fill=(180, 200, 230))
    y += SECTION_H
    canvas.paste(r1, (0, y))
    y += r1.size[1] + GAP

    # Row 2 caption strip
    d.text((40, y + 12), 'F.28  —  Inverse-Lambert albedo recovery on the scan',
           font=ft_big, fill=(245, 245, 245))
    d.text((40, y + 56), 'Original baked-in lighting  vs  recovered albedo  '
           '·  underside +17%, up-facing unchanged',
           font=ft_sm, fill=(180, 200, 230))
    y += SECTION_H
    canvas.paste(r2, (0, y))
    y += r2.size[1]

    # Footer
    d.line([(40, y + 14), (target_w - 40, y + 14)], fill=(60, 70, 90), width=2)
    d.text((40, y + 36),
           'F.23 scene contraction shipped (Gate 1 PASS) but did not beat '
           'bounded on Family (-6 dB).  See reports/F23_results.md.',
           font=ft_sm, fill=(170, 175, 190))
    d.text((40, y + 70),
           'Honest verdict: outdoor photoreal needs the full Mip-NeRF-360 '
           'recipe (3-5 days). Object-centric pipeline ships clean.',
           font=ft_sm, fill=(170, 175, 190))

    canvas.save(OUT)
    # Thumb for quick view
    thumb_w = 1600
    s = thumb_w / canvas.size[0]
    canvas.resize((thumb_w, int(canvas.size[1] * s)),
                  Image.LANCZOS).save(OUT.with_name('SHOWCASE_SESSION_F23_F28_thumb.png'))
    print(f'wrote {OUT}  ({OUT.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
