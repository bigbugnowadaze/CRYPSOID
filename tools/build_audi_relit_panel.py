"""F.28 — A/B panel: original Audi photoreal v2 vs relit (recovered albedo).

Same camera, same lighting recipe. The relit version separates capture
lighting from geometry colour so re-lighting doesn't compound shading.
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/sessions/ecstatic-sleepy-curie/mnt/Crypsoid')
ORIG  = ROOT / 'renders' / 'crypsorender_v01' / 'SHOWCASE_AUDI_PHOTOREAL_v2_2k.png'
RELIT = ROOT / 'renders' / 'crypsorender_v01' / 'SHOWCASE_AUDI_RELIT_2k.png'
OUT   = ROOT / 'renders' / 'crypsorender_v01' / 'SHOWCASE_AUDI_RELIT_AB.png'


def label(img, title, sub):
    W, H = img.size
    BAND = 110
    out = Image.new('RGB', (W, H + BAND), (12, 12, 14))
    out.paste(img, (0, BAND))
    d = ImageDraw.Draw(out)
    try:
        ft = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 38)
        fs = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    except Exception:
        ft = fs = ImageFont.load_default()
    d.text((28, 18), title, font=ft, fill=(245, 245, 245))
    d.text((28, 66), sub, font=fs, fill=(180, 180, 195))
    return out


def main():
    a = Image.open(ORIG).convert('RGB')
    b = Image.open(RELIT).convert('RGB')
    H = min(a.size[1], b.size[1])
    a = a.resize((int(a.size[0] * H / a.size[1]), H), Image.LANCZOS)
    b = b.resize((int(b.size[0] * H / b.size[1]), H), Image.LANCZOS)
    al = label(a, 'Audi photoreal v2 (baked-in)',
                  'Original sh_dc + relit by 3-point — capture light compounds')
    bl = label(b, 'Audi RELIT (recovered albedo)',
                  'F.28 inverse-Lambert: capture light divided out before relighting')
    GAP = 24
    Hf = max(al.size[1], bl.size[1])
    Wf = al.size[0] + GAP + bl.size[0]
    canvas = Image.new('RGB', (Wf, Hf + 90), (12, 12, 14))
    canvas.paste(al, (0, 0))
    canvas.paste(bl, (al.size[0] + GAP, 0))
    d = ImageDraw.Draw(canvas)
    try:
        ft = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 26)
        fs = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    except Exception:
        ft = fs = ImageFont.load_default()
    d.text((28, Hf + 18),
           'Same renderer + same lighting setup. Only the input .3dphox differs.',
           font=ft, fill=(245, 245, 245))
    d.text((28, Hf + 50),
           'Underside splats brighten ~17% after albedo recovery; up-facing stay similar.',
           font=fs, fill=(180, 200, 255))
    target_w = 2400
    if canvas.size[0] > target_w:
        s = target_w / canvas.size[0]
        canvas = canvas.resize((target_w, int(canvas.size[1] * s)), Image.LANCZOS)
    canvas.save(OUT)
    canvas.resize((1600, int(canvas.size[1] * 1600 / canvas.size[0])),
                  Image.LANCZOS).save(OUT.with_name('SHOWCASE_AUDI_RELIT_AB_thumb.png'))
    print(f'wrote {OUT}  ({OUT.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
