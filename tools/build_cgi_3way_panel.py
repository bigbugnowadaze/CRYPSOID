"""F.27 — 3-panel: scan-Audi v40 / CGI-studio (F.26) / CGI-car (F.27).

Same renderer, three .3dphox sources, three distinct aesthetics.
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/sessions/ecstatic-sleepy-curie/mnt/Crypsoid')
SCAN   = ROOT / 'renders' / 'crypsorender_v01' / 'SHOWCASE_AUDI_PHOTOREAL_v2_2k.png'
STUDIO = ROOT / 'renders' / 'crypsorender_v01' / 'SHOWCASE_CGI_STUDIO_2k.png'
CAR    = ROOT / 'renders' / 'crypsorender_v01' / 'SHOWCASE_CGI_CAR_HDRI_2k.png'
OUT    = ROOT / 'renders' / 'crypsorender_v01' / 'SHOWCASE_CGI_3WAY.png'


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
    a = Image.open(SCAN).convert('RGB')
    b = Image.open(STUDIO).convert('RGB')
    c = Image.open(CAR).convert('RGB')
    H = min(a.size[1], b.size[1], c.size[1])
    a = a.resize((int(a.size[0] * H / a.size[1]), H), Image.LANCZOS)
    b = b.resize((int(b.size[0] * H / b.size[1]), H), Image.LANCZOS)
    c = c.resize((int(c.size[0] * H / c.size[1]), H), Image.LANCZOS)

    al = label(a, 'Trained-3DGS scan',  'Real Audi scan via .3dphox + photoreal stack')
    bl = label(b, 'Procedural CGI studio',  'F.26 — clean primitives + PBR sidecar')
    cl = label(c, 'Procedural CGI car + HDRI',  'F.27 — stylized SDF car + studio HDRI environment')

    GAP = 24
    Hf = max(al.size[1], bl.size[1], cl.size[1])
    Wf = al.size[0] + GAP + bl.size[0] + GAP + cl.size[0]
    canvas = Image.new('RGB', (Wf, Hf + 90), (12, 12, 14))
    canvas.paste(al, (0, 0))
    canvas.paste(bl, (al.size[0] + GAP, 0))
    canvas.paste(cl, (al.size[0] + GAP + bl.size[0] + GAP, 0))

    d = ImageDraw.Draw(canvas)
    try:
        ft = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 30)
        fs = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    except Exception:
        ft = fs = ImageFont.load_default()
    d.text((28, Hf + 14),
           'Same CRYPSOID renderer. Same lit stack. Three distinct .3dphox inputs.',
           font=ft, fill=(245, 245, 245))
    d.text((28, Hf + 52),
           'The aesthetic is a property of the input data, not the pipeline.',
           font=fs, fill=(180, 200, 255))

    target_w = 3200
    if canvas.size[0] > target_w:
        s = target_w / canvas.size[0]
        canvas = canvas.resize((target_w, int(canvas.size[1] * s)), Image.LANCZOS)
    canvas.save(OUT)
    canvas.resize((1920, int(canvas.size[1] * 1920 / canvas.size[0])),
                  Image.LANCZOS).save(OUT.with_name('SHOWCASE_CGI_3WAY_thumb.png'))
    print(f'wrote {OUT} ({OUT.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
