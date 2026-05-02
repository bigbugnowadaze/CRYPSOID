"""Create multi-panel contact sheets."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def add_label(img: Image.Image, label: str) -> Image.Image:
    """Add a text label above an image."""
    pad = 44
    out = Image.new("RGB", (img.width, img.height + pad), (18, 18, 18))
    out.paste(img, (0, pad))
    d = ImageDraw.Draw(out)
    d.text((12, 12), label, fill=(245, 245, 245))
    return out


def make_contact_sheet_3panel(
    img_a: Image.Image,
    img_b: Image.Image,
    img_c: Image.Image,
    label_a: str = "Panel A",
    label_b: str = "Panel B",
    label_c: str = "Panel C",
    path: Path | None = None,
) -> Image.Image:
    """Create a 3-panel horizontal contact sheet.

    Args:
        img_a, img_b, img_c: PIL Images
        label_a, label_b, label_c: labels for each panel
        path: optional output path

    Returns:
        combined image
    """
    # Add labels
    labeled_a = add_label(img_a, label_a)
    labeled_b = add_label(img_b, label_b)
    labeled_c = add_label(img_c, label_c)

    # Get dimensions
    h = labeled_a.height
    w = labeled_a.width
    contact_h = h
    contact_w = w * 3

    # Create contact sheet
    contact = Image.new("RGB", (contact_w, contact_h), (0, 0, 0))
    contact.paste(labeled_a, (0, 0))
    contact.paste(labeled_b, (w, 0))
    contact.paste(labeled_c, (w * 2, 0))

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        contact.save(path)

    return contact
