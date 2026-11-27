#!/usr/bin/env python3
"""Collision-surface stills of two variants side by side (crash_outlines' renders): one sheet per
camera, a row per moment, the left column the first variant.

    PY=.../venv/bin/python   # Pillow
    $PY tools/sections/stills_beside.py OUT_PREFIX DIR_A LABEL_A DIR_B LABEL_B
"""
import os
import sys

from PIL import Image, ImageDraw

MOMENTS = ["before", "peak", "max_crush", "rest"]


def main():
    prefix, da, la, db, lb = sys.argv[1:6]
    for cam in ["side", "top"]:
        rows = []
        for m in MOMENTS:
            ims = [Image.open(os.path.join(d, f"{m}-{cam}.png")).convert("RGB") for d in (da, db)]
            w, h = ims[0].size
            row = Image.new("RGB", (2 * w, h), "white")
            for k, im in enumerate(ims):
                row.paste(im, (k * w, 0))
                ImageDraw.Draw(row).text((k * w + 12, 10), f"{(la, lb)[k]}: {m}", fill=(0, 0, 0))
            rows.append(row)
        sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
        y = 0
        for r in rows:
            sheet.paste(r, (0, y))
            y += r.height
        sheet.save(f"{prefix}_{cam}.png")
        print("wrote", f"{prefix}_{cam}.png")


if __name__ == "__main__":
    main()
