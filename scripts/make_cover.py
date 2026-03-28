#!/usr/bin/env python
"""Create a local Kaggle cover image candidate."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "cover_v1.png"
WIDTH, HEIGHT = 560, 280


def gradient_background() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#0d1b2a")
    pixels = image.load()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            mix_x = x / (WIDTH - 1)
            mix_y = y / (HEIGHT - 1)
            r = int(10 + 15 * mix_x + 20 * mix_y)
            g = int(24 + 35 * mix_x + 8 * mix_y)
            b = int(38 + 28 * mix_x + 40 * (1 - mix_x))
            pixels[x, y] = (r, g, b)
    return image


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image = gradient_background()
    draw = ImageDraw.Draw(image, "RGBA")

    draw.rounded_rectangle((24, 28, 290, 252), radius=24, fill=(8, 14, 24, 180), outline=(120, 170, 210, 80), width=2)
    draw.rounded_rectangle((320, 34, 520, 112), radius=18, fill=(230, 82, 58, 220))
    draw.rounded_rectangle((340, 128, 520, 242), radius=20, fill=(244, 245, 247, 235))

    title_font = get_font(36, bold=True)
    subtitle_font = get_font(18, bold=False)
    label_font = get_font(16, bold=True)
    small_font = get_font(14, bold=False)
    metric_font = get_font(20, bold=True)

    draw.text((40, 52), "Beyond", font=title_font, fill="#f4f7fb")
    draw.text((40, 90), "Acuity", font=title_font, fill="#f4f7fb")
    draw.text((40, 128), "Prediction", font=title_font, fill="#f4f7fb")
    draw.text((40, 182), "Interpretable triage support for", font=subtitle_font, fill="#d9e2ec")
    draw.text((40, 206), "undertriage detection", font=subtitle_font, fill="#d9e2ec")

    draw.text((338, 50), "HIGH-RISK REVIEW", font=label_font, fill="#fff8f6")
    draw.text((338, 74), "Second-reader safety layer", font=small_font, fill="#fff2ef")

    draw.text((358, 144), "Patient Card", font=label_font, fill="#122033")
    draw.text((358, 172), "Complaint: pleuritic chest pain", font=small_font, fill="#243447")
    draw.text((358, 194), "SpO2 92   NEWS2 7   GCS 14", font=small_font, fill="#243447")
    draw.text((358, 216), "Flag for urgent reassessment", font=metric_font, fill="#d94841")

    line_color = (77, 162, 212, 180)
    points = [(318, 78), (350, 64), (378, 86), (408, 58), (438, 96), (468, 72), (500, 84)]
    draw.line(points, fill=line_color, width=4)
    for px, py in points:
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill="#9ad1f5")

    draw.text((24, 14), "Emergency Triage AI", font=small_font, fill="#a8c7dc")
    image.save(OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
