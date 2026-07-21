#!/usr/bin/env python3
"""Render a documentation figure from the repository's canonical hand assets."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "temp"))

from visualize_all_right_hands_mujoco import presentation_mesh  # noqa: E402
from visualize_right_hands import load_all_hands  # noqa: E402


OUTPUT = ROOT / "assets" / "representation_assets" / "hand_to_fhsg.png"
WIDTH, HEIGHT = 1800, 720
BG = (19, 24, 34)
FG = (232, 237, 245)
MUTED = (145, 158, 178)
ACCENT = (88, 190, 255)
ACTIVE = (92, 211, 157)
INACTIVE = (70, 79, 96)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    draw.line([start, end], fill=ACCENT, width=7)
    x, y = end
    draw.polygon([(x, y), (x - 22, y - 14), (x - 22, y + 14)], fill=ACCENT)


def draw_hand_projection(
    draw: ImageDraw.ImageDraw,
    vertices: np.ndarray,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    x = vertices[:, 0]
    z = vertices[:, 2]
    span = max(float(np.ptp(x)), float(np.ptp(z)), 1e-9)
    scale = 0.86 * min(x1 - x0, y1 - y0) / span
    px = (x - (x.min() + x.max()) / 2.0) * scale + (x0 + x1) / 2.0
    py = y1 - 0.06 * (y1 - y0) - (z - z.min()) * scale
    if len(px) > 35_000:
        selection = np.linspace(0, len(px) - 1, 35_000, dtype=int)
        px, py = px[selection], py[selection]
    draw.point([(int(a), int(b)) for a, b in zip(px, py)], fill=color)


def draw_primitive_hand(draw: ImageDraw.ImageDraw, origin: tuple[int, int]) -> None:
    ox, oy = origin
    draw.rounded_rectangle((ox - 115, oy + 50, ox + 115, oy + 250), radius=35, fill=(52, 124, 161), outline=ACCENT, width=4)
    x_positions = [-82, -40, 2, 44, 86]
    lengths = [132, 182, 200, 186, 152]
    for finger, (x_offset, length) in enumerate(zip(x_positions, lengths)):
        x = ox + x_offset
        base = oy + 60
        segment = length / 3.0
        for part in range(3):
            y_bottom = base - part * segment
            y_top = base - (part + 1) * segment + 5
            draw.rounded_rectangle((x - 13, y_top, x + 13, y_bottom), radius=13, fill=(66, 174, 137), outline=ACTIVE, width=3)
            draw.ellipse((x - 7, y_bottom - 7, x + 7, y_bottom + 7), fill=(255, 187, 79))
    # Opposition thumb: a short capsule chain attached to the side of the palm.
    points = [(ox + 102, oy + 165), (ox + 154, oy + 125), (ox + 196, oy + 84)]
    draw.line(points, fill=ACTIVE, width=24, joint="curve")
    for x, y in points:
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=(255, 187, 79))


def draw_supergraph(draw: ImageDraw.ImageDraw, origin: tuple[int, int]) -> None:
    ox, oy = origin
    cell_w, cell_h = 62, 54
    masks = [
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [0, 0, 0, 0],
    ]
    for finger in range(5):
        draw.text((ox - 58, oy + finger * (cell_h + 13) + 13), f"F{finger}", font=font(24, True), fill=FG if any(masks[finger]) else MUTED)
        for segment in range(4):
            x0 = ox + segment * (cell_w + 12)
            y0 = oy + finger * (cell_h + 13)
            active = masks[finger][segment]
            draw.rounded_rectangle(
                (x0, y0, x0 + cell_w, y0 + cell_h),
                radius=9,
                fill=ACTIVE if active else INACTIVE,
                outline=(154, 241, 205) if active else (91, 100, 118),
                width=3,
            )
            # Two small candidate-DOF gates per connection.
            for axis in range(2):
                gate_color = (255, 187, 79) if active and not (finger == 0 and segment == 2 and axis == 1) else (45, 51, 64)
                gx = x0 + 13 + axis * 25
                draw.rectangle((gx, y0 + 34, gx + 13, y0 + 47), fill=gate_color)
    draw.text((ox + 8, oy + 5 * (cell_h + 13) + 8), "fixed tensor + masks", font=font(25, True), fill=FG)


def main() -> int:
    hands = {hand.hand_id: hand for hand in load_all_hands("right")}
    selected = [
        ("mano", (239, 177, 67)),
        ("unitree_dex5_1", (92, 211, 157)),
        ("inspire_rh56dfx", (88, 190, 255)),
    ]
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((48, 28), "Different real embodiments", font=font(38, True), fill=FG)
    draw.text((705, 28), "Functional primitive hand", font=font(38, True), fill=FG)
    draw.text((1265, 28), "FHSG representation", font=font(38, True), fill=FG)

    for row, (hand_id, color) in enumerate(selected):
        hand = hands[hand_id]
        mesh = presentation_mesh(hand)
        top = 94 + row * 198
        draw_hand_projection(draw, np.asarray(mesh.vertices), (55, top, 470, top + 170), color)
        draw.text((72, top + 137), hand.display_name, font=font(23, True), fill=FG)

    arrow(draw, (500, 360), (665, 360))
    draw.text((506, 300), "collapse auxiliary links", font=font(22), fill=MUTED)
    draw.text((535, 330), "fit primitives", font=font(22), fill=MUTED)
    draw_primitive_hand(draw, (895, 350))
    arrow(draw, (1105, 360), (1240, 360))
    draw.text((1126, 310), "encode", font=font(24), fill=MUTED)
    draw_supergraph(draw, (1325, 120))

    draw.text((54, 677), "Actual canonical assets from DexCoDesign", font=font(21), fill=MUTED)
    draw.text((690, 677), "box / capsule / ellipsoid + active joints", font=font(21), fill=MUTED)
    draw.text((1272, 677), "same tensor shape, different masks and DOF count", font=font(21), fill=MUTED)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
