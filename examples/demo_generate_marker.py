#!/usr/bin/env python3
"""Generate printable CCTag marker images (PNG) using only Pillow.

Draws the same concentric black/white ring pattern as the C++ reference
generator (``markersToPrint/generators/generate.py``), computed directly
from ``cctagpy.markers_bank``'s radius-ratio table instead of the C++
side's separate ``cctag3.txt``/``cctag4.txt`` files.

Usage:
    uv run python examples/demo_generate_marker.py                # marker 0, 3 crowns
    uv run python examples/demo_generate_marker.py --id 5
    uv run python examples/demo_generate_marker.py --n-crowns 4 --id 10
    uv run python examples/demo_generate_marker.py --all           # every marker for --n-crowns
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from cctagpy.markers_bank import CCTagMarkersBank


def draw_marker(marker_id: int, ratios: list[float], radius: int, margin: int) -> Image.Image:
    """One marker image: solid black outer disk, then alternating white/black
    rings (largest first) sized from ``ratios`` (``ring_radius = outer_radius
    / ratio``), matching the C++ generator's draw order and colors."""
    ring_radii = sorted(radius / ratio for ratio in ratios)[::-1]  # largest first

    size = 2 * (radius + margin)
    center = size // 2
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)

    def disk(r: float, fill: int) -> None:
        draw.ellipse((center - r, center - r, center + r, center + r), fill=fill)

    disk(radius, 0)
    fill = 255
    for r in ring_radii:
        disk(r, fill)
        fill = 0 if fill == 255 else 255

    draw.text((margin // 2, margin // 2), str(marker_id), fill=0)
    return img


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CCTag marker PNGs")
    parser.add_argument("--id", type=int, default=0, help="marker id to generate (default: 0)")
    parser.add_argument("--n-crowns", type=int, default=3, choices=[3, 4], help="ring count (default: 3)")
    parser.add_argument("--radius", type=int, default=300, help="outer circle radius in pixels (default: 300)")
    parser.add_argument("--outdir", type=str, default="markers_out", help="output directory (default: markers_out)")
    parser.add_argument("--all", action="store_true", help="generate every marker id for --n-crowns instead of just --id")
    args = parser.parse_args()

    bank = CCTagMarkersBank(n_crowns=args.n_crowns).get_markers()
    ids = range(len(bank)) if args.all else [args.id]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    margin = int(args.radius * 0.4)

    for marker_id in ids:
        img = draw_marker(marker_id, bank[marker_id], args.radius, margin)
        out_path = outdir / f"marker_{args.n_crowns}crowns_{marker_id:04d}.png"
        img.save(out_path)
        print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
