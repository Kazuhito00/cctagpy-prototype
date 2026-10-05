"""Minimal CLI mirroring the output format of
``src/applications/detection/main.cpp`` (``x y id status`` per marker),
for validation against the C++ reference build.

Image decoding is the one deliberate exception to the "NumPy/SciPy only"
scope: Pillow is used purely to turn a PNG/JPEG file into an RGB array (one
call). The grayscale conversion itself replicates OpenCV's
``cv::COLOR_BGR2GRAY`` coefficients with plain NumPy arithmetic, not
Pillow's own conversion.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from cctagpy.detection import cctag_detection
from cctagpy.params import Parameters


def load_gray_image(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.float64)

    # cv::COLOR_BGR2GRAY / RGB2GRAY coefficients (ITU-R BT.601).
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cctagpy", description="Pure-Python CCTag detection")
    parser.add_argument("-n", "--nrings", type=int, default=3, help="number of crowns (3 or 4)")
    parser.add_argument("-i", "--input", required=True, help="path to a grayscale/color image")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible RANSAC draws")
    parser.add_argument(
        "--fast-identification",
        action="store_true",
        help=(
            "use the experimental bounded-optimizer imaged-center search "
            "(faster, not bit-identical to the C++ reference) instead of "
            "the default grid search"
        ),
    )
    args = parser.parse_args(argv)

    gray = load_gray_image(args.input)
    params = Parameters(n_crowns=args.nrings)
    rng = np.random.default_rng(args.seed)

    markers = cctag_detection(gray, params, rng=rng, use_identification_optimizer=args.fast_identification)

    print("#frame 0")
    print(f"Detected {len(markers)} candidates")
    for m in markers:
        print(f"{m.x()} {m.y()} {m.id} {m.status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
