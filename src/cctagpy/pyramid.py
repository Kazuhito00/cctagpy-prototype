"""Cascaded image pyramid, ported from ``src/cctag/ImagePyramid.{hpp,cpp}``.

Each level is built by resizing the *previous* level's output (not the
original image), so downsampling error compounds level to level -- this
must be replicated, not shortcut by resizing directly from the full-res
source at every level.
"""

from __future__ import annotations

import numpy as np

from cctagpy.level import Level


class ImagePyramid:
    def __init__(self, width: int, height: int, n_levels: int) -> None:
        if n_levels < 1 or (width >> (n_levels - 1)) < 1 or (height >> (n_levels - 1)) < 1:
            raise ValueError(f"image {width}x{height} is too small for {n_levels} pyramid levels")
        self.n_levels = n_levels
        self.levels: list[Level] = []
        for i in range(n_levels):
            self.levels.append(Level(width >> i, height >> i))

    def build(self, src_gray: np.ndarray, canny_thr_low: float, canny_thr_high: float) -> None:
        prev_img = src_gray
        for level in self.levels:
            level.set_level(prev_img, canny_thr_low, canny_thr_high)
            prev_img = level.src

    def get_level(self, i: int) -> Level:
        return self.levels[i]
