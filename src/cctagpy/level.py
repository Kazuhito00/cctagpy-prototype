"""One pyramid level, ported from ``src/cctag/Level.{hpp,cpp}``.

Each level resizes the *previous* level's already-resized image (a genuine
cascaded pyramid -- see ``pyramid.py``), then runs the recoded Canny and
two-pass thinning on that resized image. There is no Gaussian blur anywhere
in this pipeline; any "smoothing" is baked into the 9x9 derivative kernel.
"""

from __future__ import annotations

import numpy as np

from cctagpy._numba_utils import HAS_NUMBA, njit, prange
from cctagpy.canny import recoded_canny
from cctagpy.thinning import thin


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Round-to-nearest, clip to [0, 255], ``uint8`` -- the quantization every
    pyramid level stores (the Numba resize fuses the same three steps)."""
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _resize_bilinear_numpy(img: np.ndarray, dst_w: int, dst_h: int) -> np.ndarray:
    src_h, src_w = img.shape
    scale_x = src_w / dst_w
    scale_y = src_h / dst_h

    dst_x = (np.arange(dst_w) + 0.5) * scale_x - 0.5
    dst_y = (np.arange(dst_h) + 0.5) * scale_y - 0.5

    x0 = np.floor(dst_x).astype(np.int64)
    y0 = np.floor(dst_y).astype(np.int64)
    fx = dst_x - x0
    fy = dst_y - y0

    x0c = np.clip(x0, 0, src_w - 1)
    x1c = np.clip(x0 + 1, 0, src_w - 1)
    y0c = np.clip(y0, 0, src_h - 1)
    y1c = np.clip(y0 + 1, 0, src_h - 1)

    src = img.astype(np.float64)
    top = src[np.ix_(y0c, x0c)] * (1 - fx)[None, :] + src[np.ix_(y0c, x1c)] * fx[None, :]
    bot = src[np.ix_(y1c, x0c)] * (1 - fx)[None, :] + src[np.ix_(y1c, x1c)] * fx[None, :]
    out = top * (1 - fy)[:, None] + bot * fy[:, None]
    return _to_uint8(out)


if HAS_NUMBA:

    @njit(cache=True, parallel=True)
    def _resize_bilinear_numba(img: np.ndarray, dst_w: int, dst_h: int) -> np.ndarray:
        """Same arithmetic as :func:`_resize_bilinear_numpy` followed by the
        caller's ``clip(rint(.), 0, 255).astype(uint8)``, fused into one
        row-parallel pass that writes ``uint8`` directly (the NumPy form
        materializes a full float64 image plus several gathered
        temporaries first). Per-column sample positions/weights are
        precomputed once; rows are independent, hence ``prange``."""
        src_h, src_w = img.shape
        scale_x = src_w / dst_w
        scale_y = src_h / dst_h
        out = np.empty((dst_h, dst_w), dtype=np.uint8)

        x0c_all = np.empty(dst_w, dtype=np.int64)
        x1c_all = np.empty(dst_w, dtype=np.int64)
        fx_all = np.empty(dst_w, dtype=np.float64)
        for j in range(dst_w):
            dst_xj = (j + 0.5) * scale_x - 0.5
            x0 = int(np.floor(dst_xj))
            fx_all[j] = dst_xj - x0
            x0c_all[j] = min(max(x0, 0), src_w - 1)
            x1c_all[j] = min(max(x0 + 1, 0), src_w - 1)

        for i in prange(dst_h):
            dst_yi = (i + 0.5) * scale_y - 0.5
            y0 = int(np.floor(dst_yi))
            fy = dst_yi - y0
            y0c = min(max(y0, 0), src_h - 1)
            y1c = min(max(y0 + 1, 0), src_h - 1)
            for j in range(dst_w):
                fx = fx_all[j]
                x0c = x0c_all[j]
                x1c = x1c_all[j]
                top = float(img[y0c, x0c]) * (1.0 - fx) + float(img[y0c, x1c]) * fx
                bot = float(img[y1c, x0c]) * (1.0 - fx) + float(img[y1c, x1c]) * fx
                out[i, j] = np.uint8(min(max(np.rint(top * (1.0 - fy) + bot * fy), 0.0), 255.0))

        return out


def resize_bilinear(img: np.ndarray, dst_w: int, dst_h: int) -> np.ndarray:
    """Bilinear resize matching OpenCV's ``cv::resize`` (``INTER_LINEAR``)
    pixel-center convention and edge-replicate border handling, returned as
    a rounded/clipped ``uint8`` image. Dispatches to a fused row-parallel
    Numba loop when available, falling back to the vectorized NumPy form
    otherwise -- verified bit-identical on real images either way.

    A same-size request is the identity (every sample lands exactly on a
    source pixel with zero fractional weight, so ``rint`` gives the source
    value back) and skips the resample; a ``uint8`` input is then returned
    as-is, *not* copied, so callers must treat the result as read-only."""
    if img.shape != (dst_h, dst_w):
        if HAS_NUMBA:
            return _resize_bilinear_numba(img, dst_w, dst_h)
        return _resize_bilinear_numpy(img, dst_w, dst_h)
    return img if img.dtype == np.uint8 else _to_uint8(img.astype(np.float64))


class Level:
    """One pyramid level's resized image, gradients, and thinned edge map."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.src: np.ndarray | None = None
        self.dx: np.ndarray | None = None
        self.dy: np.ndarray | None = None
        self.edges: np.ndarray | None = None

    def set_level(self, src_gray: np.ndarray, canny_thr_low: float, canny_thr_high: float) -> None:
        self.src = resize_bilinear(src_gray, self.width, self.height)

        edges, dx, dy = recoded_canny(self.src, canny_thr_low * 256, canny_thr_high * 256)
        self.dx = dx
        self.dy = dy
        self.edges = thin(edges)
