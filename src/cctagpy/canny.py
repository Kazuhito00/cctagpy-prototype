"""Recoded Canny edge/gradient detector, ported from ``src/cctag/filter/cvRecode.cpp``.

This replicates OpenCV's classic (2.x) ``cvCanny`` algorithm verbatim except
the derivatives come from a fixed 9x9 kernel (baked-in smoothing + derivative,
values copied from the C++ source) rather than a 3x3 Sobel operator, and the
gradient magnitude is the rounded Euclidean norm (``CV_CANNY_L2_GRADIENT``,
``USE_INTEGER_REP`` branch in the source).

The non-maxima-suppression + hysteresis stage is reimplemented here as a
connected-components pass (:func:`scipy.ndimage.label`) instead of the
original stack-based flood fill: keeping every 8-connected component of
"candidate" pixels (magnitude > low threshold and a local maximum along its
gradient sector) that contains at least one "strong" pixel (magnitude >
high threshold) is mathematically equivalent to the seeded flood-fill
hysteresis in the source, and is far more efficient to vectorize with NumPy.
That is the SciPy/NumPy fallback path (:func:`_recoded_canny_numpy`); with
Numba installed the stage runs as JIT kernels instead, see :func:`recoded_canny`.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.ndimage import correlate1d

from cctagpy._numba_utils import HAS_NUMBA, njit, prange

# 9x9 derivative-of-Gaussian-like kernel, copied verbatim from cvRecode.cpp.
_KERNEL_DX = np.array(
    [
        [-0.000000143284235, -0.000003558691641, -0.000028902492951, -0.000064765993382, 0.0,
         0.000064765993382, 0.000028902492951, 0.000003558691641, 0.000000143284235],
        [-0.000004744922188, -0.000117847682078, -0.000957119116802, -0.002144755142391, 0.0,
         0.002144755142391, 0.000957119116802, 0.000117847682078, 0.000004744922188],
        [-0.000057804985902, -0.001435678675203, -0.011660097860113, -0.026128466569370, 0.0,
         0.026128466569370, 0.011660097860113, 0.001435678675203, 0.000057804985902],
        [-0.000259063973527, -0.006434265427174, -0.052256933138740, -0.117099663048638, 0.0,
         0.117099663048638, 0.052256933138740, 0.006434265427174, 0.000259063973527],
        [-0.000427124283626, -0.010608310271112, -0.086157117207395, -0.193064705260108, 0.0,
         0.193064705260108, 0.086157117207395, 0.010608310271112, 0.000427124283626],
        [-0.000259063973527, -0.006434265427174, -0.052256933138740, -0.117099663048638, 0.0,
         0.117099663048638, 0.052256933138740, 0.006434265427174, 0.000259063973527],
        [-0.000057804985902, -0.001435678675203, -0.011660097860113, -0.026128466569370, 0.0,
         0.026128466569370, 0.011660097860113, 0.001435678675203, 0.000057804985902],
        [-0.000004744922188, -0.000117847682078, -0.000957119116802, -0.002144755142391, 0.0,
         0.002144755142391, 0.000957119116802, 0.000117847682078, 0.000004744922188],
        [-0.000000143284235, -0.000003558691641, -0.000028902492951, -0.000064765993382, 0.0,
         0.000064765993382, 0.000028902492951, 0.000003558691641, 0.000000143284235],
    ]
)
_KERNEL_DY = _KERNEL_DX.T

# The 9x9 kernel above is a separable derivative-of-Gaussian: it is rank-1
# to within ~1e-16 (SVD's 2nd singular value is ~1e-15 relative to the 1st),
# i.e. _KERNEL_DX == outer(_SMOOTH_1D, _DERIV_1D) up to float noise. Two 1D
# correlations therefore reproduce the 2D correlation's *rounded* dx/dy
# output exactly (verified on both sample images: rounded results identical,
# float divergence ~1e-13, far below the 0.5 rounding threshold) while doing
# 2*9=18 multiply-adds per pixel instead of 81 -- ~2x faster in practice.
_u, _s, _vt = np.linalg.svd(_KERNEL_DX)
_SMOOTH_1D = _u[:, 0] * np.sqrt(_s[0])  # symmetric smoothing profile
_DERIV_1D = _vt[0, :] * np.sqrt(_s[0])  # antisymmetric derivative profile

_CANNY_SHIFT = 15
_TG22 = int(0.4142135623730950488016887242097 * (1 << _CANNY_SHIFT) + 0.5)


def _mag_nms_numpy(dx: np.ndarray, dy: np.ndarray, low: int, high: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized magnitude + fixed-point-sector NMS. Mirrors the inner loop
    of ``cvRecodedCanny`` after gradient computation. Returns
    ``(candidate, strong)``."""
    mag = np.rint(np.sqrt(dx.astype(np.float64) ** 2 + dy.astype(np.float64) ** 2)).astype(np.int64)

    # zero-pad by one pixel on every side: this matches the C++ ring buffer's
    # boundary sentinels (column -1/width and the virtual row above/below
    # the image are all magnitude 0), which differs from BORDER_REPLICATE.
    mag_p = np.pad(mag, 1, mode="constant", constant_values=0)
    center = mag_p[1:-1, 1:-1]
    left = mag_p[1:-1, 0:-2]
    right = mag_p[1:-1, 2:]
    up = mag_p[0:-2, 1:-1]
    down = mag_p[2:, 1:-1]
    up_left = mag_p[0:-2, 0:-2]
    up_right = mag_p[0:-2, 2:]
    down_left = mag_p[2:, 0:-2]
    down_right = mag_p[2:, 2:]

    x = np.abs(dx)
    y = np.abs(dy)
    tg22x = x * _TG22
    tg67x = tg22x + ((x + x) << _CANNY_SHIFT)
    y_shifted = y << _CANNY_SHIFT

    horiz_mask = y_shifted < tg22x
    vert_mask = y_shifted > tg67x

    s_raw = np.bitwise_xor(dx, dy)
    sign_pos = s_raw >= 0  # s = (s_raw < 0) ? -1 : 1

    horiz_ok = (center > left) & (center >= right)
    vert_ok = (center > up) & (center >= down)
    diag1_ok = (center > up_left) & (center > down_right)  # s == 1
    diag2_ok = (center > up_right) & (center > down_left)  # s == -1
    diag_ok = np.where(sign_pos, diag1_ok, diag2_ok)

    is_local_max = np.where(horiz_mask, horiz_ok, np.where(vert_mask, vert_ok, diag_ok))

    candidate = (mag > low) & is_local_max
    strong = candidate & (mag > high)
    return candidate, strong


if HAS_NUMBA:

    @njit(cache=True, parallel=True)
    def _gradient_rows_numba(src: np.ndarray, deriv: np.ndarray, smooth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Row pass of the separable gradient: correlates every row with the
        antisymmetric ``deriv`` profile (-> ``t_d``, later smoothed along y
        to give dx) and the symmetric ``smooth`` profile (-> ``t_s``, later
        differentiated along y to give dy), reading the ``uint8`` source
        once for both. Boundary handling is edge-replicate (``mode=
        "nearest"``).

        The summation order deliberately mirrors SciPy's ``NI_Correlate1D``
        folded symmetric/antisymmetric branches (``center*w[0]`` first, then
        ``(left +/- right) * w_left`` from the outermost tap inward, using
        the *left*-side weight), which is the branch scipy takes for these
        two profiles -- so the float64 results are bit-identical to the
        ``correlate1d`` fallback, not merely equal after rounding.
        """
        rows, cols = src.shape
        half = deriv.shape[0] // 2
        t_d = np.empty((rows, cols), dtype=np.float64)
        t_s = np.empty((rows, cols), dtype=np.float64)
        for i in prange(rows):
            for j in range(cols):
                center = float(src[i, j])
                sd = center * deriv[half]
                ss = center * smooth[half]
                for jj in range(-half, 0):
                    left = float(src[i, max(j + jj, 0)])
                    right = float(src[i, min(j - jj, cols - 1)])
                    sd += (left - right) * deriv[half + jj]
                    ss += (left + right) * smooth[half + jj]
                t_d[i, j] = sd
                t_s[i, j] = ss
        return t_d, t_s

    @njit(cache=True, parallel=True)
    def _gradient_cols_numba(
        t_d: np.ndarray, t_s: np.ndarray, deriv: np.ndarray, smooth: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Column pass (same folded order as :func:`_gradient_rows_numba`),
        fused with the ``CV_16SC1`` rounding of dx/dy and the rounded-L2
        magnitude -- three full-image passes (``rint``, ``astype``, ``hypot``)
        that the NumPy path runs separately."""
        rows, cols = t_d.shape
        half = deriv.shape[0] // 2
        dx = np.empty((rows, cols), dtype=np.int16)
        dy = np.empty((rows, cols), dtype=np.int16)
        mag = np.empty((rows, cols), dtype=np.int32)
        for i in prange(rows):
            for j in range(cols):
                sx = t_d[i, j] * smooth[half]
                sy = t_s[i, j] * deriv[half]
                for ii in range(-half, 0):
                    iu = max(i + ii, 0)
                    idn = min(i - ii, rows - 1)
                    sx += (t_d[iu, j] + t_d[idn, j]) * smooth[half + ii]
                    sy += (t_s[iu, j] - t_s[idn, j]) * deriv[half + ii]
                fdx = np.rint(sx)
                fdy = np.rint(sy)
                dx[i, j] = np.int16(fdx)
                dy[i, j] = np.int16(fdy)
                mag[i, j] = np.int32(np.rint(math.sqrt(fdx * fdx + fdy * fdy)))
        return dx, dy, mag

    @njit(cache=True, parallel=True)
    def _nms_numba(
        dx: np.ndarray, dy: np.ndarray, mag: np.ndarray, low: int, high: int, tg22: int, canny_shift: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fixed-point-sector non-maxima suppression; mirrors the inner loop
        of ``cvRecodedCanny``. Rows are independent (each only reads its
        neighbours' magnitudes and writes its own row), hence ``prange``."""
        rows, cols = dx.shape
        candidate = np.zeros((rows, cols), dtype=np.bool_)
        strong = np.zeros((rows, cols), dtype=np.bool_)

        for i in prange(rows):
            for j in range(cols):
                m = np.int64(mag[i, j])
                if m <= low:
                    continue

                left = mag[i, j - 1] if j - 1 >= 0 else 0
                right = mag[i, j + 1] if j + 1 < cols else 0
                up = mag[i - 1, j] if i - 1 >= 0 else 0
                down = mag[i + 1, j] if i + 1 < rows else 0
                up_left = mag[i - 1, j - 1] if (i - 1 >= 0 and j - 1 >= 0) else 0
                up_right = mag[i - 1, j + 1] if (i - 1 >= 0 and j + 1 < cols) else 0
                down_left = mag[i + 1, j - 1] if (i + 1 < rows and j - 1 >= 0) else 0
                down_right = mag[i + 1, j + 1] if (i + 1 < rows and j + 1 < cols) else 0

                dxi = np.int64(dx[i, j])
                dyi = np.int64(dy[i, j])
                x = abs(dxi)
                y = abs(dyi)
                tg22x = x * tg22
                tg67x = tg22x + ((x + x) << canny_shift)
                y_shifted = y << canny_shift

                if y_shifted < tg22x:
                    is_max = m > left and m >= right
                elif y_shifted > tg67x:
                    is_max = m > up and m >= down
                else:
                    s_raw = dxi ^ dyi
                    if s_raw >= 0:
                        is_max = m > up_left and m > down_right
                    else:
                        is_max = m > up_right and m > down_left

                if is_max:
                    candidate[i, j] = True
                    if m > high:
                        strong[i, j] = True

        return candidate, strong

    @njit(cache=True)
    def _hysteresis_numba(candidate: np.ndarray, strong: np.ndarray) -> np.ndarray:
        """Seeded 8-connected flood fill from every strong pixel through the
        candidate mask -- the original ``cvCanny`` stack-based hysteresis,
        which is the same set the NumPy path computes with ``label`` +
        ``isin`` (a component is kept iff it contains a strong pixel), but
        touching only the candidate pixels rather than the whole image
        three times. Writes the final ``uint8`` 0/255 edge map directly."""
        rows, cols = candidate.shape
        edges = np.zeros((rows, cols), dtype=np.uint8)

        # every candidate is pushed at most once (marked on push)
        n_candidates = np.count_nonzero(candidate)
        stack_i = np.empty(n_candidates, dtype=np.int32)
        stack_j = np.empty(n_candidates, dtype=np.int32)

        for i0 in range(rows):
            for j0 in range(cols):
                if not strong[i0, j0] or edges[i0, j0] != 0:
                    continue
                edges[i0, j0] = 255
                stack_i[0] = i0
                stack_j[0] = j0
                top = 1
                while top > 0:
                    top -= 1
                    ci = stack_i[top]
                    cj = stack_j[top]
                    for ni in range(max(ci - 1, 0), min(ci + 2, rows)):
                        for nj in range(max(cj - 1, 0), min(cj + 2, cols)):
                            if candidate[ni, nj] and edges[ni, nj] == 0:
                                edges[ni, nj] = 255
                                stack_i[top] = ni
                                stack_j[top] = nj
                                top += 1
        return edges


def _recoded_canny_numpy(gray: np.ndarray, low: int, high: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = gray.astype(np.float64)
    # cv::filter2D performs correlation (not convolution), BORDER_REPLICATE.
    # Separable form of the 9x9 kernel (see _SMOOTH_1D/_DERIV_1D above):
    # _KERNEL_DX smooths along y (axis 0) and differentiates along x (axis
    # 1); _KERNEL_DY = _KERNEL_DX.T swaps those roles.
    dx_f = correlate1d(correlate1d(src, _DERIV_1D, axis=1, mode="nearest"), _SMOOTH_1D, axis=0, mode="nearest")
    dy_f = correlate1d(correlate1d(src, _SMOOTH_1D, axis=1, mode="nearest"), _DERIV_1D, axis=0, mode="nearest")
    # emulate the CV_16SC1 output of filter2D (round to nearest integer)
    dx = np.rint(dx_f).astype(np.int64)
    dy = np.rint(dy_f).astype(np.int64)

    candidate, strong = _mag_nms_numpy(dx, dy, low, high)

    structure = np.ones((3, 3), dtype=int)
    labels, num_labels = ndimage.label(candidate, structure=structure)
    if num_labels > 0:
        strong_labels = np.unique(labels[strong])
        strong_labels = strong_labels[strong_labels != 0]
        final_edge = np.isin(labels, strong_labels)
    else:
        final_edge = np.zeros_like(candidate)

    edges = np.where(final_edge, 255, 0).astype(np.uint8)
    return edges, dx.astype(np.int16), dy.astype(np.int16)


def _recoded_canny_numba(gray: np.ndarray, low: int, high: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = np.ascontiguousarray(gray, dtype=np.uint8)
    t_d, t_s = _gradient_rows_numba(src, _DERIV_1D, _SMOOTH_1D)
    dx, dy, mag = _gradient_cols_numba(t_d, t_s, _DERIV_1D, _SMOOTH_1D)
    candidate, strong = _nms_numba(dx, dy, mag, low, high, _TG22, _CANNY_SHIFT)
    edges = _hysteresis_numba(candidate, strong)
    return edges, dx, dy


def recoded_canny(gray: np.ndarray, low_thresh: float, high_thresh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Edge map + gradients for one image. Mirrors ``cvRecodedCanny``.

    ``low_thresh``/``high_thresh`` are expected already scaled by 256
    (matching the C++ call convention: ``cannyThrLow*256``,
    ``cannyThrHigh*256``), and are floored to integers after min/max
    swapping, exactly as the reference does.

    Returns ``(edges, dx, dy)``: ``edges`` is a ``uint8`` 0/255 map, ``dx``/
    ``dy`` are the raw (rounded) gradient components as in the C++
    ``CV_16SC1`` outputs.

    With Numba installed the whole stage runs as four JIT kernels (row
    pass, column pass fused with rounding + magnitude, NMS, seeded
    hysteresis), the first three parallelized over image rows; without it
    the SciPy/NumPy form (:func:`_recoded_canny_numpy`) is used. Both paths
    were verified to give bit-identical ``(edges, dx, dy)`` on every pyramid
    level of both sample images plus synthetic noise/gradient images.
    """
    low_thresh, high_thresh = sorted((low_thresh, high_thresh))
    low = int(np.floor(low_thresh))
    high = int(np.floor(high_thresh))

    if HAS_NUMBA:
        return _recoded_canny_numba(gray, low, high)
    return _recoded_canny_numpy(gray, low, high)
