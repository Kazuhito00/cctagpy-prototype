"""Marker identification, ported from ``src/cctag/Identification.{hpp,cpp}``.

After the outer ellipse has been located (Phase 4), this stage samples the
marker's concentric-ring pattern along radial "image cuts", optimizes the
imaged center/homography by making all cuts agree with each other, then
matches the resulting 1D signature against the marker bank
(``markers_bank.py``) to assign a numeric ID.

``SubPixEdgeOptimizer`` in the C++ source is entirely dead code (guarded by
a macro that is always undefined) and is not ported; the active subpixel
refinement is :func:`outer_edge_refinement` below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from cctagpy._numba_utils import HAS_NUMBA, njit, prange
from cctagpy.cctag import CCTag, status
from cctagpy.conditioner import condition_point, condition_points, conditioner_from_ellipse
from cctagpy.geometry import Circle, Ellipse, point_on_ellipse
from cctagpy.params import Parameters
from cctagpy.transform2d import projective_transform_conic

_KERNEL_A = [-0.0000, -0.0003, -0.1065, -0.7863, 0.0, 0.7863, 0.1065, 0.0003, 0.0000]  # sigma=0.5
_KERNEL_B = [-0.0044, -0.0540, -0.2376, -0.3450, 0.0, 0.3450, 0.2376, 0.0540, 0.0044]  # sigma=1
_KERNEL_C = [-0.0366, -0.1113, -0.1801, -0.1594, 0.0, 0.1594, 0.1801, 0.1113, 0.0366]  # sigma=1.5

_CUT_LENGTH_FACTOR = 3.0 * np.sqrt(2.0)  # outer_edge_refinement's search-segment length / scale


@dataclass
class ImageCut:
    """Mirrors ``cctag::ImageCut``: a radial segment ``start`` -> ``stop``
    with the sampled signal only covering ``[begin_sig, end_sig]`` of it.

    ``out_of_bounds`` is sticky: once set, nothing in this port (matching
    the C++ source) ever resets it back to ``False`` -- a cut that goes out
    of bounds for one trial homography during the center-search grid stays
    "dead" for every subsequent trial that reuses the same cut object.
    """

    start: np.ndarray
    stop: np.ndarray
    stop_grad: np.ndarray
    begin_sig: float
    end_sig: float
    img_signal: np.ndarray
    out_of_bounds: bool = False


def apply_homography(h: np.ndarray, x: float, y: float) -> tuple[float, float]:
    u = h[0, 0] * x + h[0, 1] * y + h[0, 2]
    v = h[1, 0] * x + h[1, 1] * y + h[1, 2]
    w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    return u / w, v / w


def get_pixel_bilinear(src: np.ndarray, x: float, y: float) -> float:
    """Mirrors ``getPixelBilinear`` -- note the extra ``/2`` (not standard
    bilinear interpolation), applied consistently everywhere a cut signal
    is sampled, so it does not bias anything downstream."""
    px, py = int(x), int(y)
    fx, fy = x - px, y - py
    p1 = float(src[py, px])
    p2 = float(src[py, px + 1])
    p3 = float(src[py + 1, px])
    p4 = float(src[py + 1, px + 1])
    w1, w2, w3, w4 = (1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy
    return (p1 * w1 + p2 * w2 + p3 * w3 + p4 * w4) / 2.0


def cut_interpolated(cut: ImageCut, src: np.ndarray) -> None:
    """Raw pixel-space sampling from ``cut.start`` to ``cut.stop``. Mirrors
    ``cutInterpolated``. Breaks (and marks out-of-bounds) at the first
    out-of-range sample."""
    diff = cut.stop - cut.start
    start = cut.start + diff * cut.begin_sig if cut.begin_sig != 0.0 else cut.start.copy()
    stop = cut.start + diff * cut.end_sig if cut.end_sig != 1.0 else cut.stop.copy()

    n = len(cut.img_signal)
    step = (stop - start) / (n - 1.0)
    x, y = start
    rows, cols = src.shape
    for i in range(n):
        if 1.0 <= x < cols - 1 and 1.0 <= y < rows - 1:
            cut.img_signal[i] = get_pixel_bilinear(src, x, y)
        else:
            cut.out_of_bounds = True
            break
        x += step[0]
        y += step[1]


def collect_cuts(
    src: np.ndarray,
    center: np.ndarray,
    outer_positions: np.ndarray,
    outer_gradients: np.ndarray,
    sample_cut_length: int,
    begin_sig: float,
) -> list[ImageCut]:
    """Mirrors ``collectCuts``: one cut per outer point, dropped if OOB.

    Vectorized over all candidate outer points at once instead of calling
    :func:`cut_interpolated` in a Python loop -- every cut here shares the
    same ``center``/``begin_sig``/``end_sig``, so, exactly as in
    :func:`get_signals`, sampling all of them is a handful of array ops.
    A cut dropped for being out-of-bounds is never consulted again by any
    caller, so there is no need to reproduce ``cut_interpolated``'s
    "truncate at the first OOB sample" behavior here -- only cuts that are
    in-bounds for their *entire* length are kept, matching the original's
    net effect exactly.
    """
    k = len(outer_positions)
    if k == 0:
        return []

    center = np.asarray(center, dtype=np.float64)
    stops = np.asarray(outer_positions, dtype=np.float64)  # (K, 2)
    starts = np.broadcast_to(center, stops.shape)
    diff = stops - starts
    eff_start = starts + diff * begin_sig if begin_sig != 0.0 else starts

    n = sample_cut_length
    t = np.arange(n, dtype=np.float64) / (n - 1.0)
    x = eff_start[:, 0:1] + t[None, :] * (stops[:, 0:1] - eff_start[:, 0:1])
    y = eff_start[:, 1:2] + t[None, :] * (stops[:, 1:2] - eff_start[:, 1:2])

    rows, cols = src.shape
    in_bounds = (x >= 1.0) & (x < cols - 1) & (y >= 1.0) & (y < rows - 1)
    fully_in_bounds = in_bounds.all(axis=1)
    values = _bilinear_batch(src, x, y)

    cuts: list[ImageCut] = []
    for i in range(k):
        if not fully_in_bounds[i]:
            continue
        cuts.append(
            ImageCut(
                start=center.copy(),
                stop=stops[i].copy(),
                stop_grad=np.asarray(outer_gradients[i], dtype=np.float64).copy(),
                begin_sig=begin_sig,
                end_sig=1.0,
                img_signal=values[i].copy(),
                out_of_bounds=False,
            )
        )
    return cuts


def extract_signal_using_homography(cut: ImageCut, src: np.ndarray, h: np.ndarray, h_inv: np.ndarray) -> None:
    """Mirrors ``extractSignalUsingHomography``: samples along the ray from
    the marker-plane origin through the back-projected stop point, applying
    the FORWARD homography per sample. Unlike :func:`cut_interpolated`,
    sampling continues past an out-of-bounds sample (only the flag is set)."""
    backproj_x, backproj_y = apply_homography(h_inv, cut.stop[0], cut.stop[1])

    x_start, y_start = (0.0, 0.0) if cut.begin_sig == 0.0 else (backproj_x * cut.begin_sig, backproj_y * cut.begin_sig)
    x_stop, y_stop = (backproj_x, backproj_y) if cut.end_sig == 1.0 else (backproj_x * cut.end_sig, backproj_y * cut.end_sig)

    n = len(cut.img_signal)
    step_x = (x_stop - x_start) / (n - 1.0)
    step_y = (y_stop - y_start) / (n - 1.0)

    x, y = x_start, y_start
    rows, cols = src.shape
    for i in range(n):
        x_res, y_res = apply_homography(h, x, y)
        if 0.0 <= x_res < cols - 1 and 0.0 <= y_res < rows - 1:
            cut.img_signal[i] = get_pixel_bilinear(src, x_res, y_res)
        else:
            cut.out_of_bounds = True
        x += step_x
        y += step_y


def _bilinear_batch(src: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized :func:`get_pixel_bilinear` over arrays of positions.

    Positions may lie outside ``src`` (indices are clipped to stay
    in-bounds for the gather); the result at such positions is meaningless
    and must be discarded by the caller via an independently computed
    bounds mask -- this never decides in/out-of-bounds itself."""
    rows, cols = src.shape
    px = np.floor(x).astype(np.int64)
    py = np.floor(y).astype(np.int64)
    fx = x - px
    fy = y - py
    px = np.clip(px, 0, cols - 2)
    py = np.clip(py, 0, rows - 2)
    p1 = src[py, px].astype(np.float64)
    p2 = src[py, px + 1].astype(np.float64)
    p3 = src[py + 1, px].astype(np.float64)
    p4 = src[py + 1, px + 1].astype(np.float64)
    w1, w2, w3, w4 = (1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy
    return (p1 * w1 + p2 * w2 + p3 * w3 + p4 * w4) / 2.0


if HAS_NUMBA:

    @njit(cache=True, inline="always")
    def _cut_signal_numba(
        sx: float, sy: float, bs: float, es: float, h: np.ndarray, h_inv: np.ndarray, src: np.ndarray,
        rows: int, cols: int, t_all: np.ndarray, out_row: np.ndarray,
    ) -> bool:
        """One cut under one homography: backproject its outer stop with
        ``h_inv``, walk the ``begin_sig``..``end_sig`` segment in ``len(t_all)``
        samples, map each sample forward with ``h`` and bilinearly sample
        ``src`` into ``out_row``. Returns whether every sample was in bounds.
        Shared by the single- and multi-homography kernels below so the
        fused per-sample arithmetic exists once."""
        bx = h_inv[0, 0] * sx + h_inv[0, 1] * sy + h_inv[0, 2]
        by = h_inv[1, 0] * sx + h_inv[1, 1] * sy + h_inv[1, 2]
        bw = h_inv[2, 0] * sx + h_inv[2, 1] * sy + h_inv[2, 2]
        bpx = bx / bw
        bpy = by / bw

        x_start = 0.0 if bs == 0.0 else bpx * bs
        y_start = 0.0 if bs == 0.0 else bpy * bs
        x_stop = bpx if es == 1.0 else bpx * es
        y_stop = bpy if es == 1.0 else bpy * es

        in_bounds_all = True
        for i in range(t_all.shape[0]):
            t = t_all[i]
            x = x_start + t * (x_stop - x_start)
            y = y_start + t * (y_stop - y_start)
            u = h[0, 0] * x + h[0, 1] * y + h[0, 2]
            v = h[1, 0] * x + h[1, 1] * y + h[1, 2]
            w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
            xr = u / w
            yr = v / w
            if not (0.0 <= xr < cols - 1 and 0.0 <= yr < rows - 1):
                in_bounds_all = False
                continue
            px = int(np.floor(xr))
            py = int(np.floor(yr))
            fx = xr - px
            fy = yr - py
            p1 = float(src[py, px])
            p2 = float(src[py, px + 1])
            p3 = float(src[py + 1, px])
            p4 = float(src[py + 1, px + 1])
            w1 = (1.0 - fx) * (1.0 - fy)
            w2 = fx * (1.0 - fy)
            w3 = (1.0 - fx) * fy
            w4 = fx * fy
            out_row[i] = (p1 * w1 + p2 * w2 + p3 * w3 + p4 * w4) / 2.0
        return in_bounds_all

    @njit(cache=True, inline="always")
    def _sample_positions_numba(n: int) -> np.ndarray:
        """``i / (n - 1)`` for every sample -- computed once per kernel call
        instead of once per (cut, sample); same expression, same values."""
        t_all = np.empty(n, dtype=np.float64)
        for i in range(n):
            t_all[i] = i / (n - 1.0)
        return t_all

    @njit(cache=True)
    def _get_signals_numba(
        stops: np.ndarray, begin_sig: np.ndarray, end_sig: np.ndarray, h: np.ndarray, h_inv: np.ndarray,
        src: np.ndarray, n: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fused per-(cut, sample) kernel (see :func:`_cut_signal_numba`)
        instead of the ~15 full-array temporaries (`bp`, `x_start`, `y_start`,
        `x`, `y`, `u`, `v`, `w`, `x_res`, `y_res`, `in_bounds`, ...) the NumPy
        form allocates per call. Bounds are checked before sampling, so
        (unlike :func:`_bilinear_batch`) no defensive index clipping is
        needed -- an out-of-bounds sample is simply never read."""
        c = stops.shape[0]
        rows, cols = src.shape
        values = np.zeros((c, n), dtype=np.float64)
        all_in_bounds = np.empty(c, dtype=np.bool_)
        t_all = _sample_positions_numba(n)

        for ci in range(c):
            all_in_bounds[ci] = _cut_signal_numba(
                stops[ci, 0], stops[ci, 1], begin_sig[ci], end_sig[ci], h, h_inv, src, rows, cols, t_all, values[ci]
            )

        return values, all_in_bounds


def _get_signals_numpy(cuts: list[ImageCut], h: np.ndarray, src: np.ndarray) -> None:
    live = [c for c in cuts if not c.out_of_bounds]
    if not live:
        return

    h_inv = np.linalg.inv(h)
    rows, cols = src.shape
    stops = np.array([c.stop for c in live], dtype=np.float64)
    pts_h = np.column_stack([stops, np.ones(len(live))])
    bp = (h_inv @ pts_h.T).T
    bp_xy = bp[:, :2] / bp[:, 2:3]

    begin_sig = np.array([c.begin_sig for c in live], dtype=np.float64)
    end_sig = np.array([c.end_sig for c in live], dtype=np.float64)
    x_start = np.where(begin_sig == 0.0, 0.0, bp_xy[:, 0] * begin_sig)
    y_start = np.where(begin_sig == 0.0, 0.0, bp_xy[:, 1] * begin_sig)
    x_stop = np.where(end_sig == 1.0, bp_xy[:, 0], bp_xy[:, 0] * end_sig)
    y_stop = np.where(end_sig == 1.0, bp_xy[:, 1], bp_xy[:, 1] * end_sig)

    n = len(live[0].img_signal)
    t = np.arange(n, dtype=np.float64) / (n - 1.0)
    x = x_start[:, None] + t[None, :] * (x_stop - x_start)[:, None]
    y = y_start[:, None] + t[None, :] * (y_stop - y_start)[:, None]

    u = h[0, 0] * x + h[0, 1] * y + h[0, 2]
    v = h[1, 0] * x + h[1, 1] * y + h[1, 2]
    w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    x_res = u / w
    y_res = v / w

    in_bounds = (x_res >= 0.0) & (x_res < cols - 1) & (y_res >= 0.0) & (y_res < rows - 1)
    all_in_bounds = in_bounds.all(axis=1)
    values = _bilinear_batch(src, x_res, y_res)

    for k, cut in enumerate(live):
        if all_in_bounds[k]:
            cut.img_signal[:] = values[k]
        else:
            cut.out_of_bounds = True


def get_signals(cuts: list[ImageCut], h: np.ndarray, src: np.ndarray) -> None:
    """Vectorized equivalent of calling :func:`extract_signal_using_homography`
    for every still-live cut. ``out_of_bounds`` is sticky, so cuts already
    flagged are skipped entirely -- their stale ``img_signal`` is never
    consulted by any downstream consumer (:func:`cost_function_glob`,
    :func:`orazio_distance_robust`, the ``correct_cuts`` filter in
    :func:`refine_conic_family_glob`), so recomputing them would be wasted
    work. All live cuts share ``sample_cut_length`` samples (they all
    originate from the same :func:`collect_cuts` call). Dispatches to a
    fused Numba pixel-loop kernel when available (see
    :func:`_get_signals_numba`), falling back to the vectorized NumPy form
    otherwise -- verified bit-identical on real markers either way."""
    live = [c for c in cuts if not c.out_of_bounds]
    if not live:
        return

    if not HAS_NUMBA:
        _get_signals_numpy(cuts, h, src)
        return

    h_inv = np.linalg.inv(h)
    stops = np.array([c.stop for c in live], dtype=np.float64)
    begin_sig = np.array([c.begin_sig for c in live], dtype=np.float64)
    end_sig = np.array([c.end_sig for c in live], dtype=np.float64)
    n = len(live[0].img_signal)

    values, all_in_bounds = _get_signals_numba(stops, begin_sig, end_sig, h, h_inv, src, n)

    for k, cut in enumerate(live):
        if all_in_bounds[k]:
            cut.img_signal[:] = values[k]
        else:
            cut.out_of_bounds = True


if HAS_NUMBA:

    @njit(cache=True, parallel=True)
    def _get_signals_multi_h_numba(
        stops: np.ndarray, begin_sig: np.ndarray, end_sig: np.ndarray, h_stack: np.ndarray, h_inv_stack: np.ndarray,
        src: np.ndarray, n: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """:func:`_get_signals_numba` over ``K`` candidate homographies at
        once. Every (candidate, cut) pair is independent (it writes only its
        own ``values[ki, ci]`` row and ``fresh_in_bounds[ki, ci]`` and
        allocates nothing), so the ``prange`` runs over the flattened
        ``K * C`` pairs -- a few thousand work items -- rather than over the
        ~25 candidates alone, which would leave most threads idle."""
        k_trials = h_stack.shape[0]
        c = stops.shape[0]
        rows, cols = src.shape
        values = np.zeros((k_trials, c, n), dtype=np.float64)
        fresh_in_bounds = np.empty((k_trials, c), dtype=np.bool_)
        t_all = _sample_positions_numba(n)

        for flat in prange(k_trials * c):
            ki = flat // c
            ci = flat - ki * c
            fresh_in_bounds[ki, ci] = _cut_signal_numba(
                stops[ci, 0], stops[ci, 1], begin_sig[ci], end_sig[ci], h_stack[ki], h_inv_stack[ki], src,
                rows, cols, t_all, values[ki, ci],
            )

        return fresh_in_bounds, values


def _get_signals_multi_h_numpy(live_cuts: list[ImageCut], h_stack: np.ndarray, src: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched :func:`get_signals` over ``K`` candidate homographies at once,
    for the SAME set of still-live cuts. Returns ``(fresh_in_bounds,
    values)`` of shape ``(K, C)`` / ``(K, C, n)`` -- "fresh" because these
    ignore the sticky ``out_of_bounds`` coupling *between* candidates within
    one grid iteration (the C++ reference shares one mutable ``vCuts``
    across all 25 candidates in a `for` loop, so a cut going out of bounds
    under candidate 3 stays excluded for candidates 4-25 too). That
    cross-candidate stickiness only ever *removes* cuts from later
    candidates' cost sums; since which samples are geometrically in bounds
    under a given candidate's homography never depends on any other
    candidate, it is computed here for every (candidate, cut) pair
    independently in one vectorized pass, and the caller
    (:func:`image_center_optimization_glob`) replays the cheap sticky
    bookkeeping across candidates in order afterward -- mathematically
    identical to calling :func:`get_signals` once per candidate in sequence,
    just with the expensive resampling batched 25x."""
    rows, cols = src.shape
    c = len(live_cuts)
    n = len(live_cuts[0].img_signal)

    h_inv_stack = np.linalg.inv(h_stack)  # (K, 3, 3)
    stops = np.array([cut.stop for cut in live_cuts], dtype=np.float64)  # (C, 2)
    pts_h = np.column_stack([stops, np.ones(c)])  # (C, 3)
    # `@` (batched matmul), not `einsum`: bp[k] must equal the single-h
    # `(h_inv @ pts_h.T).T` bit-for-bit, and einsum's differently-ordered
    # summation does not reproduce that here (same pitfall as fit_solver's
    # S1/S2/S3 -- verified batched `@` matches a per-item loop exactly).
    bp = pts_h[None, :, :] @ np.transpose(h_inv_stack, (0, 2, 1))  # (K, C, 3)
    bp_xy = bp[:, :, :2] / bp[:, :, 2:3]

    begin_sig = np.array([cut.begin_sig for cut in live_cuts], dtype=np.float64)  # (C,)
    end_sig = np.array([cut.end_sig for cut in live_cuts], dtype=np.float64)
    x_start = np.where(begin_sig == 0.0, 0.0, bp_xy[:, :, 0] * begin_sig[None, :])  # (K, C)
    y_start = np.where(begin_sig == 0.0, 0.0, bp_xy[:, :, 1] * begin_sig[None, :])
    x_stop = np.where(end_sig == 1.0, bp_xy[:, :, 0], bp_xy[:, :, 0] * end_sig[None, :])
    y_stop = np.where(end_sig == 1.0, bp_xy[:, :, 1], bp_xy[:, :, 1] * end_sig[None, :])

    t = np.arange(n, dtype=np.float64) / (n - 1.0)
    x = x_start[:, :, None] + t[None, None, :] * (x_stop - x_start)[:, :, None]  # (K, C, n)
    y = y_start[:, :, None] + t[None, None, :] * (y_stop - y_start)[:, :, None]

    h00 = h_stack[:, 0, 0][:, None, None]
    h01 = h_stack[:, 0, 1][:, None, None]
    h02 = h_stack[:, 0, 2][:, None, None]
    h10 = h_stack[:, 1, 0][:, None, None]
    h11 = h_stack[:, 1, 1][:, None, None]
    h12 = h_stack[:, 1, 2][:, None, None]
    h20 = h_stack[:, 2, 0][:, None, None]
    h21 = h_stack[:, 2, 1][:, None, None]
    h22 = h_stack[:, 2, 2][:, None, None]
    u = h00 * x + h01 * y + h02
    v = h10 * x + h11 * y + h12
    w = h20 * x + h21 * y + h22
    x_res = u / w
    y_res = v / w

    in_bounds = (x_res >= 0.0) & (x_res < cols - 1) & (y_res >= 0.0) & (y_res < rows - 1)
    fresh_in_bounds = in_bounds.all(axis=2)  # (K, C)
    values = _bilinear_batch(src, x_res, y_res)  # (K, C, n)
    return fresh_in_bounds, values


def _get_signals_multi_h(live_cuts: list[ImageCut], h_stack: np.ndarray, src: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched :func:`get_signals` over ``K`` candidate homographies at once,
    for the SAME set of still-live cuts. Returns ``(fresh_in_bounds,
    values)`` of shape ``(K, C)`` / ``(K, C, n)`` -- "fresh" because these
    ignore the sticky ``out_of_bounds`` coupling *between* candidates within
    one grid iteration (see :func:`_get_signals_multi_h_numpy`'s docstring
    for the full argument); the caller replays that coupling afterward.
    Dispatches to the fused Numba kernel when available, falling back to
    the vectorized NumPy form otherwise."""
    if not HAS_NUMBA:
        return _get_signals_multi_h_numpy(live_cuts, h_stack, src)

    stops = np.array([cut.stop for cut in live_cuts], dtype=np.float64)
    begin_sig = np.array([cut.begin_sig for cut in live_cuts], dtype=np.float64)
    end_sig = np.array([cut.end_sig for cut in live_cuts], dtype=np.float64)
    n = len(live_cuts[0].img_signal)
    h_inv_stack = np.linalg.inv(h_stack)
    return _get_signals_multi_h_numba(stops, begin_sig, end_sig, h_stack, h_inv_stack, src, n)


def conv_image_cut(kernel: list[float], signal: np.ndarray) -> tuple[float, float]:
    """Edge-clamped correlation + argmax. Mirrors ``convImageCut``.

    ``scipy.ndimage.correlate1d`` with ``mode="nearest"`` is exactly
    "correlate with the boundary clamped to the edge sample", which is
    what the manual per-sample loop computed by hand (verified equal to
    the loop form on 500 random signals against all three kernels)."""
    from scipy.ndimage import correlate1d

    output = correlate1d(signal, np.asarray(kernel, dtype=np.float64), mode="nearest")
    max_idx = int(np.argmax(output))
    return float(output[max_idx]), float(max_idx)


_KERNELS_STACKED = np.array([_KERNEL_A, _KERNEL_B, _KERNEL_C], dtype=np.float64)


def _outer_edge_refinement_numpy(cut: ImageCut, src: np.ndarray, scale: float, num_samples: int) -> bool:
    cut_length = _CUT_LENGTH_FACTOR * scale
    half_width = cut_length / 2.0

    grad_norm = float(np.hypot(cut.stop_grad[0], cut.stop_grad[1]))
    grad_dir = cut.stop_grad / grad_norm

    p_start = cut.stop - half_width * grad_dir
    p_stop = cut.stop + half_width * grad_dir

    mini_cut = ImageCut(
        start=p_start, stop=p_stop, stop_grad=cut.stop_grad, begin_sig=0.0, end_sig=1.0,
        img_signal=np.zeros(num_samples),
    )
    cut_interpolated(mini_cut, src)
    if mini_cut.out_of_bounds:
        return False

    results = [conv_image_cut(k, mini_cut.img_signal) for k in (_KERNEL_A, _KERNEL_B, _KERNEL_C)]
    max_location = max(results, key=lambda t: t[0])[1]

    step = cut_length / (num_samples - 1.0)
    cut.stop = p_start + step * max_location * grad_dir
    return True


if HAS_NUMBA:

    @njit(cache=True)
    def _outer_edge_refinement_numba_core(
        stop_x: float, stop_y: float, grad_x: float, grad_y: float, src: np.ndarray,
        cut_length: float, num_samples: int, kernels: np.ndarray,
    ) -> tuple[float, float, bool]:
        grad_norm = np.hypot(grad_x, grad_y)
        dir_x = grad_x / grad_norm
        dir_y = grad_y / grad_norm
        half_width = cut_length / 2.0

        p_start_x = stop_x - half_width * dir_x
        p_start_y = stop_y - half_width * dir_y
        step = cut_length / (num_samples - 1.0)
        step_x = step * dir_x
        step_y = step * dir_y

        rows, cols = src.shape
        signal = np.empty(num_samples)
        x, y = p_start_x, p_start_y
        for i in range(num_samples):
            if not (1.0 <= x < cols - 1 and 1.0 <= y < rows - 1):
                return 0.0, 0.0, False
            px = int(x)
            py = int(y)
            fx = x - px
            fy = y - py
            p1 = src[py, px]
            p2 = src[py, px + 1]
            p3 = src[py + 1, px]
            p4 = src[py + 1, px + 1]
            w1 = (1.0 - fx) * (1.0 - fy)
            w2 = fx * (1.0 - fy)
            w3 = (1.0 - fx) * fy
            w4 = fx * fy
            signal[i] = (p1 * w1 + p2 * w2 + p3 * w3 + p4 * w4) / 2.0
            x += step_x
            y += step_y

        best_val = -1.0e300
        best_loc = 0.0
        r = kernels.shape[1] // 2
        for kk in range(kernels.shape[0]):
            for i in range(num_samples):
                s = 0.0
                for t in range(kernels.shape[1]):
                    idx = i + t - r
                    if idx < 0:
                        idx = 0
                    elif idx >= num_samples:
                        idx = num_samples - 1
                    s += signal[idx] * kernels[kk, t]
                if s > best_val:
                    best_val = s
                    best_loc = float(i)

        new_x = p_start_x + step_x * best_loc
        new_y = p_start_y + step_y * best_loc
        return new_x, new_y, True


def _outer_edge_refinement_numba(cut: ImageCut, src: np.ndarray, scale: float, num_samples: int) -> bool:
    cut_length = _CUT_LENGTH_FACTOR * scale
    new_x, new_y, ok = _outer_edge_refinement_numba_core(
        float(cut.stop[0]), float(cut.stop[1]), float(cut.stop_grad[0]), float(cut.stop_grad[1]),
        src, cut_length, num_samples, _KERNELS_STACKED,
    )
    if not ok:
        return False
    cut.stop = np.array([new_x, new_y])
    return True


def outer_edge_refinement(cut: ImageCut, src: np.ndarray, scale: float, num_samples: int) -> bool:
    """Multi-scale derivative-of-Gaussian peak search along the gradient
    direction at ``cut.stop``. Mirrors ``outerEdgeRefinement`` (the active
    subpixel refinement -- ``SubPixEdgeOptimizer`` is dead code).

    Dispatches to a fused Numba kernel when available: the NumPy form makes
    several small-array NumPy/SciPy calls per cut (one bilinear-sampling
    loop, three ``scipy.ndimage.correlate1d`` calls) whose fixed per-call
    overhead dominates for these tiny (~10-20 sample) signals -- this is
    called once per candidate cut, up to ``n_samples_outer_ellipse`` times
    per candidate marker. Verified bit-identical to the NumPy form on real
    images either way (same clamp-to-edge correlation, same first-occurrence
    tie-breaking across the three kernels' peaks)."""
    if HAS_NUMBA:
        return _outer_edge_refinement_numba(cut, src, scale, num_samples)
    return _outer_edge_refinement_numpy(cut, src, scale, num_samples)


if HAS_NUMBA:

    @njit(cache=True)
    def _project_and_refine_cuts_numba(
        stops: np.ndarray, stop_grads: np.ndarray, primal: np.ndarray, dual: np.ndarray, a: float, b: float,
        src: np.ndarray, cut_length: float, num_samples: int, kernels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """For every cut: ``geometry.point_on_ellipse`` of its stop (radial
        rescale in the ellipse's canonical frame, same arithmetic), then
        :func:`_outer_edge_refinement_numba_core` from there. Returns the new
        stops (the refined one where refinement succeeded, else the
        projected one) and the per-cut success flag."""
        c = stops.shape[0]
        new_stops = np.empty((c, 2), dtype=np.float64)
        refined = np.zeros(c, dtype=np.bool_)
        for ci in range(c):
            px = stops[ci, 0]
            py = stops[ci, 1]
            c0 = dual[0, 0] * px + dual[0, 1] * py + dual[0, 2]
            c1 = dual[1, 0] * px + dual[1, 1] * py + dual[1, 2]
            c2 = dual[2, 0] * px + dual[2, 1] * py + dual[2, 2]
            x = c0 / c2
            y = c1 / c2
            denom = np.sqrt((x * x) / (a**2) + (y * y) / (b**2))
            ox = x / denom
            oy = y / denom
            i0 = primal[0, 0] * ox + primal[0, 1] * oy + primal[0, 2]
            i1 = primal[1, 0] * ox + primal[1, 1] * oy + primal[1, 2]
            i2 = primal[2, 0] * ox + primal[2, 1] * oy + primal[2, 2]
            sx = i0 / i2
            sy = i1 / i2
            new_stops[ci, 0] = sx
            new_stops[ci, 1] = sy
            nx, ny, ok = _outer_edge_refinement_numba_core(
                sx, sy, stop_grads[ci, 0], stop_grads[ci, 1], src, cut_length, num_samples, kernels
            )
            if ok:
                new_stops[ci, 0] = nx
                new_stops[ci, 1] = ny
                refined[ci] = True
        return new_stops, refined


def select_cut_cheap_uniform(
    select_size: int,
    outer_ellipse: Ellipse,
    collected_cuts: list[ImageCut],
    src: np.ndarray,
    scale: float,
    num_samples_refine: int,
) -> list[ImageCut]:
    """Mirrors ``selectCutCheapUniform``."""
    select_size = min(select_size, len(collected_cuts))
    if select_size == 0:
        return []

    var_cuts = np.var(np.stack([cut.img_signal for cut in collected_cuts]), axis=1)
    var_max = var_cuts.max()

    ind_to_add: list[int] = []
    if HAS_NUMBA:
        # every cut's "project stop onto the ellipse, then refine it" step
        # in one kernel instead of two Python-level calls per cut
        _, m_t_primal, m_t_dual = outer_ellipse.get_canonic_form()
        stops = np.array([cut.stop for cut in collected_cuts], dtype=np.float64)
        stop_grads = np.array([cut.stop_grad for cut in collected_cuts], dtype=np.float64)
        new_stops, refined = _project_and_refine_cuts_numba(
            stops, stop_grads, m_t_primal, m_t_dual, float(outer_ellipse.a), float(outer_ellipse.b), src,
            _CUT_LENGTH_FACTOR * scale, num_samples_refine, _KERNELS_STACKED,
        )
        for i, cut in enumerate(collected_cuts):
            cut.stop = new_stops[i]
            if refined[i] and var_cuts[i] / var_max > 0.5:
                ind_to_add.append(i)
    else:
        for i, cut in enumerate(collected_cuts):
            cut.stop = point_on_ellipse(outer_ellipse, cut.stop)
            if outer_edge_refinement(cut, src, scale, num_samples_refine):
                if var_cuts[i] / var_max > 0.5:
                    ind_to_add.append(i)

    step = max(1.0, len(ind_to_add) / select_size)
    selected: list[ImageCut] = []
    k = 0
    while True:
        idx = int(k * step)
        if idx < len(ind_to_add) and len(selected) < select_size:
            selected.append(collected_cuts[ind_to_add[idx]])
            k += 1
        else:
            break
    return selected


def compute_homography_from_ellipse_and_imaged_center(ellipse: Ellipse, center: np.ndarray) -> np.ndarray:
    """Closed-form homography (up to a 2D rotation) mapping the marker
    plane's unit circle to ``ellipse`` and its origin to ``center``.
    Mirrors ``computeHomographyFromEllipseAndImagedCenter``."""
    m_canonic, m_t_primal, m_t_dual = ellipse.get_canonic_form()
    xc, yc = apply_homography(m_t_dual, center[0], center[1])

    q11, q22, q33 = m_canonic[0, 0], m_canonic[1, 1], m_canonic[2, 2]

    h = np.zeros((3, 3))
    h[0, 0], h[1, 0], h[2, 0] = q33, 0.0, -q11 * xc
    h[0, 1], h[1, 1], h[2, 1] = q22 * xc * yc, -q11 * xc * xc - q33, q22 * yc
    h[0, 2], h[1, 2], h[2, 2] = -q33 * xc, -q33 * yc, -q33

    d0_arg = q22 * q33 / q11 * (q11 * xc * xc + q22 * yc * yc + q33)
    d2_arg = -q22 * (q11 * xc * xc + q33)
    if d0_arg < 0 or d2_arg < 0:
        raise ValueError("computeHomographyFromEllipseAndImagedCenter: degenerate (negative sqrt argument)")
    diag = [np.sqrt(d0_arg), q33, np.sqrt(d2_arg)]
    for j, d in enumerate(diag):
        h[:, j] *= d

    return m_t_primal @ h


def _compute_homography_batched(ellipse: Ellipse, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized :func:`compute_homography_from_ellipse_and_imaged_center`
    over ``K`` candidate centers against the same (fixed) ``ellipse`` --
    exactly the grid-search access pattern in
    :func:`image_center_optimization_glob`, which previously called the
    scalar function in a Python loop over ``nearby_points`` even though
    ``ellipse.get_canonic_form()`` (the only per-call work that isn't a
    handful of scalar ops) is already memoized and thus identical on every
    iteration. Pure elementwise arithmetic plus one batched ``@``, so this
    is the same computation, not an approximation.

    Returns ``(h_batch, valid)``: ``valid[k]`` is false exactly where the
    scalar function would have raised (``d0_arg < 0 or d2_arg < 0``) --
    note this deliberately mirrors the scalar comparison, not an ``isnan``
    check, since ``m_canonic``'s entries are NumPy floats (unlike
    ``Ellipse.a``/``.b``, which are plain Python floats): a NumPy division
    by exactly zero produces +-inf/nan rather than raising, and
    ``nan < 0``/``inf < 0`` are both false, so a degenerate ``q11`` does
    *not* raise in the scalar path either -- it silently produces a
    nan/inf-poisoned homography that any consumer's cost comparisons will
    simply lose in `<`. Replicated here rather than "fixed" to keep the
    sticky out-of-bounds bookkeeping in the caller identical either way.
    """
    m_canonic, m_t_primal, m_t_dual = ellipse.get_canonic_form()
    x, y = centers[:, 0], centers[:, 1]
    u = m_t_dual[0, 0] * x + m_t_dual[0, 1] * y + m_t_dual[0, 2]
    v = m_t_dual[1, 0] * x + m_t_dual[1, 1] * y + m_t_dual[1, 2]
    w = m_t_dual[2, 0] * x + m_t_dual[2, 1] * y + m_t_dual[2, 2]
    xc, yc = u / w, v / w

    q11, q22, q33 = m_canonic[0, 0], m_canonic[1, 1], m_canonic[2, 2]
    k = centers.shape[0]

    h = np.zeros((k, 3, 3))
    h[:, 0, 0] = q33
    h[:, 2, 0] = -q11 * xc
    h[:, 0, 1] = q22 * xc * yc
    h[:, 1, 1] = -q11 * xc * xc - q33
    h[:, 2, 1] = q22 * yc
    h[:, 0, 2] = -q33 * xc
    h[:, 1, 2] = -q33 * yc
    h[:, 2, 2] = -q33

    d0_arg = q22 * q33 / q11 * (q11 * xc * xc + q22 * yc * yc + q33)
    d2_arg = -q22 * (q11 * xc * xc + q33)
    valid = ~((d0_arg < 0) | (d2_arg < 0))

    # Left un-guarded on purpose (matches the scalar path exactly, including
    # its behavior for the pathological `nan`/`inf` case documented above):
    # `sqrt` of a negative `d0_arg`/`d2_arg` only happens on rows already
    # marked invalid and discarded by the caller, so the RuntimeWarning it
    # prints is expected noise, not a sign of a bug.
    with np.errstate(invalid="ignore"):
        h[:, :, 0] *= np.sqrt(d0_arg)[:, None]
        h[:, :, 1] *= q33
        h[:, :, 2] *= np.sqrt(d2_arg)[:, None]

    return m_t_primal[None, :, :] @ h, valid


def nearby_grid_frame(ellipse: Ellipse) -> tuple[np.ndarray, np.ndarray, float]:
    """The part of :func:`get_nearby_points` that depends only on the outer
    ellipse: its conditioner ``m_t``, the inverse, and the larger semi-axis
    of the conditioned ellipse. The coarse-to-fine search calls
    ``get_nearby_points`` ~10 times per marker with the same ellipse, so
    :func:`refine_conic_family_glob` computes this once and passes it down."""
    m_t = conditioner_from_ellipse(ellipse.center, ellipse.a, ellipse.b)
    m_inv_t = np.linalg.inv(m_t)
    transformed_ellipse = Ellipse(matrix=projective_transform_conic(m_inv_t, ellipse.matrix))
    return m_t, m_inv_t, max(transformed_ellipse.a, transformed_ellipse.b)


def get_nearby_points(
    ellipse: Ellipse,
    center: np.ndarray,
    neighbour_size: float,
    grid_n_sample: int,
    frame: tuple[np.ndarray, np.ndarray, float] | None = None,
) -> np.ndarray:
    """Regular grid of candidate centers, scaled/positioned relative to
    ``ellipse`` via the same conditioner used elsewhere. Mirrors
    ``getNearbyPoints`` (``GRID`` neighborhood only -- the other
    ``NeighborType`` variants are unused stubs in the source). ``frame`` is
    :func:`nearby_grid_frame` of ``ellipse`` if the caller already has it."""
    if frame is None:
        frame = nearby_grid_frame(ellipse)
    m_t, m_inv_t, max_axis = frame
    size = neighbour_size * max_axis

    cond_center = condition_point(center, m_t)
    half_width = size / 2.0
    step = size / (grid_n_sample - 1)

    # i outer / j inner, same order (and the same `base + k * step` float
    # expression per coordinate) as the original nested comprehension
    ii = np.repeat(np.arange(grid_n_sample), grid_n_sample)
    jj = np.tile(np.arange(grid_n_sample), grid_n_sample)
    points = np.column_stack([cond_center[0] - half_width + ii * step, cond_center[1] - half_width + jj * step])
    return condition_points(points, m_inv_t)


def cost_function_glob(h: np.ndarray, cuts: list[ImageCut], src: np.ndarray) -> tuple[float, bool]:
    """Mirrors ``costFunctionGlob``.

    The original sums ``||s_i - s_j||^2`` over every pair ``i < j`` of
    non-out-of-bounds cuts. That sum has a closed form avoiding the O(m^2)
    pairwise loop: for signals stacked as rows of ``s`` (shape (m, n)),
    ``sum_{i<j} ||s_i - s_j||^2 == m * sum(s**2) - ||sum_i s_i||^2``
    (expand the squared norms and use ``sum_{i,j} s_i.s_j == ||sum_i s_i||^2``)."""
    get_signals(cuts, h, src)

    signals = [c.img_signal for c in cuts if not c.out_of_bounds]
    m = len(signals)
    if m < 2:
        return float("inf"), False

    s = np.asarray(signals, dtype=np.float64)
    sq_sum = float(np.sum(s * s))
    sum_vec = s.sum(axis=0)
    total = m * sq_sum - float(sum_vec @ sum_vec)
    res_size = m * (m - 1) // 2
    return total / res_size, True


def _center_search_replay_numpy(fresh_in_bounds: np.ndarray, values: np.ndarray) -> tuple[int, float, bool, np.ndarray]:
    """Sequential replay of the per-candidate cost with the sticky
    out-of-bounds coupling (see :func:`image_center_optimization_glob`).
    Returns ``(best_row, min_res, has_solution, cumulative_dead)``;
    ``best_row`` is -1 when no candidate had two live cuts."""
    k_valid, c = fresh_in_bounds.shape
    cumulative_dead = np.zeros(c, dtype=bool)
    best_row = -1
    min_res = float("inf")
    has_solution = False
    for row in range(k_valid):
        row_mask = ~cumulative_dead & fresh_in_bounds[row]
        m = int(row_mask.sum())
        if m >= 2:
            s = values[row][row_mask]
            sq_sum = float(np.sum(s * s))
            sum_vec = s.sum(axis=0)
            total = m * sq_sum - float(sum_vec @ sum_vec)
            res_size = m * (m - 1) // 2
            res = total / res_size
            has_solution = True
            if res < min_res:
                min_res = res
                best_row = row
        cumulative_dead |= ~fresh_in_bounds[row]
    return best_row, min_res, has_solution, cumulative_dead


if HAS_NUMBA:

    @njit(cache=True)
    def _center_search_replay_numba(fresh_in_bounds: np.ndarray, values: np.ndarray) -> tuple[int, float, bool, np.ndarray]:
        """Same replay as :func:`_center_search_replay_numpy` as one loop
        nest: no per-candidate boolean-mask gather / temporaries. The
        closed-form pairwise cost ``m * sum(s^2) - ||sum_i s_i||^2`` is
        accumulated sequentially here whereas NumPy's ``sum`` and ``@`` use
        pairwise/BLAS orderings, so the two agree to ~1e-14 relative rather
        than bit-for-bit; a candidate can only flip on an exact tie of costs
        at that level (checked on every real call of both sample images:
        no flips)."""
        k_valid, c = fresh_in_bounds.shape
        n = values.shape[2]
        cumulative_dead = np.zeros(c, dtype=np.bool_)
        sum_vec = np.empty(n, dtype=np.float64)
        best_row = -1
        min_res = np.inf
        has_solution = False
        for row in range(k_valid):
            m = 0
            for ci in range(c):
                if not cumulative_dead[ci] and fresh_in_bounds[row, ci]:
                    m += 1
            if m >= 2:
                sq_sum = 0.0
                sum_vec[:] = 0.0
                for ci in range(c):
                    if cumulative_dead[ci] or not fresh_in_bounds[row, ci]:
                        continue
                    for i in range(n):
                        v = values[row, ci, i]
                        sq_sum += v * v
                        sum_vec[i] += v
                dot = 0.0
                for i in range(n):
                    dot += sum_vec[i] * sum_vec[i]
                total = m * sq_sum - dot
                res = total / (m * (m - 1) // 2)
                has_solution = True
                if res < min_res:
                    min_res = res
                    best_row = row
            for ci in range(c):
                if not fresh_in_bounds[row, ci]:
                    cumulative_dead[ci] = True
        return best_row, min_res, has_solution, cumulative_dead


def _center_search_replay(fresh_in_bounds: np.ndarray, values: np.ndarray) -> tuple[int, float, bool, np.ndarray]:
    if HAS_NUMBA:
        best_row, min_res, has_solution, dead = _center_search_replay_numba(fresh_in_bounds, values)
        return int(best_row), float(min_res), bool(has_solution), dead
    return _center_search_replay_numpy(fresh_in_bounds, values)


def image_center_optimization_glob(
    cuts: list[ImageCut],
    center: np.ndarray,
    outer_ellipse: Ellipse,
    neighbour_size: float,
    grid_n_sample: int,
    src: np.ndarray,
    frame: tuple[np.ndarray, np.ndarray, float] | None = None,
) -> tuple[np.ndarray | None, np.ndarray, float, bool]:
    """Mirrors ``imageCenterOptimizationGlob``: derivative-free grid search
    over candidate imaged centers, scoring each by cut-pairwise agreement.

    The C++ reference (and the original scalar port of this function) calls
    ``costFunctionGlob``/``getSignals`` once per candidate center, mutating
    ONE shared ``vCuts`` in place across all of them -- so a cut going out
    of bounds under candidate 3's homography stays excluded (sticky) for
    candidates 4-25 too, within this single grid iteration. That coupling
    is genuinely sequential, but the expensive part underneath it (the
    homography-warped bilinear resample) is not: it's a pure function of
    (cut, candidate homography), independent of every other candidate. So
    the resampling for all candidates x all live cuts is computed in one
    batched call (:func:`_get_signals_multi_h`), and only the cheap sticky
    bookkeeping + cost reduction is replayed candidate-by-candidate in
    order (:func:`_center_search_replay`) -- same winner/tie-break, just
    25x fewer resampling calls per grid iteration.
    """
    nearby_points = get_nearby_points(outer_ellipse, center, neighbour_size, grid_n_sample, frame)

    min_res = float("inf")
    optimal_point = center
    optimal_h = None
    has_solution = False

    live_cuts = [cut for cut in cuts if not cut.out_of_bounds]
    if not live_cuts:
        return optimal_h, optimal_point, min_res, has_solution

    h_batch, homography_valid = _compute_homography_batched(outer_ellipse, nearby_points)
    valid_idx = np.nonzero(homography_valid)[0].tolist()
    if not valid_idx:
        return optimal_h, optimal_point, min_res, has_solution

    h_stack = h_batch[valid_idx]  # (K_valid, 3, 3)
    fresh_in_bounds, values = _get_signals_multi_h(live_cuts, h_stack, src)  # (K_valid, C), (K_valid, C, n)

    best_row, min_res, has_solution, cumulative_dead = _center_search_replay(fresh_in_bounds, values)
    if best_row >= 0:
        optimal_point = nearby_points[valid_idx[best_row]]
        optimal_h = h_batch[valid_idx[best_row]]

    # Persist the final sticky out-of-bounds state so the NEXT (finer) grid
    # iteration correctly skips cuts that died during this one. `img_signal`
    # itself is never written here -- nothing reads it before
    # `refine_conic_family_glob` calls `get_signals` again with the chosen
    # `optimal_h` once the whole coarse-to-fine loop finishes.
    for cut, dead in zip(live_cuts, cumulative_dead):
        if dead:
            cut.out_of_bounds = True

    return optimal_h, optimal_point, min_res, has_solution


def refine_conic_family_glob(
    cuts: list[ImageCut], src: np.ndarray, outer_ellipse: Ellipse, initial_center: np.ndarray, params: Parameters
) -> tuple[np.ndarray | None, np.ndarray, float | None, bool]:
    """Coarse-to-fine grid search for the imaged center + homography, then
    residual normalization against the cuts' consensus signal. Mirrors
    ``refineConicFamilyGlob`` (CPU path)."""
    neighbour_size = params.imaged_center_neighbour_size
    grid_n_sample = params.imaged_center_n_grid_sample
    if grid_n_sample <= 3:
        # neighbour_size shrinks by (n - 1) / 2 per iteration; <= 1 never terminates.
        raise ValueError(f"imaged_center_n_grid_sample must be greater than 3, got {grid_n_sample}")
    max_semi_axis = max(outer_ellipse.a, outer_ellipse.b)

    center = np.asarray(initial_center, dtype=np.float64).copy()
    h = None
    min_res = float("inf")
    frame = nearby_grid_frame(outer_ellipse)

    while neighbour_size * max_semi_axis > 0.02:
        h, center, min_res, has_solution = image_center_optimization_glob(
            cuts, center, outer_ellipse, neighbour_size, grid_n_sample, src, frame
        )
        if not has_solution:
            return None, center, None, False
        neighbour_size /= (grid_n_sample - 1) / 2

    get_signals(cuts, h, src)

    correct_cuts = [c for c in cuts if not c.out_of_bounds]
    if not correct_cuts:
        return h, center, None, False

    barcode = np.median(np.stack([c.img_signal for c in correct_cuts]), axis=0)
    magnitude = float(barcode.max() - barcode.min())
    if magnitude == 0.0:
        return h, center, None, False

    residual = float(np.sqrt(min_res) / magnitude)
    if residual > 2.7:
        return h, center, residual, False
    return h, center, residual, True


def refine_conic_family_glob_optimizer(
    cuts: list[ImageCut], src: np.ndarray, outer_ellipse: Ellipse, initial_center: np.ndarray, params: Parameters
) -> tuple[np.ndarray | None, np.ndarray, float | None, bool]:
    """EXPERIMENTAL alternative to :func:`refine_conic_family_glob`: replaces
    the coarse-to-fine grid search (~10 shrinking iterations x 25 candidate
    centers = ~250 evaluations of :func:`cost_function_glob` per marker)
    with a single bounded Powell-method local search over the same
    ellipse-conditioned neighborhood the grid search would have covered.

    This is NOT guaranteed to reproduce the grid search's exact trial path
    or sticky ``out_of_bounds`` history on any given cut, so results are
    expected to be close but not bit-identical to
    :func:`refine_conic_family_glob` -- selected via
    ``Parameters``-independent opt-in (see ``identify_step_2``'s
    ``use_optimizer`` argument), not a default. Imports ``scipy.optimize``
    lazily (only paid the first time this path actually runs in a process)
    so callers who never opt in never pay for it.
    """
    from scipy.optimize import minimize

    neighbour_size = params.imaged_center_neighbour_size
    max_semi_axis = max(outer_ellipse.a, outer_ellipse.b)

    m_t, m_inv_t, max_axis = nearby_grid_frame(outer_ellipse)
    size = neighbour_size * max_axis
    half_width = size / 2.0

    cond_center0 = condition_point(np.asarray(initial_center, dtype=np.float64), m_t)
    bounds = [
        (cond_center0[0] - half_width, cond_center0[0] + half_width),
        (cond_center0[1] - half_width, cond_center0[1] + half_width),
    ]

    def cost(cond_xy: np.ndarray) -> float:
        point = condition_points(np.array([cond_xy]), m_inv_t)[0]
        try:
            h_candidate = compute_homography_from_ellipse_and_imaged_center(outer_ellipse, point)
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            return 1.0e12
        res, readable = cost_function_glob(h_candidate, cuts, src)
        return res if readable else 1.0e12

    # `xtol`/`ftol` scaled off the same "fine enough" threshold the grid
    # search's shrinking loop terminates at (`neighbour_size * max_semi_axis
    # <= 0.02`), converted to the conditioned coordinate frame.
    target_xtol = max(0.02 / max(max_semi_axis, 1e-9), 1e-6)
    result = minimize(
        cost,
        cond_center0,
        method="Powell",
        bounds=bounds,
        options={"xtol": target_xtol, "ftol": 1e-9, "maxiter": 60, "maxfev": 120},
    )

    best_point = condition_points(np.array([result.x]), m_inv_t)[0]
    try:
        h = compute_homography_from_ellipse_and_imaged_center(outer_ellipse, best_point)
    except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
        return None, best_point, None, False

    get_signals(cuts, h, src)
    correct_cuts = [c for c in cuts if not c.out_of_bounds]
    if not correct_cuts:
        return h, best_point, None, False

    barcode = np.median(np.stack([c.img_signal for c in correct_cuts]), axis=0)
    magnitude = float(barcode.max() - barcode.min())
    if magnitude == 0.0:
        return h, best_point, None, False

    res_final, readable_final = cost_function_glob(h, cuts, src)
    if not readable_final:
        return h, best_point, None, False

    residual = float(np.sqrt(res_final) / magnitude)
    if residual > 2.7:
        return h, best_point, residual, False
    return h, best_point, residual, True


@lru_cache(maxsize=8)
def _digit_matrix_cached(
    rr_bank_tuple: tuple[tuple[float, ...], ...], begin_sig: float, step_x: float, n: int
) -> np.ndarray:
    """Vectorized precompute of the per-(bank row, sample index) alternating
    +-1 "digit" pattern used by :func:`orazio_distance_robust`. This depends
    only on the bank table and the cut's sampling geometry (``begin_sig``,
    ``step_x``, ``n``) -- never on the image signal -- and that geometry is
    the same for every cut of every marker in a single ``cctag_detection``
    run (``begin_sig``/``n`` come from ``params``, fixed for the whole run;
    ``step_x`` is derived from them). Without this cache it would be
    recomputed once per marker (5-10x redundant on a typical image) even
    though every marker gets the identical matrix; ``lru_cache`` makes that
    cross-marker reuse automatic. Assumes every bank row has the same number
    of ratio entries (true for both the 32x5 three-crown and 128x7
    four-crown tables)."""
    ratios_arr = np.asarray(rr_bank_tuple, dtype=np.float64)
    thresholds = 1.0 / ratios_arr
    x = begin_sig + step_x * np.arange(n, dtype=np.float64)
    ldum = (thresholds[:, :, None] <= x[None, None, :]).sum(axis=1)
    return np.where(ldum % 2 == 1, -1.0, 1.0)


def _digit_matrix(rr_bank: list[list[float]], begin_sig: float, step_x: float, n: int) -> np.ndarray:
    return _digit_matrix_cached(tuple(tuple(row) for row in rr_bank), begin_sig, step_x, n)


if HAS_NUMBA:

    @njit(cache=True)
    def _orazio_scores_numba(signals: np.ndarray, digit_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-cut body of :func:`orazio_distance_robust` for ``signals``
        (C, n) that all share ``digit_matrix`` (K, n): returns the best bank
        row, its score and a validity flag per cut (False where the tail
        variance is zero and the cut is skipped)."""
        c, n = signals.shape
        k_rows = digit_matrix.shape[0]
        best_id = np.zeros(c, dtype=np.int64)
        best_v = np.zeros(c, dtype=np.float64)
        valid = np.zeros(c, dtype=np.bool_)
        distance = np.empty(k_rows, dtype=np.float64)
        for ci in range(c):
            sig = signals[ci]
            n_tail = n - 30
            mean_sig = 0.0
            for i in range(30, n):
                mean_sig += sig[i]
            mean_sig /= n_tail
            var_sig = 0.0
            for i in range(30, n):
                d = sig[i] - mean_sig
                var_sig += d * d
            var_sig /= n_tail
            if var_sig == 0.0:
                continue

            do_accumulate = False
            sum_inf = 0.0
            n_inf = 0
            sum_sup = 0.0
            n_sup = 0
            for i in range(n):
                v = sig[i]
                if not do_accumulate and v < mean_sig:
                    do_accumulate = True
                if do_accumulate:
                    if v < mean_sig:
                        sum_inf += v
                        n_inf += 1
                    else:
                        sum_sup += v
                        n_sup += 1
            muw = sum_sup / n_sup if n_sup > 0 else 0.0
            mub = sum_inf / n_inf if n_inf > 0 else 0.0

            two_var = 2.0 * var_sig
            for kr in range(k_rows):
                acc = 0.0
                for i in range(n):
                    if digit_matrix[kr, i] == -1.0:
                        dl = sig[i] - mub
                        if dl < 0.0:
                            dl = 0.0
                        acc += dl * dl / two_var
                    else:
                        dh = sig[i] - muw
                        if dh > 0.0:
                            dh = 0.0
                        acc += dh * dh / two_var
                distance[kr] = acc
            bi = 0
            bv = np.exp(-distance[0])
            for kr in range(1, k_rows):
                vv = np.exp(-distance[kr])
                if vv > bv:
                    bv = vv
                    bi = kr
            best_id[ci] = bi
            best_v[ci] = bv
            valid[ci] = True
        return best_id, best_v, valid


def _cut_sig_key(cut: ImageCut) -> tuple[float, float, int]:
    n = len(cut.img_signal)
    return (cut.begin_sig, (cut.end_sig - cut.begin_sig) / (n - 1.0), n)


def _orazio_distance_robust_numpy(
    rr_bank: list[list[float]], live: list[ImageCut], v_score: list[list[float]]
) -> None:
    """Per-cut bank matching (mirrors ``orazioDistanceRobust``'s body,
    statement for statement), appending into ``v_score`` in place. Used
    directly without Numba, and as the reference this module's kernel path
    (:func:`_orazio_distance_robust_numba`) is verified against."""
    digit_matrix = None
    cached_key: tuple[float, float, int] | None = None

    for cut in live:
        img_sig = cut.img_signal
        n = len(img_sig)
        if n <= 30:
            continue

        tail = img_sig[30:]
        median_sig = float(tail.mean())
        var_sig = float(tail.var())
        if var_sig == 0.0:
            continue

        do_accumulate = False
        acc_inf: list[float] = []
        acc_sup: list[float] = []
        for v in img_sig:
            if not do_accumulate and v < median_sig:
                do_accumulate = True
            if do_accumulate:
                (acc_inf if v < median_sig else acc_sup).append(v)
        muw = float(np.mean(acc_sup)) if acc_sup else 0.0
        mub = float(np.mean(acc_inf)) if acc_inf else 0.0

        step_x = (cut.end_sig - cut.begin_sig) / (n - 1.0)
        key = (cut.begin_sig, step_x, n)
        if key != cached_key:
            digit_matrix = _digit_matrix(rr_bank, cut.begin_sig, step_x, n)
            cached_key = key

        diff_lo = np.maximum(img_sig - mub, 0.0) ** 2 / (2.0 * var_sig)
        diff_hi = np.minimum(img_sig - muw, 0.0) ** 2 / (2.0 * var_sig)
        distance = np.where(digit_matrix == -1.0, diff_lo[None, :], diff_hi[None, :]).sum(axis=1)
        v_all = np.exp(-distance)
        best_id = int(np.argmax(v_all))
        best_v = float(v_all[best_id])

        v_score[best_id].append(best_v)


def _orazio_distance_robust_numba(
    rr_bank: list[list[float]], live: list[ImageCut], v_score: list[list[float]]
) -> None:
    """Per-group kernel form of :func:`_orazio_distance_robust_numpy`: cuts
    are grouped by sampling geometry (:func:`_cut_sig_key` -- in practice
    all cuts of one run share a single key, since they come from the same
    :func:`collect_cuts` call, so this is normally one group and one kernel
    call scoring every cut against one digit matrix instead of ~10 NumPy
    calls per cut), then scattered back into ``v_score`` in the cuts'
    original order (``np.exp``/``sum`` inside the kernel use a different
    accumulation order than the NumPy form above, so scores agree only to
    ~1e-13 relative, not bit-for-bit -- verified on real markers)."""
    groups: dict[tuple[float, float, int], list[int]] = {}
    for i, cut in enumerate(live):
        groups.setdefault(_cut_sig_key(cut), []).append(i)

    results: dict[int, tuple[int, float]] = {}
    for (begin_sig, step_x, n), idx_list in groups.items():
        if n <= 30:
            continue
        signals = np.stack([live[i].img_signal for i in idx_list]).astype(np.float64)
        best_id, best_v, valid = _orazio_scores_numba(signals, _digit_matrix(rr_bank, begin_sig, step_x, n))
        for k, i in enumerate(idx_list):
            if valid[k]:
                results[i] = (int(best_id[k]), float(best_v[k]))

    for i in range(len(live)):
        if i in results:
            best_id, best_v = results[i]
            v_score[best_id].append(best_v)


def orazio_distance_robust(
    rr_bank: list[list[float]], cuts: list[ImageCut], min_ident_proba: float
) -> list[list[float]]:
    """Per-cut bank matching (each cut votes its single best-matching bank
    row). Mirrors ``orazioDistanceRobust`` (``min_ident_proba`` is accepted
    for signature parity but, as in the source, unused here). Dispatches to
    a Numba kernel when available, falling back to the NumPy per-cut loop
    otherwise."""
    v_score: list[list[float]] = [[] for _ in range(len(rr_bank))]
    live = [cut for cut in cuts if not cut.out_of_bounds]
    if HAS_NUMBA:
        _orazio_distance_robust_numba(rr_bank, live, v_score)
    else:
        _orazio_distance_robust_numpy(rr_bank, live, v_score)
    return v_score


def identify_step_1(cctag: CCTag, src: np.ndarray, params: Parameters) -> tuple[int, list[ImageCut]]:
    """Mirrors ``identify_step_1``: generates and selects image cuts only."""
    ellipse = cctag.rescaled_outer_ellipse
    outer_points = cctag.rescaled_outer_ellipse_points
    if not outer_points:
        return status.too_few_outer_points, []

    positions = np.array([(p[0], p[1]) for p in outer_points])
    gradients = np.array([(p[2], p[3]) for p in outer_points])

    idx = Ellipse.get_sorted_outer_points(ellipse, positions, params.n_samples_outer_ellipse)
    if len(idx) < 5:
        return status.too_few_outer_points, []
    sel_positions = positions[idx]
    sel_gradients = gradients[idx]

    if params.n_crowns == 3:
        start_sig = 1.0 - (2 * params.n_crowns - 1) * 0.15
    elif params.n_crowns == 4:
        start_sig = 0.26
    else:
        start_sig = 0.0

    cuts = collect_cuts(src, ellipse.center, sel_positions, sel_gradients, params.sample_cut_length, start_sig)
    if not cuts:
        return status.no_collected_cuts, []

    selected = select_cut_cheap_uniform(
        params.num_cuts_in_ident_step, ellipse, cuts, src, cctag.scale, params.num_samples_outer_edge_points_refinement
    )
    if not selected:
        return status.no_selected_cuts, []

    return status.id_reliable, selected


def identify_step_2(
    cctag: CCTag,
    selected_cuts: list[ImageCut],
    radius_ratios: list[list[float]],
    src: np.ndarray,
    params: Parameters,
    use_optimizer: bool = False,
) -> int:
    """Mirrors ``identify_step_2``. Mutates ``cctag`` in place.

    ``use_optimizer`` (Python-port-only knob, not a C++ ``Parameters``
    field): when set, the imaged-center search uses
    :func:`refine_conic_family_glob_optimizer` (bounded Powell search,
    ~10-20x fewer cost-function evaluations) instead of the default
    :func:`refine_conic_family_glob` (coarse-to-fine grid search, faithful
    to the C++ reference). The optimizer path trades exact reproducibility
    of the search trajectory for speed -- see both functions' docstrings.
    """
    ellipse = cctag.rescaled_outer_ellipse

    refine_fn = refine_conic_family_glob_optimizer if use_optimizer else refine_conic_family_glob
    h, center, residual, converged = refine_fn(selected_cuts, src, ellipse, cctag.center_img, params)
    cctag.quality = (1.0 / residual) if residual else float("inf")
    if not converged:
        return status.opti_has_diverged

    cctag.homography = h
    cctag.center_img = center

    v_score = orazio_distance_robust(radius_ratios, selected_cuts, params.min_ident_proba)

    i_max = 0
    max_size = 0
    for i, scores in enumerate(v_score):
        if len(scores) > max_size:
            i_max = i
            max_size = len(scores)

    score = float(np.mean(v_score[i_max])) if v_score[i_max] else 0.0

    cctag.id = i_max
    cctag.id_set = []  # matches a confirmed bug in the C++ source: always empty
    cctag.radius_ratios = list(radius_ratios[i_max])

    try:
        h_inv = np.linalg.inv(cctag.homography)
        ellipses = []
        for ratio in cctag.radius_ratios:
            circle = Circle(center=np.zeros(2), radius=1.0 / ratio)
            ellipses.append(Ellipse(matrix=h_inv.T @ circle.matrix @ h_inv))
        ellipses.append(cctag.rescaled_outer_ellipse)
        cctag.ellipses = ellipses
    except (ValueError, np.linalg.LinAlgError):
        return status.degenerate

    ident_successful = score > params.min_ident_proba
    return status.id_reliable if ident_successful else status.id_not_reliable
