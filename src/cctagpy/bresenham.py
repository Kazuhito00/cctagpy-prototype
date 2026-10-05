"""Gradient-direction ray marching, ported from ``src/cctag/Bresenham.cpp``.

The C++ ``gradientDirectionDescent`` recomputes a "steering" direction
mid-walk from ``imgDx``/``imgDy`` sampled at the *original* point ``p`` --
algebraically this always reproduces the same ``dir`` it started with
(``dir_new = sign(dir_old * (imgDx(p)^2 + imgDy(p)^2)) == dir_old``), so it
is a verified no-op and is intentionally omitted here: this is a plain
straight-line Bresenham walk in the initial (possibly ``dir``-flipped)
gradient direction. ``_thrGradientMagInVote`` therefore has no effect and is
not part of this function's signature.
"""

from __future__ import annotations

import numpy as np

from cctagpy._numba_utils import HAS_NUMBA, njit, prange
from cctagpy.edge_collection import EdgePointCollection


def _sign(v: float) -> int:
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def gradient_direction_descent(
    collection: EdgePointCollection,
    px: int,
    py: int,
    direction: int,
    nmax: int,
    img_dx: np.ndarray,
    img_dy: np.ndarray,
) -> int:
    """March from edge point ``(px, py)`` along ``direction * gradient(px,py)``.

    Returns the index of the first :class:`EdgePointCollection` point found
    along the way, or -1 if none is found within ``nmax`` steps or the ray
    exits the image. Mirrors ``gradientDirectionDescent``.
    """
    width, height = collection.shape

    dx = direction * float(img_dx[py, px])
    dy = direction * float(img_dy[py, px])
    adx, ady = abs(dx), abs(dy)

    x, y = px, py
    e = 0.0
    steep = ady > adx

    def step() -> None:
        nonlocal x, y, e
        stp_x = _sign(dx)
        stp_y = _sign(dy)
        if steep:
            a = abs(dx / dy)
            e += a
            y += stp_y
            if e >= 0.5:
                x += stp_x
                e -= 1
        else:
            a = abs(dy / dx)
            e += a
            x += stp_x
            if e >= 0.5:
                y += stp_y
                e -= 1

    def in_bounds(cx: int, cy: int) -> bool:
        return 0 <= cx < width and 0 <= cy < height

    # step 1: unconditional, result never checked
    step()
    n = 1

    # step 2: checked, but no side-probe fallback yet
    step()
    n = 2
    if not in_bounds(x, y):
        return -1
    idx = collection.index_at(x, y)
    if idx != -1:
        return idx

    stp_x_last = _sign(dx)
    stp_y_last = _sign(dy)

    while n <= nmax:
        step()
        n += 1
        if not in_bounds(x, y):
            return -1
        idx = collection.index_at(x, y)
        if idx != -1:
            return idx
        if steep:
            probe_x, probe_y = x, y - stp_y_last
        else:
            probe_x, probe_y = x - stp_x_last, y
        if not in_bounds(probe_x, probe_y):
            return -1
        idx = collection.index_at(probe_x, probe_y)
        if idx != -1:
            return idx

    return -1


def _gradient_direction_descent_batch_numpy(
    collection: EdgePointCollection,
    px: np.ndarray,
    py: np.ndarray,
    direction: int,
    nmax: int,
    img_dx: np.ndarray,
    img_dy: np.ndarray,
) -> np.ndarray:
    """Vectorized :func:`gradient_direction_descent` over many starting
    points that all share the same ``direction``.

    Each point's walk is a pure function of its own starting position and
    the fixed gradient sampled there, plus read-only lookups into
    ``collection.edge_map`` -- it never depends on another point's walk.
    That makes it safe to batch here, in lockstep across all points, for
    :func:`cctagpy.vote.vote`'s phase-1 before/after link construction
    (the only caller). It is NOT safe to use for any caller that mutates
    ``edge_map``/a "processed" set mid-walk (e.g. edge linking), since
    those depend on the order individual descents are resolved in.

    Uses stream compaction: on real images, a large and growing fraction of
    walks finish (find an edge point, or leave the image) within the first
    few steps, but not all of them do -- ``done.all()`` essentially never
    becomes true before ``nmax`` (measured: ~65-90% done by the last step,
    never 100%), so a naive lockstep loop keeps re-touching full-size
    arrays every iteration even though most entries stopped changing steps
    ago. Compacting the working arrays down to only the still-active
    indices each iteration (and scattering finished results back into the
    full-size output by their original index) does the same arithmetic on
    a shrinking array instead -- not an approximation, since per-point
    results are independent of processing order or grouping, just less
    wasted work as the active set drains.
    """
    px = np.asarray(px, dtype=np.int64)
    py = np.asarray(py, dtype=np.int64)
    n_points = len(px)
    width, height = collection.shape
    if n_points == 0:
        return np.empty(0, dtype=np.int64)

    dx_full = direction * img_dx[py, px].astype(np.float64)
    dy_full = direction * img_dy[py, px].astype(np.float64)
    steep_full = np.abs(dy_full) > np.abs(dx_full)
    stp_x_full = np.sign(dx_full).astype(np.int64)
    stp_y_full = np.sign(dy_full).astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        a_full = np.where(
            steep_full,
            np.abs(np.where(dy_full != 0, dx_full / dy_full, 0.0)),
            np.abs(np.where(dx_full != 0, dy_full / dx_full, 0.0)),
        )

    result = np.full(n_points, -1, dtype=np.int64)

    # Working (compacted) state, indexed by `active`'s own position, not by
    # original point index; `active` holds the original indices these rows
    # currently correspond to.
    active = np.arange(n_points, dtype=np.int64)
    x = px.copy()
    y = py.copy()
    e = np.zeros(n_points, dtype=np.float64)
    steep = steep_full
    stp_x = stp_x_full
    stp_y = stp_y_full
    a = a_full

    def step() -> None:
        nonlocal x, y, e
        e2 = e + a
        move = e2 >= 0.5
        x_inc = np.where(steep, np.where(move, stp_x, 0), stp_x)
        y_inc = np.where(steep, stp_y, np.where(move, stp_y, 0))
        x = x + x_inc
        y = y + y_inc
        e = np.where(move, e2 - 1.0, e2)

    def edge_at(cx: np.ndarray, cy: np.ndarray, valid: np.ndarray) -> np.ndarray:
        cxc = np.clip(cx, 0, width - 1)
        cyc = np.clip(cy, 0, height - 1)
        idx = collection.edge_map[cyc, cxc].astype(np.int64)
        return np.where(valid, idx, -1)

    def compact(keep: np.ndarray) -> None:
        nonlocal active, x, y, e, steep, stp_x, stp_y, a
        active = active[keep]
        x = x[keep]
        y = y[keep]
        e = e[keep]
        steep = steep[keep]
        stp_x = stp_x[keep]
        stp_y = stp_y[keep]
        a = a[keep]

    step()  # step 1: unconditional, result never checked

    step()  # step 2: checked, but no side-probe fallback yet
    in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    idx = edge_at(x, y, in_bounds)
    found = in_bounds & (idx != -1)
    result[active[found]] = idx[found]
    compact(in_bounds & ~found)

    n = 2
    while n <= nmax and active.size > 0:
        step()
        n += 1

        in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        idx = edge_at(x, y, in_bounds)
        found = in_bounds & (idx != -1)
        result[active[found]] = idx[found]

        still_active = in_bounds & ~found
        probe_x = np.where(steep, x, x - stp_x)
        probe_y = np.where(steep, y - stp_y, y)
        probe_in_bounds = (probe_x >= 0) & (probe_x < width) & (probe_y >= 0) & (probe_y < height)
        idx_probe = edge_at(probe_x, probe_y, still_active & probe_in_bounds)
        found_probe = still_active & probe_in_bounds & (idx_probe != -1)
        result[active[found_probe]] = idx_probe[found_probe]

        compact(still_active & probe_in_bounds & ~found_probe)

    return result


if HAS_NUMBA:

    @njit(cache=True, inline="always")
    def _bresenham_step(x: int, y: int, e: float, a: float, steep: bool, stp_x: int, stp_y: int) -> tuple[int, int, float]:
        """One Bresenham step; the same float sequence as the NumPy ``step``."""
        e2 = e + a
        move = e2 >= 0.5
        if steep:
            if move:
                x += stp_x
            y += stp_y
        else:
            x += stp_x
            if move:
                y += stp_y
        return x, y, (e2 - 1.0 if move else e2)

    @njit(cache=True, inline="always")
    def _in_bounds(x: int, y: int, width: int, height: int) -> bool:
        return 0 <= x < width and 0 <= y < height

    @njit(cache=True, parallel=True)
    def _gradient_direction_descent_batch_numba(
        edge_map: np.ndarray,
        px: np.ndarray,
        py: np.ndarray,
        direction: int,
        nmax: int,
        img_dx: np.ndarray,
        img_dy: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Scalar per-point walk, parallel over points (each iteration writes
        only ``result[k]`` and allocates nothing); the same operation sequence
        as the NumPy form, see :func:`gradient_direction_descent_batch`."""
        n_points = px.shape[0]
        result = np.full(n_points, -1, dtype=np.int64)

        for k in prange(n_points):
            x = px[k]
            y = py[k]
            dxv = direction * float(img_dx[y, x])
            dyv = direction * float(img_dy[y, x])
            steep = abs(dyv) > abs(dxv)
            stp_x = 1 if dxv > 0.0 else (-1 if dxv < 0.0 else 0)
            stp_y = 1 if dyv > 0.0 else (-1 if dyv < 0.0 else 0)
            if steep:
                a = abs(dxv / dyv) if dyv != 0.0 else 0.0
            else:
                a = abs(dyv / dxv) if dxv != 0.0 else 0.0

            # step 1: unconditional, result never checked
            x, y, e = _bresenham_step(x, y, 0.0, a, steep, stp_x, stp_y)
            # step 2: checked, but no side-probe fallback yet
            x, y, e = _bresenham_step(x, y, e, a, steep, stp_x, stp_y)
            if not _in_bounds(x, y, width, height):
                continue
            idx = edge_map[y, x]
            if idx != -1:
                result[k] = idx
                continue

            n = 2
            while n <= nmax:
                x, y, e = _bresenham_step(x, y, e, a, steep, stp_x, stp_y)
                n += 1
                if not _in_bounds(x, y, width, height):
                    break
                idx = edge_map[y, x]
                if idx != -1:
                    result[k] = idx
                    break
                if steep:
                    probe_x = x
                    probe_y = y - stp_y
                else:
                    probe_x = x - stp_x
                    probe_y = y
                if not _in_bounds(probe_x, probe_y, width, height):
                    break
                idx_probe = edge_map[probe_y, probe_x]
                if idx_probe != -1:
                    result[k] = idx_probe
                    break

        return result


def gradient_direction_descent_batch(
    collection: EdgePointCollection,
    px: np.ndarray,
    py: np.ndarray,
    direction: int,
    nmax: int,
    img_dx: np.ndarray,
    img_dy: np.ndarray,
) -> np.ndarray:
    """Batched :func:`gradient_direction_descent` over many starting points
    that all share the same ``direction`` (see
    :func:`_gradient_direction_descent_batch_numpy` for the safety argument:
    walks are independent of one another, which is what makes batching --
    and, with Numba, running them on all cores -- legitimate here and NOT
    for callers that mutate ``edge_map``/a processed set mid-walk).
    Dispatches to the point-parallel Numba kernel when available, falling
    back to the stream-compacted lockstep NumPy form otherwise; verified
    bit-identical on every ``vote()`` call of both sample images."""
    if HAS_NUMBA:
        px_arr = np.ascontiguousarray(px, dtype=np.int64)
        py_arr = np.ascontiguousarray(py, dtype=np.int64)
        width, height = collection.shape
        return _gradient_direction_descent_batch_numba(
            collection.edge_map, px_arr, py_arr, int(direction), int(nmax), img_dx, img_dy, width, height
        )
    return _gradient_direction_descent_batch_numpy(collection, px, py, direction, nmax, img_dx, img_dy)
