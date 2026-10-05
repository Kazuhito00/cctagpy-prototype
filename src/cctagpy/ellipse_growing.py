"""Ellipse growing + flow assembly, ported from ``src/cctag/EllipseGrowing.{hpp,cpp}``.

``connectedPoint`` is reimplemented as an iterative flood fill (a stack
instead of recursion): membership in a flood-filled connected component
does not depend on visitation order, so this is behaviorally identical to
the source's recursion and avoids Python's recursion-depth limit on large
components.

The C++ ``_processed`` bitfield (one bit per concurrently-running
candidate/``runId``, needed only because the reference implementation
grows candidates in parallel with TBB) is replaced by a plain local
``set``/boolean-array scoped to one call, since this port processes
candidates sequentially -- there is no cross-candidate interaction to
multiplex bits for.
"""

from __future__ import annotations

import math

import numpy as np

from cctagpy._numba_utils import HAS_NUMBA, njit
from cctagpy.edge_collection import EdgePointCollection
from cctagpy.fitting import circle_fitting, ellipse_fitting, inner_prod_min

if HAS_NUMBA:
    from cctagpy.fitting import _circle_fitting_numba_core, _ellipse_fitting_numba_core
    from cctagpy.geometry import _ellipse_matrix_numba
from cctagpy.geometry import Ellipse

_XOFF = [1, 1, 0, -1, -1, -1, 0, 1]
_YOFF = [0, -1, -1, -1, 0, 1, 1, 1]


def _homogeneous(xy: np.ndarray) -> np.ndarray:
    return np.array([xy[0], xy[1], 1.0])


if HAS_NUMBA:

    @njit(cache=True, inline="always")
    def _bilinear_form_numba(m: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> float:
        """``[x1,y1,1] . (m @ [x2,y2,1])``."""
        mx0 = m[0, 0] * x2 + m[0, 1] * y2 + m[0, 2]
        mx1 = m[1, 0] * x2 + m[1, 1] * y2 + m[1, 2]
        mx2 = m[2, 0] * x2 + m[2, 1] * y2 + m[2, 2]
        return x1 * mx0 + y1 * mx1 + mx2

    @njit(cache=True)
    def _is_in_ellipse_numba(matrix: np.ndarray, cx: float, cy: float, px: float, py: float) -> bool:
        s1 = _bilinear_form_numba(matrix, px, py, px, py)
        s2 = _bilinear_form_numba(matrix, px, py, cx, cy)
        return s1 * s2 > 0.0

    @njit(cache=True)
    def _is_in_hull_numba(q_in_matrix: np.ndarray, q_out_matrix: np.ndarray, px: float, py: float) -> bool:
        s1 = _bilinear_form_numba(q_in_matrix, px, py, px, py)
        s2 = _bilinear_form_numba(q_out_matrix, px, py, px, py)
        return s1 * s2 < 0.0

    _XOFF_ARR = np.array(_XOFF, dtype=np.int64)
    _YOFF_ARR = np.array(_YOFF, dtype=np.int64)

    @njit(cache=True)
    def _connected_points_numba(
        edge_map: np.ndarray,
        positions: np.ndarray,
        gradients: np.ndarray,
        width: int,
        height: int,
        seed_indices: np.ndarray,
        q_in_matrix: np.ndarray,
        q_in_cx: float,
        q_in_cy: float,
        q_out_matrix: np.ndarray,
        processed: np.ndarray,
    ) -> np.ndarray:
        n = positions.shape[0]
        cap = n + seed_indices.shape[0]
        stack = np.empty(cap, dtype=np.int64)
        stack_top = 0
        for i in range(seed_indices.shape[0]):
            stack[stack_top] = seed_indices[i]
            stack_top += 1

        found = np.empty(n, dtype=np.int64)
        found_count = 0

        while stack_top > 0:
            stack_top -= 1
            idx = stack[stack_top]
            x = positions[idx, 0]
            y = positions[idx, 1]
            for k in range(8):
                sx = x + _XOFF_ARR[k]
                sy = y + _YOFF_ARR[k]
                if sx < 0 or sx >= width or sy < 0 or sy >= height:
                    continue
                n_idx = edge_map[sy, sx]
                if n_idx == -1 or processed[n_idx]:
                    continue
                nx = float(positions[n_idx, 0])
                ny = float(positions[n_idx, 1])
                s1 = _bilinear_form_numba(q_in_matrix, nx, ny, nx, ny)
                s2 = _bilinear_form_numba(q_out_matrix, nx, ny, nx, ny)
                if not (s1 * s2 < 0.0):
                    continue
                gx = gradients[n_idx, 0]
                gy = gradients[n_idx, 1]
                tcx = q_in_cx - nx
                tcy = q_in_cy - ny
                if gx * tcx + gy * tcy < 0.0:
                    processed[n_idx] = True
                    found[found_count] = n_idx
                    found_count += 1
                    stack[stack_top] = n_idx
                    stack_top += 1

        return found[:found_count]


def is_in_ellipse(ellipse: Ellipse, point_xy: np.ndarray) -> bool:
    """``x'Qx`` sign test: true iff ``point`` is on the same side of the
    conic as the ellipse's own center. Mirrors ``isInEllipse``.

    Dispatches to a scalar-arithmetic Numba kernel when available -- this
    (and :func:`is_in_hull`) is called tens of thousands of times per image
    from the ellipse-growing flood fill, and the NumPy form's per-call cost
    is almost entirely small-array allocation/dispatch overhead (building
    3-element homogeneous points, two 3x3 matrix-vector products) rather
    than the ~15 FLOPs actually being computed -- verified bit-identical to
    the NumPy form on real images."""
    if HAS_NUMBA:
        return bool(_is_in_ellipse_numba(ellipse.matrix, ellipse.center[0], ellipse.center[1], point_xy[0], point_xy[1]))
    p = _homogeneous(point_xy)
    center_h = _homogeneous(ellipse.center)
    s1 = p @ (ellipse.matrix @ p)
    s2 = p @ (ellipse.matrix @ center_h)
    return bool(s1 * s2 > 0)


def is_overlapping_ellipses(e1: Ellipse, e2: Ellipse) -> bool:
    return is_in_ellipse(e1, e2.center) or is_in_ellipse(e2, e1.center)


def is_in_hull(q_in: Ellipse, q_out: Ellipse, point_xy: np.ndarray) -> bool:
    """True iff ``point`` is inside ``q_out`` and outside ``q_in`` (the
    elliptical annulus between them). Mirrors ``isInHull``. See
    :func:`is_in_ellipse` for why/how the Numba dispatch helps here."""
    if HAS_NUMBA:
        return bool(_is_in_hull_numba(q_in.matrix, q_out.matrix, point_xy[0], point_xy[1]))
    p = _homogeneous(point_xy)
    s1 = p @ (q_in.matrix @ p)
    s2 = p @ (q_out.matrix @ p)
    return bool(s1 * s2 < 0)


def is_in_hull_batched(q_in: Ellipse, q_out: Ellipse, positions: np.ndarray) -> np.ndarray:
    """Vectorized :func:`is_in_hull` over ``N`` independent points against
    the same (fixed) ``q_in``/``q_out`` pair. Returns an ``(N,)`` boolean
    mask. Pure elementwise quadratic-form evaluation (no matrix
    decomposition), so this is the same arithmetic as calling
    :func:`is_in_hull` once per point, not an approximation -- just done as
    one batched computation instead of a Python loop over the points."""
    ones = np.ones((len(positions), 1))
    ph = np.hstack([positions, ones])
    s1 = (ph @ q_in.matrix.T * ph).sum(axis=1)
    s2 = (ph @ q_out.matrix.T * ph).sum(axis=1)
    return s1 * s2 < 0


def is_on_the_same_side(p1_xy: np.ndarray, p2_xy: np.ndarray, line: np.ndarray) -> bool:
    p1 = _homogeneous(p1_xy)
    p2 = _homogeneous(p2_xy)
    return bool((p1 @ line) * (p2 @ line) > 0)


def compute_hull(ellipse: Ellipse, delta: float) -> tuple[Ellipse, Ellipse]:
    """Two ellipses sharing ``ellipse``'s center/angle, with semi-axes
    ``a`` +/- ``delta`` (clamped to >= 0.001). Mirrors ``computeHull``."""
    q_in = Ellipse(
        center=ellipse.center,
        a=max(ellipse.a - delta, 0.001),
        b=max(ellipse.b - delta, 0.001),
        angle=ellipse.angle,
    )
    q_out = Ellipse(center=ellipse.center, a=ellipse.a + delta, b=ellipse.b + delta, angle=ellipse.angle)
    return q_in, q_out


def _connected_points_numpy(
    collection: EdgePointCollection,
    seed_indices: list[int],
    q_in: Ellipse,
    q_out: Ellipse,
    processed: np.ndarray,
) -> list[int]:
    width, height = collection.shape
    found: list[int] = []
    stack = list(seed_indices)

    while stack:
        idx = stack.pop()
        x, y = int(collection.positions[idx, 0]), int(collection.positions[idx, 1])
        for k in range(8):
            sx, sy = x + _XOFF[k], y + _YOFF[k]
            if not (0 <= sx < width and 0 <= sy < height):
                continue
            n_idx = collection.index_at(sx, sy)
            if n_idx == -1 or processed[n_idx]:
                continue
            if not is_in_hull(q_in, q_out, collection.positions[n_idx]):
                continue
            gx, gy = collection.gradients[n_idx]
            nx, ny = collection.positions[n_idx]
            to_center = np.array([q_in.center[0] - nx, q_in.center[1] - ny])
            if gx * to_center[0] + gy * to_center[1] < 0:
                processed[n_idx] = True
                found.append(n_idx)
                stack.append(n_idx)

    return found


def _connected_points(
    collection: EdgePointCollection,
    seed_indices: list[int],
    q_in: Ellipse,
    q_out: Ellipse,
    processed: np.ndarray,
) -> list[int]:
    """Iterative (stack-based) flood fill within the ``q_in``/``q_out``
    elliptic hull, gated by gradient direction. Mirrors ``connectedPoint``.

    ``processed`` is a boolean array (one entry per collection point) rather
    than a ``set`` -- this lets the whole fill run as a single Numba kernel
    when available (index membership becomes a plain array read instead of a
    hashed set lookup), falling back to the NumPy/Python-loop form
    otherwise. Verified to visit the identical point set, in the identical
    discovery order, on real images either way."""
    if HAS_NUMBA:
        found = _connected_points_numba(
            collection.edge_map,
            collection.positions,
            collection.gradients,
            collection.shape[0],
            collection.shape[1],
            np.asarray(seed_indices, dtype=np.int64),
            q_in.matrix,
            q_in.center[0],
            q_in.center[1],
            q_out.matrix,
            processed,
        )
        return found.tolist()
    return _connected_points_numpy(collection, seed_indices, q_in, q_out, processed)


def ellipse_hull(
    collection: EdgePointCollection,
    pts: list[int],
    ellipse: Ellipse,
    delta: float,
    processed: np.ndarray,
) -> None:
    """Flood-fill-grow ``pts`` (mutated in place) within the elliptic hull
    of ``ellipse`` +/- ``delta``. Mirrors ``ellipseHull``."""
    q_in, q_out = compute_hull(ellipse, delta)
    seeds = list(pts)  # snapshot, matching the C++ initSize bound
    new_points = _connected_points(collection, seeds, q_in, q_out, processed)
    pts.extend(new_points)


def is_good_eg_points(
    collection: EdgePointCollection, filtered_children: list[int], thr_cos_diff_max: float = 0.25
) -> bool:
    idx = np.asarray(filtered_children, dtype=np.int64)
    min_val, _, _ = inner_prod_min(collection.positions[idx], collection.gradients[idx], thr_cos_diff_max)
    return min_val <= thr_cos_diff_max


def ellipse_growing_init(collection: EdgePointCollection, filtered_children: list[int]) -> tuple[Ellipse, bool]:
    """Fit an initial ellipse (or fall back to a circle if the points don't
    span a wide enough arc). Mirrors ``ellipseGrowingInit``."""
    idx = np.asarray(filtered_children, dtype=np.int64)
    positions = collection.positions[idx].astype(np.float64)
    good_init = is_good_eg_points(collection, filtered_children)
    if good_init:
        ellipse = ellipse_fitting(positions)
    else:
        ellipse = circle_fitting(positions)
    return ellipse, good_init


def _in_hull_count(collection: EdgePointCollection, pts: list[int], q_in: Ellipse, q_out: Ellipse) -> int:
    idx_arr = np.asarray(pts, dtype=np.int64)
    return int(is_in_hull_batched(q_in, q_out, collection.positions[idx_arr].astype(np.float64)).sum())


def ellipse_growing2(
    collection: EdgePointCollection,
    filtered_children: list[int],
    ellipse: Ellipse,
    hull_width: float,
    good_init: bool,
) -> tuple[Ellipse, list[int]]:
    """Iteratively grow the supporting point set and refit. Mirrors
    ``ellipseGrowing2``. Returns ``(ellipse, outer_ellipse_points)``; raises
    ``ValueError`` when a refit degenerates (the caller rejects the
    candidate).

    With Numba the whole grow/refit iteration runs as one kernel
    (:func:`_ellipse_growing2_numba`): the hull matrices, the flood fill,
    the in-hull counts and the circle/ellipse refits (the same fitting
    kernels ``ellipse_fitting`` dispatches to) all happen in compiled code,
    instead of ~5 Python-orchestrated rounds of small NumPy/Numba calls per
    candidate. Verified against the Python form on every real call of both
    sample images (identical point sets and fits)."""
    if HAS_NUMBA:
        params, outer, ok = _ellipse_growing2_numba(
            collection.edge_map, collection.positions, collection.gradients, collection.shape[0], collection.shape[1],
            np.asarray(filtered_children, dtype=np.int64), float(ellipse.center[0]), float(ellipse.center[1]),
            float(ellipse.a), float(ellipse.b), float(ellipse.angle), float(hull_width), bool(good_init),
        )
        if not ok:
            raise ValueError("ellipse_growing2: degenerate fit")
        return Ellipse(center=params[:2].copy(), a=float(params[2]), b=float(params[3]), angle=float(params[4])), outer.tolist()
    return _ellipse_growing2_python(collection, filtered_children, ellipse, hull_width, good_init)


def _ellipse_growing2_python(
    collection: EdgePointCollection,
    filtered_children: list[int],
    ellipse: Ellipse,
    hull_width: float,
    good_init: bool,
) -> tuple[Ellipse, list[int]]:
    processed = np.zeros(collection.point_count(), dtype=bool)
    processed[np.asarray(filtered_children, dtype=np.int64)] = True
    outer_points: list[int] = list(filtered_children)

    if not good_init:
        new_size = len(outer_points)
        last_size = 0
        max_nb_points = new_size
        n_iter_max = 0
        n_iter = 1
        # `outer_points` only ever grows (`ellipse_hull` appends), so each
        # iteration's snapshot is a prefix of the next -- track lengths
        # instead of copying the whole (growing) list every iteration, and
        # recover the winning snapshot with one slice at the end instead of
        # storing every intermediate copy.
        snapshot_lengths: list[int] = [len(outer_points)]
        ellipses_sets: list[Ellipse] = [ellipse]

        while new_size - last_size > 0:
            q_in, q_out = compute_hull(ellipse, hull_width)
            last_size = _in_hull_count(collection, outer_points, q_in, q_out)

            ellipse_hull(collection, outer_points, ellipse, hull_width, processed)
            snapshot_lengths.append(len(outer_points))
            ellipses_sets.append(ellipse)

            idx = np.asarray(outer_points, dtype=np.int64)
            ellipse = circle_fitting(collection.positions[idx].astype(np.float64))

            q_in, q_out = compute_hull(ellipse, hull_width)
            new_size = _in_hull_count(collection, outer_points, q_in, q_out)

            if new_size > max_nb_points:
                max_nb_points = new_size
                n_iter_max = n_iter
            n_iter += 1

        full_points = outer_points  # final (most-grown) list, a superset of every snapshot
        outer_points = full_points[: snapshot_lengths[n_iter_max]]
        ellipse = ellipses_sets[n_iter_max]

        # Every point ever marked processed during this bootstrap phase is
        # in `full_points` (each snapshot is one of its prefixes), so
        # clearing it once covers what the original's per-snapshot loop
        # cleared redundantly; only the winning snapshot is re-marked.
        processed[np.asarray(full_points, dtype=np.int64)] = False
        processed[np.asarray(outer_points, dtype=np.int64)] = True

    idx = np.asarray(outer_points, dtype=np.int64)
    ellipse = ellipse_fitting(collection.positions[idx].astype(np.float64))
    last_size = 0

    while len(outer_points) - last_size > 0:
        last_size = len(outer_points)
        ellipse_hull(collection, outer_points, ellipse, hull_width, processed)
        idx = np.asarray(outer_points, dtype=np.int64)
        ellipse = ellipse_fitting(collection.positions[idx].astype(np.float64))

    return ellipse, outer_points


if HAS_NUMBA:

    @njit(cache=True, inline="always")
    def _hull_matrices_numba(params: np.ndarray, hull_width: float) -> tuple[np.ndarray, np.ndarray]:
        """``compute_hull``'s two matrices for ellipse ``params`` (cx, cy, a,
        b, angle) +/- ``hull_width`` (clamped to >= 0.001)."""
        cx = params[0]
        cy = params[1]
        q_in = _ellipse_matrix_numba(cx, cy, max(params[2] - hull_width, 0.001), max(params[3] - hull_width, 0.001), params[4])
        q_out = _ellipse_matrix_numba(cx, cy, params[2] + hull_width, params[3] + hull_width, params[4])
        return q_in, q_out

    @njit(cache=True)
    def _in_hull_count_numba(positions: np.ndarray, idx: np.ndarray, count: int, params: np.ndarray, hull_width: float) -> int:
        q_in, q_out = _hull_matrices_numba(params, hull_width)
        n_in = 0
        for i in range(count):
            x = float(positions[idx[i], 0])
            y = float(positions[idx[i], 1])
            if _is_in_hull_numba(q_in, q_out, x, y):
                n_in += 1
        return n_in


    @njit(cache=True)
    def _grow_once_numba(
        edge_map: np.ndarray,
        positions: np.ndarray,
        gradients: np.ndarray,
        width: int,
        height: int,
        outer: np.ndarray,
        count: int,
        params: np.ndarray,
        hull_width: float,
        processed: np.ndarray,
    ) -> int:
        """``ellipse_hull``: flood-fill-grow ``outer[:count]`` within the
        hull of ``params`` +/- ``hull_width``; returns the new count."""
        q_in, q_out = _hull_matrices_numba(params, hull_width)
        found = _connected_points_numba(
            edge_map, positions, gradients, width, height, outer[:count], q_in, params[0], params[1], q_out, processed
        )
        for i in range(found.shape[0]):
            outer[count] = found[i]
            count += 1
        return count

    @njit(cache=True)
    def _ellipse_growing2_numba(
        edge_map: np.ndarray,
        positions: np.ndarray,
        gradients: np.ndarray,
        width: int,
        height: int,
        filtered: np.ndarray,
        cx: float,
        cy: float,
        a: float,
        b: float,
        angle: float,
        hull_width: float,
        good_init: bool,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """:func:`_ellipse_growing2_python` as one kernel; see
        :func:`ellipse_growing2`."""
        n = positions.shape[0]
        processed = np.zeros(n, dtype=np.bool_)
        outer = np.empty(n + filtered.shape[0], dtype=np.int64)
        count = filtered.shape[0]
        for i in range(count):
            outer[i] = filtered[i]
            processed[filtered[i]] = True
        params = np.array([cx, cy, a, b, angle])

        if not good_init:
            new_size = count
            last_size = 0
            max_nb_points = new_size
            n_iter_max = 0
            n_iter = 1
            snapshot_lengths = [count]
            ellipses = [params.copy()]

            while new_size - last_size > 0:
                last_size = _in_hull_count_numba(positions, outer, count, params, hull_width)

                count = _grow_once_numba(edge_map, positions, gradients, width, height, outer, count, params, hull_width, processed)
                snapshot_lengths.append(count)
                ellipses.append(params.copy())

                params, ok = _circle_fitting_numba_core(positions[outer[:count]].astype(np.float64))
                if not ok:
                    return params, outer[:0], False

                new_size = _in_hull_count_numba(positions, outer, count, params, hull_width)

                if new_size > max_nb_points:
                    max_nb_points = new_size
                    n_iter_max = n_iter
                n_iter += 1

            full_count = count
            count = snapshot_lengths[n_iter_max]
            params = ellipses[n_iter_max]
            for i in range(full_count):
                processed[outer[i]] = False
            for i in range(count):
                processed[outer[i]] = True

        params, ok = _ellipse_fitting_numba_core(positions[outer[:count]].astype(np.float64))
        if not ok:
            return params, outer[:0], False
        last_size = 0
        while count - last_size > 0:
            last_size = count
            count = _grow_once_numba(edge_map, positions, gradients, width, height, outer, count, params, hull_width, processed)
            params, ok = _ellipse_fitting_numba_core(positions[outer[:count]].astype(np.float64))
            if not ok:
                return params, outer[:0], False

        return params, outer[:count].copy(), True


def _add_candidate_flow_to_cctag_numpy(
    collection: EdgePointCollection,
    filtered_children: list[int],
    outer_ellipse: Ellipse,
    num_circles: int,
    cctag_points: list[list[tuple[float, float, float, float]]],
) -> tuple[bool, list[list[tuple[float, float, float, float]]]]:
    # A fresh, call-scoped mask (matching `ellipse_growing2`'s pattern)
    # rather than a persistent buffer on `collection` -- no dirty-index
    # bookkeeping or reset-on-every-return-path needed.
    processed = np.zeros(collection.point_count(), dtype=bool)
    n_gradient_out = 0
    n_added_point = 0

    for seed_idx in filtered_children:
        direction = -1
        p_idx = seed_idx
        outer_x, outer_y = collection.positions[seed_idx].astype(np.float64)
        a = outer_x - outer_ellipse.center[0]
        b = outer_y - outer_ellipse.center[1]
        line = np.array([a, b, -a * outer_ellipse.center[0] - b * outer_ellipse.center[1]])

        for j in range(1, num_circles):
            p_idx = int(collection.before[p_idx]) if direction == -1 else int(collection.after[p_idx])
            if p_idx == -1:
                return False, []

            if not processed[p_idx]:
                processed[p_idx] = True

                gx, gy = collection.gradients[p_idx]
                norm_grad = float(np.hypot(gx, gy))
                grad_e = np.array([gx / norm_grad, gy / norm_grad])

                px, py = collection.positions[p_idx].astype(np.float64)
                to_center = np.array([outer_ellipse.center[0] - px, outer_ellipse.center[1] - py])
                dist_to_center = float(np.hypot(to_center[0], to_center[1]))
                to_center = to_center / dist_to_center

                point_xy = np.array([px, py])
                if is_in_ellipse(outer_ellipse, point_xy) and is_on_the_same_side(
                    np.array([outer_x, outer_y]), point_xy, line
                ):
                    if (-direction) * (grad_e[0] * to_center[0] + grad_e[1] * to_center[1]) < -0.5 and j >= num_circles - 2:
                        n_gradient_out += 1
                    cctag_points[num_circles - j - 1].append((px, py, float(gx), float(gy)))
                    if j >= num_circles - 2:
                        n_added_point += 1
                else:
                    return False, []

            direction = -direction

    if n_added_point > 0 and n_gradient_out / n_added_point > 0.5:
        return False, []
    return True, cctag_points


if HAS_NUMBA:

    @njit(cache=True)
    def _add_candidate_flow_to_cctag_numba_core(
        before: np.ndarray,
        after: np.ndarray,
        positions: np.ndarray,
        gradients: np.ndarray,
        processed: np.ndarray,
        filtered_children: np.ndarray,
        center_x: float,
        center_y: float,
        matrix: np.ndarray,
        num_circles: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, bool]:
        n_seeds = filtered_children.shape[0]
        max_pts = n_seeds * (num_circles - 1)
        out_bucket = np.empty(max_pts, dtype=np.int64)
        out_x = np.empty(max_pts)
        out_y = np.empty(max_pts)
        out_dx = np.empty(max_pts)
        out_dy = np.empty(max_pts)
        out_count = 0

        n_gradient_out = 0
        n_added_point = 0
        ok = True

        for si in range(n_seeds):
            seed_idx = filtered_children[si]
            direction = -1
            p_idx = seed_idx
            outer_x = float(positions[seed_idx, 0])
            outer_y = float(positions[seed_idx, 1])
            a = outer_x - center_x
            b = outer_y - center_y
            line_c = -a * center_x - b * center_y

            for j in range(1, num_circles):
                p_idx = before[p_idx] if direction == -1 else after[p_idx]
                if p_idx == -1:
                    ok = False
                    break

                if not processed[p_idx]:
                    processed[p_idx] = True

                    gx = gradients[p_idx, 0]
                    gy = gradients[p_idx, 1]
                    norm_grad = math.sqrt(gx * gx + gy * gy)
                    gex = gx / norm_grad
                    gey = gy / norm_grad

                    px = float(positions[p_idx, 0])
                    py = float(positions[p_idx, 1])
                    tcx = center_x - px
                    tcy = center_y - py
                    dist_to_center = math.sqrt(tcx * tcx + tcy * tcy)
                    tcx /= dist_to_center
                    tcy /= dist_to_center

                    s1 = _bilinear_form_numba(matrix, px, py, px, py)
                    s2 = _bilinear_form_numba(matrix, px, py, center_x, center_y)
                    in_ellipse = s1 * s2 > 0.0

                    side1 = a * outer_x + b * outer_y + line_c
                    side2 = a * px + b * py + line_c
                    same_side = side1 * side2 > 0.0

                    if in_ellipse and same_side:
                        if (-direction) * (gex * tcx + gey * tcy) < -0.5 and j >= num_circles - 2:
                            n_gradient_out += 1
                        out_bucket[out_count] = num_circles - j - 1
                        out_x[out_count] = px
                        out_y[out_count] = py
                        out_dx[out_count] = gx
                        out_dy[out_count] = gy
                        out_count += 1
                        if j >= num_circles - 2:
                            n_added_point += 1
                    else:
                        ok = False
                        break

                direction = -direction
            if not ok:
                break

        return (
            out_bucket[:out_count], out_x[:out_count], out_y[:out_count],
            out_dx[:out_count], out_dy[:out_count],
            n_gradient_out, n_added_point, ok,
        )


def _add_candidate_flow_to_cctag_numba(
    collection: EdgePointCollection,
    filtered_children: list[int],
    outer_ellipse: Ellipse,
    num_circles: int,
    cctag_points: list[list[tuple[float, float, float, float]]],
) -> tuple[bool, list[list[tuple[float, float, float, float]]]]:
    filtered_arr = np.asarray(filtered_children, dtype=np.int64)
    if len(filtered_arr) == 0:
        return True, cctag_points

    # A fresh, call-scoped mask (matching `ellipse_growing2`'s pattern) rather
    # than a persistent buffer on `collection` -- no dirty-index bookkeeping
    # or reset-on-every-return-path needed, since it's just discarded here.
    processed = np.zeros(collection.point_count(), dtype=bool)
    bucket, xs, ys, dxs, dys, n_gradient_out, n_added_point, ok = _add_candidate_flow_to_cctag_numba_core(
        collection.before, collection.after, collection.positions, collection.gradients, processed,
        filtered_arr, outer_ellipse.center[0], outer_ellipse.center[1], outer_ellipse.matrix, num_circles,
    )

    if not ok:
        return False, []

    for i in range(len(bucket)):
        cctag_points[bucket[i]].append((xs[i], ys[i], dxs[i], dys[i]))

    if n_added_point > 0 and n_gradient_out / n_added_point > 0.5:
        return False, []
    return True, cctag_points


def add_candidate_flow_to_cctag(
    collection: EdgePointCollection,
    filtered_children: list[int],
    outer_ellipse_points: list[int],
    outer_ellipse: Ellipse,
    num_circles: int,
    cctag_points: list[list[tuple[float, float, float, float]]] | None = None,
) -> tuple[bool, list[list[tuple[float, float, float, float]]]]:
    """Assemble per-ring point buckets by walking the before/after chain
    ``num_circles - 1`` times from each of ``filtered_children``. Mirrors
    ``addCandidateFlowtoCCTag``. Each bucket entry is ``(x, y, dx, dy)``.

    ``cctag_points``, if given, is appended into rather than replaced: the
    C++ ``cctagPoints.resize(numCircles)`` followed by ``push_back`` calls
    is a no-op resize when already the right size, so calling this twice
    with the same ``cctag_points`` (e.g. an ``isAnotherSegment`` call
    followed by the main call in ``cctagDetectionFromEdgesLoopTwoIteration``)
    accumulates both calls' points into the same buckets. On failure the
    source does ``cctagPoints.clear()``, wiping everything (including any
    prior call's contribution) -- replicated here by returning ``(False, [])``.

    Unlike the C++ source (which does not null-check the before/after
    chain here and could in principle dereference a null point if the
    chain runs out), this returns ``(False, [])`` in that case rather than
    risking undefined behavior -- a deliberate, documented safety
    improvement, not a behavioral difference for any chain that actually
    completes.

    Dispatches the before/after chain walk (structurally the same
    "sequential walk sharing one sticky boolean array across seeds" pattern
    as :func:`_connected_points`) to a fused Numba kernel when available,
    falling back to the pure-Python walk otherwise -- verified to produce
    identical bucket contents, in identical per-bucket order, on real
    images either way. The very first bucket-population loop (over
    ``outer_ellipse_points``, unconditional and independent of the chain
    walk) is gathered with one batched NumPy index instead of per-point
    indexing in both cases.
    """
    if cctag_points is None or len(cctag_points) != num_circles:
        cctag_points = [[] for _ in range(num_circles)]

    idx_arr = np.asarray(outer_ellipse_points, dtype=np.int64)
    if len(idx_arr) > 0:
        pos = collection.positions[idx_arr].astype(np.float64)
        grad = collection.gradients[idx_arr]
        cctag_points[num_circles - 1].extend(
            (float(pos[i, 0]), float(pos[i, 1]), float(grad[i, 0]), float(grad[i, 1])) for i in range(len(idx_arr))
        )

    if HAS_NUMBA:
        return _add_candidate_flow_to_cctag_numba(collection, filtered_children, outer_ellipse, num_circles, cctag_points)
    return _add_candidate_flow_to_cctag_numpy(collection, filtered_children, outer_ellipse, num_circles, cctag_points)
