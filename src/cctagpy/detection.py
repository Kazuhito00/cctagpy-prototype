"""Detection orchestration, ported from ``src/cctag/Detection.cpp`` and
``src/cctag/Multiresolution.cpp``.

Covers the 3-loop-per-level pipeline (``constructFlowComponentFromSeed`` ->
``completeFlowComponent`` -> ``cctagDetectionFromEdgesLoopTwoIteration``,
including ``flowComponentAssembling``/``isAnotherSegment``), the
multi-resolution driver (per-level processing + merge/dedup +
level-0 outer-ellipse refinement), and the top-level ``cctag_detection``
entry point. Identification (assigning a real ``id``/``status``) is wired
in separately by ``identification.py``; markers produced here all have
``id == -1`` and ``status == 0`` ("not yet processed").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cctagpy._numba_utils import HAS_NUMBA, njit
from cctagpy.cctag import CCTag
from cctagpy.distance import distance_point_ellipse, distance_point_ellipse_batched_median
from cctagpy.edge_collection import EdgePointCollection
from cctagpy.ellipse_growing import (
    add_candidate_flow_to_cctag,
    compute_hull,
    ellipse_growing2,
    ellipse_growing_init,
    is_in_ellipse,
    is_in_hull_batched,
)
from cctagpy.fitting import batched_ellipse_fitting, ellipse_fitting
from cctagpy.geometry import Circle, Ellipse, intersect_ellipse_with_line, rasterize_ellipse_perimeter
from cctagpy.params import INV_GRAD_WEIGHT, NO_WEIGHT, Parameters
from cctagpy.pyramid import ImagePyramid
from cctagpy.ransac import draw_subsets
from cctagpy.vote import children_of, edge_linking, outlier_removal, vote

if HAS_NUMBA:
    from cctagpy.ransac import another_segment_chunk


@dataclass
class Candidate:
    seed: int
    convex_edge_segment: list[int] = field(default_factory=list)
    outer_ellipse_points: list[int] = field(default_factory=list)
    outer_ellipse: Ellipse | None = None
    filtered_children: list[int] = field(default_factory=list)
    score: int = 0
    n_label: int = 0
    average_received_vote: float = 0.0


def construct_flow_component_from_seed(
    seed_idx: int, collection: EdgePointCollection, params: Parameters
) -> Candidate | None:
    if collection.processed_in[seed_idx]:
        return None

    candidate = Candidate(seed=seed_idx)
    segment = edge_linking(
        collection, seed_idx, params.window_size_on_inner_elliptic_segment, params.average_vote_min
    )
    candidate.convex_edge_segment = segment

    votes = collection.n_voters[np.asarray(segment, dtype=np.int64)]
    n_received_vote = int(votes.sum())
    n_voted_points = int(np.count_nonzero(votes))

    candidate.average_received_vote = float(n_received_vote * n_received_vote) / n_voted_points
    return candidate


def complete_flow_component(
    candidate: Candidate,
    collection: EdgePointCollection,
    params: Parameters,
    next_label: list[int],
    rng: np.random.Generator,
) -> bool:
    """Mirrors ``completeFlowComponent``. Mutates ``candidate`` in place and
    returns whether it survives to Loop 3."""
    try:
        children = children_of(collection, candidate.convex_edge_segment)
        if len(children) < params.min_points_segment_candidate:
            return False
        candidate.score = len(children)

        filtered, _ = outlier_removal(
            collection, children, params.thresh_robust_estimation_of_outer_ellipse, INV_GRAD_WEIGHT, 60, rng
        )
        candidate.filtered_children = filtered
        if len(filtered) < 5:
            return False

        n_label = None
        for idx in filtered:
            if collection.segment[idx] != -1:
                n_label = int(collection.segment[idx])
                break
        if n_label is None:
            n_label = next_label[0]
            next_label[0] += 1
        for idx in filtered:
            collection.segment[idx] = n_label
        candidate.n_label = n_label

        ellipse, good_init = ellipse_growing_init(collection, filtered)
        ellipse, outer_points = ellipse_growing2(
            collection, filtered, ellipse, params.ellipse_growing_elliptic_hull_width, good_init
        )
        candidate.outer_ellipse = ellipse
        candidate.outer_ellipse_points = outer_points

        idx_arr = np.asarray(outer_points, dtype=np.int64)
        dists = distance_point_ellipse(collection.positions[idx_arr].astype(np.float64), ellipse.matrix)
        sm_final = float(np.median(dists))
        if sm_final > params.thr_median_distance_ellipse:
            return False

        quality = len(outer_points) / rasterize_ellipse_perimeter(ellipse)
        if quality > 1.1:
            return False

        ratio_semi_axes = ellipse.a / ellipse.b
        if ratio_semi_axes < 0.05 or ratio_semi_axes > 20:
            return False

        return True
    except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
        return False


def _another_segment_trials_numpy(
    pts: np.ndarray, another_pts: np.ndarray, patience: int, rng: np.random.Generator
) -> tuple[float, bool]:
    """NumPy form of the RANSAC trial loop of :func:`is_another_segment`
    (used without Numba; see :mod:`cctagpy.ransac` for the kernel form).
    Returns ``(sm, found)``: the best score and whether any trial was
    accepted."""
    sm = float("inf")
    qm: Ellipse | None = None
    cnt = 0

    # Same chunked-draw strategy as `vote.outlier_removal` (see its comment):
    # `patience - cnt` trials per chunk, each trial's two 4-subsets (one per
    # point set) drawn in bulk with `ransac.draw_subsets` instead of two
    # `rng.choice` calls per trial -- same distribution, different stream.
    # `batched_ellipse_fitting` then replaces `patience` separate
    # `ellipse_fitting` calls (each doing its own tiny inv/eig/solve/svd)
    # with one batched call over the whole chunk.
    while cnt < patience:
        chunk_size = patience - cnt
        i1s = draw_subsets(rng, chunk_size, len(pts), 4)
        i2s = draw_subsets(rng, chunk_size, len(another_pts), 4)
        eight_pts_batch = np.concatenate([pts[i1s], another_pts[i2s]], axis=1)  # (chunk_size, 8, 2)

        ellipses = batched_ellipse_fitting(eight_pts_batch)

        # Score every candidate ellipse of the chunk against the (fixed)
        # `pts`/`another_pts` point sets in two batched calls instead of a
        # per-candidate Python loop -- pure elementwise broadcast of the same
        # formula `distance_point_ellipse` already uses, so this is not an
        # approximation, just the same arithmetic computed for the whole
        # chunk at once (see `distance_point_ellipse_batched_median`).
        valid_idx = [t for t, e in enumerate(ellipses) if e is not None]
        s_full = np.full(chunk_size, np.inf)
        if valid_idx:
            conics = np.stack([ellipses[t].matrix for t in valid_idx])
            s_arr = distance_point_ellipse_batched_median(pts, conics) + distance_point_ellipse_batched_median(
                another_pts, conics
            )
            s_full[valid_idx] = s_arr

        for t, e in enumerate(ellipses):
            if e is None:
                cnt += 1
                if cnt >= patience:
                    break
                continue
            ratio = e.a / e.b
            if ratio < 0.12 or ratio > 8:
                cnt += 1
                if cnt >= patience:
                    break
                continue
            s = s_full[t]

            if s < sm:
                cnt = 0
                qm = e
                sm = s
            else:
                cnt += 1
                if cnt >= patience:
                    break

    return sm, qm is not None


def is_another_segment(
    collection: EdgePointCollection,
    outer_ellipse: Ellipse,
    outer_ellipse_points: list[int],
    another_candidate: Candidate,
    num_circles: int,
    thr_median_distance_ellipse: float,
    rng: np.random.Generator,
    cctag_points: list[list[tuple[float, float, float, float]]] | None,
) -> tuple[bool, list[int] | None, Ellipse | None, list[list[tuple[float, float, float, float]]] | None]:
    """Mirrors ``isAnotherSegment``. Returns ``(ok, merged_points, merged_ellipse, cctag_points)``."""
    another_points = another_candidate.outer_ellipse_points
    pts_idx = np.asarray(outer_ellipse_points, dtype=np.int64)
    another_idx = np.asarray(another_points, dtype=np.int64)
    pts = collection.positions[pts_idx].astype(np.float64)
    another_pts = collection.positions[another_idx].astype(np.float64)

    if len(pts) < 4 or len(another_pts) < 4:
        return False, None, None, cctag_points

    dist_ref = distance_point_ellipse(pts, outer_ellipse.matrix)
    s_ref = float(np.median(dist_ref))

    sm = float("inf")
    patience = 100
    if HAS_NUMBA:
        # Whole chunk (8-point fit, ratio gate, two-median score, patience
        # replay) in one kernel; see `ransac.another_segment_chunk`.
        cnt = 0
        found = False
        while cnt < patience:
            chunk_size = patience - cnt
            i1s = draw_subsets(rng, chunk_size, len(pts), 4)
            i2s = draw_subsets(rng, chunk_size, len(another_pts), 4)
            cnt, sm, found = another_segment_chunk(pts, another_pts, i1s, i2s, patience, cnt, sm, found)
    else:
        sm, found = _another_segment_trials_numpy(pts, another_pts, patience, rng)

    if not found or not (sm < 6.0 * s_ref):
        return False, None, None, cctag_points

    merged_idx = list(outer_ellipse_points) + list(another_points)
    merged_positions = collection.positions[np.asarray(merged_idx, dtype=np.int64)].astype(np.float64)
    try:
        merged_ellipse = ellipse_fitting(merged_positions)
    except (ValueError, np.linalg.LinAlgError):
        return False, None, None, cctag_points

    quality = len(merged_idx) / rasterize_ellipse_perimeter(merged_ellipse)
    if quality >= 1.1:
        return False, None, None, cctag_points

    dist_final = distance_point_ellipse(merged_positions, merged_ellipse.matrix)
    sm_final = float(np.median(dist_final))
    if sm_final >= thr_median_distance_ellipse:
        return False, None, None, cctag_points

    ok, cctag_points = add_candidate_flow_to_cctag(
        collection, another_candidate.filtered_children, another_points, merged_ellipse, num_circles, cctag_points
    )
    if not ok:
        return False, None, None, cctag_points

    return True, merged_idx, merged_ellipse, cctag_points


def flow_component_assembling(
    collection: EdgePointCollection,
    candidate: Candidate,
    v_candidate_loop_two: list[Candidate],
    outer_ellipse: Ellipse,
    outer_ellipse_points: list[int],
    num_circles: int,
    thr_median_distance_ellipse: float,
    rng: np.random.Generator,
    cctag_points: list[list[tuple[float, float, float, float]]] | None,
) -> tuple[bool, Ellipse, list[int], float, list[list[tuple[float, float, float, float]]] | None]:
    """Mirrors ``flowComponentAssembling``."""
    seed_pos = collection.positions[candidate.seed].astype(np.float64)
    flow_length = collection.flow_length[candidate.seed]
    if flow_length <= 0:
        return False, outer_ellipse, outer_ellipse_points, 0.0, cctag_points

    research_area = Circle(center=seed_pos, radius=flow_length * 2.5)

    score = -1
    i_max = None
    for i, other in enumerate(v_candidate_loop_two):
        if other is candidate or candidate.n_label == other.n_label:
            continue
        other_flow_length = collection.flow_length[other.seed]
        ratio = other_flow_length / flow_length
        if not (0.666 < ratio < 1.5):
            continue
        other_seed_pos = collection.positions[other.seed].astype(np.float64)
        if not is_in_ellipse(research_area, other_seed_pos):
            continue
        if other.score > score:
            score = other.score
            i_max = i

    if score <= 0 or i_max is None:
        return False, outer_ellipse, outer_ellipse_points, 0.0, cctag_points

    selected = v_candidate_loop_two[i_max]
    ok, merged_points, merged_ellipse, cctag_points = is_another_segment(
        collection, outer_ellipse, outer_ellipse_points, selected, num_circles,
        thr_median_distance_ellipse, rng, cctag_points,
    )
    if ok:
        quality = len(merged_points) / rasterize_ellipse_perimeter(merged_ellipse)
        return True, merged_ellipse, merged_points, quality, cctag_points
    return False, outer_ellipse, outer_ellipse_points, 0.0, cctag_points


_HEURISTIC_BOUNDS = [
    (0.35, 300.0, float("inf")),
    (0.45, 200.0, 300.0),
    (0.5, 100.0, 200.0),
    (0.5, 70.0, 100.0),
    (0.96, 50.0, 70.0),
]


def _fails_size_heuristic(quality: float, real_size: float) -> bool:
    if real_size < 50.0:
        return True
    for q_thr, lo, hi in _HEURISTIC_BOUNDS:
        if quality <= q_thr and lo <= real_size < hi:
            return True
    return False


def cctag_detection_from_edges_loop_two_iteration(
    collection: EdgePointCollection,
    v_candidate_loop_two: list[Candidate],
    i_candidate: int,
    pyramid_level: int,
    scale: float,
    params: Parameters,
    rng: np.random.Generator,
) -> CCTag | None:
    """Mirrors ``cctagDetectionFromEdgesLoopTwoIteration``."""
    candidate = v_candidate_loop_two[i_candidate]
    outer_ellipse_points = list(candidate.outer_ellipse_points)
    outer_ellipse = candidate.outer_ellipse
    cctag_points: list[list[tuple[float, float, float, float]]] | None = None
    num_circles = params.n_crowns * 2

    try:
        quality = len(outer_ellipse_points) / rasterize_ellipse_perimeter(outer_ellipse)

        if params.search_for_another_segment and 0.25 < quality < 0.7:
            ok, new_ellipse, new_points, new_quality, cctag_points = flow_component_assembling(
                collection, candidate, v_candidate_loop_two, outer_ellipse, outer_ellipse_points,
                num_circles, params.thr_median_distance_ellipse, rng, cctag_points,
            )
            if ok:
                outer_ellipse, outer_ellipse_points, quality = new_ellipse, new_points, new_quality

        ok, cctag_points = add_candidate_flow_to_cctag(
            collection, candidate.filtered_children, candidate.outer_ellipse_points,
            outer_ellipse, num_circles, cctag_points,
        )
        if not ok:
            return None

        rescale_ellipse = Ellipse(
            center=outer_ellipse.center, a=outer_ellipse.a * scale, b=outer_ellipse.b * scale, angle=outer_ellipse.angle
        )
        real_pixel_perimeter = rasterize_ellipse_perimeter(rescale_ellipse)
        real_size_outer_ellipse_points = quality * real_pixel_perimeter

        if _fails_size_heuristic(quality, real_size_outer_ellipse_points):
            return None

        ratio_semi_axes = outer_ellipse.a / outer_ellipse.b
        if ratio_semi_axes > 8.0 or ratio_semi_axes < 0.125:
            return None

        idx_arr = np.asarray(outer_ellipse_points, dtype=np.int64)
        positions = collection.positions[idx_arr].astype(np.float64)

        q_in, q_out = compute_hull(outer_ellipse, 3.6)
        if not is_in_hull_batched(q_in, q_out, positions).all():
            return None

        quality2 = float(np.sum(collection.norm_grad[idx_arr])) * scale

        return CCTag(
            id=-1,
            center_img=outer_ellipse.center.copy(),
            points=cctag_points if cctag_points is not None else [[] for _ in range(num_circles)],
            outer_ellipse=outer_ellipse,
            homography=np.zeros((3, 3)),
            pyramid_level=pyramid_level,
            scale=scale,
            quality=quality2,
        )
    except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
        return None


def cctag_detection_from_edges(
    collection: EdgePointCollection,
    seeds: list[int],
    pyramid_level: int,
    scale: float,
    level_height: int,
    params: Parameters,
    rng: np.random.Generator,
) -> list[CCTag]:
    """Mirrors ``cctagDetectionFromEdges`` (the 3-loop-per-level driver)."""
    if not seeds:
        return []

    seeds_sorted = sorted(seeds, key=lambda i: collection.is_max[i], reverse=True)

    n_maximum_nb_seeds = max(level_height // 2, params.maximum_nb_seeds)
    n_seeds_to_process = min(len(seeds_sorted), n_maximum_nb_seeds)

    candidates_loop_one: list[Candidate] = []
    for i in range(n_seeds_to_process):
        candidate = construct_flow_component_from_seed(seeds_sorted[i], collection, params)
        if candidate is not None:
            candidates_loop_one.append(candidate)
    candidates_loop_one.sort(key=lambda c: c.average_received_vote, reverse=True)

    n_flow_component_loop_two = min(len(candidates_loop_one), params.maximum_nb_candidates_loop_two)

    next_label = [0]
    v_candidate_loop_two: list[Candidate] = []
    for i in range(n_flow_component_loop_two):
        candidate = candidates_loop_one[i]
        if complete_flow_component(candidate, collection, params, next_label, rng):
            v_candidate_loop_two.append(candidate)

    markers: list[CCTag] = []
    for i_candidate in range(len(v_candidate_loop_two)):
        tag = cctag_detection_from_edges_loop_two_iteration(
            collection, v_candidate_loop_two, i_candidate, pyramid_level, scale, params, rng
        )
        if tag is not None:
            markers.append(tag)
    return markers


def _intersect_line_to_two_ellipses(
    y: int, q_in: Ellipse, q_out: Ellipse, collection: EdgePointCollection, points_in_hull: list[int]
) -> bool:
    width, _ = collection.shape
    inter_out = intersect_ellipse_with_line(q_out, y, True)
    inter_in = intersect_ellipse_with_line(q_in, y, True)

    def _maybe_add(x: int) -> None:
        idx = collection.index_at(x, y)
        if idx == -1:
            return
        px, py = collection.positions[idx]
        center_to_point = np.array([q_in.center[0] - px, q_in.center[1] - py])
        gx, gy = collection.gradients[idx]
        if gx * center_to_point[0] + gy * center_to_point[1] < 0:
            points_in_hull.append(idx)

    if len(inter_out) == 2 and len(inter_in) == 2:
        begin1, end1 = max(0, int(inter_out[0])), min(width - 1, int(inter_in[0]))
        begin2, end2 = max(0, int(inter_in[1])), min(width - 1, int(inter_out[1]))
        for x in range(begin1, end1 + 1):
            _maybe_add(x)
        for x in range(begin2, end2 + 1):
            _maybe_add(x)
    elif len(inter_out) == 2 and len(inter_in) <= 1:
        begin, end = max(0, int(inter_out[0])), min(width - 1, int(inter_out[1]))
        for x in range(begin, end + 1):
            _maybe_add(x)
    elif len(inter_out) == 1 and len(inter_in) == 0:
        if 0 <= inter_out[0] < width:
            _maybe_add(int(inter_out[0]))
    else:
        return False
    return True


if HAS_NUMBA:

    @njit(cache=True, inline="always")
    def _intersect_horizontal_numba(m: np.ndarray, y: float) -> tuple[int, float, float]:
        """``intersect_ellipse_with_line``'s horizontal branch: returns
        ``(root_count, x0, x1)`` instead of a variable-length list."""
        a = m[0, 0]
        b = 2.0 * (y * m[0, 1] + m[0, 2])
        c = m[1, 1] * y * y + 2.0 * y * m[2, 1] + m[2, 2]
        disc = b * b / 4.0 - a * c
        if disc > 0.0:
            sq = np.sqrt(disc)
            return 2, (-b / 2.0 - sq) / a, (-b / 2.0 + sq) / a
        if disc == 0.0:
            return 1, -b / (2.0 * a), 0.0
        return 0, 0.0, 0.0

    @njit(cache=True)
    def _scan_hull_row_numba(
        y: int,
        q_in_matrix: np.ndarray,
        q_out_matrix: np.ndarray,
        edge_map: np.ndarray,
        positions: np.ndarray,
        gradients: np.ndarray,
        width: int,
        cx: float,
        cy: float,
        found: np.ndarray,
        found_count: int,
    ) -> tuple[bool, int]:
        """One row of ``_intersect_line_to_two_ellipses``: same
        two-ellipse-crossing branch logic, scanning the resulting x-range(s)
        against ``edge_map`` and appending inward-pointing points into
        ``found`` in place. Returns ``(row_matched, new_found_count)``."""
        n_out, out0, out1 = _intersect_horizontal_numba(q_out_matrix, float(y))
        n_in, in0, in1 = _intersect_horizontal_numba(q_in_matrix, float(y))

        b0, e0 = 0, -1
        b1, e1 = 0, -1
        if n_out == 2 and n_in == 2:
            b0, e0 = max(0, int(out0)), min(width - 1, int(in0))
            b1, e1 = max(0, int(in1)), min(width - 1, int(out1))
        elif n_out == 2 and n_in <= 1:
            b0, e0 = max(0, int(out0)), min(width - 1, int(out1))
        elif n_out == 1 and n_in == 0:
            xi = int(out0)
            if 0 <= xi < width:
                b0, e0 = xi, xi
        else:
            return False, found_count

        for x in range(b0, e0 + 1):
            idx = edge_map[y, x]
            if idx == -1:
                continue
            gx, gy = gradients[idx, 0], gradients[idx, 1]
            if gx * (cx - positions[idx, 0]) + gy * (cy - positions[idx, 1]) < 0.0:
                found[found_count] = idx
                found_count += 1
        for x in range(b1, e1 + 1):
            idx = edge_map[y, x]
            if idx == -1:
                continue
            gx, gy = gradients[idx, 0], gradients[idx, 1]
            if gx * (cx - positions[idx, 0]) + gy * (cy - positions[idx, 1]) < 0.0:
                found[found_count] = idx
                found_count += 1
        return True, found_count

    @njit(cache=True)
    def _select_points_in_hull_numba(
        edge_map: np.ndarray,
        positions: np.ndarray,
        gradients: np.ndarray,
        width: int,
        height: int,
        q_in_matrix: np.ndarray,
        q_out_matrix: np.ndarray,
        cx: float,
        cy: float,
        y_center: float,
    ) -> np.ndarray:
        found = np.empty(positions.shape[0], dtype=np.int64)
        found_count = 0

        max_y = max(int(y_center), 0)
        for y in range(max_y, height):
            matched, found_count = _scan_hull_row_numba(
                y, q_in_matrix, q_out_matrix, edge_map, positions, gradients, width, cx, cy, found, found_count
            )
            if not matched:
                break
        min_y = min(int(y_center), height - 1)
        for y in range(min_y, -1, -1):
            matched, found_count = _scan_hull_row_numba(
                y, q_in_matrix, q_out_matrix, edge_map, positions, gradients, width, cx, cy, found, found_count
            )
            if not matched:
                break

        return found[:found_count]


def select_edge_point_in_elliptic_hull(
    collection: EdgePointCollection, outer_ellipse: Ellipse, scale: float
) -> list[int]:
    """Mirrors ``selectEdgePointInEllipticHull``.

    Dispatches to a scalar Numba kernel when available -- this scans a thin
    elliptical-annulus band row by row against the full-resolution edge map
    (called once per level>0 marker), and the Python form pays interpreter
    overhead per scanned pixel for what is otherwise a few FLOPs and an
    array lookup. See :func:`cctagpy.ellipse_growing.is_in_ellipse` for the
    same rationale applied to the ellipse-growing flood fill."""
    q_in, q_out = compute_hull(outer_ellipse, scale)
    width, height = collection.shape
    y_center = outer_ellipse.center[1]

    if HAS_NUMBA:
        return _select_points_in_hull_numba(
            collection.edge_map,
            collection.positions,
            collection.gradients,
            width,
            height,
            q_in.matrix,
            q_out.matrix,
            outer_ellipse.center[0],
            outer_ellipse.center[1],
            y_center,
        )

    points_in_hull: list[int] = []
    max_y = max(int(y_center), 0)
    for y in range(max_y, height):
        if not _intersect_line_to_two_ellipses(y, q_in, q_out, collection, points_in_hull):
            break
    min_y = min(int(y_center), height - 1)
    for y in range(min_y, -1, -1):
        if not _intersect_line_to_two_ellipses(y, q_in, q_out, collection, points_in_hull):
            break
    return points_in_hull


def update(markers: list[CCTag], marker_to_add: CCTag) -> None:
    """Mirrors ``update`` (dedup-by-merge into ``markers``, in place)."""
    flag = False
    for i, current in enumerate(markers):
        if current.status > 0 and marker_to_add.status > 0 and current.is_equal(marker_to_add):
            if marker_to_add.quality > current.quality:
                markers[i] = marker_to_add
            flag = True
    if not flag:
        markers.append(marker_to_add)


def cctag_multires_detection(
    gray_image: np.ndarray, params: Parameters, rng: np.random.Generator
) -> list[CCTag]:
    """Mirrors ``cctagMultiresDetection``: build the pyramid, run the 3-loop
    pipeline per level (coarsest first), then refine each level>0 marker's
    outer ellipse at level-0 resolution."""
    height, width = gray_image.shape
    n_levels = params.number_of_processed_multires_layers

    pyramid = ImagePyramid(width, height, n_levels)
    pyramid.build(gray_image, params.canny_thr_low, params.canny_thr_high)

    markers: list[CCTag] = []
    # One collection, rebuilt per level (coarsest first): `build_from_edges`
    # already clears only the previous build's own cells of `edge_map`
    # (O(points), not a full-image fill), so nothing is gained by a second,
    # caller-managed buffer-handoff mechanism. Only level 0's collection
    # (the last one built, since levels run coarse-to-fine) is consulted
    # after this loop, for the re-fit below.
    collection = EdgePointCollection(width, height)
    for i in range(n_levels - 1, -1, -1):
        level = pyramid.get_level(i)
        collection.build_from_edges(level.edges, level.dx, level.dy)

        seeds = vote(collection, level.dx, level.dy, params)
        level_markers = cctag_detection_from_edges(
            collection, seeds, i, float(2**i), level.height, params, rng
        )
        markers.extend(level_markers)

    level0_collection = collection

    for marker in markers:
        if marker.pyramid_level > 0:
            scale = marker.scale
            rescaled_ellipse = marker.rescaled_outer_ellipse
            points_in_hull = select_edge_point_in_elliptic_hull(level0_collection, rescaled_ellipse, scale)
            if len(points_in_hull) < 5:
                continue

            filtered, _ = outlier_removal(level0_collection, points_in_hull, 20.0, NO_WEIGHT, 60, rng)
            if len(filtered) < 5:
                continue

            try:
                idx_arr = np.asarray(filtered, dtype=np.int64)
                positions = level0_collection.positions[idx_arr].astype(np.float64)
                refit_ellipse = ellipse_fitting(positions)
                rescaled_points = [
                    (
                        float(level0_collection.positions[i, 0]),
                        float(level0_collection.positions[i, 1]),
                        float(level0_collection.gradients[i, 0]),
                        float(level0_collection.gradients[i, 1]),
                    )
                    for i in filtered
                ]
                marker.center_img = marker.center_img * scale
                marker.rescaled_outer_ellipse = refit_ellipse
                marker.rescaled_outer_ellipse_points = rescaled_points
            except (ValueError, np.linalg.LinAlgError):
                pass
        else:
            marker.rescaled_outer_ellipse_points = marker.points[-1]

    return markers


def cctag_detection(
    gray_image: np.ndarray,
    params: Parameters,
    bank: list[list[float]] | None = None,
    rng: np.random.Generator | None = None,
    use_identification_optimizer: bool = False,
) -> list[CCTag]:
    """Top-level entry point. Mirrors ``cctagDetection`` (CPU path): builds
    the pyramid and detects geometric candidates, runs identification
    (``identify_step_1`` for every marker, then ``identify_step_2`` only for
    those whose step 1 succeeded) if ``params.do_identification``, sets
    each marker's final ``status``, then dedups.

    ``bank`` defaults to the built-in table for ``params.n_crowns`` (see
    ``markers_bank.py``) if not given.

    ``use_identification_optimizer`` (Python-port-only knob, not a C++
    ``Parameters`` field, default off): routes ``identify_step_2``'s
    imaged-center search through the experimental bounded-optimizer path
    (see ``identification.refine_conic_family_glob_optimizer``) instead of
    the grid search that mirrors the C++ reference exactly.
    """
    from cctagpy.cctag import status as cctag_status
    from cctagpy.identification import identify_step_1, identify_step_2
    from cctagpy.markers_bank import CCTagMarkersBank

    if rng is None:
        rng = np.random.default_rng()
    if bank is None:
        bank = CCTagMarkersBank(n_crowns=params.n_crowns).get_markers()

    markers = cctag_multires_detection(gray_image, params, rng)

    if params.do_identification:
        detected = [0] * len(markers)
        selected_cuts: list[list] = [[] for _ in markers]

        for i, tag in enumerate(markers):
            detected[i], selected_cuts[i] = identify_step_1(tag, gray_image, params)

        for i, tag in enumerate(markers):
            if detected[i] == cctag_status.id_reliable:
                detected[i] = identify_step_2(
                    tag, selected_cuts[i], bank, gray_image, params, use_optimizer=use_identification_optimizer
                )
            tag.status = detected[i]

    markers_prelim: list[CCTag] = []
    for m in markers:
        update(markers_prelim, m)
    markers_final: list[CCTag] = []
    for m in markers_prelim:
        update(markers_final, m)
    markers_final.sort(key=lambda m: m.id)
    return markers_final
