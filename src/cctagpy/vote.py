"""Voting procedure, ported from ``src/cctag/Vote.cpp``.

Covers: link construction + the main ``vote()`` loop, ``edge_linking`` /
``edge_linking_dir`` (convex-arc growth from a seed), ``children_of``
(voter lookup for a convex segment), and ``outlier_removal`` (RANSAC
median-distance ellipse fit used to filter a seed's children). Loop-3-only
helpers (``isAnotherSegment``/``flowComponentAssembling``) are ported
alongside ``detection.py`` instead, since they are tightly coupled to that
stage's control flow.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from cctagpy._numba_utils import HAS_NUMBA, njit, prange
from cctagpy.bresenham import gradient_direction_descent_batch
from cctagpy.distance import distance_point_ellipse, distance_point_ellipse_batched
from cctagpy.edge_collection import EdgePointCollection
from cctagpy.geometry import compute_semi_axes_ratio_batched
from cctagpy.params import INV_GRAD_WEIGHT, INV_SQUARE_GRAD_WEIGHT, NO_WEIGHT, Parameters
from cctagpy.ransac import draw_subsets

if HAS_NUMBA:
    from cctagpy.ransac import outlier_removal_chunk

EDGE_NOT_FOUND = -1
CONVEXITY_LOST = -2


def _compute_vote_targets_numpy(collection: EdgePointCollection, params: Parameters) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized crown-hop chain walk: for every point ``i``, which point it
    votes for (``choosen[i]``, or -1) and the chain's total distance if so.

    Each point's chain is a pure function of its own ``before``/``after``
    links and every point's static ``gradients``/``positions`` (all already
    computed) -- it never depends on any OTHER point's chain or on
    processing order -- so, exactly like ``gradient_direction_descent_batch``
    in ``bresenham.py``, this runs all points in lockstep for a small fixed
    number of hops (``2*(n_crowns-1)``) instead of one Python loop iteration
    per point. Only the actual vote *bookkeeping* (``voters``/
    ``flow_length``/``is_max``/``seeds``, which genuinely depends on
    processing order for its running averages and tie-breaking) stays a
    Python loop, in :func:`vote`.

    The scalar reference resets its local ``choosen`` to -1 at the top of
    every while-loop iteration and ``break``s out entirely on any failure,
    so a point's *final* ``choosen`` is non-(-1) iff it survives every hop
    of every iteration -- exactly "stays alive the whole way", tracked here
    as a single monotonic ``alive`` mask instead of replaying the reset.
    """
    n = collection.point_count()
    before = collection.before
    after = collection.after
    positions = collection.positions.astype(np.float64)
    gradients = collection.gradients
    ratio_voting = params.ratio_voting
    n_crowns = params.n_crowns

    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    def safe(idx: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.where(mask, idx, 0)

    def pair_dist(a_idx: np.ndarray, b_idx: np.ndarray) -> np.ndarray:
        diff = positions[a_idx] - positions[b_idx]
        return np.hypot(diff[:, 0], diff[:, 1])

    def cos_diff(a_grad: np.ndarray, b_grad: np.ndarray) -> np.ndarray:
        return -(a_grad[:, 0] * b_grad[:, 0] + a_grad[:, 1] * b_grad[:, 1])

    def ratio_ok_vec(v_list: list[np.ndarray]) -> np.ndarray:
        ok = np.ones(n, dtype=bool)
        for a in range(len(v_list)):
            for b in range(a + 1, len(v_list)):
                da, db = v_list[a], v_list[b]
                ok &= (da <= db * ratio_voting) & (db <= da * ratio_voting)
        return ok

    idx = np.arange(n)
    current = before.copy()
    alive = current != -1

    current_safe = safe(current, alive)
    current_grad = gradients[current_safe]
    alive &= cos_diff(gradients, current_grad) >= 0.0

    current_safe = safe(current, alive)
    dist0 = np.where(alive, pair_dist(idx, current_safe), 0.0)
    total_distance = dist0.copy()
    v_dists = [dist0]

    for _ in range(1, n_crowns):
        # hop "after"
        current_safe = safe(current, alive)
        target = np.where(alive, after[current_safe], -1)
        alive &= target != -1
        target_safe = safe(target, alive)
        target_grad = gradients[target_safe]
        alive &= cos_diff(target_grad, current_grad) >= 0.0
        dist = np.where(alive, pair_dist(target_safe, current_safe), 0.0)
        v_dists.append(dist)
        total_distance = total_distance + dist
        alive &= ratio_ok_vec(v_dists)
        current = np.where(alive, target, current)
        current_grad = np.where(alive[:, None], target_grad, current_grad)

        # hop "before"
        current_safe = safe(current, alive)
        target2 = np.where(alive, before[current_safe], -1)
        alive &= target2 != -1
        target2_safe = safe(target2, alive)
        target2_grad = gradients[target2_safe]
        alive &= cos_diff(target2_grad, current_grad) >= 0.0
        dist2 = np.where(alive, pair_dist(target2_safe, current_safe), 0.0)
        v_dists.append(dist2)
        total_distance = total_distance + dist2
        alive &= ratio_ok_vec(v_dists)
        current = np.where(alive, target2, current)
        current_grad = np.where(alive[:, None], target2_grad, current_grad)

    choosen = np.where(alive, current, -1)
    return choosen, total_distance


if HAS_NUMBA:

    @njit(cache=True, parallel=True)
    def _compute_vote_targets_numba(
        before: np.ndarray,
        after: np.ndarray,
        positions: np.ndarray,
        gradients: np.ndarray,
        ratio_voting: float,
        n_crowns: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-point scalar form of :func:`_compute_vote_targets_numpy`, one
        point per ``prange`` iteration (each writes only its own
        ``choosen[i]``/``total_distance[i]`` and a private row of the
        preallocated ``dists`` scratch). Same hop order, same float
        operations (``np.hypot`` per link, distances summed in hop order, the
        all-pairs ratio test after every hop), so the outputs are
        bit-identical to the lockstep NumPy form."""
        n = before.shape[0]
        n_links = 2 * n_crowns - 1
        choosen = np.full(n, -1, dtype=np.int64)
        total_distance = np.zeros(n, dtype=np.float64)
        dists = np.empty((n, n_links), dtype=np.float64)

        for i in prange(n):
            current = before[i]
            if current == -1:
                continue
            cg0 = gradients[current, 0]
            cg1 = gradients[current, 1]
            if -(gradients[i, 0] * cg0 + gradients[i, 1] * cg1) < 0.0:
                continue
            d = np.hypot(positions[i, 0] - positions[current, 0], positions[i, 1] - positions[current, 1])
            dists[i, 0] = d
            total = d
            n_d = 1
            alive = True

            for _ in range(n_crowns - 1):
                for direction in range(2):
                    target = after[current] if direction == 0 else before[current]
                    if target == -1:
                        alive = False
                        break
                    tg0 = gradients[target, 0]
                    tg1 = gradients[target, 1]
                    if -(tg0 * cg0 + tg1 * cg1) < 0.0:
                        alive = False
                        break
                    d = np.hypot(positions[target, 0] - positions[current, 0], positions[target, 1] - positions[current, 1])
                    dists[i, n_d] = d
                    n_d += 1
                    total = total + d
                    for a in range(n_d):
                        da = dists[i, a]
                        for b in range(a + 1, n_d):
                            db = dists[i, b]
                            if not (da <= db * ratio_voting and db <= da * ratio_voting):
                                alive = False
                    if not alive:
                        break
                    current = target
                    cg0 = tg0
                    cg1 = tg1
                if not alive:
                    break

            total_distance[i] = total
            if alive:
                choosen[i] = current

        return choosen, total_distance


def _compute_vote_targets(collection: EdgePointCollection, params: Parameters) -> tuple[np.ndarray, np.ndarray]:
    """Which point each point votes for (``choosen[i]`` or -1) and the chain's
    total distance; see :func:`_compute_vote_targets_numpy` for the
    algorithm and the independence argument that makes it batchable.
    Dispatches to the point-parallel Numba kernel when available."""
    if HAS_NUMBA:
        n = collection.point_count()
        if n == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        return _compute_vote_targets_numba(
            collection.before, collection.after, collection.positions.astype(np.float64), collection.gradients,
            float(params.ratio_voting), int(params.n_crowns),
        )
    return _compute_vote_targets_numpy(collection, params)


def vote(collection: EdgePointCollection, img_dx: np.ndarray, img_dy: np.ndarray, params: Parameters) -> list[int]:
    """Build before/after links then run the voting pass. Mirrors ``vote()``.

    Returns the list of seed point indices (points that received at least
    ``params.min_votes_to_select_candidate`` votes), in the order they first
    qualified -- callers should sort by final vote count themselves (see
    ``receivedMoreVoteThan`` in the source) since ``is_max`` keeps growing
    after a point is first added as a seed.
    """
    n = collection.point_count()
    width, height = collection.shape

    if n > 0:
        xs = collection.positions[:, 0].astype(np.int64)
        ys = collection.positions[:, 1].astype(np.int64)
        collection.before = gradient_direction_descent_batch(collection, xs, ys, -1, params.dist_search, img_dx, img_dy)
        collection.after = gradient_direction_descent_batch(collection, xs, ys, 1, params.dist_search, img_dx, img_dy)

    if params.angle_voting != 0:
        raise ValueError("angle_voting must be 0 (gradients are not normalized)")

    # The expensive per-point crown-hop chain walk (cos_diff/ratio_ok/hypot
    # over up to 2*(n_crowns-1) hops) is a pure, order-independent function
    # of each point's own before/after/gradient data, so it is batched
    # across all points in `_compute_vote_targets` (verified against this
    # loop's scalar form on ~170k real edge points at both n_crowns=3 and 4).
    #
    # The bookkeeping below looks sequential (a running average, an
    # append-on-first-qualify list) but is not actually order-dependent in
    # its *outputs*: `flow_length[choosen]`'s incremental-mean formula
    # always converges to the same value as `sum(total_distance_all[i] for
    # i voting for choosen) / count`, regardless of processing order --
    # that's just how a running mean works -- and `is_max[choosen]`'s final
    # value is simply that target's final vote count (or unset if it never
    # reached the threshold), also order-independent. Only `seeds`'
    # *insertion order* genuinely depends on processing order (each target
    # is appended at the moment its own vote count first reaches
    # `min_votes_to_select_candidate`, and callers tie-break by this order
    # for equal final vote counts) -- computed here as the original index
    # of each qualifying target's min-votes-th vote, sorted ascending,
    # which is exactly when the scalar loop would have appended it.
    choosen_all, total_distance_all = _compute_vote_targets(collection, params)

    voted_idx = np.nonzero(choosen_all != -1)[0]
    choosen_valid = choosen_all[voted_idx]
    dist_valid = total_distance_all[voted_idx]

    counts = np.bincount(choosen_valid, minlength=n)
    collection.n_voters = counts
    sums = np.bincount(choosen_valid, weights=dist_valid, minlength=n)
    has_votes = counts > 0
    collection.flow_length[has_votes] = sums[has_votes] / counts[has_votes]

    min_votes = params.min_votes_to_select_candidate
    qualifies = counts >= min_votes
    collection.is_max[qualifies] = counts[qualifies]

    # Group voter indices by target, preserving ascending original order
    # within each group (a stable sort of already-ascending `voted_idx`).
    # CSR voters: one stable sort of the voters by target gives every
    # target's voters as a contiguous ascending run; `counts` (the bincount
    # above) are exactly the run lengths, so the offsets are their cumsum.
    order = np.argsort(choosen_valid, kind="stable")
    collection.voter_index = voted_idx[order]
    voter_start = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=voter_start[1:])
    collection.voter_start = voter_start

    # seeds, in the order the scalar loop would have appended them: a target
    # qualifies the moment its `min_votes`-th voter is processed, so order
    # qualifying targets by the original index of that voter.
    qualifying_targets = np.nonzero(qualifies)[0]
    threshold_i = collection.voter_index[voter_start[qualifying_targets] + min_votes - 1]
    seeds = qualifying_targets[np.argsort(threshold_i, kind="stable")].tolist()
    return seeds


def edge_linking(
    collection: EdgePointCollection,
    seed_idx: int,
    window_size: int,
    average_vote_min: float,
) -> list[int]:
    """Grow a convex arc outward from ``seed_idx`` in both directions along
    the thinned edge image (8-connected walk). Mirrors ``edgeLinking``.
    Dispatches to a fused Numba kernel when available (the walk is a tight
    scalar loop over a few hundred pixels with a sliding angle window --
    exactly the shape that pays Python interpreter overhead per pixel),
    falling back to the pure-Python form otherwise; verified to return the
    same segment and the same ``processed_in`` side effects on every seed of
    both sample images."""
    if HAS_NUMBA:
        width, height = collection.shape
        segment = _edge_linking_numba(
            collection.positions, collection.gradients, collection.edge_map, collection.n_voters,
            collection.processed_in, seed_idx, window_size, average_vote_min, width, height,
        )
        return segment.tolist()
    return _edge_linking_python(collection, seed_idx, window_size, average_vote_min)


def _edge_linking_python(
    collection: EdgePointCollection,
    seed_idx: int,
    window_size: int,
    average_vote_min: float,
) -> list[int]:
    segment: deque[int] = deque([seed_idx])
    processed: set[tuple[int, int]] = set()
    sx, sy = int(collection.positions[seed_idx, 0]), int(collection.positions[seed_idx, 1])
    processed.add((sx, sy))

    _edge_linking_dir(collection, processed, seed_idx, 1, segment, window_size, average_vote_min)
    _edge_linking_dir(collection, processed, seed_idx, -1, segment, window_size, average_vote_min)
    return list(segment)


_XOFF = [1, 1, 0, -1, -1, -1, 0, 1]
_MAX_LINK_LENGTH = 100


def _edge_linking_dir(
    collection: EdgePointCollection,
    processed: set[tuple[int, int]],
    p_idx: int,
    direction: int,
    segment: deque[int],
    window_size: int,
    average_vote_min: float,
) -> None:
    width, height = collection.shape
    phi: deque[float] = deque()
    i = 0
    found = True
    stop = 0
    max_length = _MAX_LINK_LENGTH

    average_vote = float(collection.n_voters[p_idx])

    while i < max_length and found and average_vote >= average_vote_min:
        px, py = int(collection.positions[p_idx, 0]), int(collection.positions[p_idx, 1])
        gx, gy = collection.gradients[p_idx]
        angle = math.fmod(math.atan2(gy, gx) + 2.0 * math.pi, 2.0 * math.pi)
        phi.append(angle)
        if len(phi) > window_size:
            phi.popleft()

        shifting = round(((angle + math.pi / 4.0) / (2.0 * math.pi)) * 8.0) - 1
        yoff = [0, direction * -1, direction * -1, direction * -1, 0, direction * 1, direction * 1, direction * 1]

        stop = 0
        j = 0
        while stop == 0:
            if j >= 8:
                stop = EDGE_NOT_FOUND
                break

            k = (8 - shifting + j) % 8 if direction == 1 else (shifting + j) % 8
            sx = px + _XOFF[k]
            sy = py + yoff[k]

            if 0 <= sx < width and 0 <= sy < height:
                idx = collection.index_at(sx, sy)
                if idx != -1 and (sx, sy) not in processed:
                    if len(phi) == window_size:
                        if direction * math.sin(phi[-1] - phi[0]) < 0.0:
                            stop = CONVEXITY_LOST
                    else:
                        s = direction * math.sin(phi[-1] - phi[0])
                        c = math.cos(phi[-1] - phi[0])
                        if (s < -0.707 and c > 0.0) or (s < 0.0 and c < 0.0):
                            stop = CONVEXITY_LOST

                    if stop == 0:
                        processed.add((px, py))
                        p_idx = idx
                        if direction > 0:
                            segment.append(p_idx)
                        else:
                            segment.appendleft(p_idx)
                        average_vote = (average_vote * len(segment) + collection.n_voters[p_idx]) / (
                            len(segment) + 1.0
                        )
                        stop = 1
                    px, py = int(collection.positions[p_idx, 0]), int(collection.positions[p_idx, 1])
                    processed.add((px, py))
            j += 1

        found = stop == 1
        i += 1

    if i == max_length or stop == CONVEXITY_LOST:
        if len(segment) > window_size:
            n_mark = len(segment) - window_size
            for idx in list(segment)[:n_mark]:
                collection.processed_in[idx] = True
    elif stop == EDGE_NOT_FOUND:
        for idx in segment:
            collection.processed_in[idx] = True


if HAS_NUMBA:

    @njit(cache=True)
    def _edge_linking_dir_numba(
        positions: np.ndarray,
        gradients: np.ndarray,
        edge_map: np.ndarray,
        n_voters: np.ndarray,
        processed_in: np.ndarray,
        processed: set,
        seg_buf: np.ndarray,
        seg_head: int,
        seg_tail: int,
        p_idx: int,
        direction: int,
        window_size: int,
        average_vote_min: float,
        width: int,
        height: int,
    ) -> tuple[int, int]:
        """One direction of :func:`_edge_linking_dir`, statement for
        statement. The segment is a deque in the Python form; here it is the
        slice ``seg_buf[seg_head:seg_tail]`` of a buffer with room on both
        sides (returns the new head/tail). ``processed`` is keyed by point
        index rather than pixel: every pixel the Python form records is an
        edge pixel (the seed, or a hit in ``edge_map``), and edge pixels and
        point indices are one-to-one, so the two sets are the same set. The
        angle window is the last ``window_size`` entries of ``phi_all``."""
        xoff = (1, 1, 0, -1, -1, -1, 0, 1)
        max_length = 100
        phi_all = np.empty(max_length, dtype=np.float64)
        n_phi = 0
        i = 0
        found = True
        stop = 0

        average_vote = float(n_voters[p_idx])

        while i < max_length and found and average_vote >= average_vote_min:
            px = positions[p_idx, 0]
            py = positions[p_idx, 1]
            gx = gradients[p_idx, 0]
            gy = gradients[p_idx, 1]
            # np.fmod: C fmod semantics, same as math.fmod (which Numba lacks)
            angle = np.fmod(math.atan2(gy, gx) + 2.0 * math.pi, 2.0 * math.pi)
            phi_all[n_phi] = angle
            n_phi += 1
            window_full = n_phi >= window_size
            phi_last = phi_all[n_phi - 1]
            phi_first = phi_all[n_phi - window_size] if window_full else phi_all[0]

            shifting = int(np.rint(((angle + math.pi / 4.0) / (2.0 * math.pi)) * 8.0)) - 1

            stop = 0
            j = 0
            while stop == 0:
                if j >= 8:
                    stop = EDGE_NOT_FOUND
                    break

                k = (8 - shifting + j) % 8 if direction == 1 else (shifting + j) % 8
                sx = px + xoff[k]
                if k == 0 or k == 4:
                    sy = py
                elif k < 4:
                    sy = py - direction
                else:
                    sy = py + direction

                if 0 <= sx < width and 0 <= sy < height:
                    idx = edge_map[sy, sx]
                    if idx != -1 and idx not in processed:
                        if window_full:
                            if direction * math.sin(phi_last - phi_first) < 0.0:
                                stop = CONVEXITY_LOST
                        else:
                            s = direction * math.sin(phi_last - phi_first)
                            c = math.cos(phi_last - phi_first)
                            if (s < -0.707 and c > 0.0) or (s < 0.0 and c < 0.0):
                                stop = CONVEXITY_LOST

                        if stop == 0:
                            processed.add(p_idx)
                            p_idx = idx
                            if direction > 0:
                                seg_buf[seg_tail] = p_idx
                                seg_tail += 1
                            else:
                                seg_head -= 1
                                seg_buf[seg_head] = p_idx
                            seg_len = seg_tail - seg_head
                            average_vote = (average_vote * seg_len + n_voters[p_idx]) / (seg_len + 1.0)
                            stop = 1
                        px = positions[p_idx, 0]
                        py = positions[p_idx, 1]
                        processed.add(p_idx)
                j += 1

            found = stop == 1
            i += 1

        seg_len = seg_tail - seg_head
        if i == max_length or stop == CONVEXITY_LOST:
            if seg_len > window_size:
                n_mark = seg_len - window_size
                for q in range(seg_head, seg_head + n_mark):
                    processed_in[seg_buf[q]] = True
        elif stop == EDGE_NOT_FOUND:
            for q in range(seg_head, seg_tail):
                processed_in[seg_buf[q]] = True

        return seg_head, seg_tail

    @njit(cache=True)
    def _edge_linking_numba(
        positions: np.ndarray,
        gradients: np.ndarray,
        edge_map: np.ndarray,
        n_voters: np.ndarray,
        processed_in: np.ndarray,
        seed_idx: int,
        window_size: int,
        average_vote_min: float,
        width: int,
        height: int,
    ) -> np.ndarray:
        max_length = 100
        seg_buf = np.empty(2 * max_length + 3, dtype=np.int64)
        seg_head = max_length + 1
        seg_tail = seg_head + 1
        seg_buf[seg_head] = seed_idx
        processed = {seed_idx}

        seg_head, seg_tail = _edge_linking_dir_numba(
            positions, gradients, edge_map, n_voters, processed_in, processed, seg_buf, seg_head, seg_tail,
            seed_idx, 1, window_size, average_vote_min, width, height,
        )
        seg_head, seg_tail = _edge_linking_dir_numba(
            positions, gradients, edge_map, n_voters, processed_in, processed, seg_buf, seg_head, seg_tail,
            seed_idx, -1, window_size, average_vote_min, width, height,
        )
        return seg_buf[seg_head:seg_tail].copy()


def children_of(collection: EdgePointCollection, segment: list[int]) -> list[int]:
    """Voters of every point on ``segment`` whose vote count is not too far
    below the segment's max. Mirrors ``childrenOf``."""
    seg = np.asarray(segment, dtype=np.int64)
    counts = collection.n_voters[seg]
    vote_max = max(1, int(counts.max())) if len(seg) else 1

    keep = seg[(counts > 0) & (counts >= vote_max // 14)]
    if len(keep) == 0:
        return []
    starts = collection.voter_start[keep]
    ends = collection.voter_start[keep + 1]
    return np.concatenate([collection.voter_index[a:b] for a, b in zip(starts, ends)]).tolist()


def _outlier_removal_chunks_numpy(
    pts: np.ndarray,
    weights: np.ndarray | None,
    weighted_type: int,
    m: int,
    f: float,
    patience: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, float]:
    """NumPy form of the RANSAC trial loop of :func:`outlier_removal` (used
    without Numba; see :mod:`cctagpy.ransac` for the kernel form). Returns
    ``(qm, sm)``: the best conic matrix (``None`` if no trial was accepted)
    and its score."""
    sm = 1.0e7
    qm = None
    counter = 0
    while counter < patience:
        chunk_size = patience - counter
        perms = draw_subsets(rng, chunk_size, m, 5)
        p5 = pts[perms]  # (chunk_size, 5, 2)

        a_mat = np.empty((chunk_size, 5, 5))
        a_mat[:, :, 0] = p5[:, :, 0] ** 2
        a_mat[:, :, 1] = 2.0 * p5[:, :, 0] * p5[:, :, 1]
        a_mat[:, :, 2] = p5[:, :, 1] ** 2
        a_mat[:, :, 3] = 2.0 * f * p5[:, :, 0]
        a_mat[:, :, 4] = 2.0 * f * p5[:, :, 1]
        b_vec = np.full((chunk_size, 5), -f * f)

        det = np.linalg.det(a_mat)
        det_ok = det != 0
        temp = np.zeros((chunk_size, 5))
        if det_ok.any():
            # `b_vec[..., None]` forces the unambiguous "batch of column
            # vectors" gufunc signature ((m,m),(m,1)->(m,1)); passing a bare
            # (n, 5) array is ambiguous between that and "batch of matrices".
            temp[det_ok] = np.linalg.solve(a_mat[det_ok], b_vec[det_ok][..., None])[..., 0]
        quad_ok = det_ok & (temp[:, 0] * temp[:, 2] - temp[:, 1] * temp[:, 1] > 0)

        # Score every candidate conic of the chunk in two batched calls
        # instead of a per-trial Python loop that also built an `Ellipse`
        # object per trial just to read its axis ratio. Both
        # `compute_semi_axes_ratio_batched` and `distance_point_ellipse_batched`
        # are pure elementwise arithmetic (no matrix decomposition), so this
        # is the same arithmetic as the scalar path, computed for the whole
        # chunk at once. `invalid_arr` replicates exactly the cases where
        # the scalar path's `Ellipse(matrix=q).a / .b` would have raised
        # (degenerate conic -> `0.0 / 0.0`, or a negative semi-axis).
        q_stack = np.zeros((chunk_size, 3, 3))
        q_stack[:, 0, 0] = temp[:, 0]
        q_stack[:, 0, 1] = temp[:, 1]
        q_stack[:, 1, 0] = temp[:, 1]
        q_stack[:, 1, 1] = temp[:, 2]
        q_stack[:, 0, 2] = temp[:, 3]
        q_stack[:, 2, 0] = temp[:, 3]
        q_stack[:, 1, 2] = temp[:, 4]
        q_stack[:, 2, 1] = temp[:, 4]
        q_stack[:, 2, 2] = 1.0

        ratio_arr, invalid_arr = compute_semi_axes_ratio_batched(q_stack)
        ratio_ok = quad_ok & ~invalid_arr & (ratio_arr >= 0.04) & (ratio_arr <= 25)

        s_full = np.full(chunk_size, np.inf)
        if ratio_ok.any():
            dist_batch = distance_point_ellipse_batched(pts, q_stack[ratio_ok])
            if weighted_type != NO_WEIGHT:
                dist_batch = dist_batch * weights[None, :]
            s_full[ratio_ok] = np.median(dist_batch, axis=1)

        for t in range(chunk_size):
            if not ratio_ok[t]:
                counter += 1
                if counter >= patience:
                    break
                continue

            s = s_full[t]
            if s < sm:
                counter = 0
                qm = q_stack[t]
                sm = s
            else:
                counter += 1
                if counter >= patience:
                    break

    return qm, sm


def outlier_removal(
    collection: EdgePointCollection,
    children: list[int],
    threshold: float,
    weighted_type: int,
    max_size: int,
    rng: np.random.Generator,
) -> tuple[list[int], float | None]:
    """RANSAC median-distance ellipse fit + outlier filtering. Mirrors
    ``outlierRemoval``. Returns ``(filtered_children, sm_final)``; per the
    source, ``sm_final`` is ``None`` ("unchanged") whenever the routine
    bails out early (too few points, or no point passes the final filter).
    """
    n_children = len(children)
    n_subsample = min(n_children, max_size)
    if n_subsample < 5:
        return [], None

    step = n_children / n_subsample
    subsample_local_idx = []
    k = 0
    for i in range(n_children):
        if i == int(k * step):
            subsample_local_idx.append(i)
            k += 1

    child_idx = np.asarray(children, dtype=np.int64)
    sub_idx = child_idx[subsample_local_idx]
    pts = collection.positions[sub_idx].astype(np.float64)
    norm_grad_sub = collection.norm_grad[sub_idx]

    weights = None
    if weighted_type == INV_GRAD_WEIGHT:
        weights = 255.0 / norm_grad_sub
    elif weighted_type == INV_SQUARE_GRAD_WEIGHT:
        weights = np.full(len(sub_idx), 255.0)

    m = len(pts)
    f = 1.0
    sm = 1.0e7
    qm = None
    counter = 0
    patience = 70

    # The scalar reference draws one random point-quintuple per trial,
    # win-or-lose, and stops the instant `counter` reaches `patience`, so
    # the number of trials is data-dependent. Trials are processed in
    # chunks of `patience - counter` -- the minimum that could prove
    # termination from the current counter -- and each chunk's quintuples
    # are drawn in ONE bulk call (`ransac.draw_subsets`: a uniformly random
    # ordered 5-subset per row, the same distribution as
    # `rng.choice(m, 5, replace=False)` per trial, at a fraction of the
    # ~5us-per-call cost that had made the draws ~17% of end-to-end time). This changes which subsets a given seed produces
    # relative to per-trial `rng.choice`, i.e. results are statistically
    # equivalent rather than bit-identical to that form -- verified by
    # comparing marker ids and centers across seeds, see the README. The
    # 5x5 conic solve is then one batched det/solve call over the chunk
    # (pure gufuncs, numerically identical to per-trial calls), and only
    # the accept/reject/patience logic is replayed sequentially.
    if HAS_NUMBA:
        # Whole chunk (solve, gates, weighted median score, patience replay)
        # in one kernel; see `ransac.outlier_removal_chunk`. The best conic
        # accumulates in `qm_buf` in place.
        qm_buf = np.zeros((3, 3))
        have_qm = False
        use_weights = weighted_type != NO_WEIGHT
        weights_arr = weights if use_weights else np.empty(0)
        while counter < patience:
            chunk_size = patience - counter
            perms = draw_subsets(rng, chunk_size, m, 5)
            counter, sm, have_qm = outlier_removal_chunk(
                pts, weights_arr, use_weights, perms, patience, counter, sm, qm_buf, have_qm, f
            )
        qm = qm_buf if have_qm else None
    else:
        qm, sm = _outlier_removal_chunks_numpy(pts, weights, weighted_type, m, f, patience, rng)

    if qm is None:
        return [], None

    children_pos = collection.positions[child_idx].astype(np.float64)
    dist_final = distance_point_ellipse(children_pos, qm)
    if weighted_type == INV_GRAD_WEIGHT:
        dist_final = dist_final * 255.0 / collection.norm_grad[child_idx]
    elif weighted_type == INV_SQUARE_GRAD_WEIGHT:
        dist_final = dist_final * 255.0 / (collection.norm_grad[child_idx] ** 2)

    mask = dist_final < threshold * sm
    filtered_local = np.nonzero(mask)[0]
    if len(filtered_local) == 0:
        return [], None

    sm_final = float(np.median(dist_final[filtered_local]))
    filtered_children = child_idx[filtered_local].tolist()
    return filtered_children, sm_final
