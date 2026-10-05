import numpy as np
from helpers import build_collection, synthetic_cctag

from cctagpy.params import Parameters
from cctagpy.vote import children_of, edge_linking, outlier_removal, vote


def test_vote_finds_seeds_on_concentric_rings():
    img, center = synthetic_cctag()
    collection, dx, dy = build_collection(img)
    assert collection.point_count() > 0

    params = Parameters(n_crowns=3)
    seeds = vote(collection, dx, dy, params)

    assert len(seeds) > 0
    # seeds should be positioned somewhere within the target, not scattered
    # arbitrarily far outside it
    seed_positions = collection.positions[seeds]
    dists = np.hypot(seed_positions[:, 0] - center[0], seed_positions[:, 1] - center[1])
    assert np.all(dists < 100)


def test_edge_linking_and_children_and_outlier_removal_run_end_to_end():
    img, center = synthetic_cctag()
    collection, dx, dy = build_collection(img)
    params = Parameters(n_crowns=3)
    seeds = vote(collection, dx, dy, params)
    assert len(seeds) > 0

    seeds_sorted = sorted(seeds, key=lambda i: collection.is_max[i], reverse=True)
    seed = seeds_sorted[0]

    segment = edge_linking(collection, seed, params.window_size_on_inner_elliptic_segment, params.average_vote_min)
    assert seed in segment
    assert len(segment) >= 1

    children = children_of(collection, segment)
    if len(children) >= 5:
        rng = np.random.default_rng(0)
        filtered, sm_final = outlier_removal(
            collection, children, threshold=params.thresh_robust_estimation_of_outer_ellipse,
            weighted_type=1, max_size=60, rng=rng,
        )
        # either it found a consistent ellipse (some filtered points, sm set)
        # or bailed out cleanly (empty + None) -- both are valid outcomes,
        # what matters is it doesn't crash and keeps the two in sync.
        assert (len(filtered) == 0) == (sm_final is None)
