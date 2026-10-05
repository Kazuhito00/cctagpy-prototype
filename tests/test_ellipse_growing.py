import numpy as np
from helpers import build_collection, synthetic_cctag

from cctagpy.ellipse_growing import (
    add_candidate_flow_to_cctag,
    ellipse_growing2,
    ellipse_growing_init,
)
from cctagpy.params import Parameters
from cctagpy.vote import children_of, edge_linking, outlier_removal, vote


def _best_seed_and_children(collection, dx, dy, params):
    seeds = vote(collection, dx, dy, params)
    assert len(seeds) > 0
    seed = max(seeds, key=lambda i: collection.is_max[i])
    segment = edge_linking(collection, seed, params.window_size_on_inner_elliptic_segment, params.average_vote_min)
    children = children_of(collection, segment)
    assert len(children) >= 5
    rng = np.random.default_rng(0)
    filtered, sm_final = outlier_removal(
        collection, children, threshold=params.thresh_robust_estimation_of_outer_ellipse,
        weighted_type=1, max_size=60, rng=rng,
    )
    assert len(filtered) >= 5
    return filtered


def test_ellipse_growing_recovers_outer_boundary():
    img, center = synthetic_cctag()
    collection, dx, dy = build_collection(img)
    params = Parameters(n_crowns=3)

    filtered = _best_seed_and_children(collection, dx, dy, params)
    ellipse, good_init = ellipse_growing_init(collection, filtered)

    final_ellipse, outer_points = ellipse_growing2(
        collection, filtered, ellipse, params.ellipse_growing_elliptic_hull_width, good_init
    )

    assert len(outer_points) >= len(filtered)
    assert np.allclose(final_ellipse.center, center, atol=3.0)
    # growing is designed to expand outward, ring by ring, all the way to the
    # true outer boundary of the marker (r=90 in this synthetic target)
    assert 85 < final_ellipse.a < 95
    assert 85 < final_ellipse.b < 95


def test_add_candidate_flow_to_cctag_builds_ring_buckets():
    img, center = synthetic_cctag()
    collection, dx, dy = build_collection(img)
    params = Parameters(n_crowns=3)

    filtered = _best_seed_and_children(collection, dx, dy, params)
    ellipse, good_init = ellipse_growing_init(collection, filtered)
    final_ellipse, outer_points = ellipse_growing2(
        collection, filtered, ellipse, params.ellipse_growing_elliptic_hull_width, good_init
    )

    num_circles = params.n_crowns * 2
    ok, cctag_points = add_candidate_flow_to_cctag(collection, filtered, outer_points, final_ellipse, num_circles)

    assert ok
    assert len(cctag_points) == num_circles
    assert len(cctag_points[-1]) == len(outer_points)
    assert sum(len(bucket) for bucket in cctag_points) > 0
