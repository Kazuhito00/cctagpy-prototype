import numpy as np
from helpers import synthetic_cctag

from cctagpy.bresenham import gradient_direction_descent, gradient_direction_descent_batch
from cctagpy.canny import recoded_canny
from cctagpy.edge_collection import EdgePointCollection
from cctagpy.params import Parameters
from cctagpy.thinning import thin


def test_batched_descent_matches_scalar_on_every_edge_point():
    """`gradient_direction_descent_batch` is a vectorized reformulation used
    for `vote()`'s phase-1 before/after link construction. Pin it against
    the scalar reference (which is otherwise dead code) so a future edit to
    either implementation is caught here instead of surfacing only as a
    tolerance-masked mismatch in end-to-end results.
    """
    img, _ = synthetic_cctag()
    edges, dx, dy = recoded_canny(img, low_thresh=0.01 * 256, high_thresh=0.04 * 256)
    thinned = thin(edges)
    h, w = img.shape
    collection = EdgePointCollection(w, h)
    collection.build_from_edges(thinned, dx, dy)

    n = collection.point_count()
    assert n > 0
    px = collection.positions[:, 0].astype(np.int64)
    py = collection.positions[:, 1].astype(np.int64)
    params = Parameters(n_crowns=3)

    for direction in (-1, 1):
        scalar = np.array(
            [
                gradient_direction_descent(collection, int(x), int(y), direction, params.dist_search, dx, dy)
                for x, y in zip(px, py)
            ],
            dtype=np.int64,
        )
        batch = gradient_direction_descent_batch(collection, px, py, direction, params.dist_search, dx, dy)
        assert np.array_equal(scalar, batch)


def test_batched_descent_handles_empty_input():
    img, _ = synthetic_cctag()
    edges, dx, dy = recoded_canny(img, low_thresh=0.01 * 256, high_thresh=0.04 * 256)
    thinned = thin(edges)
    h, w = img.shape
    collection = EdgePointCollection(w, h)
    collection.build_from_edges(thinned, dx, dy)
    params = Parameters(n_crowns=3)

    result = gradient_direction_descent_batch(
        collection, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), 1, params.dist_search, dx, dy
    )
    assert result.shape == (0,)
