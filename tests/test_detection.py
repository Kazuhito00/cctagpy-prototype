import numpy as np
from helpers import synthetic_cctag

from cctagpy.detection import cctag_detection
from cctagpy.params import Parameters


def test_cctag_detection_geometric_stage_finds_marker_near_true_center():
    """With identification disabled, this exercises pure Phase 4 geometric
    detection (pyramid -> vote -> ellipse growing -> candidate assembly)."""
    img, center = synthetic_cctag(size=320)
    params = Parameters(n_crowns=3)
    params.do_identification = False
    rng = np.random.default_rng(0)

    markers = cctag_detection(img, params, rng=rng)

    assert len(markers) > 0
    best = min(markers, key=lambda m: np.hypot(m.x() - center[0], m.y() - center[1]))
    assert np.hypot(best.x() - center[0], best.y() - center[1]) < 5.0
    assert 80 < best.outer_ellipse.a < 100
    assert 80 < best.outer_ellipse.b < 100
    # identification disabled: id == -1, status == 0 (untouched)
    assert all(m.id == -1 for m in markers)
    assert all(m.status == 0 for m in markers)
