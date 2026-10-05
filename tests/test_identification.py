import numpy as np
from helpers import synthetic_cctag

from cctagpy.cctag import status as cctag_status
from cctagpy.detection import cctag_detection
from cctagpy.params import Parameters


def test_cctag_detection_with_identification_runs_end_to_end():
    img, center = synthetic_cctag(size=320)
    params = Parameters(n_crowns=3)
    rng = np.random.default_rng(0)

    markers = cctag_detection(img, params, rng=rng)

    assert len(markers) > 0
    valid_statuses = {
        cctag_status.id_reliable,
        cctag_status.too_few_outer_points,
        cctag_status.no_collected_cuts,
        cctag_status.no_selected_cuts,
        cctag_status.opti_has_diverged,
        cctag_status.id_not_reliable,
        cctag_status.degenerate,
    }
    for m in markers:
        assert m.status in valid_statuses
        if m.status == cctag_status.id_reliable:
            assert 0 <= m.id < 32
            assert np.isfinite(m.homography).all()
            det = np.linalg.det(m.homography)
            assert abs(det) > 1e-9
