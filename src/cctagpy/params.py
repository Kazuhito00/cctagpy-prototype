"""Detection parameters, ported from ``src/cctag/Params.hpp``/``Params.cpp``.

Every default below is transcribed directly from the ``kDefault*`` constants
in ``Params.hpp``. Fields that are CUDA-only in the C++ library
(``_useCuda``, ``_pinnedCounters``, ``_pinnedNearbyPoints``) are omitted:
this port is CPU-only, so they never take effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: weighting scheme constants (see ``INV_GRAD_WEIGHT`` etc. in Params.hpp)
NO_WEIGHT = 0
INV_GRAD_WEIGHT = 1
INV_SQRT_GRAD_WEIGHT = 2
INV_SQUARE_GRAD_WEIGHT = 3


@dataclass
class Parameters:
    """Mirrors ``cctag::Parameters``. Construct with ``Parameters(n_crowns=3)``."""

    n_crowns: int = 3

    canny_thr_low: float = 0.01
    canny_thr_high: float = 0.04
    dist_search: int = 30
    thr_gradient_mag_in_vote: int = 2500
    angle_voting: float = 0.0
    ratio_voting: float = 4.0
    average_vote_min: float = 0.0
    thr_median_distance_ellipse: float = 3.0
    maximum_nb_seeds: int = 500
    maximum_nb_candidates_loop_two: int = 40
    min_points_segment_candidate: int = 10
    min_votes_to_select_candidate: int = 3
    thresh_robust_estimation_of_outer_ellipse: float = 30.0
    ellipse_growing_elliptic_hull_width: float = 2.3
    window_size_on_inner_elliptic_segment: int = 20
    number_of_multires_layers: int = 4
    number_of_processed_multires_layers: int = 4
    n_samples_outer_ellipse: int = 150
    num_cuts_in_ident_step: int = 22
    num_samples_outer_edge_points_refinement: int = 20
    cuts_selection_trials: int = 500
    sample_cut_length: int = 100
    imaged_center_n_grid_sample: int = 5
    imaged_center_neighbour_size: float = 0.20
    min_ident_proba: float = 1e-6
    use_lm_dif: bool = True
    search_for_another_segment: bool = True
    write_output: bool = False
    do_identification: bool = True
    max_edges: int = 20000
    debug_dir: str = ""

    #: derived, always ``2 * n_crowns`` (Params.cpp sets this in the ctor)
    n_circles: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_circles = 2 * self.n_crowns
