"""Final marker result class, ported from ``src/cctag/CCTag.{hpp,cpp}`` and
``src/cctag/ICCTag.hpp``.

Only the geometric-detection fields are populated by ``detection.py``
(Phase 4); ``id``/``id_set``/``radius_ratios``/``ellipses``/``homography``/
``quality`` (identification-derived) and the final ``status`` are set by
``identification.py`` (Phase 5) -- until then a tag's ``status`` stays at
its post-construction default (0, "not yet processed").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cctagpy.geometry import Ellipse

MarkerID = int
UNDEFINED_MARKER_ID: MarkerID = -1

# Default placeholder radius ratios (5 values), copied from
# CCTag::_radiusRatiosInit = {29/9, 29/13, 29/17, 29/21, 29/25}.
RADIUS_RATIOS_INIT = [29.0 / 9, 29.0 / 13, 29.0 / 17, 29.0 / 21, 29.0 / 25]


class status:
    """Mirrors the ``cctag::status`` namespace. Only ``id_reliable`` (1) is
    "valid" -- everything else should be treated as rejected/unreliable."""

    id_reliable = 1
    too_few_outer_points = -1
    no_collected_cuts = -1
    no_selected_cuts = -2
    opti_has_diverged = -3
    id_not_reliable = -4
    degenerate = -5


@dataclass
class CCTag:
    id: MarkerID
    center_img: np.ndarray  # (x, y)
    points: list[list[tuple[float, float, float, float]]]  # per-ring (x,y,dx,dy) buckets
    outer_ellipse: Ellipse
    homography: np.ndarray
    pyramid_level: int
    scale: float
    quality: float = 1.0

    rescaled_outer_ellipse: Ellipse = field(init=False)
    rescaled_outer_ellipse_points: list[tuple[float, float, float, float]] = field(default_factory=list)
    radius_ratios: list[float] = field(default_factory=lambda: list(RADIUS_RATIOS_INIT))
    n_circles: int = field(init=False)
    id_set: list[tuple[MarkerID, float]] = field(default_factory=list)
    ellipses: list[Ellipse] = field(default_factory=list)
    status: int = 0

    def __post_init__(self) -> None:
        self.n_circles = len(RADIUS_RATIOS_INIT) + 1
        # "+0.5" offset, verbatim from the C++ constructor (marked
        # "todo@Lilian" there too -- kept for parity).
        self.outer_ellipse = Ellipse(
            center=self.outer_ellipse.center + 0.5,
            a=self.outer_ellipse.a,
            b=self.outer_ellipse.b,
            angle=self.outer_ellipse.angle,
        )
        self.rescaled_outer_ellipse = self.outer_ellipse.scale(self.scale)

    def x(self) -> float:
        return float(self.center_img[0])

    def y(self) -> float:
        return float(self.center_img[1])

    def is_equal(self, other: "CCTag") -> bool:
        """Mirrors ``CCTag::isEqual``: shrink each ellipse's semi-major axis
        to half its semi-minor axis, then check if either shrunk ellipse's
        center lies inside the other."""
        from cctagpy.ellipse_growing import is_overlapping_ellipses

        e_self = self.rescaled_outer_ellipse
        e_other = other.rescaled_outer_ellipse
        center_a = Ellipse(center=e_self.center, a=e_self.b * 0.5, b=e_self.b, angle=e_self.angle)
        center_b = Ellipse(center=e_other.center, a=e_other.b * 0.5, b=e_other.b, angle=e_other.angle)
        return is_overlapping_ellipses(center_a, center_b)

    def __lt__(self, other: "CCTag") -> bool:
        return self.id < other.id
