"""Point/ellipse conditioning, ported from ``src/cctag/optimization/conditioner.hpp``.

Only ``conditioner_from_ellipse`` is exercised by the active pipeline
(``Identification.cpp::getNearbyPoints``); ``conditionerFromImage`` in the
C++ source is confirmed dead code and is not ported.
"""

from __future__ import annotations

import numpy as np


def conditioner_from_ellipse(center: np.ndarray, a: float, b: float) -> np.ndarray:
    """Isotropic similarity conditioner derived from an ellipse's mean radius.

    Mirrors ``optimization::conditionerFromEllipse``: translate by ``-center``
    then scale isotropically by ``sqrt(2) / mean(a, b)``.
    """
    mean_ab = (a + b) / 2.0
    s = np.sqrt(2.0) / mean_ab
    cx, cy = center[0], center[1]
    return np.array(
        [
            [s, 0.0, -s * cx],
            [0.0, s, -s * cy],
            [0.0, 0.0, 1.0],
        ]
    )


def condition_point(point: np.ndarray, m_transform: np.ndarray) -> np.ndarray:
    """Apply a homogeneous 3x3 transform to a single 2D point."""
    v = m_transform @ np.array([point[0], point[1], 1.0])
    return v[:2] / v[2]


def condition_points(points: np.ndarray, m_transform: np.ndarray) -> np.ndarray:
    """Apply a homogeneous 3x3 transform to an (N, 2) array of points."""
    pts = np.atleast_2d(points)
    homogeneous = np.column_stack([pts, np.ones(len(pts))])
    transformed = homogeneous @ m_transform.T
    return transformed[:, :2] / transformed[:, 2:3]
