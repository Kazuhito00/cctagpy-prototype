"""Conic transforms, ported from ``src/cctag/geometry/2DTransform.cpp``.

Also used as ``Ellipse.transform`` in ``geometry.py`` (same math, exposed here
as a free function operating directly on 3x3 conic matrices to avoid a
circular import).
"""

from __future__ import annotations

import numpy as np


def projective_transform_conic(m_transform: np.ndarray, conic: np.ndarray) -> np.ndarray:
    """Apply a general 3x3 projective transform to a conic matrix.

    Mirrors ``viewGeometry::projectiveTransform`` / ``Ellipse::transform``:
    ``M' = T^T @ M @ T``.
    """
    return m_transform.T @ conic @ m_transform
