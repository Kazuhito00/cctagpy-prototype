"""Point-to-conic distance, ported from ``src/cctag/geometry/Distance.cpp``.

Note this is *not* the true orthogonal geometric distance: it is the
first-order (Sampson-like) approximation used throughout the C++ pipeline --
squared algebraic residual divided by the squared local gradient norm of the
conic. Kept as an approximation here for parity with the reference.
"""

from __future__ import annotations

import numpy as np


def distance_point_ellipse(points: np.ndarray, conic: np.ndarray) -> np.ndarray:
    """Sampson-approximate distance from each 2D point to the conic ``Q``.

    ``points`` is an (N, 2) array; ``conic`` is the 3x3 symmetric conic
    matrix (``Ellipse.matrix``). Returns an (N,) array of squared-residual
    distances, mirroring ``distancePointEllipseScalar``/``distancePointEllipse``.
    """
    pts = np.atleast_2d(points)
    x = pts[:, 0]
    y = pts[:, 1]

    q00, q01, q02 = conic[0, 0], conic[0, 1], conic[0, 2]
    q11, q12, q22 = conic[1, 1], conic[1, 2], conic[2, 2]

    # algebraic residual F(p) = [x^2, 2xy, 2x, y^2, 2y, 1] . [Q00,Q01,Q02,Q11,Q12,Q22]
    residual = (
        x * x * q00
        + 2 * x * y * q01
        + 2 * x * q02
        + y * y * q11
        + 2 * y * q12
        + q22
    )

    # (grad F / 2) first two components: row0 and row1 of Q @ [x, y, 1]
    tmp1 = q00 * x + q01 * y + q02
    tmp2 = q01 * x + q11 * y + q12
    denom = tmp1 * tmp1 + tmp2 * tmp2

    return (residual * residual) / denom


def distance_point_ellipse_batched(points: np.ndarray, conics: np.ndarray) -> np.ndarray:
    """:func:`distance_point_ellipse` from a *fixed* point set to each of
    ``K`` candidate conics, computed in one vectorized call.

    ``points`` is (N, 2); ``conics`` is (K, 3, 3). Returns (K, N) distances.
    Pure elementwise broadcast of the same formula as the scalar function
    (no matrix products/LAPACK involved) -- bit-identical to calling
    :func:`distance_point_ellipse` once per conic, just done as one batched
    computation instead of a Python loop over the candidates. Used by
    RANSAC-style hot loops (e.g. ``vote.outlier_removal``,
    ``detection.is_another_segment``) that score many candidate ellipses
    against the same point set per call.
    """
    pts = np.atleast_2d(points)
    x = pts[:, 0][None, :]  # (1, N)
    y = pts[:, 1][None, :]

    q00 = conics[:, 0, 0][:, None]  # (K, 1)
    q01 = conics[:, 0, 1][:, None]
    q02 = conics[:, 0, 2][:, None]
    q11 = conics[:, 1, 1][:, None]
    q12 = conics[:, 1, 2][:, None]
    q22 = conics[:, 2, 2][:, None]

    residual = x * x * q00 + 2 * x * y * q01 + 2 * x * q02 + y * y * q11 + 2 * y * q12 + q22
    tmp1 = q00 * x + q01 * y + q02
    tmp2 = q01 * x + q11 * y + q12
    denom = tmp1 * tmp1 + tmp2 * tmp2

    return (residual * residual) / denom  # (K, N)


def distance_point_ellipse_batched_median(points: np.ndarray, conics: np.ndarray) -> np.ndarray:
    """Median of :func:`distance_point_ellipse_batched` along the point
    axis. Returns (K,)."""
    return np.median(distance_point_ellipse_batched(points, conics), axis=1)
