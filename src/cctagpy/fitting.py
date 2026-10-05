"""Ellipse/circle fitting, ported from ``src/cctag/Fitting.cpp``.

Despite its name, ``src/cctag/geometry/EllipseFromPoints.cpp`` in the C++
source does *not* implement the fit -- it only holds rasterization helpers.
The actual fit (Halir & Flusser's "Numerically Stable Direct Least Squares
Fitting of Ellipses") lives in ``Fitting.cpp``, which is what this module
mirrors: :func:`fit_solver` + :func:`to_ellipse` (-> :func:`ellipse_fitting`),
:func:`circle_fitting`, and :func:`inner_prod_min`.
"""

from __future__ import annotations

import numpy as np

from cctagpy._numba_utils import HAS_NUMBA, njit
from cctagpy.geometry import Ellipse

_EPS = np.finfo(float).eps
_EPS_F = float(_EPS)


def _fit_solver_numpy(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if len(pts) < 5:
        raise ValueError("fit_solver: at least 5 points are needed to estimate an ellipse")

    offset = pts.mean(axis=0)
    pc = pts - offset
    x, y = pc[:, 0], pc[:, 1]

    d1 = np.column_stack([x * x, x * y, y * y])
    d2 = np.column_stack([x, y, np.ones_like(x)])

    s1 = d1.T @ d1
    s2 = d1.T @ d2
    s3 = d2.T @ d2

    c1_inv = np.array(
        [
            [0.0, 0.0, 0.5],
            [0.0, -1.0, 0.0],
            [0.5, 0.0, 0.0],
        ]
    )

    try:
        s3_inv = np.linalg.inv(s3)
    except np.linalg.LinAlgError as exc:
        raise ValueError("fit_solver: points appear to be linearly dependent") from exc

    t = -s3_inv @ s2.T
    m = c1_inv @ (s1 + s2 @ t)

    eigvals, eigvecs = np.linalg.eig(m)
    eigvecs = eigvecs.real

    best_idx = None
    best_cond = None
    for j in range(3):
        a1, b1, c1 = eigvecs[0, j], eigvecs[1, j], eigvecs[2, j]
        cond = 4 * a1 * c1 - b1 * b1
        if cond > _EPS and (best_cond is None or cond < best_cond):
            best_cond = cond
            best_idx = j

    if best_idx is None:
        raise ValueError("fit_solver: degeneracy (no valid ellipse root)")

    a1 = eigvecs[:, best_idx]
    a2 = t @ a1
    coeffs = np.concatenate([a1, a2])
    return coeffs, offset


if HAS_NUMBA:

    @njit(cache=True)
    def _fit_solver_numba_core(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        n = pts.shape[0]
        offset = np.zeros(2)
        for i in range(n):
            offset[0] += pts[i, 0]
            offset[1] += pts[i, 1]
        offset[0] /= n
        offset[1] /= n

        d1 = np.empty((n, 3))
        d2 = np.empty((n, 3))
        for i in range(n):
            x = pts[i, 0] - offset[0]
            y = pts[i, 1] - offset[1]
            d1[i, 0] = x * x
            d1[i, 1] = x * y
            d1[i, 2] = y * y
            d2[i, 0] = x
            d2[i, 1] = y
            d2[i, 2] = 1.0

        s1 = d1.T @ d1
        s2 = d1.T @ d2
        s3 = d2.T @ d2

        c1_inv = np.array([[0.0, 0.0, 0.5], [0.0, -1.0, 0.0], [0.5, 0.0, 0.0]])

        det_s3 = np.linalg.det(s3)
        if det_s3 == 0.0:
            return np.zeros(6), offset, False

        s3_inv = np.linalg.inv(s3)
        t = -s3_inv @ s2.T
        m = c1_inv @ (s1 + s2 @ t)

        eigvals, eigvecs = np.linalg.eig(m)
        eigvecs_real = np.real(eigvecs)

        best_idx = -1
        best_cond = 0.0
        for j in range(3):
            a1 = eigvecs_real[0, j]
            b1 = eigvecs_real[1, j]
            c1 = eigvecs_real[2, j]
            cond = 4 * a1 * c1 - b1 * b1
            if cond > _EPS_F and (best_idx == -1 or cond < best_cond):
                best_cond = cond
                best_idx = j

        if best_idx == -1:
            return np.zeros(6), offset, False

        a1vec = np.ascontiguousarray(eigvecs_real[:, best_idx])
        a2vec = t @ a1vec
        coeffs = np.empty(6)
        coeffs[0:3] = a1vec
        coeffs[3:6] = a2vec
        return coeffs, offset, True


def _fit_solver_numba(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if len(pts) < 5:
        raise ValueError("fit_solver: at least 5 points are needed to estimate an ellipse")
    coeffs, offset, ok = _fit_solver_numba_core(pts)
    if not ok:
        raise ValueError("fit_solver: degeneracy (linearly dependent points or no valid ellipse root)")
    return coeffs, offset


def fit_solver(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Halir-Flusser direct least-squares conic fit.

    Returns ``(coeffs, offset)`` where ``coeffs = (a, b, c, d, e, f)`` are the
    conic coefficients ``a*x^2 + b*x*y + c*y^2 + d*x + e*y + f = 0`` in the
    *centered* frame (``offset`` = centroid of ``points``, to be added back
    to the ellipse center by :func:`to_ellipse`). Dispatches to a fused Numba
    kernel when available, falling back to the vectorized NumPy form
    otherwise -- verified end-to-end result-identical on real RANSAC point
    sets either way (see module tests)."""
    if HAS_NUMBA:
        return _fit_solver_numba(points)
    return _fit_solver_numpy(points)


def _to_ellipse_numpy(coeffs: np.ndarray, offset: np.ndarray) -> Ellipse:
    a, b, c, d, e, f = coeffs

    idet = a * c - b * b / 4.0
    if idet <= _EPS:
        raise ValueError("to_ellipse: singularity 1 (idet <= eps)")
    scale = np.sqrt(idet / 4.0)
    if scale <= _EPS:
        raise ValueError("to_ellipse: singularity 1 (scale <= eps)")

    a, b, c, d, e, f = a * scale, b * scale, c * scale, d * scale, e * scale, f * scale

    center_mat = np.array([[2 * a, b], [b, 2 * c]])
    try:
        x0, y0 = np.linalg.solve(center_mat, np.array([-d, -e]))
    except np.linalg.LinAlgError as exc:
        raise ValueError("to_ellipse: singularity 1 (singular center system)") from exc

    f0 = a * x0 * x0 + b * x0 * y0 + c * y0 * y0 + d * x0 + e * y0 + f
    if abs(f0) <= _EPS:
        raise ValueError("to_ellipse: singularity 2 (|f0| <= eps)")

    s_mat = np.array([[a, b / 2.0], [b / 2.0, c]]) / (-f0)
    u, sv, _ = np.linalg.svd(s_mat)
    if sv[0] <= 0 or sv[1] <= 0:
        raise ValueError("to_ellipse: degenerate ellipse => line or point")

    radius0 = np.sqrt(1.0 / sv[0])
    radius1 = np.sqrt(1.0 / sv[1])
    angle = np.pi - np.arctan2(u[0, 1], u[1, 1])
    center = np.array([x0, y0]) + offset

    if radius0 <= 0 or radius1 <= 0:
        raise ValueError("to_ellipse: degenerate ellipse => line or point")

    return Ellipse(center=center, a=float(radius0), b=float(radius1), angle=float(angle))


if HAS_NUMBA:

    @njit(cache=True)
    def _to_ellipse_numba_core(coeffs: np.ndarray, offset: np.ndarray) -> tuple[np.ndarray, bool]:
        a, b, c, d, e, f = coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5]

        idet = a * c - b * b / 4.0
        if idet <= _EPS_F:
            return np.zeros(5), False
        scale = np.sqrt(idet / 4.0)
        if scale <= _EPS_F:
            return np.zeros(5), False

        a, b, c, d, e, f = a * scale, b * scale, c * scale, d * scale, e * scale, f * scale

        cm00, cm01, cm10, cm11 = 2 * a, b, b, 2 * c
        det_cm = cm00 * cm11 - cm01 * cm10
        if det_cm == 0.0:
            return np.zeros(5), False
        rhs0, rhs1 = -d, -e
        x0 = (cm11 * rhs0 - cm01 * rhs1) / det_cm
        y0 = (cm00 * rhs1 - cm10 * rhs0) / det_cm

        f0 = a * x0 * x0 + b * x0 * y0 + c * y0 * y0 + d * x0 + e * y0 + f
        if abs(f0) <= _EPS_F:
            return np.zeros(5), False

        neg_f0 = -f0
        s_mat = np.array([[a / neg_f0, (b / 2.0) / neg_f0], [(b / 2.0) / neg_f0, c / neg_f0]])
        u, sv, _ = np.linalg.svd(s_mat)
        if sv[0] <= 0 or sv[1] <= 0:
            return np.zeros(5), False

        radius0 = np.sqrt(1.0 / sv[0])
        radius1 = np.sqrt(1.0 / sv[1])
        angle = np.pi - np.arctan2(u[0, 1], u[1, 1])
        cx = x0 + offset[0]
        cy = y0 + offset[1]

        if radius0 <= 0 or radius1 <= 0:
            return np.zeros(5), False

        out = np.empty(5)
        out[0] = cx
        out[1] = cy
        out[2] = radius0
        out[3] = radius1
        out[4] = angle
        return out, True


def _to_ellipse_numba(coeffs: np.ndarray, offset: np.ndarray) -> Ellipse:
    out, ok = _to_ellipse_numba_core(np.asarray(coeffs, dtype=np.float64), np.asarray(offset, dtype=np.float64))
    if not ok:
        raise ValueError("to_ellipse: singularity or degenerate ellipse")
    cx, cy, radius0, radius1, angle = out
    return Ellipse(center=np.array([cx, cy]), a=float(radius0), b=float(radius1), angle=float(angle))


def to_ellipse(coeffs: np.ndarray, offset: np.ndarray) -> Ellipse:
    """Conic coefficients (centered frame) + offset -> :class:`Ellipse`.

    Standard conic-to-ellipse conversion: solve for the center via the
    conic's stationary point, evaluate the constant term there, then
    eigendecompose the resulting 2x2 quadratic-form matrix for radii/angle.
    Dispatches to a Numba kernel when available (the 2x2 center-system solve
    is done via Cramer's rule instead of ``np.linalg.solve`` to avoid
    LAPACK's gufunc dispatch overhead on such a tiny system), falling back
    to the NumPy form otherwise -- verified end-to-end result-identical on
    real RANSAC point sets either way (see module tests)."""
    if HAS_NUMBA:
        return _to_ellipse_numba(coeffs, offset)
    return _to_ellipse_numpy(coeffs, offset)


def ellipse_fitting(points: np.ndarray) -> Ellipse:
    """Fit an ellipse to a set of (>=5) 2D points. Mirrors ``ellipseFitting``."""
    coeffs, offset = fit_solver(points)
    return to_ellipse(coeffs, offset)


def _batched_ellipse_fitting_numpy(points_batch: np.ndarray) -> list[Ellipse | None]:
    """Vectorized :func:`ellipse_fitting` over ``K`` independent point sets
    of identical size, for RANSAC-style hot loops (e.g.
    ``detection.is_another_segment``) that fit many candidate ellipses per
    call. Each entry of the returned list is ``None`` exactly where the
    scalar :func:`ellipse_fitting` would have raised for that point set
    (linear dependency, degeneracy, or no valid conic root).

    Every ``np.linalg`` primitive used here (``det``, ``inv``, ``eig``,
    ``solve``, ``svd``) is a gufunc that loops the same per-matrix LAPACK
    kernel when given a stacked ``(K, ...)`` input -- verified empirically
    to be bit-for-bit identical to calling it once per matrix, not merely
    an equivalent reformulation -- so this is not an approximation of
    :func:`fit_solver`/:func:`to_ellipse`, just the same arithmetic done in
    one batched call instead of ``K`` separate ones. Degeneracy checks that
    raise ``ValueError`` in the scalar version become boolean masks here so
    a bad trial only drops out of the batch instead of aborting it.
    """
    pts = np.asarray(points_batch, dtype=np.float64)
    k_trials, n_pts, _ = pts.shape

    offset = pts.mean(axis=1)  # (K, 2)
    pc = pts - offset[:, None, :]
    x, y = pc[:, :, 0], pc[:, :, 1]

    d1 = np.stack([x * x, x * y, y * y], axis=-1)  # (K, N, 3)
    d2 = np.stack([x, y, np.ones_like(x)], axis=-1)  # (K, N, 3)

    # `@` (batched matmul) rather than `einsum`: both compute the same sum
    # but a differently-ordered summation can round differently in the last
    # bit for tiny N -- verified empirically that batched `@` matches a
    # per-trial `d1[k].T @ d1[k]` loop bit-for-bit, whereas `einsum` does not.
    d1t = np.transpose(d1, (0, 2, 1))
    d2t = np.transpose(d2, (0, 2, 1))
    s1 = d1t @ d1
    s2 = d1t @ d2
    s3 = d2t @ d2

    c1_inv = np.array([[0.0, 0.0, 0.5], [0.0, -1.0, 0.0], [0.5, 0.0, 0.0]])

    det_s3 = np.linalg.det(s3)
    valid = det_s3 != 0

    t = np.zeros((k_trials, 3, 3))
    m = np.zeros((k_trials, 3, 3))
    if valid.any():
        s3_inv = np.linalg.inv(s3[valid])
        t[valid] = -s3_inv @ np.transpose(s2[valid], (0, 2, 1))
        m[valid] = c1_inv[None, :, :] @ (s1[valid] + s2[valid] @ t[valid])

    eigvecs_real = np.zeros((k_trials, 3, 3))
    if valid.any():
        _, eigvecs = np.linalg.eig(m[valid])
        eigvecs_real[valid] = eigvecs.real

    a1v, b1v, c1v = eigvecs_real[:, 0, :], eigvecs_real[:, 1, :], eigvecs_real[:, 2, :]
    cond = 4 * a1v * c1v - b1v * b1v  # (K, 3)
    cond_ok = cond > _EPS
    best_idx = np.argmin(np.where(cond_ok, cond, np.inf), axis=1)
    valid = valid & cond_ok.any(axis=1)

    a1 = eigvecs_real[np.arange(k_trials), :, best_idx]  # (K, 3)
    a2 = (t @ a1[:, :, None])[:, :, 0]
    coeffs = np.concatenate([a1, a2], axis=1)  # (K, 6)

    return _batched_to_ellipse(coeffs, offset, valid)


def batched_ellipse_fitting(points_batch: np.ndarray) -> list[Ellipse | None]:
    """Vectorized :func:`ellipse_fitting` over ``K`` independent point sets,
    for RANSAC-style hot loops (see :func:`_batched_ellipse_fitting_numpy`).

    Deliberately NumPy-only, unlike the other fitting functions in this
    module: a Numba version that loops the scalar :func:`fit_solver`/
    :func:`to_ellipse` kernels over the batch was implemented and measured
    -- it was ~15% *slower* than this vectorized form on realistic batch
    sizes (K=100, N=8), because NumPy's gufunc dispatch already runs the
    per-matrix LAPACK kernel across the whole stacked batch in one call,
    which beats invoking LAPACK K separate times from inside a compiled
    loop. Not every hot loop benefits from Numba -- this one was measured
    and rejected rather than assumed."""
    return _batched_ellipse_fitting_numpy(points_batch)


def _batched_to_ellipse(coeffs: np.ndarray, offset: np.ndarray, valid: np.ndarray) -> list[Ellipse | None]:
    """Batched counterpart of :func:`to_ellipse`; see :func:`batched_ellipse_fitting`."""
    k_trials = coeffs.shape[0]
    a, b, c, d, e, f = (coeffs[:, i] for i in range(6))

    idet = a * c - b * b / 4.0
    valid = valid & (idet > _EPS)
    scale = np.sqrt(np.where(idet > 0, idet, 0.0) / 4.0)
    valid = valid & (scale > _EPS)

    a2, b2, c2 = a * scale, b * scale, c * scale
    d2, e2, f2 = d * scale, e * scale, f * scale

    center_mat = np.zeros((k_trials, 2, 2))
    center_mat[:, 0, 0] = 2 * a2
    center_mat[:, 0, 1] = b2
    center_mat[:, 1, 0] = b2
    center_mat[:, 1, 1] = 2 * c2
    rhs = np.stack([-d2, -e2], axis=1)

    solve_ok = valid & (np.linalg.det(center_mat) != 0)
    x0 = np.zeros(k_trials)
    y0 = np.zeros(k_trials)
    if solve_ok.any():
        sol = np.linalg.solve(center_mat[solve_ok], rhs[solve_ok][..., None])[..., 0]
        x0[solve_ok] = sol[:, 0]
        y0[solve_ok] = sol[:, 1]
    valid = solve_ok

    f0 = a2 * x0 * x0 + b2 * x0 * y0 + c2 * y0 * y0 + d2 * x0 + e2 * y0 + f2
    valid = valid & (np.abs(f0) > _EPS)
    neg_f0 = np.where(valid, -f0, 1.0)

    s_mat = np.zeros((k_trials, 2, 2))
    s_mat[:, 0, 0] = a2 / neg_f0
    s_mat[:, 0, 1] = (b2 / 2.0) / neg_f0
    s_mat[:, 1, 0] = (b2 / 2.0) / neg_f0
    s_mat[:, 1, 1] = c2 / neg_f0

    u = np.zeros((k_trials, 2, 2))
    sv = np.zeros((k_trials, 2))
    if valid.any():
        u_valid, sv_valid, _ = np.linalg.svd(s_mat[valid])
        u[valid] = u_valid
        sv[valid] = sv_valid
    valid = valid & (sv[:, 0] > 0) & (sv[:, 1] > 0)

    sv0_safe = np.where(sv[:, 0] > 0, sv[:, 0], 1.0)
    sv1_safe = np.where(sv[:, 1] > 0, sv[:, 1], 1.0)
    radius0 = np.sqrt(1.0 / sv0_safe)
    radius1 = np.sqrt(1.0 / sv1_safe)
    angle = np.pi - np.arctan2(u[:, 0, 1], u[:, 1, 1])
    center = np.stack([x0, y0], axis=1) + offset
    valid = valid & (radius0 > 0) & (radius1 > 0)

    ellipses: list[Ellipse | None] = [None] * k_trials
    for k in np.nonzero(valid)[0]:
        ellipses[k] = Ellipse(center=center[k], a=float(radius0[k]), b=float(radius1[k]), angle=float(angle[k]))
    return ellipses


if HAS_NUMBA:

    @njit(cache=True)
    def _circle_fitting_numba_core(pts: np.ndarray) -> tuple[np.ndarray, bool]:
        """``fitting.circle_fitting`` (same SVD-based Kasa fit); params are
        ``(cx, cy, a, b, angle)`` with ``a == b``, ``angle == 0``."""
        n = pts.shape[0]
        a_mat = np.empty((n, 4), dtype=np.float64)
        for i in range(n):
            x = pts[i, 0]
            y = pts[i, 1]
            a_mat[i, 0] = x
            a_mat[i, 1] = y
            a_mat[i, 2] = 1.0
            a_mat[i, 3] = x * x + y * y
        _, _, vt = np.linalg.svd(a_mat, False)
        v0 = vt[3, 0]
        v1 = vt[3, 1]
        v2 = vt[3, 2]
        v3 = vt[3, 3]
        xc = -0.5 * v0 / v3
        yc = -0.5 * v1 / v3
        radius = np.sqrt(xc * xc + yc * yc - v2 / v3)
        out = np.zeros(5)
        if not (radius > 0):
            return out, False
        out[0] = xc
        out[1] = yc
        out[2] = radius
        out[3] = radius
        out[4] = 0.0
        return out, True

    @njit(cache=True)
    def _ellipse_fitting_numba_core(pts: np.ndarray) -> tuple[np.ndarray, bool]:
        """``fitting.ellipse_fitting`` via the same two Numba cores."""
        if pts.shape[0] < 5:
            return np.zeros(5), False
        coeffs, offset, ok = _fit_solver_numba_core(pts)
        if not ok:
            return np.zeros(5), False
        out, ok2 = _to_ellipse_numba_core(coeffs, offset)
        if not ok2:
            return np.zeros(5), False
        return out, True


def circle_fitting(points: np.ndarray) -> Ellipse:
    """Linear algebraic circle fit (Kasa-style, lifted to a homogeneous
    4-parameter system solved via SVD). Mirrors ``circleFitting``."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    x, y = pts[:, 0], pts[:, 1]
    a_mat = np.column_stack([x, y, np.ones_like(x), x * x + y * y])
    _, _, vt = np.linalg.svd(a_mat, full_matrices=False)
    v0, v1, v2, v3 = vt[-1, :]
    xc = -0.5 * v0 / v3
    yc = -0.5 * v1 / v3
    radius = np.sqrt(xc * xc + yc * yc - v2 / v3)
    if radius <= 0:
        raise ValueError("circle_fitting: degenerate (radius <= 0)")
    return Ellipse(center=np.array([xc, yc]), a=float(radius), b=float(radius), angle=0.0)


def _inner_prod_min_python(
    positions: np.ndarray, gradients: np.ndarray, thr_cos_diff_max: float
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """Two-pass gradient-direction-spread check, ported from ``innerProdMin``.

    ``positions``/``gradients`` are (N, 2) arrays (raw, unnormalized
    gradients); index 0 plays the role of ``filteredChildren.front()``.
    Returns ``(min_inner_prod, p1, p2)`` where ``p1``/``p2`` are the points
    farthest from the reference point of each pass (only ``min_inner_prod``
    is actually consulted by callers such as ``ellipseGrowingInit``).
    """
    positions = np.atleast_2d(positions)
    gradients = np.atleast_2d(gradients)
    n = len(positions)

    p0_pos = positions[0]
    g0 = gradients[0]
    norm0 = np.hypot(g0[0], g0[1])
    gx0, gy0 = g0[0] / norm0, g0[1] / norm0

    min_val = 1.1
    dist_max = 0.0
    p_angle1_idx = None
    p1 = None

    for i in range(1, n):
        g = gradients[i]
        norm = np.hypot(g[0], g[1])
        gx, gy = g[0] / norm, g[1] / norm
        inner = gx0 * gx + gy0 * gy
        if inner <= thr_cos_diff_max:
            return float(inner), p1, None
        if inner < min_val:
            min_val = inner
            p_angle1_idx = i
        dist = np.hypot(positions[i][0] - p0_pos[0], positions[i][1] - p0_pos[1])
        if dist > dist_max:
            dist_max = dist
            p1 = positions[i]

    if p_angle1_idx is None:
        return float(min_val), p1, None

    g_min = gradients[p_angle1_idx]
    norm = np.hypot(g_min[0], g_min[1])
    gxmin, gymin = g_min[0] / norm, g_min[1] / norm

    min_val = 1.0
    dist_max = 0.0
    p2 = None
    p1_ref = p1 if p1 is not None else p0_pos

    for i in range(n):
        g = gradients[i]
        norm = np.hypot(g[0], g[1])
        chgx, chgy = g[0] / norm, g[1] / norm
        inner = gxmin * chgx + gymin * chgy
        if inner <= thr_cos_diff_max:
            return float(inner), p1, p2
        if inner < min_val:
            min_val = inner
        dist = np.hypot(positions[i][0] - p1_ref[0], positions[i][1] - p1_ref[1])
        if dist > dist_max:
            dist_max = dist
            p2 = positions[i]

    return float(min_val), p1, p2


if HAS_NUMBA:

    @njit(cache=True)
    def _inner_prod_min_numba(positions: np.ndarray, gradients: np.ndarray, thr_cos_diff_max: float) -> tuple[float, int, int]:
        """:func:`_inner_prod_min_python` statement for statement, returning
        the two farthest points as indices (-1 for ``None``)."""
        n = positions.shape[0]
        p0x = positions[0, 0]
        p0y = positions[0, 1]
        norm0 = np.hypot(gradients[0, 0], gradients[0, 1])
        gx0 = gradients[0, 0] / norm0
        gy0 = gradients[0, 1] / norm0

        min_val = 1.1
        dist_max = 0.0
        p_angle1_idx = -1
        p1 = -1

        for i in range(1, n):
            norm = np.hypot(gradients[i, 0], gradients[i, 1])
            gx = gradients[i, 0] / norm
            gy = gradients[i, 1] / norm
            inner = gx0 * gx + gy0 * gy
            if inner <= thr_cos_diff_max:
                return inner, p1, -1
            if inner < min_val:
                min_val = inner
                p_angle1_idx = i
            dist = np.hypot(positions[i, 0] - p0x, positions[i, 1] - p0y)
            if dist > dist_max:
                dist_max = dist
                p1 = i

        if p_angle1_idx == -1:
            return min_val, p1, -1

        norm = np.hypot(gradients[p_angle1_idx, 0], gradients[p_angle1_idx, 1])
        gxmin = gradients[p_angle1_idx, 0] / norm
        gymin = gradients[p_angle1_idx, 1] / norm

        min_val = 1.0
        dist_max = 0.0
        p2 = -1
        if p1 != -1:
            refx = positions[p1, 0]
            refy = positions[p1, 1]
        else:
            refx = p0x
            refy = p0y

        for i in range(n):
            norm = np.hypot(gradients[i, 0], gradients[i, 1])
            chgx = gradients[i, 0] / norm
            chgy = gradients[i, 1] / norm
            inner = gxmin * chgx + gymin * chgy
            if inner <= thr_cos_diff_max:
                return inner, p1, p2
            if inner < min_val:
                min_val = inner
            dist = np.hypot(positions[i, 0] - refx, positions[i, 1] - refy)
            if dist > dist_max:
                dist_max = dist
                p2 = i

        return min_val, p1, p2


def inner_prod_min(
    positions: np.ndarray, gradients: np.ndarray, thr_cos_diff_max: float
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """Two-pass gradient-direction-spread check, ported from ``innerProdMin``
    (see :func:`_inner_prod_min_python` for the algorithm). Dispatches to a
    Numba kernel when available -- same scalar operations, bit-identical."""
    if HAS_NUMBA:
        pos = np.ascontiguousarray(np.atleast_2d(positions), dtype=np.float64)
        grad = np.ascontiguousarray(np.atleast_2d(gradients), dtype=np.float64)
        min_val, i1, i2 = _inner_prod_min_numba(pos, grad, float(thr_cos_diff_max))
        return float(min_val), (pos[i1] if i1 >= 0 else None), (pos[i2] if i2 >= 0 else None)
    return _inner_prod_min_python(positions, gradients, thr_cos_diff_max)
