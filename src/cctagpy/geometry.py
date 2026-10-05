"""Ellipse/Circle representation, ported from ``src/cctag/geometry/Ellipse.{hpp,cpp}``
and ``Circle.{hpp,cpp}``.

An :class:`Ellipse` keeps two representations in sync: a 3x3 conic matrix
(``[x y 1] . M . [x y 1]^T = 0``) and parametric form (center, semi-axes
``a``/``b``, rotation ``angle`` in radians). The matrix<->parameter
conversions replicate the classic conic-to-ellipse decomposition used by
old OpenCV (``cvFitEllipse``), which the C++ ``computeParameters`` /
``computeMatrix`` are themselves adapted from.
"""

from __future__ import annotations

import numpy as np

from cctagpy._numba_utils import HAS_NUMBA, njit

if HAS_NUMBA:

    @njit(cache=True, inline="always")
    def _ellipse_matrix_numba(cx: float, cy: float, a: float, b: float, angle: float) -> np.ndarray:
        """``compute_matrix``, scalar Numba form (see its dispatch for why)."""
        cost = np.cos(angle)
        sint = np.sin(angle)
        t = np.array([[cost, -sint, cx], [sint, cost, cy], [0.0, 0.0, 1.0]])
        t_inv = np.linalg.inv(t)
        diag = np.diag(np.array([1.0 / a**2, 1.0 / b**2, -1.0]))
        return np.ascontiguousarray(t_inv.T) @ diag @ t_inv


def compute_parameters(matrix: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Conic matrix -> (center, a, b, angle). Mirrors ``Ellipse::computeParameters``."""
    m = matrix
    par0 = m[0, 0]
    par1 = 2 * m[0, 1]
    par2 = m[1, 1]
    par3 = 2 * m[0, 2]
    par4 = 2 * m[1, 2]
    par5 = m[2, 2]

    theta = 0.5 * np.arctan2(par1, par0 - par2)
    cost = np.cos(theta)
    sint = np.sin(theta)

    a0 = par5
    au = par3 * cost + par4 * sint
    av = -par3 * sint + par4 * cost
    auu = par0 * cost**2 + par2 * sint**2 + par1 * cost * sint
    avv = par0 * sint**2 + par2 * cost**2 - par1 * cost * sint

    if auu == 0 or avv == 0:
        # degenerate conic
        return np.zeros(2), 0.0, 0.0, 0.0

    tu_centre = -au / (2 * auu)
    tv_centre = -av / (2 * avv)
    w_centre = a0 - auu * tu_centre**2 - avv * tv_centre**2

    center = np.array(
        [
            tu_centre * cost - tv_centre * sint,
            tu_centre * sint + tv_centre * cost,
        ]
    )

    ru = -w_centre / auu
    rv = -w_centre / avv
    a = np.sqrt(abs(ru)) * np.sign(ru)
    b = np.sqrt(abs(rv)) * np.sign(rv)

    if a < 0 or b < 0:
        raise ValueError("degenerate ellipse: negative semi-axis after matrix decomposition")

    return center, float(a), float(b), float(theta)


def compute_semi_axes_ratio_batched(conics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized counterpart of the ``a``/``b`` (semi-axes) portion of
    :func:`compute_parameters`, for ``K`` independent conic matrices at
    once. Returns ``(ratio, invalid)`` where ``ratio = a / b`` and
    ``invalid[k]`` is true exactly where the scalar path (via
    ``Ellipse(matrix=conics[k])`` then ``ellipse.a / ellipse.b``) would have
    raised: a degenerate conic (``auu == 0 or avv == 0``, giving ``a == b ==
    0.0`` and thus a ``ZeroDivisionError`` on the ratio) or a negative
    semi-axis (the ``ValueError`` ``compute_parameters`` raises itself).
    ``ratio`` is meaningless (not necessarily finite) wherever ``invalid``
    is true -- callers must check it. Pure elementwise arithmetic, no
    matrix decomposition, so this is not an approximation, just the same
    formula evaluated for the whole batch in one call instead of a Python
    loop over conics. Used by RANSAC-style hot loops (e.g.
    ``vote.outlier_removal``) that only need the axis ratio to filter many
    candidate conics per call, not the full ``Ellipse`` (center/angle).
    """
    par0 = conics[:, 0, 0]
    par1 = 2 * conics[:, 0, 1]
    par2 = conics[:, 1, 1]
    par3 = 2 * conics[:, 0, 2]
    par4 = 2 * conics[:, 1, 2]
    par5 = conics[:, 2, 2]

    theta = 0.5 * np.arctan2(par1, par0 - par2)
    cost = np.cos(theta)
    sint = np.sin(theta)

    a0 = par5
    au = par3 * cost + par4 * sint
    av = -par3 * sint + par4 * cost
    auu = par0 * cost**2 + par2 * sint**2 + par1 * cost * sint
    avv = par0 * sint**2 + par2 * cost**2 - par1 * cost * sint

    degenerate = (auu == 0) | (avv == 0)
    auu_safe = np.where(degenerate, 1.0, auu)
    avv_safe = np.where(degenerate, 1.0, avv)

    tu_centre = -au / (2 * auu_safe)
    tv_centre = -av / (2 * avv_safe)
    w_centre = a0 - auu_safe * tu_centre**2 - avv_safe * tv_centre**2

    ru = -w_centre / auu_safe
    rv = -w_centre / avv_safe
    a_arr = np.sqrt(np.abs(ru)) * np.sign(ru)
    b_arr = np.sqrt(np.abs(rv)) * np.sign(rv)

    # `b_arr == 0` (not just `< 0`) must also count as invalid: the scalar
    # path's `ellipse_q.a / ellipse_q.b` divides two plain Python floats,
    # where dividing by exactly zero raises `ZeroDivisionError` regardless
    # of the numerator (unlike NumPy float division, which would silently
    # produce +-inf/nan).
    invalid = degenerate | (a_arr < 0) | (b_arr <= 0)
    b_safe = np.where(invalid, 1.0, b_arr)
    ratio = a_arr / b_safe
    return ratio, invalid


def compute_matrix(center: np.ndarray, a: float, b: float, angle: float) -> np.ndarray:
    """(center, a, b, angle) -> conic matrix. Mirrors ``Ellipse::computeMatrix``.

    Dispatches to a scalar Numba kernel when available: ``Ellipse`` is
    constructed constantly throughout the pipeline (every RANSAC trial,
    every growing step, every hull), and this call's cost is almost
    entirely small-array allocation/dispatch overhead for a 3x3 inverse and
    two matrix products, not the arithmetic itself -- same rationale as
    ``ellipse_growing.is_in_ellipse``/``is_in_hull``. ``t`` is a rigid
    rotation+translation (determinant 1), so it is never singular in
    practice; the ``LinAlgError`` guard below is defensive and only
    exercised by the NumPy fallback."""
    if a < 0 or b < 0:
        raise ValueError("semi-axes must be non-negative")
    if HAS_NUMBA:
        return _ellipse_matrix_numba(float(center[0]), float(center[1]), float(a), float(b), float(angle))
    cost = np.cos(angle)
    sint = np.sin(angle)
    t = np.array(
        [
            [cost, -sint, center[0]],
            [sint, cost, center[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    try:
        t_inv = np.linalg.inv(t)
    except np.linalg.LinAlgError as exc:
        raise ValueError("singular ellipse transform") from exc
    diag = np.diag([1.0 / a**2, 1.0 / b**2, -1.0])
    return t_inv.T @ diag @ t_inv


class Ellipse:
    """Conic-matrix / (center, a, b, angle) dual representation of an ellipse."""

    __slots__ = ("_matrix", "_center", "_a", "_b", "_angle", "_canonic_form")

    def __init__(
        self,
        matrix: np.ndarray | None = None,
        center: np.ndarray | None = None,
        a: float | None = None,
        b: float | None = None,
        angle: float | None = None,
    ) -> None:
        if matrix is not None:
            self.set_matrix(matrix)
        elif center is not None and a is not None and b is not None and angle is not None:
            self.set_parameters(center, a, b, angle)
        else:
            self.set_parameters(np.zeros(2), 0.0, 0.0, 0.0)

    # -- accessors -------------------------------------------------------
    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def center(self) -> np.ndarray:
        return self._center

    @property
    def a(self) -> float:
        return self._a

    @property
    def b(self) -> float:
        return self._b

    @property
    def angle(self) -> float:
        return self._angle

    # -- setters (keep matrix/params in sync, like the C++ setters) ------
    def set_matrix(self, matrix: np.ndarray) -> None:
        self._matrix = np.array(matrix, dtype=float)
        self._center, self._a, self._b, self._angle = compute_parameters(self._matrix)
        self._canonic_form = None

    def set_parameters(self, center: np.ndarray, a: float, b: float, angle: float) -> None:
        if a < 0 or b < 0:
            raise ValueError("semi-axes must be non-negative")
        self._center = np.asarray(center, dtype=float)
        self._a = float(a)
        self._b = float(b)
        self._angle = float(angle)
        self._matrix = compute_matrix(self._center, self._a, self._b, self._angle)
        self._canonic_form = None

    def set_a(self, a: float) -> None:
        self.set_parameters(self._center, a, self._b, self._angle)

    def set_b(self, b: float) -> None:
        self.set_parameters(self._center, self._a, b, self._angle)

    def set_angle(self, angle: float) -> None:
        self.set_parameters(self._center, self._a, self._b, angle)

    def set_center(self, center: np.ndarray) -> None:
        self.set_parameters(center, self._a, self._b, self._angle)

    # -- derived operations ------------------------------------------------
    def transform(self, m_transform: np.ndarray) -> "Ellipse":
        """Conic-conjugation transform: ``M' = T^T @ M @ T``."""
        from cctagpy.transform2d import projective_transform_conic

        return Ellipse(matrix=projective_transform_conic(m_transform, self._matrix))

    def scale(self, s: float) -> "Ellipse":
        return Ellipse(center=self._center * s, a=self._a * s, b=self._b * s, angle=self._angle)

    def get_canonic_form(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (m_canonic, m_t_primal, m_t_dual).

        ``m_t_primal`` maps the centered/axis-aligned canonical frame to the
        image frame (same rigid transform used by :func:`compute_matrix`);
        ``m_t_dual = m_t_primal^-1``; ``m_canonic`` is the conic expressed in
        the canonical frame, i.e. ``matrix == m_t_dual.T @ m_canonic @ m_t_dual``.

        Memoized: this ellipse's identification-stage callers (notably
        ``compute_homography_from_ellipse_and_imaged_center``) recompute it
        for the *same* ellipse dozens to hundreds of times per marker during
        the center-search grid (the ellipse itself never changes across that
        search, only the candidate center does), so caching it here removes
        that redundant work for every caller at once. Invalidated by
        ``set_matrix``/``set_parameters`` (the only two mutators).
        """
        if self._canonic_form is None:
            cost = np.cos(self._angle)
            sint = np.sin(self._angle)
            m_t_primal = np.array(
                [
                    [cost, -sint, self._center[0]],
                    [sint, cost, self._center[1]],
                    [0.0, 0.0, 1.0],
                ]
            )
            m_t_dual = np.linalg.inv(m_t_primal)
            m_canonic = np.diag([1.0 / self._a**2, 1.0 / self._b**2, -1.0])
            self._canonic_form = (m_canonic, m_t_primal, m_t_dual)
        return self._canonic_form

    @staticmethod
    def get_sorted_outer_points(ellipse: "Ellipse", points: np.ndarray, requested_size: int) -> np.ndarray:
        """Sort points by polar angle around the ellipse center, then
        uniformly subsample by index (float stride ``max(1, N/(M-1))``,
        cast to int per selected index -- NOT integer-divided once).
        Returns the selected *indices* into ``points`` so callers can gather
        any associated per-point data (e.g. gradients) consistently.
        Mirrors the free function ``getSortedOuterPoints`` in ``Ellipse.cpp``.
        """
        pts = np.atleast_2d(points)
        n = len(pts)
        cx, cy = ellipse.center
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        order = np.argsort(angles, kind="stable")

        n_outer_points = min(requested_size, n)
        if n_outer_points <= 1:
            step = 1.0
        else:
            step = max(1.0, n / (n_outer_points - 1))

        selected: list[int] = []
        k = 0
        while True:
            i_to_add = int(k * step)
            if i_to_add < n:
                selected.append(order[i_to_add])
                k += 1
            else:
                break
        return np.asarray(selected, dtype=np.int64)


def point_on_ellipse(ellipse: Ellipse, point: np.ndarray) -> np.ndarray:
    """Radially rescale ``point`` onto the ellipse boundary (not a true
    orthogonal projection). Mirrors ``pointOnEllipse``."""
    _, m_t_primal, m_t_dual = ellipse.get_canonic_form()
    p_h = np.array([point[0], point[1], 1.0])
    canonical = m_t_dual @ p_h
    canonical = canonical[:2] / canonical[2]
    x, y = canonical
    denom = np.sqrt((x * x) / (ellipse.a**2) + (y * y) / (ellipse.b**2))
    canonical_on = canonical / denom
    image_h = m_t_primal @ np.array([canonical_on[0], canonical_on[1], 1.0])
    return image_h[:2] / image_h[2]


def extract_ellipse_point_at_angle(ellipse: Ellipse, theta: float) -> np.ndarray:
    """Point on the ellipse boundary at parametric angle ``theta`` (in its
    own rotated frame). Mirrors ``extractEllipsePointAtAngle``/``ellipsePoint``."""
    x = ellipse.a * np.cos(theta)
    y = ellipse.b * np.sin(theta)
    cost, sint = np.cos(ellipse.angle), np.sin(ellipse.angle)
    return np.array(
        [
            x * cost - y * sint + ellipse.center[0],
            x * sint + y * cost + ellipse.center[1],
        ]
    )


def compute_intermediate_points(ellipse: Ellipse) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Four extremal boundary points (axis-aligned tangent points, rounded
    to integer pixel coordinates). Mirrors ``computeIntermediatePoints``."""
    angle = ellipse.angle
    a_coef = -ellipse.b * np.sin(angle) - ellipse.b * np.cos(angle)
    b_coef = -ellipse.a * np.cos(angle) + ellipse.a * np.sin(angle)
    t11 = np.arctan2(-a_coef, b_coef)
    t12 = t11 + np.pi

    a_coef = -ellipse.b * np.sin(angle) + ellipse.b * np.cos(angle)
    b_coef = -ellipse.a * np.cos(angle) - ellipse.a * np.sin(angle)
    t21 = np.arctan2(-a_coef, b_coef)
    t22 = t21 + np.pi

    pt11 = np.rint(extract_ellipse_point_at_angle(ellipse, t11))
    pt12 = np.rint(extract_ellipse_point_at_angle(ellipse, t12))
    pt21 = np.rint(extract_ellipse_point_at_angle(ellipse, t21))
    pt22 = np.rint(extract_ellipse_point_at_angle(ellipse, t22))
    return pt11, pt12, pt21, pt22


def rasterize_ellipse_perimeter(ellipse: Ellipse) -> float:
    """Cheap perimeter estimate from the 4 axis-extremal boundary points
    (an inscribed-octagon-like approximation, NOT literal pixel counting
    despite the name). Mirrors ``rasterizeEllipsePerimeter``."""
    pt11, pt12, pt21, pt22 = compute_intermediate_points(ellipse)
    diff1 = max(abs(pt22[0] - pt11[0]), abs(pt22[1] - pt11[1]))
    diff2 = max(abs(pt12[0] - pt22[0]), abs(pt12[1] - pt22[1]))
    return (diff1 + diff2) * 2.0


def intersect_ellipse_with_line(ellipse: Ellipse, value: float, horizontal: bool) -> list[float]:
    """Intersection(s) of a horizontal (``y = value``) or vertical
    (``x = value``) line with the ellipse's conic. Mirrors
    ``intersectEllipseWithLine``."""
    ec = ellipse.matrix
    if horizontal:
        a = ec[0, 0]
        b = 2 * (value * ec[0, 1] + ec[0, 2])
        c = ec[1, 1] * value**2 + 2 * value * ec[2, 1] + ec[2, 2]
    else:
        a = ec[1, 1]
        b = 2 * (value * ec[0, 1] + ec[1, 2])
        c = ec[0, 0] * value**2 + 2 * value * ec[0, 2] + ec[2, 2]

    discriminant = b**2 / 4.0 - a * c
    if discriminant > 0.0:
        sqrt_disc = np.sqrt(discriminant)
        return [(-b / 2.0 - sqrt_disc) / a, (-b / 2.0 + sqrt_disc) / a]
    if discriminant == 0.0:
        return [-b / (2.0 * a)]
    return []


class Circle(Ellipse):
    """A circle is an :class:`Ellipse` with ``a == b`` and ``angle == 0``."""

    def __init__(self, center: np.ndarray, radius: float) -> None:
        super().__init__(center=np.asarray(center, dtype=float), a=radius, b=radius, angle=0.0)

    @property
    def radius(self) -> float:
        return self._a

    @classmethod
    def from_three_points(cls, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> "Circle":
        """Exact circumcircle through three points (perpendicular-bisector
        linear system). Mirrors the ``Circle(p1, p2, p3)`` C++ constructor.

        Note: like the C++ source, a singular (collinear-points) system is
        not specially guarded against here.
        """
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        p3 = np.asarray(p3, dtype=float)
        a_mat = np.array(
            [
                [p2[0] - p1[0], p2[1] - p1[1]],
                [p3[0] - p1[0], p3[1] - p1[1]],
            ]
        )
        b_vec = np.array(
            [
                (p1[0] + p2[0]) / 2 * (p2[0] - p1[0]) + (p1[1] + p2[1]) / 2 * (p2[1] - p1[1]),
                (p1[0] + p3[0]) / 2 * (p3[0] - p1[0]) + (p1[1] + p3[1]) / 2 * (p3[1] - p1[1]),
            ]
        )
        center = np.linalg.solve(a_mat, b_vec)
        radius = float(np.hypot(p1[0] - center[0], p1[1] - center[1]))
        return cls(center=center, radius=radius)
