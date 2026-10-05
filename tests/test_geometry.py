import numpy as np
import pytest

from cctagpy.fitting import circle_fitting, ellipse_fitting, inner_prod_min
from cctagpy.geometry import Circle, Ellipse, compute_matrix, compute_parameters


def make_ellipse_points(center, a, b, angle, n=40, noise=0.0, rng=None):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = a * np.cos(t)
    y = b * np.sin(t)
    cost, sint = np.cos(angle), np.sin(angle)
    xr = x * cost - y * sint + center[0]
    yr = x * sint + y * cost + center[1]
    pts = np.column_stack([xr, yr])
    if noise:
        pts = pts + rng.normal(scale=noise, size=pts.shape)
    return pts


def test_matrix_param_roundtrip():
    # The conic<->parameter decomposition has a legitimate axis-choice
    # ambiguity (which principal axis is called "a" vs "b", with the angle
    # shifted by pi/2 accordingly) -- this ambiguity exists in the C++
    # reference algorithm too. So round-trip through the matrix and check
    # the reconstructed conic is the same ellipse (matrix equal up to an
    # overall scale), not that a/b/angle come back unchanged verbatim.
    center = np.array([12.0, -7.0])
    a, b, angle = 5.0, 3.0, 0.4
    m = compute_matrix(center, a, b, angle)
    center2, a2, b2, angle2 = compute_parameters(m)
    assert np.allclose(center2, center, atol=1e-6)
    m2 = compute_matrix(center2, a2, b2, angle2)
    ratio = m2[0, 0] / m[0, 0]
    assert np.allclose(m2, m * ratio, atol=1e-6)
    assert {round(a2, 6), round(b2, 6)} == {round(a, 6), round(b, 6)}


def test_ellipse_class_roundtrip():
    e = Ellipse(center=np.array([1.0, 2.0]), a=4.0, b=2.0, angle=0.3)
    e2 = Ellipse(matrix=e.matrix)
    assert np.allclose(e2.center, e.center, atol=1e-6)
    ratio = e2.matrix[0, 0] / e.matrix[0, 0]
    assert np.allclose(e2.matrix, e.matrix * ratio, atol=1e-6)
    assert {round(e2.a, 6), round(e2.b, 6)} == {round(e.a, 6), round(e.b, 6)}


@pytest.mark.parametrize("angle", [0.0, 0.3, 1.1, -0.7])
def test_ellipse_fitting_recovers_known_ellipse(angle):
    center = np.array([50.0, -20.0])
    a, b = 30.0, 18.0
    pts = make_ellipse_points(center, a, b, angle, n=60)
    fitted = ellipse_fitting(pts)
    assert np.allclose(fitted.center, center, atol=1e-3)
    assert fitted.a == pytest.approx(max(a, b), abs=1e-3) or fitted.b == pytest.approx(
        max(a, b), abs=1e-3
    )


def test_ellipse_fitting_with_noise():
    rng = np.random.default_rng(0)
    center = np.array([0.0, 0.0])
    a, b, angle = 100.0, 60.0, 0.5
    pts = make_ellipse_points(center, a, b, angle, n=80, noise=0.3, rng=rng)
    fitted = ellipse_fitting(pts)
    assert np.allclose(fitted.center, center, atol=1.0)


def test_ellipse_fitting_requires_five_points():
    with pytest.raises(ValueError):
        ellipse_fitting(np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float))


def test_circle_fitting_recovers_known_circle():
    center = np.array([5.0, -5.0])
    r = 10.0
    pts = make_ellipse_points(center, r, r, 0.0, n=30)
    fitted = circle_fitting(pts)
    assert np.allclose(fitted.center, center, atol=1e-6)
    assert fitted.a == pytest.approx(r, abs=1e-6)
    assert fitted.b == pytest.approx(r, abs=1e-6)


def test_circle_from_three_points():
    center = np.array([2.0, 3.0])
    r = 7.0
    angles = [0.1, 2.0, 4.5]
    pts = [center + r * np.array([np.cos(t), np.sin(t)]) for t in angles]
    circle = Circle.from_three_points(*pts)
    assert np.allclose(circle.center, center, atol=1e-6)
    assert circle.radius == pytest.approx(r, abs=1e-6)


def test_get_sorted_outer_points_subsamples():
    ellipse = Ellipse(center=np.array([0.0, 0.0]), a=10.0, b=10.0, angle=0.0)
    pts = make_ellipse_points(np.array([0.0, 0.0]), 10.0, 10.0, 0.0, n=100)
    rng = np.random.default_rng(1)
    shuffled = pts[rng.permutation(len(pts))]
    idx = Ellipse.get_sorted_outer_points(ellipse, shuffled, 20)
    assert len(idx) <= 21  # the source's stride loop can slightly overshoot
    sampled = shuffled[idx]
    # angles of the result should be monotonically increasing (sorted)
    angles = np.arctan2(sampled[:, 1], sampled[:, 0])
    assert np.all(np.diff(angles) >= 0)


def test_inner_prod_min_same_direction_gradients():
    positions = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    gradients = np.array([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=float)
    min_val, p1, p2 = inner_prod_min(positions, gradients, thr_cos_diff_max=0.25)
    assert min_val == pytest.approx(1.0, abs=1e-6)


def test_inner_prod_min_opposite_gradient_triggers_early_exit():
    positions = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
    gradients = np.array([[1, 0], [-1, 0], [1, 0]], dtype=float)
    min_val, p1, p2 = inner_prod_min(positions, gradients, thr_cos_diff_max=0.25)
    assert min_val == pytest.approx(-1.0, abs=1e-6)
