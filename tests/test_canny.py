import numpy as np

from cctagpy.canny import recoded_canny
from cctagpy.level import resize_bilinear
from cctagpy.pyramid import ImagePyramid
from cctagpy.thinning import thin


def _synthetic_disc(size=120, radius=40, value_in=220, value_out=30):
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = size / 2.0
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    img = np.full((size, size), value_out, dtype=np.uint8)
    img[mask] = value_in
    return img


def test_recoded_canny_finds_disc_boundary():
    img = _synthetic_disc()
    edges, dx, dy = recoded_canny(img, low_thresh=0.01 * 256, high_thresh=0.04 * 256)

    assert edges.shape == img.shape
    assert set(np.unique(edges)) <= {0, 255}
    assert edges.sum() > 0

    ys, xs = np.nonzero(edges)
    size = img.shape[0]
    cx = cy = size / 2.0
    radial_dist = np.hypot(xs - cx, ys - cy)
    # all detected edge points should sit close to the true radius (40)
    assert np.all(np.abs(radial_dist - 40) < 3)
    # and should roughly cover the boundary (many distinct angles hit)
    angles = np.arctan2(ys - cy, xs - cx)
    assert len(np.unique(np.round(angles, 1))) > 20


def test_thinning_reduces_or_preserves_edge_pixels():
    img = _synthetic_disc()
    edges, _, _ = recoded_canny(img, low_thresh=0.01 * 256, high_thresh=0.04 * 256)
    thinned = thin(edges)
    assert thinned.shape == edges.shape
    assert set(np.unique(thinned)) <= {0, 255}
    assert thinned.sum() <= edges.sum()
    assert thinned.sum() > 0


def test_resize_bilinear_identity_same_size():
    img = _synthetic_disc(size=50)
    out = resize_bilinear(img.astype(np.float64), 50, 50)
    assert np.allclose(out, img, atol=1e-6)


def test_resize_bilinear_halves_dimensions():
    img = _synthetic_disc(size=64)
    out = resize_bilinear(img.astype(np.float64), 32, 32)
    assert out.shape == (32, 32)


def test_pyramid_build_produces_cascaded_levels():
    img = _synthetic_disc(size=128)
    pyramid = ImagePyramid(128, 128, n_levels=4)
    pyramid.build(img, canny_thr_low=0.01, canny_thr_high=0.04)

    expected_sizes = [(128, 128), (64, 64), (32, 32), (16, 16)]
    for level, (w, h) in zip(pyramid.levels, expected_sizes):
        assert level.width == w
        assert level.height == h
        assert level.src.shape == (h, w)
        assert level.edges.shape == (h, w)
