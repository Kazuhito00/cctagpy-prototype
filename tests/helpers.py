import numpy as np

from cctagpy.canny import recoded_canny
from cctagpy.edge_collection import EdgePointCollection
from cctagpy.thinning import thin


def synthetic_cctag(size=240, boundaries=(30, 40, 55, 65, 80, 90), value_lo=20, value_hi=230):
    """A concentric-rings target matching a real CCTag's structure: a white
    center, alternating black/white rings, on a white background. With 6
    boundaries this is 3 black rings, i.e. an ``n_crowns=3`` marker shape --
    the voting algorithm's crown-hop chain needs this full alternation to
    find seeds (a simpler few-ring pattern is not enough).
    """
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = size / 2.0
    r = np.hypot(xx - cx, yy - cy)
    img = np.full((size, size), value_hi, dtype=np.float64)
    bounds = sorted(boundaries)
    for k in range(len(bounds) - 1, -1, -1):
        color = value_hi if k % 2 == 0 else value_lo
        img[r <= bounds[k]] = color
    return img.astype(np.uint8), (cx, cy)


def build_collection(img):
    edges, dx, dy = recoded_canny(img, low_thresh=0.01 * 256, high_thresh=0.04 * 256)
    thinned = thin(edges)
    h, w = img.shape
    collection = EdgePointCollection(w, h)
    collection.build_from_edges(thinned, dx, dy)
    return collection, dx, dy
