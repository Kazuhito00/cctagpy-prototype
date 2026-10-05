"""Edge-point graph, ported from ``src/cctag/Types.{hpp,cpp}`` (``EdgePointCollection``)
and ``src/cctag/EdgePoint.{hpp,cpp}``.

The C++ source uses a flat, pre-sized struct-of-arrays layout shared with
the CUDA path (``MAX_POINTS``-sized raw arrays, a CSR "voters" packing,
bit-packed "processed" flags). Most of that is not needed in a pure-Python
port: this reimplementation uses plain NumPy arrays sized to the actual
point count and a CSR packing for voters, but preserves the two
behaviorally-important properties of the original:

1. Points are inserted in raster-scan order (row-major), since several
   downstream algorithms are order-sensitive for tie-breaking.
2. The per-pixel lookup grid (``edge_map``) is always sized to the *full*
   original image resolution, even when only one pyramid level's
   sub-rectangle is populated -- bounds checks during ray marching/voting
   must be done against this full size, not the level's own (smaller) size.
"""

from __future__ import annotations

import numpy as np


class EdgePointCollection:
    """Per-pixel edge map + per-point attribute arrays for one pyramid level.

    ``full_width``/``full_height`` is the *original* image size (see the
    module docstring); the edge/gradient arrays passed to
    :meth:`build_from_edges` may be smaller (one level's own resolution).
    """

    def __init__(self, full_width: int, full_height: int) -> None:
        self.shape = (full_width, full_height)  # (width, height), matches C++ convention
        self.edge_map = np.full((full_height, full_width), -1, dtype=np.int32)

        self.positions = np.empty((0, 2), dtype=np.int64)
        self.gradients = np.empty((0, 2), dtype=np.float64)
        self.norm_grad = np.empty((0,), dtype=np.float64)
        self.before = np.empty((0,), dtype=np.int64)
        self.after = np.empty((0,), dtype=np.int64)
        self.flow_length = np.empty((0,), dtype=np.float64)
        self.is_max = np.empty((0,), dtype=np.int64)
        self.segment = np.empty((0,), dtype=np.int64)
        # Voters in CSR form, filled by vote(): the voters of point ``i`` are
        # ``voter_index[voter_start[i]:voter_start[i + 1]]`` (ascending), and
        # ``n_voters[i]`` is their count. (The C++ ``EdgePointCollection``
        # packs voters the same way.)
        self.voter_index = np.empty((0,), dtype=np.int64)
        self.voter_start = np.zeros(1, dtype=np.int64)
        self.n_voters = np.empty((0,), dtype=np.int64)
        self.processed_in = np.empty((0,), dtype=bool)

    def build_from_edges(self, edges: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> None:
        """Populate the collection from a thinned 0/255 edge map + gradients
        (all at the SAME resolution as this level; may be smaller than the
        full image size the collection itself was sized for).
        """
        # np.nonzero on a C-contiguous 2D array yields indices in row-major
        # (y outer, x inner) order -- i.e. exactly the raster-scan insertion
        # order of ``edgesPointsFromCanny``.
        ys, xs = np.nonzero(edges)
        n = len(xs)

        # `edge_map` is all -1 straight out of `__init__`, so only the cells a
        # previous build wrote (its `positions`) need clearing -- O(points),
        # not a full-image fill, and a no-op on a fresh collection.
        prev = self.positions
        self.edge_map[prev[:, 1], prev[:, 0]] = -1

        self.positions = np.column_stack([xs, ys]).astype(np.int64)
        self.gradients = np.column_stack(
            [dx[ys, xs].astype(np.float64), dy[ys, xs].astype(np.float64)]
        )
        self.norm_grad = np.hypot(self.gradients[:, 0], self.gradients[:, 1])
        self.before = np.full(n, -1, dtype=np.int64)
        self.after = np.full(n, -1, dtype=np.int64)
        self.flow_length = np.zeros(n, dtype=np.float64)
        self.is_max = np.full(n, -1, dtype=np.int64)
        self.segment = np.full(n, -1, dtype=np.int64)
        self.voter_index = np.empty((0,), dtype=np.int64)
        self.voter_start = np.zeros(n + 1, dtype=np.int64)
        self.n_voters = np.zeros(n, dtype=np.int64)
        self.processed_in = np.zeros(n, dtype=bool)

        self.edge_map[ys, xs] = np.arange(n, dtype=np.int32)

    def point_count(self) -> int:
        return len(self.positions)

    def index_at(self, x: int, y: int) -> int:
        """Point index at pixel ``(x, y)``, or -1 if none/out of (full-image) bounds."""
        width, height = self.shape
        if x < 0 or x >= width or y < 0 or y >= height:
            return -1
        return int(self.edge_map[y, x])
