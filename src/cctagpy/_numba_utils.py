"""Numba-accelerated dispatch. Numba is a required dependency (``uv sync``
installs it), but every stage that uses it keeps its original NumPy
implementation alongside, dispatched via ``HAS_NUMBA`` -- useful for the
bit-identical cross-checks in this codebase's tests, and as a fallback
if Numba itself ever fails to import in some environment.

A handful of stages in this pipeline (see ``thinning.py``, ``canny.py``)
that are already fully vectorized in NumPy still allocate many full-image
temporary arrays per call; rewriting them as explicit pixel loops and JIT
compiling those with Numba avoids that allocation traffic.

``HAS_NUMBA`` tells call sites which implementation to dispatch to; there is
deliberately no fallback ``njit`` no-op decorator here, because running an
explicit-pixel-loop function as plain interpreted Python would be far
*slower* than the NumPy-vectorized version it replaces -- the two
implementations are only equivalent once Numba actually compiles one of
them, so callers must keep both and pick one, not decorate a single
function and hope.

``prange`` is re-exported for the kernels compiled with ``parallel=True``.
Those are deliberately restricted to loops whose bodies allocate nothing
(every iteration writes into a preallocated output slice): Numba's runtime
allocator serializes allocations across threads, so a kernel that allocates
per iteration gets *slower* under threading, not faster -- measured on this
pipeline before choosing which loops to parallelize.

Threading layer: Numba's OpenMP layer lets its worker threads spin-wait after
every parallel region, and this pipeline alternates many short parallel
kernels with long single-threaded Python/NumPy stretches, so the spinning
workers slow the sequential stages substantially (measured on this pipeline).
``OMP_WAIT_POLICY=PASSIVE`` makes them sleep instead. It must
be in the environment before the OpenMP runtime is first loaded -- which
Numba does lazily, on the first parallel call -- so it is set here at import
time, only when Numba may actually pick that layer and the user has not set
it themselves. The ``workqueue`` layer has no spin problem but aborts the
process on concurrent entry from two Python threads, so it is not a safe
library default.
"""

from __future__ import annotations

import os

try:
    import numba
    from numba import njit, prange

    HAS_NUMBA = True
    if numba.config.THREADING_LAYER in ("default", "omp"):
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
except ImportError:
    HAS_NUMBA = False
    njit = None  # type: ignore[assignment]
    prange = None  # type: ignore[assignment]
