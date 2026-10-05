"""Pure-Python (NumPy/SciPy/Numba) port of the CCTag CPU detection pipeline."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

try:
    __version__ = version("cctagpy")
except PackageNotFoundError:
    __version__ = "0.0.0"

if TYPE_CHECKING:
    from cctagpy.cctag import CCTag
    from cctagpy.cli import load_gray_image
    from cctagpy.detection import cctag_detection
    from cctagpy.params import Parameters

# Resolved lazily so that ``import cctagpy.<submodule>`` does not pay the
# Numba/SciPy import cost of the whole pipeline.
_LAZY = {
    "CCTag": "cctagpy.cctag",
    "Parameters": "cctagpy.params",
    "cctag_detection": "cctagpy.detection",
    "load_gray_image": "cctagpy.cli",
}

__all__ = [*_LAZY, "__version__"]


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
