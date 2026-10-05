[[Japanese](README.md)/[English](README_EN.md)]

# cctagpy

https://github.com/user-attachments/assets/be13113f-e8fb-4436-af6c-928edb6b2e1b

A pure-Python (NumPy/SciPy/Numba) port of the CPU detection pipeline of [CCTag](https://github.com/alicevision/CCTag).<br>
It detects and identifies concentric-circle fiducial markers, and does not depend on OpenCV or CUDA.

# Features
- No dependency on OpenCV or CUDA (NumPy/SciPy/Numba only). Pillow is used only to decode image files
- Every step of the algorithm, including grayscale conversion, is implemented with NumPy/SciPy/Numba
- Hot paths are sped up with Numba JIT and multi-core parallelism
- Results are not bit-identical to the C++ reference (see "Differences from C++ Version")

# Purpose of This Repository
This repository tests a Python port of the CPU pipeline and measures how much NumPy/Numba can speed it up.

# Requirements
```
Python 3.14 or later

numba          0.67.0    or later
numpy          2.5.3     or later
pillow         12.3.0    or later
scipy          1.18.1    or later
pytest         9.1.1     or later   # for tests
opencv-python  4.9       or later   # for the webcam demo (not a dependency of cctagpy itself)
```

# Installation
Not published on PyPI. Install directly from GitHub.

```bash
pip install git+https://github.com/Kazuhito00/cctagpy-prototype.git
# with uv
uv add git+https://github.com/Kazuhito00/cctagpy-prototype.git
```

For development, clone the repository and install the dependencies.
```bash
git clone https://github.com/Kazuhito00/cctagpy-prototype
cd cctagpy-prototype

# with uv (also installs development dependencies)
uv sync

# with pip
pip install -e .
pip install "pytest>=9.1.1"
```

The command examples below use `uv run`. In an environment set up with pip, drop `uv run`.

# Usage

### Quick Start
Generate a marker image and detect it.
```bash
uv run python examples/demo_generate_marker.py --id 5
uv run python -m cctagpy -n 3 -i markers_out/marker_3crowns_0005.png
```

Example output. The columns are the center x, center y, marker ID, and status (1 means a reliable identification).
```text
#frame 0
Detected 1 candidates
419.91525528048277 419.754688594152 5 1
```

### CLI
```bash
uv run python -m cctagpy -n 3 -i path/to/image.png
```

One line `x y id status` is printed per detected marker (same format as the C++ `detection` sample app).<br>
The output starts with the lines `#frame 0` and `Detected N candidates`.

- `-i, --input`: path to the input image (required)
- `-n, --nrings`: number of marker rings (3 or 4, default 3)
- `--seed N`: fix the RANSAC RNG seed
- `--fast-identification`: switch the identification center search to an experimental optimized version. It is faster but not bit-identical to the C++ reference

### Python
```python
import numpy as np
from cctagpy import Parameters, cctag_detection, load_gray_image

gray = load_gray_image("image.png")  # 2-D uint8 array (height x width)
markers = cctag_detection(gray, Parameters(n_crowns=3), rng=np.random.default_rng(0))
for m in markers:
    print(m.x(), m.y(), m.id, m.status)  # status == 1 means a reliable identification
```

`x` is the horizontal and `y` the vertical pixel coordinate. Marker IDs start at 0.

### Demo
The webcam demo needs OpenCV.
```bash
uv pip install -r examples/requirements.txt
```

The webcam demo requests 960x540 capture and overlays the detections on each frame.
```bash
uv run python examples/demo_webcam.py
```

Generate printable marker images (PNG) in `markers_out/` (`--id` / `--n-crowns` / `--all`, etc.)
```bash
uv run python examples/demo_generate_marker.py
```

# Test
```bash
uv run pytest
```

The tests cover each processing stage and the detection/identification of synthetic concentric-ring marker images.
There is no test that compares against the output of the C++ reference implementation.

# Performance

The implementation is based on NumPy array operations. Stages whose bottleneck was temporary whole-array allocation
(pyramid resize, Canny edge detection, thinning, homography resampling in identification, flood fill in ellipse growing,
RANSAC candidate extraction and scoring, etc.) are implemented with Numba JIT.

The main optimizations are:

- Replaced Python loops (`is_another_segment`, the scoring in `vote.outlier_removal`, etc.) with batched NumPy calls
- Multi-core parallelism with Numba `prange` (pyramid construction, ray marching, identification resampling, etc.)
- Implemented the RANSAC 5-point/8-point fits directly (Gaussian elimination and analytic eigendecomposition)
  instead of going through LAPACK
- Replaced many `rng.choice` calls with bulk sampling based on `rng.permuted` / rejection sampling
- Moved the point selection inside the elliptic hull used when refitting at level 0
  (`detection.select_edge_point_in_elliptic_hull`) to a Numba scanline traversal
- Moved the conic matrix computation of `geometry.Ellipse` (`compute_matrix`, 3x3 inverse) to Numba. `Ellipse` is built
  repeatedly in RANSAC trials, each growing step, hull computation, etc.

RANSAC sampling, the RANSAC numerical solvers, and the ellipse conic-matrix computation can produce results that differ
bitwise from the earlier implementation (the closed-form solvers round differently from LAPACK). For the conic-matrix
change, I compared against the earlier implementation with 8 seeds x both sample images. The sets of marker IDs matched,
the mean center shift was at most 0.367 px (the seed-to-seed variation of that marker itself is 2.5-2.9 px), and other
markers shifted by 0.02 px or less.

On a 14-core machine, a 1920x1440 sample image takes about 2.2 s with the NumPy-only implementation and about 0.15-0.20 s
with the current one, roughly 11-15x faster.

Downscaling the input image reduces processing time roughly in proportion to the pixel count. However, when the marker
ring width becomes too thin in pixels, recall drops non-linearly, depending on the image. A safe downscale factor differs
per image, so this library does not downscale. If you need speed, measure recall on your own images and downscale on the
caller side before passing the image to `cctag_detection`.

The runtime of `python -m cctagpy` includes importing `numba` (about 0.3 s) and the first JIT compilation or cache
loading. To process many images, call `cctag_detection()` repeatedly in the same process.

# Differences from C++ Version

- Results are not bit-identical. RANSAC candidate extraction uses the NumPy RNG, which differs from the C++ PCG32 stream.
  The bilinear resize of the image pyramid may round differently from the C++ `cv::resize`. The conic matrix computation
  of `Ellipse` also differs in floating-point operation order between Numba and NumPy (see "Performance").
  Validation is tolerance-based (matching marker IDs and centers), not bit-wise.
- The CUDA pipeline is not supported (CPU-only reimplementation).
- Dead code on the C++ side (`SubPixEdgeOptimizer`, the unused cut-selection cost function, `conditionerFromImage`) is
  not ported.
- The following C++ behaviors are ported as they are. A marker detected on a coarse pyramid level keeps that level's
  coordinates in `x()`/`y()` if the level-0 refit fails. `n_circles` always has the value for three rings, regardless of the ring count.
- To keep the output comparable with the reference implementation, `id_set` stays empty after identification, matching
  the known C++ behavior (`CCTag::_idSet` is always empty after identification).

# Project Structure

```text
README.md                # README (Japanese)
README_EN.md             # README (English)
LICENSE                  # MPL-2.0
pyproject.toml           # package definition
src/cctagpy/
  detection.py           # top-level detection (cctag_detection)
  identification.py      # identification against the marker bank
  ...                    # canny/thinning/vote/ransac/ellipse_growing, etc. (each docstring names the corresponding C++ file)
examples/                # webcam demo and marker image generator
tests/                   # pytest test cases
```

# Author
Kazuhito Takahashi (https://x.com/KzhtTkhs)

# License
cctagpy is under [Mozilla Public License 2.0](LICENSE) (same as the original CCTag).<br>
