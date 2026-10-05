"""Numba kernels and subset-draw helper for the two RANSAC hot loops
(``vote.outlier_removal`` and ``detection.is_another_segment``).

Both loops draw a random point subset per trial, fit a conic to it, score
the conic by the median Sampson distance of the whole point set, and keep
the best with a patience counter. With Numba available, a whole chunk of
trials runs as one kernel here: the per-trial linear algebra is done in
closed form (partial-pivot elimination for the 5-point conic, the
characteristic cubic + cross-product eigenvectors for the Halir-Flusser
3x3 eigenproblem, the analytic eigendecomposition of the symmetric 2x2 in
the conic-to-ellipse conversion) instead of one LAPACK call per trial, and
the scoring/patience logic follows inline -- no per-trial NumPy call, no
per-trial ``Ellipse`` object. :func:`draw_subsets`, the bulk subset-draw
helper both calling modules use, is plain NumPy and needs no Numba.

These kernels are deliberately NOT bit-identical to the NumPy/LAPACK
fallbacks kept in the calling modules: the closed-form solvers round
differently in the last bits, so an individual trial's accept/reject can
in principle flip on a near-tie. RANSAC output is already only
*statistically* reproducible here (the subset draws themselves moved from
per-trial ``rng.choice`` to bulk ``draw_subsets``), and that is the
standard these kernels are verified against -- identical marker ids and
centers within the RANSAC-induced spread across seeds, see the README.
The non-RANSAC fits (``fitting.ellipse_fitting`` on final point sets) are
untouched and still go through LAPACK.

``error_model="numpy"``: a degenerate trial may divide by zero inside the
Sampson distance; NumPy yields inf/nan there and the ``s < sm`` test then
simply rejects the trial, and these kernels must do the same rather than
raise.
"""

from __future__ import annotations

import math

import numpy as np

from cctagpy._numba_utils import HAS_NUMBA, njit

_EPS = float(np.finfo(float).eps)


def draw_subsets(rng: np.random.Generator, n_rows: int, n: int, k: int) -> np.ndarray:
    """``n_rows`` independent uniformly random *ordered* ``k``-subsets of
    ``range(n)`` as an ``(n_rows, k)`` int64 array -- the distribution of
    ``rng.choice(n, k, replace=False)`` per row.

    Drawn by rejection: ``k`` independent uniform indices per row, redrawing
    only the rows that contain a repeat until none does. Conditioned on
    being distinct, ``k`` independent uniforms are a uniformly random
    ordered ``k``-subset, so this is exact, and for the ``k <= 5``,
    ``n >> k`` cases here it is far cheaper than permuting all ``n``
    indices per row (``rng.permuted`` of an ``(n_rows, n)`` grid: ~300us
    for 100x200 vs ~15us here) or than one ``rng.choice`` call per row.
    When ``n`` is not much larger than ``k`` (small RANSAC point sets:
    ``n`` of 10-13 with ``k = 5`` is common) repeats are frequent and the
    redraw rounds cost more than the permutation, so that case permutes."""
    if n < 8 * k:
        return np.ascontiguousarray(rng.permuted(np.tile(np.arange(n), (n_rows, 1)), axis=1)[:, :k])
    idx = rng.integers(0, n, (n_rows, k))
    bad = np.nonzero((np.diff(np.sort(idx, axis=1), axis=1) == 0).any(axis=1))[0]
    while len(bad):
        idx[bad] = rng.integers(0, n, (len(bad), k))
        bad = bad[(np.diff(np.sort(idx[bad], axis=1), axis=1) == 0).any(axis=1)]
    return idx


if HAS_NUMBA:

    @njit(cache=True, inline="always")
    def _median_inplace(buf: np.ndarray, m: int) -> float:
        """``np.median`` of ``buf[:m]`` (sorts that slice in place)."""
        v = buf[:m]
        v.sort()
        if m % 2 == 1:
            return v[m // 2]
        return (v[m // 2 - 1] + v[m // 2]) / 2.0

    @njit(cache=True, inline="always")
    def _sampson_median(pts: np.ndarray, q: np.ndarray, weights: np.ndarray, use_weights: bool, buf: np.ndarray) -> float:
        """Median over ``pts`` of ``distance.distance_point_ellipse`` (times
        ``weights`` when ``use_weights``), same per-point arithmetic."""
        m = pts.shape[0]
        q00 = q[0, 0]
        q01 = q[0, 1]
        q02 = q[0, 2]
        q11 = q[1, 1]
        q12 = q[1, 2]
        q22 = q[2, 2]
        for i in range(m):
            x = pts[i, 0]
            y = pts[i, 1]
            residual = x * x * q00 + 2 * x * y * q01 + 2 * x * q02 + y * y * q11 + 2 * y * q12 + q22
            tmp1 = q00 * x + q01 * y + q02
            tmp2 = q01 * x + q11 * y + q12
            d = (residual * residual) / (tmp1 * tmp1 + tmp2 * tmp2)
            if use_weights:
                d = d * weights[i]
            buf[i] = d
        return _median_inplace(buf, m)

    @njit(cache=True, inline="always")
    def _conic_semi_axes(q: np.ndarray, out_ab: np.ndarray) -> bool:
        """The semi-axis part of ``geometry.compute_parameters``; False where
        that path is degenerate or would raise (negative / zero axis)."""
        par0 = q[0, 0]
        par1 = 2 * q[0, 1]
        par2 = q[1, 1]
        par3 = 2 * q[0, 2]
        par4 = 2 * q[1, 2]
        par5 = q[2, 2]
        theta = 0.5 * math.atan2(par1, par0 - par2)
        cost = math.cos(theta)
        sint = math.sin(theta)
        au = par3 * cost + par4 * sint
        av = -par3 * sint + par4 * cost
        auu = par0 * cost**2 + par2 * sint**2 + par1 * cost * sint
        avv = par0 * sint**2 + par2 * cost**2 - par1 * cost * sint
        if auu == 0 or avv == 0:
            return False
        tu = -au / (2 * auu)
        tv = -av / (2 * avv)
        w = par5 - auu * tu**2 - avv * tv**2
        ru = -w / auu
        rv = -w / avv
        a = math.sqrt(abs(ru)) * np.sign(ru)
        b = math.sqrt(abs(rv)) * np.sign(rv)
        if a < 0 or b <= 0:
            return False
        out_ab[0] = a
        out_ab[1] = b
        return True

    @njit(cache=True, inline="always")
    def _solve5(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> bool:
        """Gaussian elimination with partial pivoting on the 5x5 system
        ``a x = b`` (``a``/``b`` are destroyed). False if singular."""
        n = 5
        for col in range(n):
            piv = col
            best = abs(a[col, col])
            for r in range(col + 1, n):
                v = abs(a[r, col])
                if v > best:
                    best = v
                    piv = r
            if best == 0.0:
                return False
            if piv != col:
                for c in range(n):
                    tmp = a[col, c]
                    a[col, c] = a[piv, c]
                    a[piv, c] = tmp
                tmp = b[col]
                b[col] = b[piv]
                b[piv] = tmp
            for r in range(col + 1, n):
                fct = a[r, col] / a[col, col]
                if fct != 0.0:
                    for c in range(col, n):
                        a[r, c] -= fct * a[col, c]
                    b[r] -= fct * b[col]
        for r in range(n - 1, -1, -1):
            acc = b[r]
            for c in range(r + 1, n):
                acc -= a[r, c] * x[c]
            x[r] = acc / a[r, r]
        return True

    @njit(cache=True, error_model="numpy")
    def outlier_removal_chunk(
        pts: np.ndarray,
        weights: np.ndarray,
        use_weights: bool,
        perms: np.ndarray,
        patience: int,
        counter: int,
        sm: float,
        qm: np.ndarray,
        have_qm: bool,
        f: float,
    ) -> tuple[int, float, bool]:
        """One chunk of ``vote.outlier_removal`` trials: for each row of
        ``perms`` (5 point indices) solve the 5-point conic, apply the
        quadratic-form / semi-axis-ratio gates, score by weighted median
        Sampson distance, and replay the accept/reject/patience logic. The
        best conic so far is kept in ``qm`` (in place). Returns the updated
        ``(counter, sm, have_qm)``."""
        chunk = perms.shape[0]
        m = pts.shape[0]
        a_mat = np.empty((5, 5))
        rhs = np.empty(5)
        temp = np.empty(5)
        q = np.empty((3, 3))
        ab = np.empty(2)
        buf = np.empty(m)

        for t in range(chunk):
            for r in range(5):
                x = pts[perms[t, r], 0]
                y = pts[perms[t, r], 1]
                a_mat[r, 0] = x * x
                a_mat[r, 1] = 2.0 * x * y
                a_mat[r, 2] = y * y
                a_mat[r, 3] = 2.0 * f * x
                a_mat[r, 4] = 2.0 * f * y
                rhs[r] = -f * f
            ok = _solve5(a_mat, rhs, temp)
            if ok:
                ok = temp[0] * temp[2] - temp[1] * temp[1] > 0
            if ok:
                q[0, 0] = temp[0]
                q[0, 1] = temp[1]
                q[1, 0] = temp[1]
                q[1, 1] = temp[2]
                q[0, 2] = temp[3]
                q[2, 0] = temp[3]
                q[1, 2] = temp[4]
                q[2, 1] = temp[4]
                q[2, 2] = 1.0
                ok = _conic_semi_axes(q, ab)
            if ok:
                ratio = ab[0] / ab[1]
                ok = ratio >= 0.04 and ratio <= 25
            if not ok:
                counter += 1
                if counter >= patience:
                    break
                continue

            s = _sampson_median(pts, q, weights, use_weights, buf)
            if s < sm:
                counter = 0
                sm = s
                qm[:, :] = q
                have_qm = True
            else:
                counter += 1
                if counter >= patience:
                    break
        return counter, sm, have_qm

    @njit(cache=True, inline="always")
    def _cubic_real_roots(c2: float, c1: float, c0: float, roots: np.ndarray) -> int:
        """Real roots of ``x^3 - c2 x^2 + c1 x - c0 = 0`` (the characteristic
        polynomial of a 3x3 matrix with trace ``c2``, sum of principal 2x2
        minors ``c1`` and determinant ``c0``). Fills ``roots`` and returns
        how many (1 or 3)."""
        shift = c2 / 3.0
        p = c1 - c2 * c2 / 3.0
        qq = -c0 + c1 * c2 / 3.0 - 2.0 * c2 * c2 * c2 / 27.0
        disc = qq * qq / 4.0 + p * p * p / 27.0
        if disc <= 0.0:
            if p >= 0.0:
                # p == 0 and disc == 0: triple root
                roots[0] = shift
                return 1
            r = 2.0 * math.sqrt(-p / 3.0)
            arg = (-qq / 2.0) / math.sqrt(-(p * p * p) / 27.0)
            if arg > 1.0:
                arg = 1.0
            elif arg < -1.0:
                arg = -1.0
            phi = math.acos(arg)
            roots[0] = r * math.cos(phi / 3.0) + shift
            roots[1] = r * math.cos((phi - 2.0 * math.pi) / 3.0) + shift
            roots[2] = r * math.cos((phi - 4.0 * math.pi) / 3.0) + shift
            return 3
        sq = math.sqrt(disc)
        roots[0] = np.cbrt(-qq / 2.0 + sq) + np.cbrt(-qq / 2.0 - sq) + shift
        return 1

    @njit(cache=True, inline="always")
    def _null_vector(a: np.ndarray, out: np.ndarray) -> bool:
        """Unit vector in the null space of the (rank-2) 3x3 ``a``: the
        largest cross product of two of its rows. False if all vanish."""
        best = -1.0
        for i in range(3):
            for j in range(i + 1, 3):
                vx = a[i, 1] * a[j, 2] - a[i, 2] * a[j, 1]
                vy = a[i, 2] * a[j, 0] - a[i, 0] * a[j, 2]
                vz = a[i, 0] * a[j, 1] - a[i, 1] * a[j, 0]
                nrm = vx * vx + vy * vy + vz * vz
                if nrm > best:
                    best = nrm
                    out[0] = vx
                    out[1] = vy
                    out[2] = vz
        if best <= 0.0:
            return False
        inv = 1.0 / math.sqrt(best)
        out[0] *= inv
        out[1] *= inv
        out[2] *= inv
        return True

    @njit(cache=True, error_model="numpy")
    def fit_trial_conic(pts: np.ndarray, q_out: np.ndarray, ab_out: np.ndarray) -> bool:
        """``fitting.ellipse_fitting`` for one RANSAC trial without LAPACK:
        Halir-Flusser direct least squares (closed-form 3x3 inverse,
        eigenvalues from the characteristic cubic, eigenvectors as null
        vectors), then the same conic-to-ellipse conversion and degeneracy
        gates as ``fitting.to_ellipse`` (analytic symmetric 2x2 eigen
        decomposition in place of the SVD). Writes the ellipse's conic
        matrix (``geometry.compute_matrix`` of center/radii/angle, as
        ``Ellipse.matrix`` would be) to ``q_out`` and ``(a, b)`` = the two
        radii in ``to_ellipse``'s order to ``ab_out``. False wherever the
        scalar path would have raised."""
        n = pts.shape[0]
        ox = 0.0
        oy = 0.0
        for i in range(n):
            ox += pts[i, 0]
            oy += pts[i, 1]
        ox /= n
        oy /= n

        s1 = np.zeros((3, 3))
        s2 = np.zeros((3, 3))
        s3 = np.zeros((3, 3))
        d1 = np.empty(3)
        d2 = np.empty(3)
        for i in range(n):
            x = pts[i, 0] - ox
            y = pts[i, 1] - oy
            d1[0] = x * x
            d1[1] = x * y
            d1[2] = y * y
            d2[0] = x
            d2[1] = y
            d2[2] = 1.0
            for r in range(3):
                for c in range(3):
                    s1[r, c] += d1[r] * d1[c]
                    s2[r, c] += d1[r] * d2[c]
                    s3[r, c] += d2[r] * d2[c]

        # s3^{-1} by adjugate
        det3 = (
            s3[0, 0] * (s3[1, 1] * s3[2, 2] - s3[1, 2] * s3[2, 1])
            - s3[0, 1] * (s3[1, 0] * s3[2, 2] - s3[1, 2] * s3[2, 0])
            + s3[0, 2] * (s3[1, 0] * s3[2, 1] - s3[1, 1] * s3[2, 0])
        )
        if det3 == 0.0:
            return False
        inv = np.empty((3, 3))
        inv[0, 0] = (s3[1, 1] * s3[2, 2] - s3[1, 2] * s3[2, 1]) / det3
        inv[0, 1] = (s3[0, 2] * s3[2, 1] - s3[0, 1] * s3[2, 2]) / det3
        inv[0, 2] = (s3[0, 1] * s3[1, 2] - s3[0, 2] * s3[1, 1]) / det3
        inv[1, 0] = (s3[1, 2] * s3[2, 0] - s3[1, 0] * s3[2, 2]) / det3
        inv[1, 1] = (s3[0, 0] * s3[2, 2] - s3[0, 2] * s3[2, 0]) / det3
        inv[1, 2] = (s3[0, 2] * s3[1, 0] - s3[0, 0] * s3[1, 2]) / det3
        inv[2, 0] = (s3[1, 0] * s3[2, 1] - s3[1, 1] * s3[2, 0]) / det3
        inv[2, 1] = (s3[0, 1] * s3[2, 0] - s3[0, 0] * s3[2, 1]) / det3
        inv[2, 2] = (s3[0, 0] * s3[1, 1] - s3[0, 1] * s3[1, 0]) / det3

        # t = -s3_inv @ s2.T ; nmat = s1 + s2 @ t ; m = c1_inv @ nmat
        t = np.empty((3, 3))
        for r in range(3):
            for c in range(3):
                acc = 0.0
                for k in range(3):
                    acc += inv[r, k] * s2[c, k]
                t[r, c] = -acc
        nmat = np.empty((3, 3))
        for r in range(3):
            for c in range(3):
                acc = s1[r, c]
                for k in range(3):
                    acc += s2[r, k] * t[k, c]
                nmat[r, c] = acc
        m = np.empty((3, 3))
        for c in range(3):
            m[0, c] = 0.5 * nmat[2, c]
            m[1, c] = -nmat[1, c]
            m[2, c] = 0.5 * nmat[0, c]

        trace = m[0, 0] + m[1, 1] + m[2, 2]
        minors = (
            (m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])
            + (m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0])
            + (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
        )
        detm = (
            m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
            - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
            + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
        )
        roots = np.empty(3)
        n_roots = _cubic_real_roots(trace, minors, detm, roots)

        shifted = np.empty((3, 3))
        vec = np.empty(3)
        a1 = np.empty(3)
        best_cond = 0.0
        found = False
        for j in range(n_roots):
            lam = roots[j]
            for r in range(3):
                for c in range(3):
                    shifted[r, c] = m[r, c]
                shifted[r, r] -= lam
            if not _null_vector(shifted, vec):
                continue
            cond = 4 * vec[0] * vec[2] - vec[1] * vec[1]
            if cond > _EPS and (not found or cond < best_cond):
                best_cond = cond
                found = True
                a1[0] = vec[0]
                a1[1] = vec[1]
                a1[2] = vec[2]
        if not found:
            return False

        a = a1[0]
        b = a1[1]
        c = a1[2]
        d = t[0, 0] * a + t[0, 1] * b + t[0, 2] * c
        e = t[1, 0] * a + t[1, 1] * b + t[1, 2] * c
        f = t[2, 0] * a + t[2, 1] * b + t[2, 2] * c

        # --- to_ellipse ---
        idet = a * c - b * b / 4.0
        if idet <= _EPS:
            return False
        scale = math.sqrt(idet / 4.0)
        if scale <= _EPS:
            return False
        a *= scale
        b *= scale
        c *= scale
        d *= scale
        e *= scale
        f *= scale

        det_cm = (2 * a) * (2 * c) - b * b
        if det_cm == 0.0:
            return False
        x0 = ((2 * c) * (-d) - b * (-e)) / det_cm
        y0 = ((2 * a) * (-e) - b * (-d)) / det_cm

        f0 = a * x0 * x0 + b * x0 * y0 + c * y0 * y0 + d * x0 + e * y0 + f
        if abs(f0) <= _EPS:
            return False
        neg_f0 = -f0
        s00 = a / neg_f0
        s01 = (b / 2.0) / neg_f0
        s11 = c / neg_f0

        half_tr = (s00 + s11) / 2.0
        half_dif = (s00 - s11) / 2.0
        disc = math.sqrt(half_dif * half_dif + s01 * s01)
        l1 = half_tr + disc
        l2 = half_tr - disc
        abs1 = abs(l1)
        abs2 = abs(l2)
        if abs1 >= abs2:
            sv0 = abs1
            sv1 = abs2
            lam_small = l2
        else:
            sv0 = abs2
            sv1 = abs1
            lam_small = l1
        if sv0 <= 0 or sv1 <= 0:
            return False
        radius0 = math.sqrt(1.0 / sv0)
        radius1 = math.sqrt(1.0 / sv1)

        # eigenvector of the smaller-|lambda| eigenvalue (= the SVD's u[:, 1])
        ux = s01
        uy = lam_small - s00
        vx = lam_small - s11
        vy = s01
        if vx * vx + vy * vy > ux * ux + uy * uy:
            ux = vx
            uy = vy
        nrm = math.sqrt(ux * ux + uy * uy)
        if nrm == 0.0:
            ux = 0.0
            uy = 1.0
        else:
            ux /= nrm
            uy /= nrm
        angle = math.pi - math.atan2(ux, uy)
        cx = x0 + ox
        cy = y0 + oy

        # --- compute_matrix(center, radius0, radius1, angle) ---
        ca = math.cos(angle)
        sa = math.sin(angle)
        # t_inv rows (inverse of the rigid transform [[ca,-sa,cx],[sa,ca,cy],[0,0,1]])
        i00 = ca
        i01 = sa
        i02 = -(ca * cx + sa * cy)
        i10 = -sa
        i11 = ca
        i12 = sa * cx - ca * cy
        w0 = 1.0 / (radius0 * radius0)
        w1 = 1.0 / (radius1 * radius1)
        # Q = t_inv^T diag(w0, w1, -1) t_inv
        q_out[0, 0] = w0 * i00 * i00 + w1 * i10 * i10
        q_out[0, 1] = w0 * i00 * i01 + w1 * i10 * i11
        q_out[0, 2] = w0 * i00 * i02 + w1 * i10 * i12
        q_out[1, 0] = q_out[0, 1]
        q_out[1, 1] = w0 * i01 * i01 + w1 * i11 * i11
        q_out[1, 2] = w0 * i01 * i02 + w1 * i11 * i12
        q_out[2, 0] = q_out[0, 2]
        q_out[2, 1] = q_out[1, 2]
        q_out[2, 2] = w0 * i02 * i02 + w1 * i12 * i12 - 1.0
        ab_out[0] = radius0
        ab_out[1] = radius1
        return True

    @njit(cache=True, error_model="numpy")
    def another_segment_chunk(
        pts: np.ndarray,
        another_pts: np.ndarray,
        i1s: np.ndarray,
        i2s: np.ndarray,
        patience: int,
        cnt: int,
        sm: float,
        found: bool,
    ) -> tuple[int, float, bool]:
        """One chunk of ``detection.is_another_segment`` trials: 4 points
        from each set per row of ``i1s``/``i2s``, an 8-point ellipse fit
        (:func:`fit_trial_conic`), the axis-ratio gate, the score
        ``median(dist to pts) + median(dist to another_pts)`` and the
        accept/reject/patience replay. Returns ``(cnt, sm, found)``."""
        chunk = i1s.shape[0]
        eight = np.empty((8, 2))
        q = np.empty((3, 3))
        ab = np.empty(2)
        buf1 = np.empty(pts.shape[0])
        buf2 = np.empty(another_pts.shape[0])
        no_weights = np.empty(1)

        for t in range(chunk):
            for r in range(4):
                eight[r, 0] = pts[i1s[t, r], 0]
                eight[r, 1] = pts[i1s[t, r], 1]
                eight[4 + r, 0] = another_pts[i2s[t, r], 0]
                eight[4 + r, 1] = another_pts[i2s[t, r], 1]
            ok = fit_trial_conic(eight, q, ab)
            if ok:
                ratio = ab[0] / ab[1]
                ok = not (ratio < 0.12 or ratio > 8)
            if not ok:
                cnt += 1
                if cnt >= patience:
                    break
                continue
            s = _sampson_median(pts, q, no_weights, False, buf1) + _sampson_median(another_pts, q, no_weights, False, buf2)
            if s < sm:
                cnt = 0
                sm = s
                found = True
            else:
                cnt += 1
                if cnt >= patience:
                    break
        return cnt, sm, found
