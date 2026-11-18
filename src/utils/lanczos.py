"""Extreme eigenvalues of a symmetric operator by Lanczos (scipy ARPACK), numpy only (no torch import).

Why this exists: `curvature.hessian_extremes` (power iteration + shift) FAILS when the spectrum has a large
negative outlier and a small positive one: after shifting by the dominant (negative) eigenvalue the whole bulk of the
spectrum sits at ~|lambda_min| and the top eigenvalue is only a tiny relative bump, so the iteration stops after 2
steps on the bulk while reporting 'converged'. (Checked on a synthetic spectrum: true lambda_max 3.88 -> reported 0.003.)
Lanczos targets the algebraically largest / smallest eigenvalues directly ('LA' / 'SA') and does not have that problem.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh


def lanczos_extremes(matvec: Callable[[np.ndarray], np.ndarray], n: int, ncv: int = 60, tol: float = 1e-4,
                     maxiter: int = 300, seed: int = 0) -> dict:
    """Return {'lam_max','lam_min','converged_max','converged_min','n_matvec'} for a symmetric operator of size n."""
    count = {"n": 0}

    def mv(v: np.ndarray) -> np.ndarray:
        count["n"] += 1
        return np.asarray(matvec(np.asarray(v).ravel()), dtype=np.float64)

    op = LinearOperator((n, n), matvec=mv, dtype=np.float64)
    out: dict = {}
    for key, which in (("lam_max", "LA"), ("lam_min", "SA")):
        v0 = np.random.default_rng(seed).standard_normal(n)
        try:
            w = eigsh(op, k=1, which=which, ncv=min(ncv, n - 1), tol=tol, maxiter=maxiter, v0=v0,
                      return_eigenvectors=False)
            out[key], out["converged_" + key[4:]] = float(w[0]), True
        except ArpackNoConvergence as e:                 # keep the best partial estimate, but flag it
            w = e.eigenvalues
            out[key] = float((w.max() if which == "LA" else w.min())) if len(w) else float("nan")
            out["converged_" + key[4:]] = False
    out["n_matvec"] = count["n"]
    return out
