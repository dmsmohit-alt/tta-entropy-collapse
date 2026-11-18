"""Lanczos extremes vs known spectra (numpy/scipy only, runs without torch)."""
import numpy as np
import pytest

from src.utils.lanczos import lanczos_extremes


def _op(ev, seed=0):
    d = len(ev)
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    return (lambda v: Q @ (ev * (Q.T @ v))), d


@pytest.mark.parametrize("true_max", [0.02, 3.88])
def test_large_negative_outlier_and_small_positive_one(true_max):
    # the case where shifted power iteration fails (reports ~0 for lam_max)
    ev = np.random.default_rng(1).normal(0, 0.005, 800)
    ev[0], ev[1] = -17.0, true_max
    mv, d = _op(ev)
    r = lanczos_extremes(mv, d)
    assert r["converged_max"] and r["converged_min"]
    assert r["lam_min"] == pytest.approx(-17.0, rel=1e-3)
    assert r["lam_max"] == pytest.approx(max(true_max, ev.max()), rel=2e-2, abs=2e-3)


def test_psd_matches_eigvalsh():
    X = np.random.default_rng(2).standard_normal((50, 30))
    H = X.T @ X / 50
    r = lanczos_extremes(lambda v: H @ v, 30, ncv=29)
    ev = np.linalg.eigvalsh(H)
    assert r["lam_max"] == pytest.approx(ev[-1], rel=1e-4) and r["lam_min"] == pytest.approx(ev[0], rel=1e-3)
