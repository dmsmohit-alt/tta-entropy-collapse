"""Tests for stream metrics and the collapse detector (numpy only)."""
import numpy as np
import pytest

from src.metrics.collapse import CollapseDetector, EmaMarginal, batch_metrics, entropy_of
from tests import reference as ref


def test_batch_metrics_match_reference():
    rng = np.random.default_rng(0)
    z = rng.normal(scale=3.0, size=(64, 10))
    y = rng.integers(0, 10, 64)
    m, pbar = batch_metrics(z, y)
    assert np.isclose(m["mean_H"], ref.mean_entropy(z))
    assert np.isclose(m["H_bar_batch"], ref.marginal_entropy(z))
    assert np.allclose(pbar, ref.marginal(z)) and np.isclose(pbar.sum(), 1.0)
    pred = z.argmax(1)
    assert np.isclose(m["online_err"], (pred != y).mean())
    assert m["top_class"] == np.bincount(pred, minlength=10).argmax()
    assert np.isclose(m["top_frac"], np.bincount(pred, minlength=10).max() / 64)


def test_batch_metrics_collapsed_batch_and_extreme_logits():
    z = np.tile(np.array([1e4, 0.0, -1e4]), (20, 1))
    y = np.zeros(20, dtype=int)
    m, pbar = batch_metrics(z, y)
    assert m["top_frac"] == 1.0 and m["top_class"] == 0 and m["online_err"] == 0.0
    assert np.isfinite(m["mean_H"]) and m["H_bar_batch"] < 1e-9 and np.isfinite(pbar).all()


def test_batch_metrics_balanced_confident_batch():
    k = 5
    z = 50.0 * np.eye(k)[np.arange(50) % k]
    m, _ = batch_metrics(z, np.arange(50) % k)
    assert np.isclose(m["H_bar_batch"], np.log(k), atol=1e-6) and m["mean_H"] < 1e-6
    assert np.isclose(m["top_frac"], 1 / k) and m["online_err"] == 0.0


def test_ema_marginal():
    e = EmaMarginal(decay=0.5)
    uniform, onehot = np.full(4, 0.25), np.array([1.0, 0, 0, 0])
    assert np.isclose(e.update(uniform), np.log(4))             # first call = that batch
    h = e.update(onehot)                                        # 0.5*uniform + 0.5*onehot = [.625,.125,.125,.125]
    assert np.isclose(h, entropy_of(np.array([0.625, 0.125, 0.125, 0.125])))
    assert EmaMarginal(decay=0.0).update(onehot) == 0.0
    assert entropy_of(np.array([0.5, 0.5, 0.0])) == pytest.approx(np.log(2))


def test_detector_requires_consecutive_batches():
    d = CollapseDetector(top_frac_threshold=0.9, patience=3)
    seq = [0.2, 0.95, 0.95, 0.5, 0.95, 0.95, 0.95, 0.99]       # streak broken at step 3; real streak starts at 4
    fired = [d.update(i, v) for i, v in enumerate(seq)]
    assert fired == [False] * 6 + [True, True]
    assert d.start_step == 4 and d.detect_step == 6 and d.collapsed


def test_detector_strict_inequality_never_fires_and_keeps_first_detection():
    d = CollapseDetector(0.9, 2)
    assert not any(d.update(i, 0.9) for i in range(10))         # exactly at threshold is NOT > threshold
    assert not d.collapsed and d.start_step is None
    d = CollapseDetector(0.9, 2)
    for i, v in enumerate([0.95, 0.95, 0.1, 0.95, 0.95]):
        d.update(i, v)
    assert (d.start_step, d.detect_step) == (0, 1)               # later streaks do not overwrite
    assert CollapseDetector(0.9, 1).update(7, 0.91) is True
