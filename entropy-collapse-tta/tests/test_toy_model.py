"""Tests for the toy model (numpy; the autograd cross-check runs only if torch is installed)."""
import numpy as np
import pytest
from scipy.special import log_softmax

from src.theory import toy_model as T
from tests import reference as ref

SPEC = T.MixtureSpec(K=4, d=8, n_per_class=60, sep=3.0, sigma=1.0)


def _mix(seed=0):
    return T.make_mixture(SPEC, np.random.default_rng(seed))


def test_mixture_is_balanced_with_equal_norm_orthogonal_means():
    m = _mix()
    assert np.all(np.bincount(m.y) == SPEC.n_per_class)
    assert np.allclose(np.linalg.norm(m.means, axis=1), SPEC.sep)
    g = m.means @ m.means.T
    assert np.allclose(g - np.diag(np.diag(g)), 0, atol=1e-10)


def test_bayes_init_accurate_and_balanced_marginal():
    m = _mix()
    W, b = T.bayes_init(m.means, SPEC.sigma)
    s = T.entropy_stats(T.forward_logprobs(m.X, W, b), m.y)
    assert s["acc"] > 0.9
    assert s["H_bar"] > 0.98 * np.log(SPEC.K)


def test_entropy_stats_match_scipy_reference():
    rng = np.random.default_rng(1)
    Z = rng.normal(scale=2.0, size=(50, 4))
    s = T.entropy_stats(log_softmax(Z, axis=1))
    assert np.isclose(s["mean_H"], ref.mean_entropy(Z))
    assert np.isclose(s["H_bar"], ref.marginal_entropy(Z))


def test_analytic_gradient_matches_finite_differences():
    m = _mix()
    rng = np.random.default_rng(2)
    W = rng.normal(scale=0.5, size=(SPEC.K, SPEC.d))
    b = rng.normal(scale=0.5, size=SPEC.K)
    _, gW, gb = T.entropy_loss_and_grad(m.X, W, b)
    f = lambda W_, b_: T.entropy_loss_and_grad(m.X, W_, b_)[0]  # noqa: E731
    eps = 1e-6
    for k in range(SPEC.K):
        e = np.zeros(SPEC.K); e[k] = eps
        assert np.isclose((f(W, b + e) - f(W, b - e)) / (2 * eps), gb[k], atol=1e-8)
    for (i, j) in [(0, 0), (1, 3), (3, 7)]:
        E = np.zeros_like(W); E[i, j] = eps
        assert np.isclose((f(W + E, b) - f(W - E, b)) / (2 * eps), gW[i, j], atol=1e-8)


def test_gradient_matches_torch_autograd_if_available():
    torch = pytest.importorskip("torch")
    m = _mix()
    rng = np.random.default_rng(3)
    W = rng.normal(scale=0.5, size=(SPEC.K, SPEC.d)); b = rng.normal(scale=0.5, size=SPEC.K)
    Wt = torch.tensor(W, requires_grad=True); bt = torch.tensor(b, requires_grad=True)
    z = torch.tensor(m.X) @ Wt.T + bt
    lp = torch.log_softmax(z, 1)
    (-(lp.exp() * lp).sum(1).mean()).backward()
    _, gW, gb = T.entropy_loss_and_grad(m.X, W, b)
    assert np.allclose(Wt.grad.numpy(), gW, atol=1e-10) and np.allclose(bt.grad.numpy(), gb, atol=1e-10)


def test_small_step_gd_decreases_entropy_loss():
    m = _mix()
    tr = T.run_with_bias(m, SPEC, 2.0, T.GDConfig(eta=0.2, steps=300, record_every=10))
    assert np.all(np.diff(tr.mean_H) <= 1e-12)


def test_no_bias_does_not_collapse_and_huge_bias_does():
    m = _mix()
    cfg = T.GDConfig()
    ok = T.run_with_bias(m, SPEC, 0.0, cfg)
    assert not T.is_collapsed(ok.H_bar[-1], SPEC.K) and ok.acc[-1] > 0.9
    bad = T.run_with_bias(m, SPEC, 3 * SPEC.sep**2, cfg)
    assert T.is_collapsed(bad.H_bar[-1], SPEC.K) and bad.top_frac[-1] > 0.9


def test_collapse_threshold_brackets_the_flip():
    m = _mix()
    cfg = T.GDConfig()
    r = T.collapse_threshold(m, SPEC, cfg, delta_max=1.2 * SPEC.sep**2, n_grid=9, bisect_iters=6)
    assert r["status"] == "ok" and r["n_switches"] == 1
    lo = T.run_with_bias(m, SPEC, r["delta_star"] - 0.2, cfg).H_bar[-1]
    hi = T.run_with_bias(m, SPEC, r["delta_star"] + 0.2, cfg).H_bar[-1]
    assert not T.is_collapsed(lo, SPEC.K) and T.is_collapsed(hi, SPEC.K)
