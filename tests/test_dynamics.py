"""Tests for the derived bias-dynamics formulas (numpy; torch cross-check if installed)."""
import numpy as np
import pytest
from scipy.special import log_softmax

from src.theory import dynamics as D
from src.theory import stability as S
from src.theory.toy_model import MixtureSpec, bayes_init, make_mixture

K = 5


def _logp(seed=0, n=300, scale=2.0, bscale=0.7):
    rng = np.random.default_rng(seed)
    z0 = rng.normal(scale=scale, size=(n, K))
    b = rng.normal(scale=bscale, size=K)
    return z0, b


@pytest.mark.parametrize("lam", [0.0, 0.8])
def test_flow_is_minus_gradient_of_value(lam):
    z0, b = _logp()
    f = lambda bb: D.infomax_value(log_softmax(z0 + bb, axis=1), lam)  # noqa: E731
    g = -D.bias_flow_from_logp(log_softmax(z0 + b, axis=1), lam)
    eps = 1e-6
    fd = np.array([(f(b + eps * e) - f(b - eps * e)) / (2 * eps) for e in np.eye(K)])
    assert np.allclose(g, fd, atol=1e-8)
    assert abs(g.sum()) < 1e-12                               # gauge: gradient is orthogonal to 1


@pytest.mark.parametrize("lam", [0.0, 0.8])
def test_hessian_matches_finite_differences_and_has_gauge_zero_mode(lam):
    z0, b = _logp(1)
    gf = lambda bb: -D.bias_flow_from_logp(log_softmax(z0 + bb, axis=1), lam)  # noqa: E731
    h = D.hessian_from_logp(log_softmax(z0 + b, axis=1), lam)
    eps = 1e-6
    hfd = np.array([(gf(b + eps * e) - gf(b - eps * e)) / (2 * eps) for e in np.eye(K)]).T
    assert np.allclose(h, hfd, atol=1e-8)
    assert np.allclose(h, h.T, atol=1e-14)
    assert np.allclose(h @ np.ones(K), 0, atol=1e-12)


def test_population_reduction_matches_real_mixture_logit_statistics():
    spec = MixtureSpec(K=K, d=10, n_per_class=20000, sep=3.0, sigma=1.0)
    mix = make_mixture(spec, np.random.default_rng(0))
    w, _ = bayes_init(mix.means, spec.sigma)
    z = mix.X @ w.T
    m = spec.sep**2 / spec.sigma**2
    zj = z[mix.y == 2]
    assert np.allclose(zj.mean(0), m * np.eye(K)[2], atol=0.15)
    assert np.allclose(np.cov(zj.T), m * np.eye(K), atol=0.4)


def test_balanced_point_hessian_structure_and_constants():
    pop = D.Population(K, 9.0, n_xi=30_000, seed=0)
    h = pop.hessian(np.zeros(K))
    q = S.perp_basis(K)
    ev = np.linalg.eigvalsh(q.T @ h @ q)
    assert np.allclose(ev, pop.alpha_L(), rtol=0.1)           # single eigenvalue alpha_L (K-1 times)
    hj = pop.hessian(np.zeros(K), lam=1.0)
    evj = np.linalg.eigvalsh(q.T @ hj @ q)
    assert np.allclose(evj, pop.alpha_J(1.0), rtol=0.1)       # alpha_J = alpha_L + lam K s^2


def test_sign_of_alpha_L_matches_phase1_observation():
    # sep=1,2 collapsed already at delta=0 in Phase 1; sep=3 did not.
    a = {sep: D.Population(K, sep**2, n_xi=60_000, seed=0).alpha_L() for sep in (1.0, 2.0, 3.0)}
    assert a[1.0] < 0 and a[2.0] < 0 and a[3.0] > 0


def test_eta_star_formula_and_lam_critical():
    assert D.eta_star(0.5) == 4.0 and np.isnan(D.eta_star(-0.1))
    pop = D.Population(K, 1.0, n_xi=30_000, seed=0)
    lc = pop.lam_critical()
    assert lc > 0 and pop.alpha_J(1.05 * lc) > 0 > pop.alpha_J(0.95 * lc)


def test_numeric_eta_star_matches_hessian_prediction():
    spec = MixtureSpec(K=K, d=10, n_per_class=100, sep=3.0)
    mix = make_mixture(spec, np.random.default_rng(0))
    w, _ = bayes_init(mix.means, spec.sigma)
    prob = S.SampleProblem(mix.X @ w.T, lam=1.0)
    b, gn = S.find_stationary(prob)
    assert gn < 1e-10
    ev = S.perp_eigs(prob.hess(b))
    assert ev.min() > 0
    pred = 2.0 / ev.max()
    num = S.eta_star_numeric(prob, b, pred, rel_tol=2e-3)
    assert abs(num - pred) / pred < 0.02
    h = prob.hess(b)
    assert S.map_spectral_radius(h, 0.9 * pred) < 1 < S.map_spectral_radius(h, 1.1 * pred)
    assert np.abs(S.fd_hessian(prob, b) - h).max() < 1e-7


def test_hessian_matches_torch_autograd_if_available():
    pytest.importorskip("torch")
    z0, b = _logp(2, n=50)
    prob = S.SampleProblem(z0, lam=0.6)
    ht = S.torch_hessian(prob, b)
    assert np.allclose(ht, prob.hess(b), atol=1e-9)
