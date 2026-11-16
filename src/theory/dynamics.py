"""Bias dynamics of the linear-softmax toy model under entropy / information-max objectives.

Objective (bias-only adaptation, W frozen):
    J(b) = E_x[ H(p(.|x)) ] - lam * H(p_bar),     p = softmax(z),  z = W x + b,
    lam = 0 is plain entropy minimisation (TENT); lam > 0 is the information-max fix.

Gradient flow:  b_dot = -grad_b J.  Derived formulas (see THEORY.md):
    b_dot_j = E[ p_j (log p_j + H) ]  -  lam * E[ p_j (log pbar_j - <log pbar>_p) ]
where <f>_p = sum_k p_k f_k.   First term: per-sample "rich get richer"; second: marginal
"rich get poorer" feedback that opposes it.

Everything takes log-probabilities `logp` of shape (N, K) so it works on a real sample or
on the Monte-Carlo `Population` below.  Status labels: [derived], [assumption], [observed].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_softmax, logsumexp


# ----------------------------------------------------------------- value, flow, Hessian
def infomax_value(logp: np.ndarray, lam: float) -> float:
    """J = mean_i H(p_i) - lam * H(p_bar)."""
    n = logp.shape[0]
    p = np.exp(logp)
    mean_h = float(-(p * logp).sum(1).mean())
    log_pbar = logsumexp(logp, axis=0) - np.log(n)
    h_bar = float(-(np.exp(log_pbar) * log_pbar).sum())
    return mean_h - lam * h_bar


def bias_flow_from_logp(logp: np.ndarray, lam: float) -> np.ndarray:
    """b_dot = -grad_b J  (K,).  [derived]"""
    n = logp.shape[0]
    p = np.exp(logp)
    h = -(p * logp).sum(1, keepdims=True)
    term_ent = (p * (logp + h)).mean(0)
    if lam == 0.0:
        return term_ent
    log_pbar = logsumexp(logp, axis=0) - np.log(n)
    centred = log_pbar[None, :] - (p * log_pbar[None, :]).sum(1, keepdims=True)
    return term_ent - lam * (p * centred).mean(0)


def hessian_from_logp(logp: np.ndarray, lam: float) -> np.ndarray:
    """Hessian of J wrt b, valid at ANY b (not only the balanced point).  [derived]

    Entropy part, per sample (A = Hessian of H wrt z):
        A_kj = -delta_kj p_k (l_k + H + 1) + p_k p_j (l_k + l_j + 2H + 1),   l = log p.
    Marginal part: with S_ki = E[p_k (delta_ki - p_i)] and
        T_kij = p_k (delta_kj - p_j)(delta_ki - p_i) - p_k p_i (delta_ij - p_j),
        Hess H(p_bar)_ij = -sum_k S_ki S_kj / pbar_k - sum_k (log pbar_k + 1) E[T_kij].
    """
    n, k = logp.shape
    p = np.exp(logp)
    h = -(p * logp).sum(1, keepdims=True)
    # entropy part
    u = p * (logp + h + 0.5)
    hess_l = (u.T @ p + p.T @ u) / n
    hess_l[np.diag_indices(k)] -= (p * (logp + h + 1.0)).mean(0)
    if lam == 0.0:
        return hess_l
    pbar = p.mean(0)
    log_pbar = np.log(pbar)
    s = np.diag(pbar) - p.T @ p / n
    big_b = np.eye(k)[None] - p[:, None, :]                   # B[n,a,b] = delta_ab - p_nb
    t_mean = (np.einsum("nk,nkj,nki->kij", p, big_b, big_b)
              - np.einsum("nk,ni,nij->kij", p, p, big_b)) / n
    hess_h = -np.einsum("ki,kj,k->ij", s, s, 1.0 / pbar) - np.einsum("k,kij->ij", log_pbar + 1.0, t_mean)
    return hess_l - lam * hess_h


# ------------------------------------------------------------- population (exact reduction)
class Population:
    """Monte-Carlo population of logits for the balanced orthogonal mixture.  [derived]

    With orthogonal equal-norm means (needs d >= K) and x | y=j ~ N(mu_j, sigma^2 I):
        z_k = m * 1[k=j] + sqrt(m) * xi_k + b_k,   xi_k iid N(0,1),   m = sep^2 / sigma^2.
    So the whole population problem depends only on (K, m, b). The same xi draws are reused
    for every b (common random numbers) so b -> flow(b) is smooth and deterministic.
    """

    def __init__(self, K: int, m: float, n_xi: int = 100_000, seed: int = 0) -> None:
        self.K, self.m = K, m
        xi = np.random.default_rng(seed).standard_normal((n_xi, K))
        base = np.sqrt(m) * xi
        z0 = np.tile(base, (K, 1, 1))                         # (K classes, n, K)
        for j in range(K):
            z0[j, :, j] += m
        self.z0 = z0.reshape(K * n_xi, K)

    def logp(self, b: np.ndarray) -> np.ndarray:
        return log_softmax(self.z0 + b, axis=1)

    def flow(self, b: np.ndarray, lam: float = 0.0) -> np.ndarray:
        return bias_flow_from_logp(self.logp(b), lam)

    def hessian(self, b: np.ndarray, lam: float = 0.0) -> np.ndarray:
        return hessian_from_logp(self.logp(b), lam)

    def reduced_flow(self, g: float, lam: float = 0.0) -> float:
        """g_dot for b = (g, 0, ..., 0), where g = b_0 - b_others.  [derived reduction]

        By the S_{K-1} symmetry (classes 1..K-1 are interchangeable) the subspace
        b_1 = ... = b_{K-1} is invariant, so the K-dim flow reduces EXACTLY to this 1-D flow
        (exact for the population; here evaluated by Monte Carlo).
        """
        b = np.zeros(self.K)
        b[0] = g
        bd = self.flow(b, lam)
        return float(bd[0] - bd[1:].mean())

    # balanced-point curvature constants  [derived formulas, evaluated by Monte Carlo]
    def alpha_L(self) -> float:
        """Hess of E[H] at b=0 equals alpha_L (I - 11^T/K);  alpha_L = E[tau(p)]/(K-1),
        tau(p) = -1 + sum_k p_k^2 (2 l_k + 2H + 1)."""
        logp = self.logp(np.zeros(self.K))
        p = np.exp(logp)
        h = -(p * logp).sum(1, keepdims=True)
        tau = -1.0 + (p**2 * (2 * logp + 2 * h + 1.0)).sum(1)
        return float(tau.mean() / (self.K - 1))

    def s_const(self) -> float:
        """S = E[diag(p) - p p^T] = s (I - 11^T/K) at b=0;  s = E[1 - sum p^2]/(K-1).
        Hess H(p_bar) = -K s^2 (I - 11^T/K), so alpha_J = alpha_L + lam K s^2."""
        p = np.exp(self.logp(np.zeros(self.K)))
        return float((1.0 - (p**2).sum(1)).mean() / (self.K - 1))

    def alpha_J(self, lam: float) -> float:
        return self.alpha_L() + lam * self.K * self.s_const() ** 2

    def lam_critical(self) -> float:
        """Smallest lam with alpha_J > 0 (0 if TENT is already locally stable)."""
        a = self.alpha_L()
        return max(0.0, -a / (self.K * self.s_const() ** 2))


def eta_star(alpha: float) -> float:
    """GD map on 1-perp has eigenvalue 1 - eta*alpha; |.|<1 iff 0 < eta < 2/alpha.  [derived]"""
    return 2.0 / alpha if alpha > 0 else float("nan")


@dataclass(frozen=True)
class GapRoot:
    g_c: float
    status: str      # "ok" | "unstable_at_zero" | "no_root"
    n_up_crossings: int = 0


def gap_root(flow_1d, m: float, g_max_rel: float = 1.5, n_scan: int = 31,
             which: str = "first", xtol: float = 1e-4) -> GapRoot:
    """Upward zero-crossing of a 1-D flow g_dot = flow_1d(g): the basin boundary g_c.

    [derived reduction; root found numerically]  g_0 < g_c -> g returns to balance,
    g_0 > g_c -> g keeps growing (collapse).  `which="first"` is right for the smooth
    population flow; on a finite sample the flow can be non-monotone, and `which="last"`
    (the crossing beyond which the flow stays positive) is the relevant escape point.
    """
    grid = np.linspace(0.0, g_max_rel * m, n_scan)
    vals = np.array([flow_1d(g) for g in grid])
    if vals[1] > 0:                                           # g grows from the very start
        return GapRoot(float("nan"), "unstable_at_zero")
    idx = np.where((vals[:-1] <= 0) & (vals[1:] > 0))[0]
    if len(idx) == 0:
        return GapRoot(float("nan"), "no_root")
    i = int(idx[0] if which == "first" else idx[-1])
    return GapRoot(float(brentq(flow_1d, grid[i], grid[i + 1], xtol=xtol)), "ok", len(idx))


def critical_gap(pop: Population, lam: float = 0.0, g_max_rel: float = 1.5, n_scan: int = 31) -> GapRoot:
    """Basin boundary of the population reduced flow (see `gap_root`)."""
    return gap_root(lambda g: pop.reduced_flow(g, lam), pop.m, g_max_rel, n_scan, "first")
