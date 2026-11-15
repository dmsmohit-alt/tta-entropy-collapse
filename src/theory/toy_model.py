"""Toy model: linear-softmax classifier on a balanced Gaussian mixture.

Model:      p(y|x) = softmax(W x + b),   W in R^{K x d}, b in R^K
Adaptation: gradient descent on  L = mean_i H(p(.|x_i))   (entropy minimisation / TENT)
Question:   starting from the Bayes-optimal classifier plus an extra bias
            `delta` toward class 0, does GD drive H(p_bar) -> 0 (collapse)?

Everything here is numpy with analytic gradients (fast on a CPU laptop). The
analytic gradient is checked against finite differences in tests, and (in
Phase 2) against PyTorch autograd.

Status labels used in comments/docs:  [derived] = follows from the math,
[observed] = only seen numerically, [assumption] = chosen by us.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import log_softmax

Adapt = Literal["b", "Wb"]


# --------------------------------------------------------------------------- data
@dataclass(frozen=True)
class MixtureSpec:
    """Balanced K-class isotropic Gaussian mixture  x | y=k ~ N(mu_k, sigma^2 I)."""

    K: int = 5
    d: int = 10
    n_per_class: int = 200
    sep: float = 3.0     # ||mu_k||: distance of each class mean from the origin
    sigma: float = 1.0


@dataclass(frozen=True)
class Mixture:
    X: np.ndarray       # (N, d)
    y: np.ndarray       # (N,)
    means: np.ndarray   # (K, d)


def make_mixture(spec: MixtureSpec, rng: np.random.Generator) -> Mixture:
    """Sample a balanced mixture. Means are orthogonal (if d >= K) with equal norm `sep`.

    [assumption] Equal norms + orthogonality keep the problem symmetric across
    classes, so any asymmetry in the dynamics comes only from the initial bias.
    """
    if spec.d >= spec.K:
        q, _ = np.linalg.qr(rng.normal(size=(spec.d, spec.K)))  # orthonormal columns
        dirs = q.T                                              # (K, d)
    else:  # fall back to random unit directions
        dirs = rng.normal(size=(spec.K, spec.d))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    means = spec.sep * dirs
    y = np.repeat(np.arange(spec.K), spec.n_per_class)
    X = means[y] + spec.sigma * rng.normal(size=(len(y), spec.d))
    return Mixture(X=X, y=y, means=means)


def bayes_init(means: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Bayes-optimal (LDA) classifier: W_k = mu_k / sigma^2, b_k = -||mu_k||^2 / (2 sigma^2).

    With equal-norm means every b_k is the same constant, which softmax ignores,
    so we return b = 0 (same predictions, balanced marginal at initialisation).
    """
    return means / sigma**2, np.zeros(means.shape[0])


# ------------------------------------------------------------------ entropy + grad
def forward_logprobs(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return log_softmax(X @ W.T + b, axis=1)


def entropy_stats(logp: np.ndarray, y: np.ndarray | None = None) -> dict[str, float]:
    """Mean entropy, marginal entropy H(p_bar), top-class fraction, accuracy."""
    n, k = logp.shape
    p = np.exp(logp)
    h_i = -(p * logp).sum(1)
    log_pbar = np.logaddexp.reduce(logp, axis=0) - np.log(n)  # stable log of mean prob
    pbar = np.exp(log_pbar)
    pred = logp.argmax(1)
    out = {
        "mean_H": float(h_i.mean()),
        "H_bar": float(-(pbar * log_pbar).sum()),
        "top_frac": float(np.bincount(pred, minlength=k).max() / n),
    }
    if y is not None:
        out["acc"] = float((pred == y).mean())
    return out


def entropy_loss_and_grad(
    X: np.ndarray, W: np.ndarray, b: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """L = mean_i H(p_i) and its gradients wrt W and b.

    [derived] For one sample, with p = softmax(z) and H = -sum_k p_k log p_k:
        dH/dz_k = -p_k (log p_k + H).
    Chain rule through z = W x + b gives  gb = mean_i dH/dz,  gW = mean_i (dH/dz) x^T.
    """
    logp = forward_logprobs(X, W, b)
    p = np.exp(logp)
    h = -(p * logp).sum(1, keepdims=True)
    dz = -p * (logp + h) / X.shape[0]          # (N, K), already divided by N
    return float(h.mean()), dz.T @ X, dz.sum(0)


# ------------------------------------------------------------------------ dynamics
@dataclass(frozen=True)
class GDConfig:
    eta: float = 2.0
    steps: int = 2000
    adapt: Adapt = "b"          # adapt only the bias, or W and b together
    record_every: int = 20
    # Stop early when the gradient norm falls below this. Note: in COLLAPSED runs the
    # gradient only decays ~1/t, so they usually run the full budget (converged=False);
    # [observed] H(p_bar) is then already ~1e-3 or less, far below the collapse threshold.
    grad_tol: float = 1e-7


@dataclass
class Trajectory:
    step: np.ndarray
    mean_H: np.ndarray
    H_bar: np.ndarray
    top_frac: np.ndarray
    acc: np.ndarray
    b_gap: np.ndarray           # b_0 - mean(b_1..b_{K-1}): bias of class 0 relative to the rest
    converged: bool             # grad norm fell below tol before the step budget ran out
    W: np.ndarray
    b: np.ndarray


def run_gd(
    X: np.ndarray, y: np.ndarray, W0: np.ndarray, b0: np.ndarray, cfg: GDConfig
) -> Trajectory:
    """Full-batch gradient descent on the entropy loss. Returns a recorded trajectory."""
    W, b = W0.copy(), b0.copy()
    rec: dict[str, list[float]] = {k: [] for k in ("step", "mean_H", "H_bar", "top_frac", "acc", "b_gap")}
    converged = False

    def record(t: int) -> None:
        s = entropy_stats(forward_logprobs(X, W, b), y)
        rec["step"].append(t)
        for k in ("mean_H", "H_bar", "top_frac", "acc"):
            rec[k].append(s[k])
        rec["b_gap"].append(float(b[0] - b[1:].mean()))

    for t in range(cfg.steps + 1):
        _, gW, gb = entropy_loss_and_grad(X, W, b)
        if t % cfg.record_every == 0:
            record(t)
        gnorm = np.sqrt((gb**2).sum() + ((gW**2).sum() if cfg.adapt == "Wb" else 0.0))
        if gnorm < cfg.grad_tol:
            converged = True
            if rec["step"][-1] != t:
                record(t)
            break
        if t == cfg.steps:
            break
        b -= cfg.eta * gb
        if cfg.adapt == "Wb":
            W -= cfg.eta * gW
    return Trajectory(**{k: np.array(v) for k, v in rec.items()}, converged=converged, W=W, b=b)


# ----------------------------------------------------------------------- collapse
def is_collapsed(final_H_bar: float, K: int, tau: float = 0.1) -> bool:
    """[assumption] Collapse := final H(p_bar) < tau * log K (default tau = 0.1).

    H(p_bar) = log K is the balanced maximum; values near 0 mean one class holds
    almost all predicted mass. tau is a threshold we chose; the sweep script also
    logs the top-class fraction so the conclusion can be checked against tau.
    """
    return bool(final_H_bar < tau * np.log(K))


def run_with_bias(
    mix: Mixture, spec: MixtureSpec, delta: float, cfg: GDConfig
) -> Trajectory:
    """Start from the Bayes classifier, add `delta` to b_0, run entropy-minimisation GD."""
    W0, b0 = bayes_init(mix.means, spec.sigma)
    b0 = b0.copy()
    b0[0] += delta
    return run_gd(mix.X, mix.y, W0, b0, cfg)


def collapse_threshold(
    mix: Mixture,
    spec: MixtureSpec,
    cfg: GDConfig,
    delta_max: float,
    n_grid: int = 25,
    tau: float = 0.1,
    bisect_iters: int = 12,
) -> dict[str, float]:
    """Smallest initial bias delta* at which GD collapses (grid scan + bisection).

    [assumption] Collapse is monotone in delta. We do not assume it blindly: the
    returned `n_switches` counts how often the collapse indicator flips along the
    grid (1 = clean single threshold, 0 = never collapses, >1 = non-monotone).
    """
    grid = np.linspace(0.0, delta_max, n_grid)
    flags = np.array(
        [is_collapsed(run_with_bias(mix, spec, d, cfg).H_bar[-1], spec.K, tau) for d in grid]
    )
    n_switches = int(np.abs(np.diff(flags.astype(int))).sum())
    if not flags.any():
        return {"delta_star": float("nan"), "n_switches": n_switches, "status": "never_collapses"}
    first = int(np.argmax(flags))
    if first == 0:
        return {"delta_star": 0.0, "n_switches": n_switches, "status": "collapses_at_zero"}
    lo, hi = float(grid[first - 1]), float(grid[first])
    for _ in range(bisect_iters):
        mid = 0.5 * (lo + hi)
        if is_collapsed(run_with_bias(mix, spec, mid, cfg).H_bar[-1], spec.K, tau):
            hi = mid
        else:
            lo = mid
    return {"delta_star": 0.5 * (lo + hi), "n_switches": n_switches, "status": "ok"}
