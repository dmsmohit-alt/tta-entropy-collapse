"""Local stability of the balanced point on a finite sample (bias-only adaptation, W frozen).

We find the stationary point b* of J near b = 0, compute the Hessian there three ways
(analytic formula / central finite differences / PyTorch autograd if installed), and
compare the predicted step-size limit eta* with a direct nonlinear GD experiment.

Gauge direction: J(b + c*1) = J(b) (softmax is shift invariant), so the Hessian always has
a zero eigenvalue along 1. That direction is neutral by construction, not an instability,
so all stability statements are about the (K-1)-dim subspace 1-perp.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import log_softmax

from src.theory.dynamics import bias_flow_from_logp, hessian_from_logp, infomax_value


@dataclass
class SampleProblem:
    """J(b) = mean H - lam * H(p_bar) on a fixed sample with frozen logits Z0 = X W^T."""

    Z0: np.ndarray
    lam: float

    @property
    def K(self) -> int:
        return self.Z0.shape[1]

    def logp(self, b: np.ndarray) -> np.ndarray:
        return log_softmax(self.Z0 + b, axis=1)

    def value(self, b: np.ndarray) -> float:
        return infomax_value(self.logp(b), self.lam)

    def grad(self, b: np.ndarray) -> np.ndarray:
        return -bias_flow_from_logp(self.logp(b), self.lam)

    def hess(self, b: np.ndarray) -> np.ndarray:
        return hessian_from_logp(self.logp(b), self.lam)


def perp_basis(k: int) -> np.ndarray:
    """Orthonormal basis (K, K-1) of the subspace orthogonal to the all-ones vector."""
    q, _ = np.linalg.qr(np.vstack([np.ones((1, k)), np.eye(k)[: k - 1]]).T)
    return q[:, 1:]


def find_stationary(prob: SampleProblem, b0: np.ndarray | None = None, iters: int = 300,
                    tol: float = 1e-12, max_step: float = 0.5) -> tuple[np.ndarray, float]:
    """Step-capped Newton iterations from b0 (default 0); pseudo-inverse handles the gauge zero mode.

    The cap matters: when the balanced point is unstable, a full Newton step can jump to a far-away
    flat (collapsed) stationary point where the Hessian is ~0, which says nothing about the balanced
    point. Callers should still check `|b_gap|` of the returned point (see `run_stability.py`).
    Returns (b*, |grad|).
    """
    b = np.zeros(prob.K) if b0 is None else b0.copy()
    for _ in range(iters):
        g = prob.grad(b)
        if np.linalg.norm(g) < tol:
            break
        step = np.linalg.pinv(prob.hess(b), rcond=1e-9) @ g
        n = np.linalg.norm(step)
        if n > max_step:
            step *= max_step / n
        b = b - step
        b -= b.mean()                                         # fix the gauge
    return b, float(np.linalg.norm(prob.grad(b)))


def fd_hessian(prob: SampleProblem, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central finite differences of the analytic gradient."""
    cols = [(prob.grad(b + eps * e) - prob.grad(b - eps * e)) / (2 * eps) for e in np.eye(prob.K)]
    h = np.array(cols).T
    return 0.5 * (h + h.T)


def torch_hessian(prob: SampleProblem, b: np.ndarray) -> np.ndarray | None:
    """Hessian by PyTorch autograd, reusing the Phase-0 `info_max_loss`. None if no torch."""
    try:
        import torch
    except ImportError:
        return None
    from src.metrics.entropy import info_max_loss

    z0 = torch.tensor(prob.Z0, dtype=torch.float64)
    f = lambda bb: info_max_loss(z0 + bb, prob.lam)  # noqa: E731
    return torch.autograd.functional.hessian(f, torch.tensor(b, dtype=torch.float64)).numpy()


def perp_eigs(h: np.ndarray) -> np.ndarray:
    """Eigenvalues of the Hessian restricted to 1-perp, ascending."""
    q = perp_basis(h.shape[0])
    return np.linalg.eigvalsh(q.T @ h @ q)


def map_spectral_radius(h: np.ndarray, eta: float) -> float:
    """max |eig| of the GD-map Jacobian I - eta*H on 1-perp (stable iff < 1)."""
    return float(np.abs(1.0 - eta * perp_eigs(h)).max())


def gd_growth_rate(prob: SampleProblem, b_star: np.ndarray, eta: float, seed: int = 0,
                   eps: float = 1e-6, max_steps: int = 6000, floor: float = 1e-9,
                   escape: float = 1e-3) -> float:
    """Asymptotic per-step growth factor of a small perturbation under NONLINEAR plain GD.

    Returns 0.0 if the perturbation decays below `floor` (stable), inf if it grows past
    `escape` (unstable), else the measured geometric rate over the second half of the run.
    `floor` must sit well above the accuracy of b_star (~1e-12): otherwise the perturbation
    stalls at the solver's noise level and the measured rate is ~1.000000 (a bug we hit once).
    Independent of the Jacobian: uses only the gradient function and GD steps.
    """
    q = perp_basis(prob.K)
    v = np.random.default_rng(seed).normal(size=q.shape[1])
    b = b_star + eps * (q @ (v / np.linalg.norm(v)))
    half = None
    for t in range(max_steps):
        b = b - eta * prob.grad(b)
        d = float(np.linalg.norm(q.T @ (b - b_star)))
        if d > escape:
            return float("inf")
        if d < floor:
            return 0.0
        if t == max_steps // 2:
            half = d
    return float((d / half) ** (1.0 / (max_steps - max_steps // 2 - 1)))


def eta_star_numeric(prob: SampleProblem, b_star: np.ndarray, eta_pred: float,
                     rel_tol: float = 1e-3, max_iter: int = 40) -> float:
    """Largest stable step size found by bisection on `gd_growth_rate` (NaN if not bracketed)."""
    lo, hi = 0.2 * eta_pred, 5.0 * eta_pred
    stable = lambda e: gd_growth_rate(prob, b_star, e) < 1.0 - 1e-6  # noqa: E731
    if not (stable(lo) and not stable(hi)):
        return float("nan")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if stable(mid):
            lo = mid
        else:
            hi = mid
        if (hi - lo) / hi < rel_tol:
            break
    return 0.5 * (lo + hi)
