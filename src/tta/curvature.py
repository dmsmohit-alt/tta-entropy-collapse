"""Extreme Hessian eigenvalues of a TTA loss w.r.t. the BN affine parameters (power iteration + Hessian-vector products).

Why: the toy theory (THEORY.md) says plain gradient descent with step eta is locally stable iff eta < 2/lambda_max
(and needs lambda_min > 0 at the point of interest). For a real network we do not know lambda_max, so we MEASURE it
and pick eta = c * 2/lambda_max for the SGD arm of Phase 4.

What this is and is not:
- Hessian of ONE batch's loss at the given parameters (BN in batch-statistics mode, so the coupling through the batch
  mean/var is included, via double backward). Not the population Hessian, and only at the parameters you pass in.
- Power iteration returns the eigenvalue of largest MAGNITUDE; we get the other extreme by a second pass on H - s*I.
  It is an estimate: `converged` says whether successive Rayleigh quotients agreed to `tol` (relative).
- Real networks are piecewise-linear (ReLU): this is the Hessian of the smooth almost-everywhere function.
- eta* = 2/lambda_max applies to plain GD (momentum 0). It does NOT directly apply to Adam.
- KNOWN FLAW of `hessian_extremes` (power iteration): if |lambda_min| >> the rest of the spectrum, the shifted pass stops after ~2 steps
  on the spectral bulk, reports lambda_max ~ 0 and still says 'converged'. Use `hessian_extremes_lanczos` (default in measure_curvature.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn


@dataclass
class Extremes:
    lam_max: float          # largest algebraic eigenvalue (estimate)
    lam_min: float          # smallest algebraic eigenvalue (estimate)
    converged_max: bool
    converged_min: bool
    n_hvp: int


def _hvp_fn(loss: Tensor, params: list[Tensor]) -> Callable[[list[Tensor]], list[Tensor]]:
    grads = torch.autograd.grad(loss, params, create_graph=True)

    def hvp(v: list[Tensor]) -> list[Tensor]:
        return list(torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True))

    return hvp


def _dot(a: list[Tensor], b: list[Tensor]) -> Tensor:
    return sum((x * y).sum() for x, y in zip(a, b))


def _power(apply: Callable[[list[Tensor]], list[Tensor]], params: list[Tensor], n_iter: int, tol: float,
           seed: int) -> tuple[float, bool, int]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = [torch.randn(p.shape, generator=g).to(p.device) for p in params]
    nrm = _dot(v, v).sqrt()
    v = [x / nrm for x in v]
    prev, eig, n = None, 0.0, 0
    for _ in range(n_iter):
        hv = apply(v)
        n += 1
        eig = float(_dot(v, hv))
        nrm = _dot(hv, hv).sqrt()
        if float(nrm) == 0.0:
            return 0.0, True, n
        v = [x / nrm for x in hv]
        if prev is not None and abs(eig - prev) <= tol * max(abs(eig), 1e-12):
            return eig, True, n
        prev = eig
    return eig, False, n


def hessian_extremes(model: nn.Module, loss_fn: Callable[[Tensor], Tensor], x: Tensor, n_iter: int = 50,
                     tol: float = 1e-3, seed: int = 0) -> Extremes:
    """lambda_max / lambda_min of d^2 loss_fn(model(x)) / d(trainable params)^2 (see module docstring)."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("model has no trainable parameters")
    loss = loss_fn(model(x))
    hvp = _hvp_fn(loss, params)
    dom, ok1, n1 = _power(hvp, params, n_iter, tol, seed)
    # second pass on (H - dom*I): its dominant eigenvalue is the opposite extreme (shifted)
    shifted = lambda v: [h - dom * vi for h, vi in zip(hvp(v), v)]  # noqa: E731
    other_shifted, ok2, n2 = _power(shifted, params, n_iter, tol, seed + 1)
    other = other_shifted + dom
    if dom >= other:
        return Extremes(dom, other, ok1, ok2, n1 + n2)
    return Extremes(other, dom, ok2, ok1, n1 + n2)


def hessian_extremes_lanczos(model: nn.Module, loss_fn: Callable[[Tensor], Tensor], x: Tensor, ncv: int = 60,
                             tol: float = 1e-4, maxiter: int = 300, seed: int = 0) -> Extremes:
    """Same quantities as `hessian_extremes`, via Lanczos (src/utils/lanczos.py). PREFER THIS ONE (see that module)."""
    import numpy as np

    from src.utils.lanczos import lanczos_extremes

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("model has no trainable parameters")
    hvp = _hvp_fn(loss_fn(model(x)), params)
    shapes, sizes = [p.shape for p in params], [p.numel() for p in params]
    dev, dt = params[0].device, params[0].dtype

    def matvec(v: "np.ndarray") -> "np.ndarray":
        t = torch.from_numpy(np.ascontiguousarray(v)).to(device=dev, dtype=dt)
        parts = [c.reshape(sh) for c, sh in zip(torch.split(t, sizes), shapes)]
        return torch.cat([h.reshape(-1) for h in hvp(parts)]).detach().double().cpu().numpy()

    r = lanczos_extremes(matvec, sum(sizes), ncv=ncv, tol=tol, maxiter=maxiter, seed=seed)
    return Extremes(r["lam_max"], r["lam_min"], r["converged_max"], r["converged_min"], r["n_matvec"])
