"""Hessian-extreme estimator vs known Hessians (needs torch; NOT run by the author's sandbox: run pytest on your machine)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from src.tta.curvature import hessian_extremes  # noqa: E402


def test_psd_quadratic_matches_eigvalsh():
    torch.manual_seed(0)
    lin = nn.Linear(4, 1, bias=False)
    x = torch.randn(30, 4)
    H = (x.T @ x / 30).numpy()                         # loss = 0.5 * mean((x w)^2)  ->  Hessian = X^T X / n
    ev = np.linalg.eigvalsh(H)
    e = hessian_extremes(lin, lambda out: 0.5 * (out ** 2).mean(), x, n_iter=500, tol=1e-9)
    assert e.lam_max == pytest.approx(ev[-1], rel=1e-3)
    assert e.lam_min == pytest.approx(ev[0], rel=1e-2)


class _Saddle(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.tensor([0.3, -0.2]))

    def forward(self, x):                              # out = (p0*p1 + 0.5*p0^2) * x ; Hessian [[1,1],[1,0]]
        return (self.p[0] * self.p[1] + 0.5 * self.p[0] ** 2) * x


def test_indefinite_hessian_gets_both_extremes():
    e = hessian_extremes(_Saddle(), lambda out: out.sum(), torch.ones(1), n_iter=500, tol=1e-10)
    assert e.lam_max == pytest.approx((1 + 5 ** 0.5) / 2, rel=1e-3)
    assert e.lam_min == pytest.approx((1 - 5 ** 0.5) / 2, rel=1e-3)
