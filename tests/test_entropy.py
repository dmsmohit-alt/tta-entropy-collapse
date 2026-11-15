"""Tests for the PyTorch entropy functions. Skipped automatically if torch is missing."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.metrics import entropy as E  # noqa: E402
from tests import reference as ref  # noqa: E402


def _logits(seed: int, n: int = 32, k: int = 10, scale: float = 3.0) -> "torch.Tensor":
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, k, generator=g, dtype=torch.float64) * scale


def test_uniform_entropy_is_log_k():
    k = 7
    logits = torch.zeros(4, k, dtype=torch.float64)
    assert torch.allclose(E.entropy_from_logits(logits), torch.full((4,), math.log(k), dtype=torch.float64))
    assert torch.isclose(E.marginal_entropy_from_logits(logits), torch.tensor(math.log(k), dtype=torch.float64))


def test_one_hot_entropy_is_zero_and_finite():
    logits = torch.tensor([[1000.0, 0.0, -1000.0]], dtype=torch.float32)
    h = E.entropy_from_logits(logits)
    assert torch.isfinite(h).all() and h.item() < 1e-6


@pytest.mark.parametrize("seed", range(5))
def test_matches_scipy_reference(seed):
    logits = _logits(seed)
    np_logits = logits.numpy()
    assert np.allclose(E.entropy_from_logits(logits).numpy(), ref.per_sample_entropy(np_logits), atol=1e-10)
    assert np.isclose(E.mean_entropy(logits).item(), ref.mean_entropy(np_logits), atol=1e-10)
    assert np.allclose(E.marginal_probs(logits).numpy(), ref.marginal(np_logits), atol=1e-10)
    assert np.isclose(E.marginal_entropy_from_logits(logits).item(), ref.marginal_entropy(np_logits), atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_decomposition_identity(seed):
    # E[H(p)] = H(p_bar) - I(X;Y), with I >= 0 (Jensen).
    logits = _logits(seed)
    mi = E.mutual_information(logits)
    assert mi.item() >= -1e-10
    assert torch.isclose(E.mean_entropy(logits), E.marginal_entropy_from_logits(logits) - mi)


def test_marginal_entropy_extremes():
    k = 5
    collapsed = torch.tile(torch.tensor([50.0, 0, 0, 0, 0], dtype=torch.float64), (8, 1))
    assert E.marginal_entropy_from_logits(collapsed).item() < 1e-9  # collapse: H(p_bar) -> 0
    diverse = 50.0 * torch.eye(k, dtype=torch.float64)[torch.arange(10) % k]
    assert math.isclose(E.marginal_entropy_from_logits(diverse).item(), math.log(k), abs_tol=1e-6)


def test_marginal_entropy_from_probs_handles_zeros():
    p = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)
    assert math.isclose(E.marginal_entropy_from_probs(p).item(), math.log(2), abs_tol=1e-12)


def test_info_max_loss_lambda_zero_is_tent_loss():
    logits = _logits(0)
    assert torch.isclose(E.info_max_loss(logits, 0.0), E.mean_entropy(logits))
    lam = 0.7
    expected = E.mean_entropy(logits) - lam * E.marginal_entropy_from_logits(logits)
    assert torch.isclose(E.info_max_loss(logits, lam), expected)


def test_gradcheck():
    # Analytic autograd gradients vs finite differences (float64).
    x = _logits(1, n=6, k=4, scale=1.0).requires_grad_(True)
    assert torch.autograd.gradcheck(E.mean_entropy, (x,))
    assert torch.autograd.gradcheck(E.marginal_entropy_from_logits, (x,))
    assert torch.autograd.gradcheck(lambda z: E.info_max_loss(z, 0.5), (x,))


def test_marginal_entropy_gradient_zero_at_uniform_marginal():
    # H(p_bar) is maximal at the uniform marginal, so its gradient wrt logits vanishes there.
    x = torch.zeros(6, 4, dtype=torch.float64, requires_grad=True)
    E.marginal_entropy_from_logits(x).backward()
    assert torch.allclose(x.grad, torch.zeros_like(x.grad), atol=1e-12)
