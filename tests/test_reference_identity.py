"""Numpy-only checks of the math (run even when PyTorch is not installed)."""
import numpy as np
import pytest

from tests import reference as ref


@pytest.mark.parametrize("seed", range(5))
def test_decomposition_identity_and_nonnegative_mi(seed):
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=3.0, size=(64, 10))
    h_bar = ref.marginal_entropy(logits)
    h_cond = ref.mean_entropy(logits)
    mi = ref.mutual_information(logits)
    # E[H(p)] = H(p_bar) - I   (I is defined as the difference, so also check I >= 0, Jensen)
    assert np.isclose(h_cond, h_bar - mi)
    assert mi >= -1e-12
    assert h_bar <= np.log(10) + 1e-12  # entropy is at most log K


def test_uniform_and_collapsed_cases():
    k = 5
    uniform = np.zeros((8, k))
    assert np.isclose(ref.mean_entropy(uniform), np.log(k))
    assert np.isclose(ref.marginal_entropy(uniform), np.log(k))
    collapsed = np.tile(np.array([50.0, 0, 0, 0, 0]), (8, 1))  # everyone -> class 0
    assert ref.marginal_entropy(collapsed) < 1e-9
    # Confident but diverse predictions: H(p_bar)=log K while E[H(p)]~0 (max information)
    diverse = 50.0 * np.eye(k)[np.arange(10) % k]
    assert np.isclose(ref.marginal_entropy(diverse), np.log(k), atol=1e-6)
    assert ref.mean_entropy(diverse) < 1e-6
