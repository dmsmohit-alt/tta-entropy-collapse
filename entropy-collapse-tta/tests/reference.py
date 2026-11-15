"""Independent float64 numpy/scipy reference for entropy quantities.

Used as an oracle: the PyTorch implementation must agree with this, and the
decomposition E[H(p)] = H(p_bar) - I(X;Y) is checked here without needing torch.
"""
from __future__ import annotations

import numpy as np
from scipy.special import softmax
from scipy.stats import entropy as scipy_entropy


def probs(logits: np.ndarray) -> np.ndarray:
    return softmax(logits.astype(np.float64), axis=-1)


def per_sample_entropy(logits: np.ndarray) -> np.ndarray:
    return scipy_entropy(probs(logits), axis=-1)  # nats


def marginal(logits: np.ndarray) -> np.ndarray:
    return probs(logits).mean(axis=0)


def marginal_entropy(logits: np.ndarray) -> float:
    return float(scipy_entropy(marginal(logits)))


def mean_entropy(logits: np.ndarray) -> float:
    return float(per_sample_entropy(logits).mean())


def mutual_information(logits: np.ndarray) -> float:
    return marginal_entropy(logits) - mean_entropy(logits)
