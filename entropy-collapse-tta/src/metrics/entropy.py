"""Entropy quantities used throughout the project (PyTorch).

Notation (matches the problem statement):
    p_theta(y|x) = softmax(logits)          per-sample prediction
    H(p)         = -sum_y p_y log p_y       prediction entropy (what TENT minimises)
    p_bar        = E_x[p_theta(.|x)]        marginal class distribution
    H(p_bar)     = entropy of the marginal  (collapse <=> H(p_bar) -> 0)
    I(X;Y)       = H(p_bar) - E_x[H(p)]     mutual information under the model

Identity checked in tests:  E[H(p)] = H(p_bar) - I(X;Y)   (so I = H(p_bar) - E[H(p)] >= 0).

Numerics: everything is computed from log-probabilities (log_softmax /
logsumexp), never from log(softmax(.)), so extreme logits give no NaN/inf.
Convention: 0 * log 0 = 0. Entropies are in nats (natural log).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def entropy_from_logits(logits: Tensor) -> Tensor:
    """Per-sample entropy H(softmax(logits)), shape (N, K) -> (N,)."""
    log_p = torch.log_softmax(logits, dim=-1)
    # xlogy-style safety: p * log p where p == 0 must be 0, not nan. With
    # log_softmax, log_p is finite (very negative) so exp(log_p)*log_p -> 0 fine.
    return -(log_p.exp() * log_p).sum(dim=-1)


def mean_entropy(logits: Tensor) -> Tensor:
    """E_x[H(p_theta(.|x))] over the batch: the TENT loss. (N, K) -> scalar."""
    return entropy_from_logits(logits).mean()


def marginal_log_probs(logits: Tensor) -> Tensor:
    """log p_bar, where p_bar = mean over the batch of softmax(logits). (N, K) -> (K,)."""
    n = logits.shape[0]
    log_p = torch.log_softmax(logits, dim=-1)
    # log(mean_i p_i) = logsumexp_i(log p_i) - log N   (stable)
    return torch.logsumexp(log_p, dim=0) - math.log(n)


def marginal_probs(logits: Tensor) -> Tensor:
    """p_bar = (1/N) sum_i softmax(logits_i). (N, K) -> (K,)."""
    return marginal_log_probs(logits).exp()


def marginal_entropy_from_logits(logits: Tensor) -> Tensor:
    """H(p_bar) for a batch of logits. (N, K) -> scalar.

    This is the term the 'information-maximisation' fix ADDS (with weight lambda):
        loss = mean_entropy - lambda * marginal_entropy
    """
    return marginal_entropy_from_log_probs(marginal_log_probs(logits))


def marginal_entropy_from_log_probs(log_p_bar: Tensor) -> Tensor:
    """H(p_bar) given log p_bar, shape (K,) -> scalar."""
    return -(log_p_bar.exp() * log_p_bar).sum()


def marginal_entropy_from_probs(p_bar: Tensor) -> Tensor:
    """H(p_bar) given a probability vector (e.g. an EMA of p_bar), shape (K,) -> scalar.

    Uses xlogy so that entries equal to 0 contribute exactly 0.
    """
    return -torch.special.xlogy(p_bar, p_bar).sum()


def mutual_information(logits: Tensor) -> Tensor:
    """I(X;Y_theta) = H(p_bar) - E[H(p)] estimated on a batch. Always >= 0 (up to fp error)."""
    return marginal_entropy_from_logits(logits) - mean_entropy(logits)


def info_max_loss(logits: Tensor, lam: float) -> Tensor:
    """Information-maximisation loss: E[H(p)] - lam * H(p_bar)  (SHOT-style fix).

    lam = 0 recovers plain TENT. The loss is a function of the same batch of
    logits, so the marginal term costs essentially nothing extra.
    """
    return mean_entropy(logits) - lam * marginal_entropy_from_logits(logits)
