"""Per-batch stream metrics and the collapse detector (pure numpy; no torch needed).

Definitions used everywhere in the project (all entropies in nats):
    online_err   : fraction of the batch predicted wrongly (prediction = argmax of the logits the
                   method returned for that batch, i.e. BEFORE that batch's own update).
    mean_H       : E_x[H(p(.|x))] over the batch.
    H_bar_batch  : H(p_bar) where p_bar = mean of the batch's softmax outputs.
    H_bar_ema    : H of an exponential moving average of p_bar over batches (smoother signal).
    top_frac     : fraction of the batch whose predicted class is the batch's most-predicted class.
    top_class    : which class that is.

Collapse (precise definition, [assumption] chosen by us): a run has collapsed at the first batch `s`
such that top_frac > `top_frac_threshold` for `patience` CONSECUTIVE batches s, s+1, ..., s+patience-1.
We report `start_step = s` (where the streak began) and `detect_step = s + patience - 1` (when an online
detector could have fired). Caveat: on a class-imbalanced stream the true top-class fraction can already
be high, so top_frac must then be compared with the label-based top fraction (Phase 5).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import log_softmax, logsumexp


def batch_metrics(logits: np.ndarray, labels: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    """Metrics for one batch. logits (N, K), labels (N,). Returns (metrics, p_bar of shape (K,))."""
    z = np.asarray(logits, dtype=np.float64)
    n, k = z.shape
    logp = log_softmax(z, axis=1)                       # stable: never log(softmax)
    p = np.exp(logp)
    ent = -(p * logp).sum(axis=1)
    log_pbar = logsumexp(logp, axis=0) - np.log(n)
    pbar = np.exp(log_pbar)
    pred = z.argmax(axis=1)
    counts = np.bincount(pred, minlength=k)
    metrics = {
        "online_err": float((pred != np.asarray(labels)).mean()),
        "mean_H": float(ent.mean()),
        "H_bar_batch": float(-(pbar * log_pbar).sum()),
        "top_frac": float(counts.max() / n),
        "top_class": int(counts.argmax()),
    }
    return metrics, pbar


def entropy_of(pbar: np.ndarray) -> float:
    """H(p) of a probability vector (0 log 0 = 0)."""
    p = np.asarray(pbar, dtype=np.float64)
    nz = p > 0
    return float(-(p[nz] * np.log(p[nz])).sum())


@dataclass
class EmaMarginal:
    """EMA of the marginal vector p_bar over batches; `update` returns H of the smoothed marginal.

    First call initialises the EMA to that batch's p_bar. decay=0 means 'no smoothing'.
    """

    decay: float = 0.9
    value: np.ndarray | None = field(default=None, repr=False)

    def update(self, pbar: np.ndarray) -> float:
        pbar = np.asarray(pbar, dtype=np.float64)
        self.value = pbar.copy() if self.value is None else self.decay * self.value + (1 - self.decay) * pbar
        return entropy_of(self.value)


@dataclass
class CollapseDetector:
    """First streak of `patience` consecutive batches with top_frac > threshold (see module docstring)."""

    top_frac_threshold: float = 0.9
    patience: int = 5
    start_step: int | None = None
    detect_step: int | None = None
    _streak: int = 0
    _streak_start: int | None = None

    def update(self, step: int, top_frac: float) -> bool:
        """Feed one batch (steps must be fed in stream order). Returns True once collapse is detected."""
        if self.detect_step is not None:
            return True                                  # keep the FIRST detection
        if top_frac > self.top_frac_threshold:
            if self._streak == 0:
                self._streak_start = step
            self._streak += 1
            if self._streak >= self.patience:
                self.start_step, self.detect_step = self._streak_start, step
                return True
        else:
            self._streak, self._streak_start = 0, None
        return False

    @property
    def collapsed(self) -> bool:
        return self.detect_step is not None
