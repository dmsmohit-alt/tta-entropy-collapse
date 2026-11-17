"""Periodic reset wrapper (RDumb-style baseline; Press et al., 2023, arXiv 2306.05401).

Verified from the paper's page: RDumb resets the model to the pre-trained weights every T steps (T=1000 there,
chosen on ImageNet-C holdout noises, batch 64, ETA-style loss, SGD lr 2.5e-4). The paper gives no CIFAR-10-C
settings, so OUR baseline is "RDumb-style": any inner TENT-type method + a fixed-period reset. It is not a
reproduction of the paper's numbers.

Design choices (ours): the reset happens BEFORE the batch with index t (t > 0, t % T == 0) is processed, so
that batch is predicted by the source weights; the inner optimiser state (Adam moments) is reset as well.
"""
from __future__ import annotations

from torch import Tensor

from src.tta.base import TTAMethod


class PeriodicReset(TTAMethod):
    name = "periodic_reset"

    def __init__(self, inner: TTAMethod, period: int) -> None:
        if period < 1:
            raise ValueError(f"reset period must be >= 1, got {period}")
        super().__init__(inner.model)
        self.inner = inner
        self.period = int(period)
        self.t = 0                 # number of batches seen so far
        self.n_resets = 0

    def forward(self, x: Tensor) -> Tensor:
        if self.t > 0 and self.t % self.period == 0:
            self.inner.reset()
            self.n_resets += 1
        self.t += 1
        return self.inner(x)

    def reset(self) -> None:
        self.inner.reset()
        self.t = 0
        self.n_resets = 0
