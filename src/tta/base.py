"""Common interface for test-time adaptation methods.

Calling a method on a batch `x` (float [0,1], NCHW) returns the logits USED FOR PREDICTION, and (for
adapting methods) also updates the model. Convention (as in the original TENT code): the returned logits
come from the forward pass BEFORE this batch's own update. So the online error of batch t reflects
adaptation on batches < t only; this keeps the evaluation honest (no peeking at its own update).
"""
from __future__ import annotations

from torch import Tensor, nn

from src.models.bn import count_trainable


class TTAMethod(nn.Module):
    name: str = "base"

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self) -> None:
        """Restore the initial (source) model state. Used by episodic / periodic-reset variants."""

    @property
    def n_adapted_params(self) -> int:
        return count_trainable(self.model)
