"""TENT: test-time entropy minimisation of the BatchNorm affine parameters (Wang et al., ICLR 2021).

Per batch: forward with batch statistics -> loss = mean prediction entropy -> one optimiser step on the BN
gamma/beta. The loss function is injectable (`loss_fn`) so that the marginal-entropy variant of Phase 4 can
reuse this class; the default is the plain entropy used by TENT.
"""
from __future__ import annotations

import copy
from typing import Callable

import torch
from torch import Tensor, nn

from src.metrics.entropy import mean_entropy
from src.models.bn import collect_bn_params, configure_model_for_tta
from src.tta.base import TTAMethod


def make_optimizer(params: list[Tensor], name: str, lr: float, betas: tuple[float, float], weight_decay: float,
                   momentum: float = 0.9) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"optimizer must be 'adam' or 'sgd', got {name!r}")


class Tent(TTAMethod):
    name = "tent"

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        steps: int = 1,
        optimizer: str = "adam",
        loss_fn: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__(configure_model_for_tta(model))
        params, self.param_names = collect_bn_params(self.model)
        self.optimizer = make_optimizer(params, optimizer, lr, tuple(betas), weight_decay)
        self.loss_fn = loss_fn or mean_entropy
        self.steps = steps
        # snapshot of the initial state, for reset()
        self._model_state0 = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self._opt_state0 = self._clone_opt_state()

    def _clone_opt_state(self) -> dict:
        return copy.deepcopy(self.optimizer.state_dict())

    def forward(self, x: Tensor) -> Tensor:
        """Adapt on `x` and return the logits of the first (pre-update) forward pass, detached."""
        first: Tensor | None = None
        for _ in range(self.steps):
            logits = self.model(x)
            if first is None:
                first = logits.detach()
            loss = self.loss_fn(logits)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        assert first is not None
        return first

    def reset(self) -> None:
        self.model.load_state_dict(self._model_state0)
        self.optimizer.load_state_dict(copy.deepcopy(self._opt_state0))
