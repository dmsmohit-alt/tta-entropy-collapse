"""No-adaptation baselines.

Source   : the pre-trained model in eval mode (BN uses its stored training statistics). The reference
           for 'how bad is the shift'.
BNAdapt  : [optional extra] same weights, but BN uses the current batch's statistics; no parameter is
           updated. Separates the effect of 'normalise with target statistics' (free) from the effect
           of 'entropy gradient steps' (TENT).
Do not call .train()/.eval() on these wrappers; the mode of the inner model is fixed on construction.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models.bn import configure_model_for_tta
from src.tta.base import TTAMethod


class Source(TTAMethod):
    name = "source"

    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class BNAdapt(TTAMethod):
    name = "bn_adapt"

    def __init__(self, model: nn.Module) -> None:
        super().__init__(configure_model_for_tta(model))
        self.model.requires_grad_(False)                    # nothing is trained

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
