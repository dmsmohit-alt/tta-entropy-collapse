"""BatchNorm handling for test-time adaptation.

TENT's recipe (Wang et al.): (1) normalise with the CURRENT test batch's statistics instead of the stored
running statistics, and (2) update only the BN affine parameters (gamma = weight, beta = bias) by
gradient descent on the prediction entropy. Everything else stays frozen.
"""
from __future__ import annotations

from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm


def configure_model_for_tta(model: nn.Module) -> nn.Module:
    """In place: BN uses batch statistics, only BN affine params require grad, all other params frozen.

    How it works (for the learning notes): a BatchNorm layer whose `running_mean`/`running_var` are None
    always normalises with the current batch's mean/variance, even in eval mode. We also call
    `model.train()` so the layers are in training mode (there is no dropout in these models).
    """
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def collect_bn_params(model: nn.Module) -> tuple[list[Tensor], list[str]]:
    """The BN affine parameters (weight, bias) and their names."""
    params: list[Tensor] = []
    names: list[str] = []
    for mod_name, m in model.named_modules():
        if isinstance(m, _BatchNorm):
            for p_name, p in m.named_parameters(recurse=False):
                if p_name in ("weight", "bias"):
                    params.append(p)
                    names.append(f"{mod_name}.{p_name}")
    if not params:
        raise ValueError("Model has no BatchNorm affine parameters to adapt.")
    return params, names


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
