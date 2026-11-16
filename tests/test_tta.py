"""Torch tests for BN selection and the Source / BNAdapt / TENT methods (tiny random network; CPU is fine).

These need PyTorch, so they are skipped where torch is missing. They could NOT be run by the author's
sandbox: run `pytest tests/test_tta.py -q` on a machine with torch.
"""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from src.metrics.entropy import mean_entropy  # noqa: E402
from src.models.bn import collect_bn_params, configure_model_for_tta, count_trainable  # noqa: E402
from src.models.resnet_cifar import ResNet18CIFAR  # noqa: E402
from src.tta import BNAdapt, Source, Tent, build_method  # noqa: E402


def _tiny(seed=0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1),
                         nn.BatchNorm2d(8), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 5)).eval()


def _batch(seed=1, n=32):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, 8, 8, generator=g)


def _snapshot(m: nn.Module):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def test_configure_model_trains_only_bn_affine():
    m = configure_model_for_tta(_tiny())
    params, names = collect_bn_params(m)
    assert len(params) == 4 and all(n.endswith(("weight", "bias")) for n in names)
    assert count_trainable(m) == 4 * 8
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable == set(names)
    bn = m[1]
    assert bn.running_mean is None and bn.running_var is None and m.training


def test_batch_statistics_are_used():
    m = configure_model_for_tta(_tiny())
    x = _batch()
    with torch.no_grad():
        full, flipped, half = m(x), m(x.flip(0)), m(x[:16])
    assert torch.allclose(full.flip(0), flipped, atol=1e-5)     # batch-norm over a batch ignores sample order ...
    assert not torch.allclose(full[:16], half, atol=1e-5)       # ... but depends on WHICH samples are in the batch


def test_source_is_unchanged_and_deterministic():
    m = _tiny()
    s = Source(copy.deepcopy(m))
    before = _snapshot(s.model)
    x = _batch()
    a, b = s(x), s(x)
    assert torch.equal(a, b) and not s.model.training
    assert all(torch.equal(before[k], v) for k, v in _snapshot(s.model).items())
    assert not any(p.requires_grad for p in s.model.parameters())
    assert torch.allclose(a, m(x), atol=1e-6)                   # equals plain eval-mode forward


def test_bn_adapt_changes_nothing_but_uses_batch_stats():
    m = _tiny()
    t = BNAdapt(copy.deepcopy(m))
    before = _snapshot(t.model)
    x = _batch() * 3 + 1                                        # shifted input: eval-BN and batch-BN must differ
    out = t(x)
    assert all(torch.equal(before[k], v) for k, v in _snapshot(t.model).items())
    assert not torch.allclose(out, m(x), atol=1e-4)


def test_tent_updates_only_bn_affine_and_returns_pre_update_logits():
    m = _tiny()
    t = Tent(copy.deepcopy(m), lr=1e-2, optimizer="sgd")
    before = _snapshot(t.model)
    x = _batch()
    with torch.no_grad():
        expected = copy.deepcopy(t.model)(x)                    # forward of the not-yet-updated model
    out = t(x)
    assert torch.allclose(out, expected, atol=1e-6)             # returned logits are PRE-update
    after = _snapshot(t.model)
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    bn_keys = {"1.weight", "1.bias", "4.weight", "4.bias"}
    assert changed and changed <= bn_keys
    assert t.n_adapted_params == 32


def test_tent_step_decreases_entropy_of_the_same_batch():
    t = Tent(_tiny(), lr=1e-2, optimizer="sgd")
    x = _batch()
    with torch.no_grad():
        h0 = mean_entropy(t.model(x)).item()
    t(x)
    with torch.no_grad():
        h1 = mean_entropy(t.model(x)).item()
    assert h1 < h0


def test_tent_reset_restores_initial_state():
    t = Tent(_tiny(), lr=1e-1, optimizer="adam")
    s0 = _snapshot(t.model)
    for seed in range(3):
        t(_batch(seed))
    assert any(not torch.equal(s0[k], v) for k, v in _snapshot(t.model).items())
    t.reset()
    assert all(torch.equal(s0[k], v) for k, v in _snapshot(t.model).items())
    assert len(t.optimizer.state_dict()["state"]) == 0          # optimiser moments cleared


def test_tent_with_custom_loss_fn_and_factory():
    calls = []
    t = Tent(_tiny(), lr=1e-2, optimizer="sgd", loss_fn=lambda z: (calls.append(1), mean_entropy(z))[1])
    t(_batch())
    assert calls == [1]
    assert isinstance(build_method("tent", _tiny(), {"lr": 5e-4}), Tent)
    assert isinstance(build_method("source", _tiny(), {}), Source)
    with pytest.raises(ValueError):
        build_method("nope", _tiny(), {})


def test_resnet18_cifar_shapes_params_and_bn_selection():
    m = ResNet18CIFAR().eval()
    x = torch.rand(4, 3, 32, 32)
    assert m(x).shape == (4, 10)
    n_params = sum(p.numel() for p in m.parameters())
    assert 11_000_000 < n_params < 11_400_000                   # about 11.2M for CIFAR ResNet-18
    configure_model_for_tta(m)
    params, names = collect_bn_params(m)
    assert len(params) == 2 * sum(isinstance(x_, nn.BatchNorm2d) for x_ in m.modules())
    assert count_trainable(m) == sum(p.numel() for p in params) and count_trainable(m) < 30_000
    assert np.isfinite(m(x).detach().numpy()).all()
