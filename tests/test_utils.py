"""Tests for config, seeding and logging utilities (no torch needed)."""
import csv
import json
import random

import numpy as np
import pytest
import yaml

from src.utils.config import (
    apply_overrides, config_hash, deep_merge, load_config, make_run_dir, save_config,
)
from src.utils.run_logger import RunLogger
from src.utils.seeding import get_rng, seed_everything


def test_default_config_loads_and_overrides(tmp_path):
    exp = tmp_path / "exp.yaml"
    exp.write_text(yaml.safe_dump({"experiment": {"name": "t"}, "stream": {"batch_size": 50}}))
    cfg = load_config(exp, overrides=["stream.num_cycles=3", "metrics.ema_decay=0.5"],
                      default_path="configs/default.yaml")
    assert cfg["experiment"]["name"] == "t"
    assert cfg["stream"]["batch_size"] == 50          # experiment file beats default
    assert cfg["stream"]["num_cycles"] == 3           # override beats both
    assert cfg["metrics"]["ema_decay"] == 0.5
    assert cfg["metrics"]["collapse_patience"] == 5   # untouched default survives


def test_override_typos_raise():
    cfg = {"stream": {"batch_size": 1}}
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["stream.batchsize=2"])
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["nope.x=1"])
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["no_equals_sign"])


def test_deep_merge_does_not_mutate_inputs():
    a, b = {"x": {"y": 1}}, {"x": {"z": 2}}
    out = deep_merge(a, b)
    assert out == {"x": {"y": 1, "z": 2}} and a == {"x": {"y": 1}}


def test_hash_stable_and_run_dir_saves_config(tmp_path):
    cfg = {"experiment": {"name": "demo", "results_dir": str(tmp_path)}, "a": 1}
    assert config_hash(cfg) == config_hash(dict(reversed(list(cfg.items()))))
    run_dir = make_run_dir(cfg, seed=2)
    assert run_dir.name.endswith("_seed2")
    assert yaml.safe_load((run_dir / "config.yaml").read_text()) == cfg
    assert save_config(cfg, tmp_path / "x").exists()


def test_seeding_reproducible():
    seed_everything(123, deterministic=False)
    a = (random.random(), np.random.rand())
    seed_everything(123, deterministic=False)
    b = (random.random(), np.random.rand())
    assert a == b
    assert get_rng(1).random() == get_rng(1).random()
    assert get_rng(1).random() != get_rng(2).random()


def test_seeding_torch_if_available():
    torch = pytest.importorskip("torch")
    seed_everything(7, deterministic=False)
    a = torch.rand(3)
    seed_everything(7, deterministic=False)
    assert torch.equal(a, torch.rand(3))


def test_run_logger_writes_csv_and_summary(tmp_path):
    with RunLogger(tmp_path / "run") as log:
        log.log_step(0, {"err": 0.5, "h_bar": 2.3})
        log.log_step(1, {"err": 0.4, "h_bar": 2.2})
        with pytest.raises(KeyError):               # column mismatch is rejected
            log.log_step(2, {"err": 0.3})
        path = log.write_summary({"final_acc": np.float64(0.6)})
    rows = list(csv.DictReader(open(tmp_path / "run" / "metrics.csv")))
    assert [r["step"] for r in rows] == ["0", "1"]
    assert json.loads(path.read_text())["final_acc"] == 0.6
