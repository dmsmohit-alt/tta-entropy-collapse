"""degradation_metrics and compare_runs_paired on synthetic run folders (numpy/pandas only)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


deg = _load("degradation_metrics")
pair = _load("compare_runs_paired")


def _mk(root, method, h, seed, errs, num_cycles=None, tta=None):
    d = root / method / f"{h}_seed{seed}"
    d.mkdir(parents=True)
    cfg = {"stream": {"batch_size": 200, "severity": 5, "num_cycles": num_cycles or len(errs)}, "compute": {"input_layout": "nchw"},
           "model": {"source": "checkpoint"}, "data": {"root": "x"}, "tta": tta or {}}
    (d / "config.yaml").write_text(yaml.safe_dump(cfg))
    (d / "summary.json").write_text(json.dumps({"seed": seed, "overall_err": sum(errs) / len(errs), "err_last_cycle": errs[-1],
                                                "err_by_cycle": errs}))


def test_judge_exact_values():
    ref = {0: [0.2, 0.2, 0.2, 0.2], 1: [0.2, 0.2, 0.2, 0.2]}
    m = {0: [0.15, 0.2, 0.3, 0.4], 1: [0.15, 0.2, 0.2, 0.4]}     # seed-mean excess: -.05, 0, .05, .2
    r = deg.judge(m, ref, 0.01)
    assert r["onset"] == 2 and r["sustained"] is True and r["n_degraded"] == 2
    assert r["per_seed_onset"] == [2, 3]
    assert deg.judge(m, ref, 0.5)["onset"] is None


def test_not_sustained_when_it_recovers():
    ref = {0: [0.2] * 4}
    r = deg.judge({0: [0.3, 0.2, 0.2, 0.3]}, ref, 0.01)
    assert r["onset"] == 0 and r["sustained"] is False


def test_reference_match_ignores_num_cycles(tmp_path):
    for s in (0, 1):
        _mk(tmp_path, "bn_adapt", "r", s, [0.226] * 4)
        _mk(tmp_path, "source", "q", s, [0.447] * 4)
        _mk(tmp_path, "tent", "t", s, [0.2, 0.226, 0.3, 0.5], tta={"optimizer": "adam", "lr": 0.001})
    groups = deg.load_groups(tmp_path)
    assert groups[("tent", "t")]["ctx"] == groups[("bn_adapt", "r")]["ctx"]
    _mk(tmp_path, "tent_long", "u", 0, [0.2] * 8, num_cycles=8)      # longer stream, same ctx: cycles beyond the reference are not judged
    g = deg.load_groups(tmp_path)
    assert g[("tent_long", "u")]["ctx"] == g[("bn_adapt", "r")]["ctx"]
    assert deg.judge(g[("tent_long", "u")]["seeds"], g[("bn_adapt", "r")]["seeds"], 0.01)["n_cycles_judged"] == 4


def test_paired_compare_methods_and_cycle(tmp_path, capsys, monkeypatch):
    for s, (a, b) in enumerate([(0.226, 0.198), (0.226, 0.200), (0.227, 0.196)]):
        _mk(tmp_path, "bn_adapt", "r", s, [a, a])
        _mk(tmp_path, "tent_im_lam1", "t", s, [b, 0.3])
    monkeypatch.setattr(sys, "argv", ["x", "--exp-dir", str(tmp_path), "--method-a", "bn_adapt", "--method-b", "tent_im_lam1", "--cycle", "0"])
    pair.main()
    out = capsys.readouterr().out
    assert "seeds with B < A: 3 of 3" in out and "mean(B-A) = -0.0283" in out
