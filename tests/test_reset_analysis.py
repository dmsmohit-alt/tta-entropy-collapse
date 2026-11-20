import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("esr", SCRIPTS / "error_vs_steps_since_reset.py")
esr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(esr)


def test_steps_since_reset_offsets():
    st = np.arange(12)
    assert esr.steps_since_reset(st, 4, 0).tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    # offset 1: first reset before t = 3 -> t=0,1,2 have had 0,1,2 steps; t=3 is s=0
    assert esr.steps_since_reset(st, 4, 1).tolist() == [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0]


def test_lasting_break_even():
    b = np.array([0, 25, 50, 75, 100])
    assert esr.lasting_break_even(b, np.array([0.0, -0.02, -0.01, 0.01, 0.03])) == 75
    assert esr.lasting_break_even(b, np.array([0.01, -0.02, 0.01, 0.02, 0.03])) == 50      # an earlier positive bin does not count
    assert esr.lasting_break_even(b, np.array([-0.1, -0.1, -0.1, -0.1, -0.1])) is None


def test_per_run_table_diff_zero_at_s0():
    T, n = 4, 12
    ref = pd.DataFrame({"step": range(n), "online_err": 0.2})
    m = pd.DataFrame({"step": range(n), "online_err": 0.2 + 0.01 * (np.arange(n) % T)})   # error grows with s, equals ref at s = 0
    t = esr.per_run_table(m, ref, T, 0, 1)
    assert t["max_abs_diff_s0"].iloc[0] == pytest.approx(0.0)
    assert t["diff"].tolist() == pytest.approx([0.0, 0.01, 0.02, 0.03])
    assert t["n"].tolist() == [3, 3, 3, 3]


def test_cli_end_to_end(tmp_path, monkeypatch, capsys):
    T, n = 4, 12
    for name, errs, tta in (("bn_adapt", [0.2] * n, {}), ("rdumb_T4", list(0.2 + 0.01 * (np.arange(n) % T)), {"reset_every": T})):
        d = tmp_path / name / "h_seed0"
        d.mkdir(parents=True)
        pd.DataFrame({"step": range(n), "online_err": errs}).to_csv(d / "metrics.csv", index=False)
        (d / "config.yaml").write_text(yaml.safe_dump({"tta": tta}))
    monkeypatch.setattr(sys, "argv", ["x", "--exp-dir", str(tmp_path), "--methods", "rdumb_T4", "--bin", "1"])
    esr.main()
    assert "max|diff at s=0| = 0.00e+00" in capsys.readouterr().out
    assert (tmp_path / "error_vs_steps_since_reset.csv").exists() and (tmp_path / "error_vs_steps_since_reset.png").exists()
