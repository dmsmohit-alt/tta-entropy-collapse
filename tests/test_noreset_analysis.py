import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("esn", SCRIPTS / "error_vs_step_noreset.py")
esn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(esn)


def test_trailing_mean():
    r = esn.trailing_mean(np.array([1.0, 2.0, 3.0, 4.0]), 2)
    assert np.isnan(r[0]) and r[1:].tolist() == [1.5, 2.5, 3.5]


def test_break_even_first_and_lasting():
    end = np.arange(6) * 25 + 24
    roll = np.array([np.nan, -0.03, -0.01, 0.01, -0.005, 0.02])         # positive at 3, dips at 4, positive again at 5
    assert esn.break_even(end, roll) == (int(end[3]), int(end[5]))
    assert esn.break_even(end, np.array([np.nan, -1, -1, -1, -1, -1.0])) == (None, None)
    assert esn.break_even(end, np.array([np.nan, -1, 1, 1, 1, 1.0])) == (int(end[2]), int(end[2]))


def test_cli(tmp_path, monkeypatch, capsys):
    n = 200
    step = np.arange(n)
    for name, err in (("bn_adapt", np.full(n, 0.2)), ("tent", 0.2 + np.where(step < 100, -0.03, 0.04))):
        d = tmp_path / name / "h_seed0"
        d.mkdir(parents=True)
        pd.DataFrame({"step": step, "cycle": step // 100, "corruption": np.where(step % 100 < 50, "a", "b"), "online_err": err}).to_csv(d / "metrics.csv", index=False)
        (d / "config.yaml").write_text(yaml.safe_dump({}))
    monkeypatch.setattr(sys, "argv", ["x", "--exp-dir", str(tmp_path), "--methods", "tent", "--bin", "25", "--roll", "2"])
    esn.main()
    out = capsys.readouterr().out
    assert "first > 0 at step 124" in out and "lasting > 0 from step 124" in out     # trailing mean of bins [75-99] (-0.03) and [100-124] (+0.04) is +0.005 > 0: first positive window ends at step 124
    assert "c0:-0.0300" in out and "c1:+0.0400" in out
