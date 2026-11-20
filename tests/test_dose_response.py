import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("plot_dose_response", SCRIPTS / "plot_dose_response.py")
dr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dr)


def test_t_axis_is_lr_times_steps_at_cycle_midpoints():
    t = dr.t_axis(0.1, 750, 3)
    assert np.allclose(t, [0.1 * 750 * 0.5, 0.1 * 750 * 1.5, 0.1 * 750 * 2.5])


def test_interp_no_extrapolation_and_difference():
    curves = {"a": (np.array([0.0, 10.0]), np.array([0.2, 0.4])), "b": (np.array([5.0, 20.0]), np.array([0.3, 0.6]))}
    df = dr.interp_table(curves, np.array([0.0, 5.0, 10.0, 15.0]), ref="a")
    assert df["a"].tolist()[:3] == pytest.approx([0.2, 0.3, 0.4]) and np.isnan(df["a"].iloc[3])
    assert np.isnan(df["b"].iloc[0]) and df["b"].iloc[1] == pytest.approx(0.3)
    assert df["b"].iloc[2] == pytest.approx(0.4)             # 0.3 + (10-5)/(20-5)*(0.6-0.3)
    assert df["b - a"].iloc[2] == pytest.approx(0.0) and np.isnan(df["b - a"].iloc[0])


def test_plot_writes_png(tmp_path):
    curves = {"x": (np.array([1.0, 2.0]), np.array([0.2, 0.3])), "y": (np.array([2.0, 4.0]), np.array([0.25, 0.5]))}
    out = tmp_path / "p.png"
    dr.plot_curves(curves, {"x": 0.1, "y": 0.2}, {"bn_adapt": 0.226}, out, "note")
    assert out.exists() and out.stat().st_size > 1000


def test_segment_vs_curvature_analyse():
    import pandas as pd
    spec2 = importlib.util.spec_from_file_location("segment_vs_curvature", SCRIPTS / "segment_vs_curvature.py")
    sv = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(sv)
    names = ["a", "b", "c", "d", "e"]
    lo = pd.Series([0.2] * 5, index=names)
    lam = [2.5, 3.0, 5.0, 6.0, 7.0]                                   # local eta* = 0.8, 0.67, 0.4, 0.33, 0.29
    hi = pd.Series([0.2, 0.2, 0.3, 0.35, 0.4], index=names)           # excess only where eta_hi = 0.5 > eta*_local
    curv = pd.DataFrame({"corruption": names, "lam_max": lam})
    r = sv.analyse(lo, hi, curv, eta_hi=0.5)
    assert r["n_above"] == 3 and r["excess_above"] == pytest.approx(0.15) and r["excess_below"] == pytest.approx(0.0)
    assert r["rho_lam_max"] == pytest.approx(0.9747, abs=1e-3)       # ties in the two zero-excess segments keep it below 1
