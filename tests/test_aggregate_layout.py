"""aggregate_runs.layout_of: summary first, then config.yaml, else 'unrecorded'; disagreement is flagged."""
import importlib.util
import json
from pathlib import Path

import yaml

spec = importlib.util.spec_from_file_location("aggregate_runs", Path(__file__).resolve().parents[1] / "scripts" / "aggregate_runs.py")
agg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg)


def _run(tmp_path, cfg_layout, summary):
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"compute": {"input_layout": cfg_layout} if cfg_layout else {}}))
    return agg.layout_of(tmp_path, summary)


def test_layout_sources(tmp_path):
    assert _run(tmp_path, None, {}) == "unrecorded"
    assert _run(tmp_path, "nchw", {}) == "nchw"                       # old run: only config.yaml has it
    assert _run(tmp_path, None, {"input_layout": "as_is"}) == "as_is"
    assert _run(tmp_path, "nchw", {"input_layout": "nchw"}) == "nchw"
    assert _run(tmp_path, "nchw", {"input_layout": "as_is"}).startswith("CONFLICT")


def test_missing_config_file(tmp_path):
    assert agg.layout_of(tmp_path, {}) == "unrecorded"
