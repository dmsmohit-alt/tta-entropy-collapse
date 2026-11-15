"""YAML config system: load, override from the command line, save per run.

Why: every run must save the exact config it used, so each number in the README
can be traced back to (config, log). Configs are plain nested dicts.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

Config = dict[str, Any]


def load_yaml(path: str | Path) -> Config:
    """Read a YAML file into a nested dict (empty file -> empty dict)."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_merge(base: Config, new: Config) -> Config:
    """Return a copy of `base` with `new` merged in recursively (new wins)."""
    out = copy.deepcopy(base)
    for key, val in new.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def apply_overrides(cfg: Config, overrides: Iterable[str]) -> Config:
    """Apply command-line overrides like ``stream.batch_size=64``.

    Values are parsed with YAML, so ``1e-3`` / ``true`` / ``[0,1,2]`` / ``null``
    work as expected. A key must already exist in the config: this catches typos
    (e.g. ``stream.batchsize=64``) instead of silently ignoring them.
    """
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        node = out
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise KeyError(f"Unknown config section {p!r} in override {item!r}")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"Unknown config key {key!r}")
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def load_config(
    path: str | Path,
    overrides: Iterable[str] = (),
    default_path: str | Path | None = "configs/default.yaml",
) -> Config:
    """Load `default_path`, merge the experiment file `path` on top, apply overrides."""
    cfg: Config = {}
    if default_path is not None and Path(default_path).exists():
        cfg = load_yaml(default_path)
    cfg = deep_merge(cfg, load_yaml(path))
    return apply_overrides(cfg, overrides)


def config_hash(cfg: Config) -> str:
    """Short stable hash of a config (used in run folder names)."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


def save_config(cfg: Config, out_dir: str | Path) -> Path:
    """Write the resolved config to ``<out_dir>/config.yaml`` and return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    return path


def make_run_dir(cfg: Config, seed: int | None = None) -> Path:
    """Create ``<results_dir>/<experiment.name>/<hash>[_seedN]`` and save the config there."""
    exp = cfg.get("experiment", {})
    name = exp.get("name", "run")
    root = Path(exp.get("results_dir", "results"))
    tag = config_hash(cfg) + (f"_seed{seed}" if seed is not None else "")
    run_dir = root / name / tag
    save_config(cfg, run_dir)
    return run_dir
