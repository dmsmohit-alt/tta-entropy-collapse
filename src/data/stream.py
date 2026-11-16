"""Continual-shift stream builder: corruptions concatenated in sequence, repeated for many cycles.

One cycle = the chosen corruptions one after another (each: all 10k test images at one severity,
in batches). `num_cycles` repeats the whole sequence to make long streams, which is where
entropy-minimisation collapse is expected to show up.

Design decisions (comments are for the learning notes):
- Images inside a (cycle, corruption) segment are shuffled with a generator seeded by (seed, cycle,
  corruption index): different seeds give different batches, so "mean +/- std over seeds" is meaningful;
  the same seed always gives the identical stream (reproducible and identical across methods).
- `drop_last`: the incomplete final batch of a segment is dropped (10000 // batch_size full batches),
  because BatchNorm statistics from a tiny batch are noisy. With batch_size=200 nothing is dropped.
- Yields uint8 NHWC numpy; conversion to float tensors happens in the runner (this module needs no torch).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from src.data.cifar10c import CORRUPTIONS_15, N_PER_SEVERITY, load_corruption


@dataclass(frozen=True)
class StreamSpec:
    data_root: str
    corruptions: tuple[str, ...] = CORRUPTIONS_15
    severity: int = 5
    num_cycles: int = 1
    batch_size: int = 200
    shuffle: bool = True
    seed: int = 0
    max_steps: int | None = None
    n_per_severity: int = N_PER_SEVERITY   # only tests change this


@dataclass
class Batch:
    step: int            # global batch index, 0-based, across all cycles
    cycle: int
    corruption_idx: int  # position inside the cycle
    corruption: str
    x: np.ndarray        # uint8 (B, 32, 32, 3)
    y: np.ndarray        # int64 (B,)


def batches_per_segment(spec: StreamSpec) -> int:
    return spec.n_per_severity // spec.batch_size


def steps_per_cycle(spec: StreamSpec) -> int:
    return len(spec.corruptions) * batches_per_segment(spec)


def total_steps(spec: StreamSpec) -> int:
    full = spec.num_cycles * steps_per_cycle(spec)
    return full if spec.max_steps is None else min(full, spec.max_steps)


def iter_stream(spec: StreamSpec) -> Iterator[Batch]:
    """Yield the stream batch by batch (deterministic given the spec)."""
    if spec.batch_size <= 0 or batches_per_segment(spec) == 0:
        raise ValueError(f"batch_size={spec.batch_size} must be in 1..{spec.n_per_severity}")
    step = 0
    for cycle in range(spec.num_cycles):
        for ci, name in enumerate(spec.corruptions):
            x, y = load_corruption(spec.data_root, name, spec.severity, spec.n_per_severity)
            order = (np.random.default_rng([spec.seed, cycle, ci]).permutation(spec.n_per_severity)
                     if spec.shuffle else np.arange(spec.n_per_severity))
            for b in range(batches_per_segment(spec)):
                if spec.max_steps is not None and step >= spec.max_steps:
                    return
                idx = np.sort(order[b * spec.batch_size:(b + 1) * spec.batch_size])  # batch is a set: sort is harmless
                yield Batch(step, cycle, ci, name, x[idx], y[idx])
                step += 1


def spec_from_cfg(cfg: dict[str, Any], seed: int) -> StreamSpec:
    """Build a StreamSpec from the resolved config (`stream.*`, `data.root`)."""
    s = cfg["stream"]
    names = tuple(s["corruptions"]) if s.get("corruptions") else CORRUPTIONS_15[: int(s["num_corruptions"])]
    return StreamSpec(
        data_root=cfg["data"]["root"], corruptions=names, severity=int(s["severity"]),
        num_cycles=int(s["num_cycles"]), batch_size=int(s["batch_size"]),
        shuffle=bool(s.get("shuffle", True)), seed=int(seed),
        max_steps=None if s.get("max_steps") is None else int(s["max_steps"]),
    )
