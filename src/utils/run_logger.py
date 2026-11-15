"""Per-run logging: step metrics -> CSV, final summary -> JSON, text -> stdlib logger.

Why: the project rule is "every number comes from a saved log". The CSV is the
source of truth that make_plots.py and the README tables will read from.
(File is named run_logger.py, not logging.py, to avoid confusion with stdlib.)
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Mapping


def get_text_logger(name: str = "ectta", log_file: str | Path | None = None) -> logging.Logger:
    """Console (+ optional file) logger with a simple format. Safe to call repeatedly."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if log_file is not None:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


class RunLogger:
    """Append per-step metric rows to ``metrics.csv`` and write ``summary.json``.

    The column set is fixed by the first row logged; later rows must use the
    same keys. This prevents silently misaligned CSV columns.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        self.summary_path = self.run_dir / "summary.json"
        self._fieldnames: list[str] | None = None
        self._file = None
        self._writer: csv.DictWriter | None = None

    def log_step(self, step: int, metrics: Mapping[str, float]) -> None:
        row: dict[str, Any] = {"step": step, **metrics}
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
        elif set(row.keys()) != set(self._fieldnames or []):
            raise KeyError(
                f"Row keys {sorted(row)} differ from CSV columns {sorted(self._fieldnames or [])}"
            )
        self._writer.writerow(row)
        self._file.flush()  # keep the log valid even if the run is interrupted

    def write_summary(self, summary: Mapping[str, Any]) -> Path:
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(dict(summary), f, indent=2, sort_keys=True, default=float)
        return self.summary_path

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
