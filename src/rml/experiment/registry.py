"""Append-only CSV experiment registry.

Each line is an event: ``train`` (written by scripts/train.py, validation
metrics only) or ``final_test`` (written once per run by
scripts/evaluate_final.py). Existing lines are never rewritten.

Runs are ranked with :func:`best_by_validation`, which only accepts
validation metrics; test metrics are for reporting and never for selection.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

REGISTRY_COLUMNS = (
    "event",
    "timestamp_utc",
    "run_id",
    "dataset",
    "experiment",
    "model",
    "seed",
    "git_sha",
    "git_dirty",
    "split_sha256_train",
    "split_sha256_val",
    "split_sha256_test",
    "epochs_run",
    "best_epoch",
    "val_highest_snr",
    "val_peak_accuracy_highest_snr",
    "val_overall_accuracy",
    "test_highest_snr",
    "test_peak_accuracy_highest_snr",
    "test_overall_accuracy",
    "checkpoint_sha256",
    "run_dir",
)

EVENTS = ("train", "final_test")


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def append_registry_row(path: str | Path, row: Mapping) -> None:
    unknown = set(row) - set(REGISTRY_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown registry columns: {sorted(unknown)}")
    if row.get("event") not in EVENTS:
        raise ValueError(f"Registry event must be one of {EVENTS}, got {row.get('event')!r}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    has_header = path.exists() and path.stat().st_size > 0
    if has_header:
        with open(path, encoding="utf-8", newline="") as f:
            header = next(csv.reader(f))
        if tuple(header) != REGISTRY_COLUMNS:
            raise ValueError(f"Registry {path} has an unexpected header; refusing to append")

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS, lineterminator="\n")
        if not has_header:
            writer.writeheader()
        writer.writerow({c: _fmt(row.get(c)) for c in REGISTRY_COLUMNS})


def read_registry(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def best_by_validation(
    rows: list[Mapping[str, str]],
    metric: str = "val_peak_accuracy_highest_snr",
    dataset: str | None = None,
) -> Mapping[str, str] | None:
    """Best ``train`` event by a validation metric. Test metrics are rejected."""
    if not metric.startswith("val_"):
        raise ValueError(f"Model selection must use a validation metric, got {metric!r}")
    candidates = [
        r for r in rows if r["event"] == "train" and r.get(metric) and (dataset is None or r["dataset"] == dataset)
    ]
    return max(candidates, key=lambda r: float(r[metric]), default=None)
