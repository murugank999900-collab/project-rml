"""Checkpoint selection and early stopping on validation metrics only."""

from __future__ import annotations

from typing import Mapping


def _require_val(name: str | None, role: str) -> None:
    if name is not None and not name.startswith("val_"):
        raise ValueError(f"{role} must be a validation metric (val_*), got {name!r}")


class BestTracker:
    """Track the best epoch by ``metric`` (higher is better), then ``tie_breaker``.

    A new best requires a strict improvement of (metric, tie_breaker).
    ``should_stop`` becomes true after ``patience`` epochs without one.
    """

    def __init__(self, metric: str, tie_breaker: str | None = None, patience: int | None = None):
        _require_val(metric, "Selection metric")
        _require_val(tie_breaker, "Tie-breaker")
        self.metric = metric
        self.tie_breaker = tie_breaker
        self.patience = patience
        self.best_epoch: int | None = None
        self.best_key: tuple[float, ...] | None = None
        self.epochs_since_improvement = 0

    def _key(self, metrics: Mapping[str, float]) -> tuple[float, ...]:
        key: tuple[float, ...] = (float(metrics[self.metric]),)
        if self.tie_breaker is not None:
            key += (float(metrics[self.tie_breaker]),)
        return key

    def update(self, epoch: int, metrics: Mapping[str, float]) -> bool:
        """Record an epoch's validation metrics; return True if it is the new best."""
        key = self._key(metrics)
        if self.best_key is None or key > self.best_key:
            self.best_key, self.best_epoch = key, epoch
            self.epochs_since_improvement = 0
            return True
        self.epochs_since_improvement += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience is not None and self.epochs_since_improvement >= self.patience
