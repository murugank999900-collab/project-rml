"""Split-specific views of a loaded dataset.

Training code uses :func:`make_train_val`, which reads only the train and
validation indices of a split. The test subset is available only through
:func:`make_test`, which is called by the final-evaluation script and never by
training code (enforced by tests).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rml.data.rml2016 import RMLDataset
from rml.data.splits import Split


@dataclass(frozen=True)
class Subset:
    name: str  # "train", "val" or "test"
    X: np.ndarray
    y: np.ndarray
    snr: np.ndarray

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def with_inputs(self, X: np.ndarray) -> "Subset":
        """Same labels and SNRs, transformed inputs."""
        if X.shape[0] != len(self):
            raise ValueError("Transformed inputs must keep the number of samples")
        return Subset(self.name, X, self.y, self.snr)


@dataclass(frozen=True)
class TrainVal:
    train: Subset
    val: Subset
    classes: tuple[str, ...]


def check_split_matches_dataset(ds: RMLDataset, split: Split) -> None:
    if split.group_sizes != ds.group_sizes:
        raise ValueError("Split groups/sizes do not match the loaded dataset")


def _subset(ds: RMLDataset, name: str, idx: np.ndarray) -> Subset:
    X, y, snr = ds.take(idx)  # fancy indexing copies, so ``ds`` can be released afterwards
    return Subset(name, X, y, snr)


def make_train_val(ds: RMLDataset, split: Split) -> TrainVal:
    """Train and validation subsets. Does not read the split's test indices."""
    check_split_matches_dataset(ds, split)
    return TrainVal(_subset(ds, "train", split.train), _subset(ds, "val", split.val), ds.classes)


def make_test(ds: RMLDataset, split: Split) -> Subset:
    """Held-out test subset. For final evaluation only."""
    check_split_matches_dataset(ds, split)
    return _subset(ds, "test", split.test)
