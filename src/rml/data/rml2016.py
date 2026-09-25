"""RML2016.10a loader.

The source file (e.g. ``RML2016.10a_dict_optimized.pkl``) is a pickled dict
mapping ``(modulation, snr) -> array of shape (n, 2, 128)``. Samples are laid
out in canonical order: groups sorted by (modulation, SNR), samples in their
original within-group order. Split indices (see :mod:`rml.data.splits`) refer
to this order.

Only load pickle files from a trusted source: unpickling can execute code.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from rml.data.splits import GroupKey, canonical_group_order

SAMPLE_SHAPE = (2, 128)


@dataclass(frozen=True)
class RMLDataset:
    X: np.ndarray  # (N, 2, 128) float32
    y: np.ndarray  # (N,) int64 index into ``classes``
    snr: np.ndarray  # (N,) int64 SNR in dB
    classes: tuple[str, ...]  # modulation names, sorted
    snrs: tuple[int, ...]  # distinct SNRs, ascending
    group_sizes: dict[GroupKey, int]

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def take(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (X, y, snr) for the given global indices."""
        return self.X[idx], self.y[idx], self.snr[idx]


def _normalize_key(key) -> GroupKey:
    if not (isinstance(key, tuple) and len(key) == 2):
        raise ValueError(f"Expected (modulation, snr) tuple key, got {key!r}")
    mod, snr = key
    if isinstance(mod, bytes):
        mod = mod.decode("utf-8")
    snr_f = float(snr)
    if not snr_f.is_integer():
        raise ValueError(f"Non-integer SNR in key {key!r}")
    return str(mod), int(snr_f)


def from_groups(groups: Mapping, sample_shape: Sequence[int] = SAMPLE_SHAPE) -> RMLDataset:
    """Build an :class:`RMLDataset` from a ``{(mod, snr): samples}`` mapping."""
    normalized: dict[GroupKey, np.ndarray] = {}
    for key, samples in groups.items():
        k = _normalize_key(key)
        if k in normalized:
            raise ValueError(f"Duplicate group after key normalization: {k}")
        arr = np.asarray(samples)
        if arr.ndim != 3 or tuple(arr.shape[1:]) != tuple(sample_shape):
            raise ValueError(f"Group {k}: expected shape (n, {', '.join(map(str, sample_shape))}), got {arr.shape}")
        normalized[k] = arr

    if not normalized:
        raise ValueError("Dataset contains no groups")

    order = canonical_group_order(normalized)
    classes = tuple(sorted({mod for mod, _ in order}))
    class_index = {c: i for i, c in enumerate(classes)}

    X = np.concatenate([normalized[k] for k in order]).astype(np.float32, copy=False)
    y = np.concatenate([np.full(len(normalized[k]), class_index[k[0]], dtype=np.int64) for k in order])
    snr = np.concatenate([np.full(len(normalized[k]), k[1], dtype=np.int64) for k in order])

    return RMLDataset(
        X=X,
        y=y,
        snr=snr,
        classes=classes,
        snrs=tuple(sorted({s for _, s in order})),
        group_sizes={k: len(normalized[k]) for k in order},
    )


def load_rml2016(path: str | Path, sample_shape: Sequence[int] = SAMPLE_SHAPE) -> RMLDataset:
    """Load an RML2016.10a-style pickled dict (Python 2 pickles supported via latin1)."""
    with open(path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a dict of (modulation, snr) -> samples, got {type(obj).__name__}")
    return from_groups(obj, sample_shape)


def validate_structure(
    ds: RMLDataset,
    num_classes: int | None = None,
    snrs: Sequence[int] | None = None,
    samples_per_group: int | None = None,
) -> None:
    """Raise ValueError if the dataset does not match the expected structure."""
    errors = []
    if num_classes is not None and len(ds.classes) != num_classes:
        errors.append(f"expected {num_classes} classes, found {len(ds.classes)}: {ds.classes}")
    if snrs is not None and tuple(sorted(snrs)) != ds.snrs:
        errors.append(f"expected SNRs {sorted(snrs)}, found {list(ds.snrs)}")
    if len(ds.group_sizes) != len(ds.classes) * len(ds.snrs):
        errors.append("not every (modulation, SNR) combination is present")
    if samples_per_group is not None:
        bad = {k: n for k, n in ds.group_sizes.items() if n != samples_per_group}
        if bad:
            errors.append(f"{len(bad)} groups do not have {samples_per_group} samples, e.g. {next(iter(bad.items()))}")
    if ds.X.dtype != np.float32:
        errors.append(f"expected float32 samples, found {ds.X.dtype}")
    if errors:
        raise ValueError("Dataset structure check failed:\n  - " + "\n  - ".join(errors))
