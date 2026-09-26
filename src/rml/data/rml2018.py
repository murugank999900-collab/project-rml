"""RML2018.01a loader and frozen-split reader.

The source file (e.g. ``GOLD_XYZ_OSC.0001_1024.hdf5``) holds three datasets:

* ``X``: (N, 1024, 2) I/Q samples
* ``Y``: (N, 24) one-hot modulation labels
* ``Z``: (N, 1) SNR in dB

Rows are never reordered: sample index ``i`` is HDF5 row ``i``, and the frozen
split (``splits/rml2018.01a_seed42.json``) stores these row numbers directly.
Labels and SNRs are read into memory; X (~21 GB as float32) is read only for
the rows requested through :meth:`RML2018Dataset.take`, transposed to
(n, 2, 1024) so the shared transforms and models apply unchanged.

The class label is the one-hot column index. The HDF5 file stores no class
names; :data:`CLASSES` follows the corrected ``classes-fixed.json`` shipped with
the Kaggle copy of the dataset (DeepSig's original ``classes.txt`` order does
not match the columns). Names are for display only.

``h5py`` is imported lazily (optional extra ``rml2018``), so the split reader
works without it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from rml.data.splits import GroupKey, _hash_indices
from rml.data.views import Subset, TrainVal, check_split_matches_dataset
from rml.experiment.metadata import sha256_file

# Display names by one-hot column, as in the dataset's corrected ``classes-fixed.json``
# (DeepSig's original ``classes.txt`` order does not match the columns).
CLASSES = (
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM", "128QAM",
    "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC", "FM", "GMSK", "OQPSK",
)  # fmt: skip
RAW_SAMPLE_SHAPE = (1024, 2)  # per-row shape of X in the HDF5 file
SAMPLE_SHAPE = (2, 1024)  # per-sample shape returned by ``take``
SPLIT_STRATEGY = "per_class_snr"
PARTS = ("train", "val", "test")
_CHUNK_ROWS = 1 << 16


def onehot_to_class_ids(Y) -> np.ndarray:
    """(N, C) one-hot labels -> (N,) int64 class ids. Rejects anything not strictly one-hot."""
    Y = np.asarray(Y)
    if Y.ndim != 2 or Y.shape[1] < 1:
        raise ValueError(f"Expected one-hot labels of shape (N, C), got {Y.shape}")
    ones = Y == 1
    if not np.all(ones | (Y == 0)):
        raise ValueError("One-hot labels contain values other than 0 and 1")
    bad = np.flatnonzero(ones.sum(axis=1) != 1)
    if bad.size:
        raise ValueError(f"{bad.size} label rows do not have exactly one hot entry, e.g. row {bad[0]}")
    return ones.argmax(axis=1).astype(np.int64)


def snr_to_int(Z) -> np.ndarray:
    """(N, 1) or (N,) SNR values -> (N,) int64 dB. Rejects non-integer values."""
    Z = np.asarray(Z)
    if Z.ndim == 2 and Z.shape[1] == 1:
        Z = Z[:, 0]
    if Z.ndim != 1:
        raise ValueError(f"Expected SNR of shape (N, 1) or (N,), got {Z.shape}")
    as_float = Z.astype(np.float64)
    if not np.all(np.isfinite(as_float)) or not np.all(as_float == np.round(as_float)):
        raise ValueError("SNR values must be finite integers (dB)")
    return as_float.astype(np.int64)


@dataclass(frozen=True)
class RML2018Dataset:
    """Labels and SNRs in memory; samples read from the HDF5 file on demand.

    Provides the attributes :mod:`rml.data.views` uses (``y``, ``snr``,
    ``classes``, ``group_sizes``, ``take``).
    """

    path: Path
    y: np.ndarray  # (N,) int64 one-hot column index
    snr: np.ndarray  # (N,) int64 SNR in dB
    classes: tuple[str, ...]  # display names, indexed by class id
    snrs: tuple[int, ...]  # distinct SNRs, ascending
    group_sizes: dict[GroupKey, int]  # (class name, snr) -> count

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def take(self, idx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (X, y, snr) for the given row indices; X is (n, 2, 1024) float32."""
        idx = self._check_indices(idx)
        return self.read_rows({"x": idx})[0]["x"], self.y[idx], self.snr[idx]

    def _check_indices(self, idx) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError("Indices must be one-dimensional")
        if idx.size and (idx.min() < 0 or idx.max() >= len(self)):
            raise IndexError("Row index out of range")
        return idx

    def read_rows(
        self,
        parts: Mapping[str, np.ndarray],
        dtype=np.float32,
        chunk_rows: int | None = None,
        log: Callable[[str], None] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Read X for several index sets in one sequential pass over the file.

        Returns ``{name: (n, 2, 1024) array}`` in the order of each index set,
        plus conversion stats. X is read in row chunks (only the span of rows
        that is needed from each chunk), so memory holds one chunk of float32
        rows plus the outputs, never the full dataset. For ``float16`` outputs
        the stats report overflow/non-finite values and the relative RMS
        rounding error, so callers can reject a lossy conversion.
        """
        import h5py

        chunk_rows = chunk_rows or _CHUNK_ROWS
        plan = {}
        for name, idx in parts.items():
            idx = self._check_indices(idx)
            order = np.argsort(idx, kind="stable")
            plan[name] = (idx[order], order, np.empty((idx.size, *SAMPLE_SHAPE), dtype=dtype))
        needed = np.unique(np.concatenate([s // chunk_rows for s, _, _ in plan.values()] or [np.empty(0, np.int64)]))
        sq_sum = err_sq_sum = 0.0
        max_abs = 0.0
        with h5py.File(self.path, "r") as f:
            X = f["X"]
            for i, c in enumerate(needed):
                start, stop = int(c) * chunk_rows, min((int(c) + 1) * chunk_rows, len(self))
                active = {}
                for name, (s, _, _) in plan.items():
                    lo, hi = np.searchsorted(s, [start, stop])
                    if hi > lo:
                        active[name] = (lo, hi)
                first = min(int(plan[n][0][lo]) for n, (lo, _) in active.items())
                last = max(int(plan[n][0][hi - 1]) for n, (_, hi) in active.items())
                rows = X[first : last + 1]
                for name, (lo, hi) in active.items():
                    s, order, out = plan[name]
                    src = rows[s[lo:hi] - first].transpose(0, 2, 1).astype(np.float32, copy=False)
                    dst = src.astype(dtype)
                    if dst.dtype != src.dtype:
                        err = dst.astype(np.float32) - src
                        sq_sum += float(np.sum(src * src, dtype=np.float64))
                        err_sq_sum += float(np.sum(err * err, dtype=np.float64))
                        max_abs = max(max_abs, float(np.max(np.abs(src))))
                    out[order[lo:hi]] = dst
                if log and (i + 1) % 8 == 0:
                    log(f"  read {i + 1}/{needed.size} chunks")
        outputs = {name: out for name, (_, _, out) in plan.items()}
        stats = {"dtype": np.dtype(dtype).name, "chunks_read": int(needed.size)}
        if sq_sum:
            finite = all(bool(np.all(np.isfinite(o))) for o in outputs.values())
            stats.update(
                source_max_abs=max_abs,
                all_finite=finite,
                relative_rms_rounding_error=float(np.sqrt(err_sq_sum / sq_sum)),
            )
        return outputs, stats


def load_rml2018(path: str | Path, classes: Sequence[str] = CLASSES, chunk_rows: int = _CHUNK_ROWS) -> RML2018Dataset:
    """Open an RML2018.01a HDF5 file; read labels (from one-hot) and SNRs, check shapes."""
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as f:
        missing = [k for k in ("X", "Y", "Z") if k not in f]
        if missing:
            raise ValueError(f"{path.name}: missing datasets {missing}")
        X, Y, Z = f["X"], f["Y"], f["Z"]
        n = X.shape[0]
        if X.ndim != 3 or tuple(X.shape[1:]) != RAW_SAMPLE_SHAPE:
            raise ValueError(f"Expected X of shape (N, {RAW_SAMPLE_SHAPE[0]}, {RAW_SAMPLE_SHAPE[1]}), got {X.shape}")
        if Y.shape != (n, len(classes)):
            raise ValueError(f"Expected Y of shape ({n}, {len(classes)}), got {Y.shape}")
        if Z.shape[0] != n:
            raise ValueError(f"Z has {Z.shape[0]} rows, X has {n}")
        y = np.empty(n, dtype=np.int64)
        snr = np.empty(n, dtype=np.int64)
        for start in range(0, n, chunk_rows):
            stop = min(start + chunk_rows, n)
            y[start:stop] = onehot_to_class_ids(Y[start:stop])
            snr[start:stop] = snr_to_int(Z[start:stop])

    pairs, counts = np.unique(np.stack([y, snr], axis=1), axis=0, return_counts=True)
    group_sizes = {(classes[int(c)], int(s)): int(k) for (c, s), k in zip(pairs, counts)}
    snrs = tuple(int(s) for s in np.unique(snr))
    return RML2018Dataset(path, y, snr, tuple(classes), snrs, group_sizes)


def validate_rml2018_structure(
    ds: RML2018Dataset,
    num_classes: int,
    snrs: Sequence[int],
    samples_per_group: int,
) -> None:
    """Raise ValueError unless every class x SNR group is present with ``samples_per_group``
    rows and the rows form contiguous, aligned blocks of one group each (the layout the
    frozen split's per-block counts rely on)."""
    errors = []
    if len(ds.classes) != num_classes or len(np.unique(ds.y)) != num_classes:
        errors.append(f"expected {num_classes} classes, found {len(np.unique(ds.y))} in labels")
    if tuple(sorted(snrs)) != ds.snrs:
        errors.append(f"expected SNRs {sorted(snrs)}, found {list(ds.snrs)}")
    if len(ds.group_sizes) != num_classes * len(snrs):
        errors.append("not every (class, SNR) combination is present")
    bad = {k: v for k, v in ds.group_sizes.items() if v != samples_per_group}
    if bad:
        errors.append(f"{len(bad)} groups do not have {samples_per_group} samples, e.g. {next(iter(bad.items()))}")
    if len(ds) % samples_per_group:
        errors.append(f"{len(ds)} rows is not a multiple of {samples_per_group}")
    else:
        for name, a in (("classes", ds.y), ("SNRs", ds.snr)):
            blocks = a.reshape(-1, samples_per_group)
            mixed = np.flatnonzero(np.any(blocks != blocks[:, :1], axis=1))
            if mixed.size:
                errors.append(f"{mixed.size} row blocks of {samples_per_group} mix {name}, e.g. block {mixed[0]}")
    if errors:
        raise ValueError("Dataset structure check failed:\n  - " + "\n  - ".join(errors))


# --- Frozen split -------------------------------------------------------------


@dataclass(frozen=True)
class RML2018Split:
    """Frozen global row indices. Provides what :mod:`rml.data.views` uses."""

    seed: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    group_sizes: dict[GroupKey, int]

    def sha256(self) -> dict[str, str]:
        return {part: _hash_indices(getattr(self, part)) for part in PARTS}

    def counts(self) -> dict[str, int]:
        return {part: int(getattr(self, part).size) for part in PARTS}


def load_rml2018_split(
    path: str | Path,
    expected_sha256: Mapping[str, str] | None = None,
    classes: Sequence[str] = CLASSES,
) -> RML2018Split:
    """Load and verify a frozen RML2018 split file. Never writes.

    Checks: recorded sizes, indices sorted/unique/in range, train/val/test
    disjoint and covering every row, per (class, SNR) block counts, and, if
    given, the SHA-256 of the file (``file``) and of each index array.
    """
    path = Path(path)
    expected = dict(expected_sha256 or {})
    if "file" in expected and sha256_file(path) != expected["file"]:
        raise ValueError(f"{path.name}: file SHA-256 does not match the recorded hash")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("strategy") != SPLIT_STRATEGY:
        raise ValueError(f"Unsupported split strategy: {data.get('strategy')!r}")

    num_classes, snrs, per_group = int(data["num_classes"]), [int(s) for s in data["snrs"]], int(data["samples_per_class_snr"])
    if num_classes != len(classes):
        raise ValueError(f"Split has {num_classes} classes, expected {len(classes)}")
    n_groups = num_classes * len(snrs)
    total = n_groups * per_group

    parts = {}
    for part in PARTS:
        idx = np.asarray(data[f"{part}_indices"], dtype=np.int64)
        if idx.ndim != 1 or idx.size != int(data[f"{part}_size"]):
            raise ValueError(f"{part}: {idx.size} indices, recorded size {data[f'{part}_size']}")
        if idx.size and (np.any(np.diff(idx) <= 0) or idx[0] < 0 or idx[-1] >= total):
            raise ValueError(f"{part}: indices must be strictly increasing and within [0, {total})")
        per_block = np.bincount(idx // per_group, minlength=n_groups)
        if np.any(per_block != int(data[f"{part}_per_class_snr"])):
            raise ValueError(f"{part}: not {data[f'{part}_per_class_snr']} indices in every class x SNR block")
        parts[part] = idx

    union = np.concatenate(list(parts.values()))
    if union.size != total or np.unique(union).size != total:
        raise ValueError("Split parts overlap or do not cover every row exactly once")

    split = RML2018Split(
        seed=int(data["seed"]),
        group_sizes={(c, s): per_group for c in classes for s in snrs},
        **parts,
    )
    hashes = split.sha256()
    wrong = [p for p in PARTS if p in expected and hashes[p] != expected[p]]
    if wrong:
        raise ValueError(f"Split index hashes do not match the recorded sha256 values: {wrong}")
    return split


def check_split_against_labels(
    split: RML2018Split,
    y: np.ndarray,
    snr: np.ndarray,
    per_group: Mapping[str, int],
    parts: Sequence[str] = PARTS,
) -> None:
    """Raise unless each of ``parts`` has ``per_group[part]`` samples of every (class, SNR).

    Training passes ``parts=("train", "val")`` so it never reads the test indices.
    """
    snr_values, snr_idx = np.unique(snr, return_inverse=True)
    n_classes = int(y.max()) + 1
    for part in parts:
        idx = getattr(split, part)
        counts = np.bincount(y[idx] * len(snr_values) + snr_idx[idx], minlength=n_classes * len(snr_values))
        if np.any(counts != per_group[part]):
            raise ValueError(f"{part}: not {per_group[part]} samples in every (class, SNR) group")


# --- Split views (compact in-memory copies, one sequential read) ---------------


def make_train_val_compact(
    ds: RML2018Dataset, split: RML2018Split, dtype=np.float16, log: Callable[[str], None] | None = None
) -> tuple[TrainVal, dict]:
    """Train and validation subsets read in one pass. Does not read the split's test indices.

    With ``float16`` the train+val samples take ~9.4 GB instead of ~18.8 GB
    (float32); the returned stats give the rounding error so callers can check it.
    """
    check_split_matches_dataset(ds, split)
    train_idx, val_idx = split.train, split.val
    arrays, stats = ds.read_rows({"train": train_idx, "val": val_idx}, dtype=dtype, log=log)
    tv = TrainVal(
        Subset("train", arrays["train"], ds.y[train_idx], ds.snr[train_idx]),
        Subset("val", arrays["val"], ds.y[val_idx], ds.snr[val_idx]),
        ds.classes,
    )
    return tv, stats


def make_test_compact(
    ds: RML2018Dataset, split: RML2018Split, dtype=np.float16, log: Callable[[str], None] | None = None
) -> tuple[Subset, dict]:
    """Held-out test subset. For final evaluation only."""
    check_split_matches_dataset(ds, split)
    idx = split.test
    arrays, stats = ds.read_rows({"test": idx}, dtype=dtype, log=log)
    return Subset("test", arrays["test"], ds.y[idx], ds.snr[idx]), stats
