"""Deterministic, stratified train/val/test split for (modulation, SNR) datasets.

Each (modulation, SNR) group is split independently. The permutation for a
group is drawn from an RNG seeded by ``(seed, crc32("<mod>|<snr>"))``, so a
group's split depends only on the seed, its key and its size -- not on dict
iteration order or on which other groups exist.

Global sample indices refer to the canonical ordering used by the loader:
groups sorted by (modulation, SNR), samples in their original within-group
order. See :func:`canonical_group_order`.

The split is exported as JSON (validation/test indices per group; train is the
complement) so it can be frozen in Git without storing any samples.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

SPLIT_ALGORITHM = "per-group-permutation/v1"

GroupKey = tuple[str, int]


def canonical_group_order(keys: Iterable[GroupKey]) -> list[GroupKey]:
    """Sort group keys by (modulation name, SNR). Shared by loader and splitter."""
    return sorted(keys, key=lambda k: (k[0], k[1]))


def group_rng(seed: int, mod: str, snr: int) -> np.random.Generator:
    key = zlib.crc32(f"{mod}|{snr}".encode("utf-8"))
    return np.random.default_rng(np.random.SeedSequence([seed, key]))


def group_counts(n: int, val_fraction: float, test_fraction: float) -> tuple[int, int, int]:
    """Return (n_train, n_val, n_test) for a group of size ``n``."""
    n_val = int(round(n * val_fraction))
    n_test = int(round(n * test_fraction))
    n_train = n - n_val - n_test
    if min(n_train, n_val, n_test) < 0:
        raise ValueError(f"Invalid split fractions for group size {n}")
    return n_train, n_val, n_test


@dataclass(frozen=True)
class GroupSplit:
    mod: str
    snr: int
    n: int
    offset: int  # global index of this group's first sample
    train: np.ndarray  # sorted local indices
    val: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class Split:
    seed: int
    val_fraction: float
    test_fraction: float
    groups: tuple[GroupSplit, ...]

    def _global(self, part: str) -> np.ndarray:
        if not self.groups:
            return np.empty(0, dtype=np.int64)
        return np.concatenate([g.offset + getattr(g, part) for g in self.groups]).astype(np.int64)

    @property
    def train(self) -> np.ndarray:
        return self._global("train")

    @property
    def val(self) -> np.ndarray:
        return self._global("val")

    @property
    def test(self) -> np.ndarray:
        return self._global("test")

    @property
    def group_sizes(self) -> dict[GroupKey, int]:
        return {(g.mod, g.snr): g.n for g in self.groups}

    def sha256(self) -> dict[str, str]:
        return {part: _hash_indices(getattr(self, part)) for part in ("train", "val", "test")}

    def counts(self) -> dict[str, int]:
        return {part: int(getattr(self, part).size) for part in ("train", "val", "test")}

    def same_indices(self, other: "Split") -> bool:
        return self.group_sizes == other.group_sizes and self.sha256() == other.sha256()


def _hash_indices(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, dtype="<i8").tobytes()).hexdigest()


def make_split(
    group_sizes: Mapping[GroupKey, int],
    seed: int,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> Split:
    """Split every (modulation, SNR) group into train/val/test by the given fractions."""
    groups = []
    offset = 0
    for mod, snr in canonical_group_order(group_sizes):
        n = int(group_sizes[(mod, snr)])
        n_train, n_val, _ = group_counts(n, val_fraction, test_fraction)
        perm = group_rng(seed, mod, snr).permutation(n)
        groups.append(
            GroupSplit(
                mod=mod,
                snr=snr,
                n=n,
                offset=offset,
                train=np.sort(perm[:n_train]),
                val=np.sort(perm[n_train : n_train + n_val]),
                test=np.sort(perm[n_train + n_val :]),
            )
        )
        offset += n
    return Split(seed, val_fraction, test_fraction, tuple(groups))


def export_split(split: Split, path: str | Path, dataset_name: str) -> Path:
    """Write the split as JSON. Refuses to overwrite an existing file (splits are frozen)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "dataset": dataset_name,
        "algorithm": SPLIT_ALGORITHM,
        "seed": split.seed,
        "fractions": {
            "train": round(1.0 - split.val_fraction - split.test_fraction, 10),
            "val": split.val_fraction,
            "test": split.test_fraction,
        },
        "index_order": "groups sorted by (modulation, snr); samples in original within-group order",
        "counts": split.counts(),
        "sha256": split.sha256(),
        "note": "train indices are the complement of val and test within each group",
    }
    head = json.dumps(header, indent=2)[:-2]  # drop the closing "\n}"
    group_lines = [
        "    "
        + json.dumps(
            {"mod": g.mod, "snr": g.snr, "n": g.n, "val": g.val.tolist(), "test": g.test.tolist()},
            separators=(",", ":"),
        )
        for g in split.groups
    ]
    text = head + ',\n  "groups": [\n' + ",\n".join(group_lines) + "\n  ]\n}\n"
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def load_split(path: str | Path) -> Split:
    """Load a split JSON and verify its recorded counts and hashes."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("algorithm") != SPLIT_ALGORITHM:
        raise ValueError(f"Unsupported split algorithm: {data.get('algorithm')!r}")

    by_key = {(g["mod"], int(g["snr"])): g for g in data["groups"]}
    if len(by_key) != len(data["groups"]):
        raise ValueError("Duplicate (mod, snr) groups in split file")

    groups = []
    offset = 0
    for key in canonical_group_order(by_key):
        g = by_key[key]
        n = int(g["n"])
        val = np.asarray(sorted(g["val"]), dtype=np.int64)
        test = np.asarray(sorted(g["test"]), dtype=np.int64)
        held_out = np.concatenate([val, test])
        if np.unique(held_out).size != held_out.size:
            raise ValueError(f"Group {key}: val/test indices overlap or repeat")
        if held_out.size and (held_out.min() < 0 or held_out.max() >= n):
            raise ValueError(f"Group {key}: index out of range for group size {n}")
        train = np.setdiff1d(np.arange(n, dtype=np.int64), held_out)
        groups.append(GroupSplit(key[0], key[1], n, offset, train, val, test))
        offset += n

    split = Split(
        seed=int(data["seed"]),
        val_fraction=float(data["fractions"]["val"]),
        test_fraction=float(data["fractions"]["test"]),
        groups=tuple(groups),
    )
    if split.counts() != data["counts"]:
        raise ValueError(f"Split counts {split.counts()} != recorded {data['counts']}")
    if split.sha256() != data["sha256"]:
        raise ValueError("Split index hashes do not match the recorded sha256 values")
    return split


def check_disjoint(split: Split) -> None:
    """Raise if any two of train/val/test share an index or a group is not fully covered."""
    tr, va, te = split.train, split.val, split.test
    for a_name, a, b_name, b in (("train", tr, "val", va), ("train", tr, "test", te), ("val", va, "test", te)):
        overlap = np.intersect1d(a, b)
        if overlap.size:
            raise AssertionError(f"{a_name}/{b_name} overlap: {overlap.size} indices")
    total = sum(g.n for g in split.groups)
    union = np.concatenate([tr, va, te])
    if union.size != total or np.unique(union).size != total:
        raise AssertionError("Split does not cover every sample exactly once")
