"""RML2018.01a: frozen split file (real, in repo) and loader (synthetic HDF5 with the real layout)."""

import dataclasses
import json

import numpy as np
import pytest

from rml.config import PROJECT_ROOT, load_config, resolve_project_path
from rml.data import make_train_val
from rml.data.rml2018 import (
    CLASSES,
    check_split_against_labels,
    load_rml2018,
    load_rml2018_split,
    onehot_to_class_ids,
    snr_to_int,
    validate_rml2018_structure,
)
from rml.data.views import make_test
from rml.experiment.metadata import sha256_file

BASE = PROJECT_ROOT / "configs" / "rml2018" / "base.yaml"
SNRS = list(range(-20, 32, 2))


@pytest.fixture(scope="module")
def dcfg():
    return load_config(BASE)["dataset"]


# --- Config ---------------------------------------------------------------------


def test_base_config_values(dcfg):
    assert dcfg["name"] == "RML2018.01a"
    assert dcfg["target_peak_accuracy"] == 0.96
    assert dcfg["split_seed"] == 42
    assert dcfg["root"] is None  # no runtime path baked into the repo
    assert dcfg["root_env"] == "RML2018_ROOT"
    exp = dcfg["expected"]
    assert exp["num_classes"] == len(CLASSES) == 24
    assert exp["snrs"] == SNRS and len(exp["snrs"]) == 26
    assert exp["highest_snr"] == max(exp["snrs"]) == 30
    assert exp["sample_shape"] == [2, 1024]
    assert exp["num_samples"] == 24 * 26 * exp["samples_per_group"] == 2555904


# --- Real frozen split ----------------------------------------------------------


@pytest.fixture(scope="module")
def frozen(dcfg):
    return load_rml2018_split(resolve_project_path(dcfg["split"]["file"]), expected_sha256=dcfg["split"]["sha256"])


def _layout_labels(n_classes=24, snrs=SNRS, per_group=4096, block_order=None):
    """(y, snr) for rows in contiguous (class, SNR) blocks, in ``block_order`` (default class-major)."""
    groups = [(c, s) for c in range(n_classes) for s in snrs]
    if block_order is not None:
        groups = [groups[i] for i in block_order]
    y = np.repeat([c for c, _ in groups], per_group).astype(np.int64)
    snr = np.repeat([s for _, s in groups], per_group).astype(np.int64)
    return y, snr


def test_frozen_split_file_sha256(dcfg):
    path = resolve_project_path(dcfg["split"]["file"])
    assert sha256_file(path) == dcfg["split"]["sha256"]["file"]


def test_frozen_split_index_hashes(dcfg, frozen):
    expected = {k: v for k, v in dcfg["split"]["sha256"].items() if k != "file"}
    assert frozen.sha256() == expected
    assert frozen.seed == 42


def test_frozen_split_counts(dcfg, frozen):
    assert frozen.counts() == dcfg["expected"]["counts"] == {"train": 2044224, "val": 255840, "test": 255840}


def test_frozen_split_disjoint_and_complete(frozen):
    tr, va, te = frozen.train, frozen.val, frozen.test
    assert np.intersect1d(tr, va).size == 0
    assert np.intersect1d(tr, te).size == 0
    assert np.intersect1d(va, te).size == 0
    union = np.concatenate([tr, va, te])
    assert union.size == 2555904
    np.testing.assert_array_equal(np.sort(union), np.arange(2555904))


def test_frozen_split_stratified_per_block(frozen):
    for part, n in (("train", 3276), ("val", 410), ("test", 410)):
        assert set(np.bincount(getattr(frozen, part) // 4096, minlength=624).tolist()) == {n}


@pytest.mark.parametrize("shuffle_blocks", [False, True], ids=["class_major", "any_block_order"])
def test_frozen_split_highest_snr_counts(dcfg, frozen, shuffle_blocks):
    # Holds for any assignment of (class, SNR) groups to the 624 row blocks;
    # scripts/prepare_rml2018.py checks it against the real labels.
    order = np.random.default_rng(0).permutation(624) if shuffle_blocks else None
    y, snr = _layout_labels(block_order=order)
    check_split_against_labels(frozen, y, snr, dcfg["expected"]["per_group"])
    for part in ("val", "test"):
        assert int(np.sum(snr[getattr(frozen, part)] == 30)) == dcfg["expected"]["highest_snr_counts"][part] == 9840


def test_frozen_split_loads_deterministically(dcfg, frozen):
    again = load_rml2018_split(resolve_project_path(dcfg["split"]["file"]), expected_sha256=dcfg["split"]["sha256"])
    for part in ("train", "val", "test"):
        np.testing.assert_array_equal(getattr(again, part), getattr(frozen, part))


# --- Label conversion -----------------------------------------------------------


def test_onehot_to_class_ids():
    Y = np.eye(24, dtype=np.int64)[[3, 0, 23, 3]]
    np.testing.assert_array_equal(onehot_to_class_ids(Y), [3, 0, 23, 3])
    np.testing.assert_array_equal(onehot_to_class_ids(Y.astype(np.float32)), [3, 0, 23, 3])
    assert onehot_to_class_ids(Y).dtype == np.int64


@pytest.mark.parametrize(
    "row, match",
    [
        (np.zeros(24), "exactly one"),
        (np.r_[1, 1, np.zeros(22)], "exactly one"),
        (np.r_[2, np.zeros(23)], "other than 0 and 1"),
        (np.r_[0.5, 0.5, np.zeros(22)], "other than 0 and 1"),
    ],
)
def test_onehot_rejects_invalid_rows(row, match):
    Y = np.eye(24)[[0, 1]]
    Y[1] = row
    with pytest.raises(ValueError, match=match):
        onehot_to_class_ids(Y)


def test_onehot_rejects_wrong_rank():
    with pytest.raises(ValueError, match="shape"):
        onehot_to_class_ids(np.zeros(24))


def test_snr_to_int():
    np.testing.assert_array_equal(snr_to_int(np.array([[-20], [30]], dtype=np.int64)), [-20, 30])
    np.testing.assert_array_equal(snr_to_int(np.array([-20.0, 30.0])), [-20, 30])
    with pytest.raises(ValueError, match="integers"):
        snr_to_int(np.array([[0.5]]))
    with pytest.raises(ValueError, match="shape"):
        snr_to_int(np.zeros((3, 2)))


# --- Synthetic HDF5 with the real layout ----------------------------------------

N_PER_GROUP = 4  # per (class, SNR); real file: 4096
SYNTH_PER_GROUP = {"train": 2, "val": 1, "test": 1}


def _write_hdf5(path, n_per_group=N_PER_GROUP, x_shape=(1024, 2)):
    import h5py

    y, snr = _layout_labels(per_group=n_per_group)
    n = y.size
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, *x_shape)).astype(np.float32)
    X[:, :, 0] = np.arange(n, dtype=np.float32)[:, None]  # I channel encodes the row number
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("Y", data=np.eye(24, dtype=np.int64)[y])
        f.create_dataset("Z", data=snr[:, None])
    return X, y, snr


def _write_split(path, n_per_group=N_PER_GROUP, seed=42):
    rng = np.random.default_rng(seed)
    parts = {p: [] for p in SYNTH_PER_GROUP}
    for g in range(24 * 26):
        perm = g * n_per_group + rng.permutation(n_per_group)
        parts["train"] += perm[:2].tolist()
        parts["val"] += perm[2:3].tolist()
        parts["test"] += perm[3:].tolist()
    data = {
        "dataset": "RML2018.01a",
        "seed": seed,
        "strategy": "per_class_snr",
        "samples_per_class_snr": n_per_group,
        **{f"{p}_per_class_snr": n for p, n in SYNTH_PER_GROUP.items()},
        "num_classes": 24,
        "snrs": SNRS,
        **{f"{p}_size": len(v) for p, v in parts.items()},
        **{f"{p}_indices": sorted(v) for p, v in parts.items()},
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    pytest.importorskip("h5py")
    d = tmp_path_factory.mktemp("rml2018")
    X, y, snr = _write_hdf5(d / "GOLD_XYZ_OSC.0001_1024.hdf5")
    _write_split(d / "split.json")
    return d, X, y, snr


def test_synthetic_load_structure(synth):
    d, _, y, snr = synth
    ds = load_rml2018(d / "GOLD_XYZ_OSC.0001_1024.hdf5")
    validate_rml2018_structure(ds, num_classes=24, snrs=SNRS, samples_per_group=N_PER_GROUP)
    assert len(ds) == 24 * 26 * N_PER_GROUP
    assert ds.classes == CLASSES and len(ds.classes) == 24
    assert ds.snrs == tuple(SNRS) and len(ds.snrs) == 26
    np.testing.assert_array_equal(ds.y, y)
    np.testing.assert_array_equal(ds.snr, snr)
    assert ds.y.dtype == ds.snr.dtype == np.int64
    assert set(ds.group_sizes.values()) == {N_PER_GROUP} and len(ds.group_sizes) == 624


def test_synthetic_take_returns_1024_iq_in_requested_order(synth):
    d, X, y, snr = synth
    ds = load_rml2018(d / "GOLD_XYZ_OSC.0001_1024.hdf5")
    idx = np.array([2495, 0, 17, 17, 1300])  # unsorted, with a repeat
    Xs, ys, ss = ds.take(idx)
    assert Xs.shape == (5, 2, 1024) and Xs.dtype == np.float32
    np.testing.assert_array_equal(Xs, X[idx].transpose(0, 2, 1))
    np.testing.assert_array_equal(Xs[:, 0, 0], idx)
    np.testing.assert_array_equal(ys, y[idx])
    np.testing.assert_array_equal(ss, snr[idx])
    with pytest.raises(IndexError):
        ds.take([len(ds)])


def test_synthetic_loading_is_deterministic(synth, monkeypatch):
    d, *_ = synth
    path = d / "GOLD_XYZ_OSC.0001_1024.hdf5"
    a = load_rml2018(path)
    b = load_rml2018(path, chunk_rows=100)  # chunking does not change the result
    np.testing.assert_array_equal(a.y, b.y)
    np.testing.assert_array_equal(a.snr, b.snr)
    assert a.group_sizes == b.group_sizes
    idx = np.arange(0, len(a), 7)
    ref = a.take(idx)[0]
    monkeypatch.setattr("rml.data.rml2018._CHUNK_ROWS", 64)  # block-wise reads across many chunks
    np.testing.assert_array_equal(a.take(idx)[0], ref)
    np.testing.assert_array_equal(load_rml2018(path).take(idx)[0], ref)


def test_synthetic_split_views_and_highest_snr(synth):
    d, *_ = synth
    ds = load_rml2018(d / "GOLD_XYZ_OSC.0001_1024.hdf5")
    split = load_rml2018_split(d / "split.json")
    check_split_against_labels(split, ds.y, ds.snr, SYNTH_PER_GROUP)
    tv = make_train_val(ds, split)
    test = make_test(ds, split)
    assert (len(tv.train), len(tv.val), len(test)) == (1248, 624, 624)
    assert tv.classes == CLASSES
    ids = [s.X[:, 0, 0].astype(np.int64) for s in (tv.train, tv.val, test)]
    assert sorted(np.concatenate(ids).tolist()) == list(range(len(ds)))  # disjoint and complete
    for s in (tv.val, test):
        assert s.X.shape[1:] == (2, 1024)
        assert int(np.sum(s.snr == 30)) == 24 * SYNTH_PER_GROUP[s.name]


class NoTestAccess:
    """Wrap a split; any access to its test indices fails the test."""

    def __init__(self, split):
        self._split = split

    def __getattr__(self, name):
        if name == "test":
            raise AssertionError("training code accessed split.test")
        return getattr(self._split, name)


def test_make_train_val_never_reads_test_indices(synth):
    d, *_ = synth
    ds = load_rml2018(d / "GOLD_XYZ_OSC.0001_1024.hdf5")
    split = load_rml2018_split(d / "split.json")
    tv = make_train_val(ds, NoTestAccess(split))
    assert (len(tv.train), len(tv.val)) == (split.train.size, split.val.size)


def test_split_detects_tampering(synth, tmp_path):
    d, *_ = synth
    good = load_rml2018_split(d / "split.json")
    hashes = {**good.sha256(), "file": sha256_file(d / "split.json")}
    assert load_rml2018_split(d / "split.json", expected_sha256=hashes).sha256() == good.sha256()

    data = json.loads((d / "split.json").read_text(encoding="utf-8"))
    # Swap one val and one test index inside the same block: counts stay valid, hashes change.
    v, t = data["val_indices"][0], data["test_indices"][0]
    data["val_indices"][0], data["test_indices"][0] = t, v
    tampered = tmp_path / "split.json"
    tampered.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="file SHA-256"):
        load_rml2018_split(tampered, expected_sha256=hashes)
    with pytest.raises(ValueError, match="hash"):
        load_rml2018_split(tampered, expected_sha256={k: v for k, v in hashes.items() if k != "file"})


def test_split_rejects_overlap(synth, tmp_path):
    d, *_ = synth
    data = json.loads((d / "split.json").read_text(encoding="utf-8"))
    data["test_indices"][0] = data["val_indices"][0]  # same block, so per-block counts still pass
    data["test_indices"].sort()
    bad = tmp_path / "split.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        load_rml2018_split(bad)


def test_structure_check_rejects_mixed_blocks(synth):
    d, *_ = synth
    ds = load_rml2018(d / "GOLD_XYZ_OSC.0001_1024.hdf5")
    y = ds.y.copy()
    y[[0, N_PER_GROUP * 26]] = y[[N_PER_GROUP * 26, 0]]  # swap rows between class 0 and class 1 blocks
    with pytest.raises(ValueError, match="mix classes"):
        validate_rml2018_structure(dataclasses.replace(ds, y=y), 24, SNRS, N_PER_GROUP)
    with pytest.raises(ValueError, match="samples"):
        validate_rml2018_structure(ds, 24, SNRS, samples_per_group=4096)


def test_loader_rejects_wrong_sample_shape(tmp_path):
    pytest.importorskip("h5py")
    path = tmp_path / "bad.hdf5"
    _write_hdf5(path, n_per_group=1, x_shape=(128, 2))
    with pytest.raises(ValueError, match="shape"):
        load_rml2018(path)
