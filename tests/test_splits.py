import json
import random

import numpy as np
import pytest

from conftest import MODS, N_PER_GROUP, SNRS, make_groups
from rml.data import check_disjoint, export_split, from_groups, load_split, make_split


def sizes(mods=MODS, snrs=SNRS, n=N_PER_GROUP):
    return {(m, s): n for m in mods for s in snrs}


def test_split_is_reproducible():
    a = make_split(sizes(), seed=42)
    b = make_split(sizes(), seed=42)
    for part in ("train", "val", "test"):
        np.testing.assert_array_equal(getattr(a, part), getattr(b, part))
    assert a.sha256() == b.sha256()


def test_split_depends_on_seed():
    assert make_split(sizes(), seed=42).sha256() != make_split(sizes(), seed=43).sha256()


def test_split_independent_of_dict_order():
    items = list(sizes().items())
    random.Random(1).shuffle(items)
    assert make_split(dict(items), seed=42).same_indices(make_split(sizes(), seed=42))


def test_group_split_independent_of_other_groups():
    full = make_split(sizes(), seed=42)
    only_bpsk = make_split(sizes(mods=["BPSK"]), seed=42)
    for g_small in only_bpsk.groups:
        g_full = next(g for g in full.groups if (g.mod, g.snr) == (g_small.mod, g_small.snr))
        np.testing.assert_array_equal(g_full.test, g_small.test)
        np.testing.assert_array_equal(g_full.val, g_small.val)


def test_no_overlap_and_full_coverage():
    split = make_split(sizes(), seed=42)
    check_disjoint(split)
    assert np.intersect1d(split.train, split.test).size == 0
    assert np.intersect1d(split.train, split.val).size == 0
    assert np.intersect1d(split.val, split.test).size == 0
    total = len(MODS) * len(SNRS) * N_PER_GROUP
    assert split.train.size + split.val.size + split.test.size == total


def test_stratified_80_10_10_per_group():
    split = make_split(sizes(), seed=42)
    for g in split.groups:
        assert (g.train.size, g.val.size, g.test.size) == (16, 2, 2)


def test_rml2016_scale_counts():
    mods = [f"M{i}" for i in range(11)]
    snrs = list(range(-20, 20, 2))
    split = make_split({(m, s): 1000 for m in mods for s in snrs}, seed=42)
    assert split.counts() == {"train": 176000, "val": 22000, "test": 22000}
    check_disjoint(split)


def test_split_indices_select_correct_group():
    ds = from_groups(make_groups())
    split = make_split(ds.group_sizes, seed=42)
    for g in split.groups:
        idx = g.offset + g.test
        assert set(ds.snr[idx].tolist()) == {g.snr}
        assert {ds.classes[c] for c in ds.y[idx]} == {g.mod}


def test_export_roundtrip(tmp_path):
    split = make_split(sizes(), seed=42)
    path = export_split(split, tmp_path / "split.json", "synthetic")
    data = json.loads(path.read_text())
    assert data["seed"] == 42 and data["counts"] == split.counts()
    assert load_split(path).same_indices(split)


def test_export_refuses_overwrite(tmp_path):
    split = make_split(sizes(), seed=42)
    path = export_split(split, tmp_path / "split.json", "synthetic")
    with pytest.raises(FileExistsError):
        export_split(make_split(sizes(), seed=7), path, "synthetic")
    assert load_split(path).same_indices(split)


def test_load_detects_tampering(tmp_path):
    path = export_split(make_split(sizes(), seed=42), tmp_path / "split.json", "synthetic")
    data = json.loads(path.read_text())
    g = data["groups"][0]
    g["test"][0] = next(i for i in range(g["n"]) if i not in g["val"] + g["test"])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash"):
        load_split(path)
