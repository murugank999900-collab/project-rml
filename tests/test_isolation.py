"""The test split must stay isolated from training code."""

import ast
import re

import numpy as np
import pytest

from conftest import make_groups
from rml.config import PROJECT_ROOT
from rml.data import from_groups, make_split, make_train_val
from rml.data.views import make_test

TRAINING_SOURCES = [
    PROJECT_ROOT / "scripts" / "train.py",
    *sorted((PROJECT_ROOT / "src" / "rml" / "training").glob("*.py")),
]
FORBIDDEN = [
    (r"\bmake_test\b", "make_test"),
    (r"\.test\b", "access to split.test"),
    (r"split\s*=\s*['\"]test['\"]", "evaluation on the test split"),
    (r"test_metrics", "test metrics"),
]


class NoTestAccess:
    """Wrap a Split; any access to its test indices fails the test."""

    def __init__(self, split):
        self._split = split

    def __getattr__(self, name):
        if name == "test":
            raise AssertionError("training code accessed split.test")
        return getattr(self._split, name)


def _rows(X):
    return {x.tobytes() for x in X}


@pytest.fixture
def ds_split():
    ds = from_groups(make_groups())
    return ds, make_split(ds.group_sizes, seed=42)


def test_make_train_val_never_reads_test_indices(ds_split):
    ds, split = ds_split
    tv = make_train_val(ds, NoTestAccess(split))
    assert (len(tv.train), len(tv.val)) == (split.train.size, split.val.size)


def test_train_val_contain_no_test_samples(ds_split):
    ds, split = ds_split
    tv = make_train_val(ds, split)
    test = make_test(ds, split)
    test_rows = _rows(test.X)
    assert len(test_rows) == len(test)  # synthetic samples are unique
    assert not (_rows(tv.train.X) & test_rows)
    assert not (_rows(tv.val.X) & test_rows)
    assert not (_rows(tv.train.X) & _rows(tv.val.X))
    assert len(tv.train) + len(tv.val) + len(test) == len(ds)


def test_subsets_are_copies(ds_split):
    ds, split = ds_split
    tv = make_train_val(ds, split)
    assert not np.shares_memory(tv.train.X, ds.X)
    assert not np.shares_memory(tv.val.X, ds.X)


def test_subsets_are_labelled(ds_split):
    ds, split = ds_split
    tv = make_train_val(ds, split)
    assert (tv.train.name, tv.val.name, make_test(ds, split).name) == ("train", "val", "test")


def test_make_train_val_rejects_mismatched_split(ds_split):
    ds, _ = ds_split
    other = make_split({k: n + 1 for k, n in ds.group_sizes.items()}, seed=42)
    with pytest.raises(ValueError, match="do not match"):
        make_train_val(ds, other)


@pytest.mark.parametrize("path", TRAINING_SOURCES, ids=lambda p: p.name)
def test_training_sources_do_not_touch_test_split(path):
    source = path.read_text(encoding="utf-8")
    for pattern, what in FORBIDDEN:
        assert not re.search(pattern, source), f"{path.name} contains {what}"


def test_fit_has_no_test_parameters():
    tree = ast.parse((PROJECT_ROOT / "src" / "rml" / "training" / "trainer.py").read_text(encoding="utf-8"))
    fit = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "fit")
    names = [a.arg for a in fit.args.args + fit.args.kwonlyargs]
    assert names[:3] == ["model", "train", "val"]
    assert not [n for n in names if "test" in n]
