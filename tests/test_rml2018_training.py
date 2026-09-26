"""RML2018.01a training path: device transforms, compact split views, train/resume/final-eval (CPU, synthetic)."""

import ast
import csv
import importlib.util
import json
import re

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("h5py")

from test_isolation import FORBIDDEN  # noqa: E402
from test_rml2018 import N_PER_GROUP, SNRS, _write_hdf5, _write_split  # noqa: E402

from rml.config import PROJECT_ROOT  # noqa: E402
from rml.data import rml2018 as rml2018_module  # noqa: E402
from rml.data.rml2018 import load_rml2018, load_rml2018_split, make_test_compact, make_train_val_compact  # noqa: E402
from rml.data.transforms import FEATURES, NORMALIZERS, prepare_inputs  # noqa: E402
from rml.experiment.metadata import sha256_file  # noqa: E402
from rml.experiment.registry import read_registry  # noqa: E402
from rml.training import batched as batched_module  # noqa: E402
from rml.training.torch_transforms import prepare_inputs_torch  # noqa: E402

N_TOTAL = 24 * 26 * N_PER_GROUP


class NoTestAccess:
    def __init__(self, split):
        self._split = split

    def __getattr__(self, name):
        if name == "test":
            raise AssertionError("training code accessed split.test")
        return getattr(self._split, name)


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- Device transforms match the NumPy reference ---------------------------------


@pytest.mark.parametrize("normalize", sorted(NORMALIZERS))
@pytest.mark.parametrize("features", sorted(FEATURES))
def test_torch_transforms_match_numpy(normalize, features):
    X = (np.random.default_rng(0).standard_normal((8, 2, 1024)) * 0.01).astype(np.float32)
    X[0, :, :5] = 0.0  # zero samples: phase/IF edge cases
    ref = prepare_inputs(X, normalize, features)
    got = prepare_inputs_torch(torch.from_numpy(X), normalize, features).numpy()
    assert got.shape == ref.shape and got.dtype == np.float32
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)
    # float16 storage in, float32 features out
    half = prepare_inputs_torch(torch.from_numpy(X.astype(np.float16)), normalize, features)
    assert half.dtype == torch.float32


def test_torch_transforms_reject_unknown_names():
    x = torch.zeros(2, 2, 16)
    with pytest.raises(ValueError, match="normalization"):
        prepare_inputs_torch(x, "zscore", "iq")
    with pytest.raises(ValueError, match="features"):
        prepare_inputs_torch(x, "none", "spectrogram")


# --- Compact split views ------------------------------------------------------------


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("rml2018_train")
    X, y, snr = _write_hdf5(d / "GOLD_XYZ_OSC.0001_1024.hdf5")
    _write_split(d / "split.json")
    return d, X


def test_compact_views_float16(synth):
    d, X = synth
    ds = load_rml2018(d / "GOLD_XYZ_OSC.0001_1024.hdf5")
    split = load_rml2018_split(d / "split.json")
    tv, stats = make_train_val_compact(ds, NoTestAccess(split), dtype=np.float16)
    assert tv.train.X.dtype == np.float16 and tv.train.X.shape == (1248, 2, 1024)
    assert (tv.train.name, tv.val.name) == ("train", "val")
    np.testing.assert_array_equal(tv.val.X, X[split.val].transpose(0, 2, 1).astype(np.float16))
    np.testing.assert_array_equal(tv.train.y, ds.y[split.train])
    assert stats["all_finite"] and 0 < stats["relative_rms_rounding_error"] < 1e-3
    test, _ = make_test_compact(ds, split, dtype=np.float32)
    np.testing.assert_array_equal(test.X, X[split.test].transpose(0, 2, 1))
    assert test.name == "test"


def test_read_rows_chunking_invariant(synth, monkeypatch):
    d, X = synth
    ds = load_rml2018(d / "GOLD_XYZ_OSC.0001_1024.hdf5")
    idx = {"a": np.arange(0, N_TOTAL, 3), "b": np.array([N_TOTAL - 1, 5, 7])}
    big, _ = ds.read_rows(idx)
    small, stats = ds.read_rows(idx, chunk_rows=50)
    assert stats["chunks_read"] > 1
    for k in idx:
        np.testing.assert_array_equal(big[k], small[k])
        np.testing.assert_array_equal(big[k], X[idx[k]].transpose(0, 2, 1))


# --- End-to-end: train_rml2018.py / evaluate_final_rml2018.py ---------------------


@pytest.fixture
def setup(tmp_path, synth):
    d, _ = synth
    split = load_rml2018_split(d / "split.json")
    base = yaml.safe_load((PROJECT_ROOT / "configs" / "rml2018" / "base.yaml").read_text())
    base["dataset"]["split"]["file"] = str(d / "split.json")
    base["dataset"]["split"]["sha256"] = {**split.sha256(), "file": sha256_file(d / "split.json")}
    base["dataset"]["expected"].update(
        num_samples=N_TOTAL,
        samples_per_group=N_PER_GROUP,
        counts={"train": 1248, "val": 624, "test": 624},
        per_group={"train": 2, "val": 1, "test": 1},
        highest_snr_counts={"val": 24, "test": 24},
    )
    base_file = tmp_path / "base.yaml"
    base_file.write_text(yaml.safe_dump(base))

    exp = yaml.safe_load((PROJECT_ROOT / "configs" / "rml2018" / "exp001_cldnn.yaml").read_text())
    exp["base"] = str(base_file)
    exp["model"]["params"].update(lstm_hidden=8, attention_hidden=8, conv_channels=[8, 8, 8], lstm_layers=1)
    exp["train"].update(epochs=3, batch_size=128, eval_batch_size=256, amp=False)
    exp["selection"]["early_stopping_patience"] = 50
    exp["output"] = {"root": str(tmp_path / "experiments"), "registry": str(tmp_path / "results" / "registry.csv")}
    exp_file = tmp_path / "exp.yaml"
    exp_file.write_text(yaml.safe_dump(exp))
    return {"data": d / "GOLD_XYZ_OSC.0001_1024.hdf5", "exp": exp_file, "registry": exp["output"]["registry"]}


def _train(setup, *extra, config=True):
    args = (["--config", str(setup["exp"])] if config else []) + ["--data", str(setup["data"]), "--device", "cpu"]
    return _load_script("train_rml2018").main(args + list(extra))


def _evaluate(setup, run_dir, *extra):
    return _load_script("evaluate_final_rml2018").main(
        ["--run", str(run_dir), "--data", str(setup["data"]), "--device", "cpu", *extra]
    )


def test_train_then_final_eval(setup, monkeypatch):
    real_load = rml2018_module.load_rml2018_split

    def forbidden(*a, **k):
        raise AssertionError("training built the test subset")

    monkeypatch.setattr(rml2018_module, "make_test_compact", forbidden)
    monkeypatch.setattr(rml2018_module, "load_rml2018_split", lambda *a, **k: NoTestAccess(real_load(*a, **k)))
    script = _load_script("train_rml2018")  # binds the patched names
    run_dir = script.main(["--config", str(setup["exp"]), "--data", str(setup["data"]), "--device", "cpu"])
    monkeypatch.undo()

    for name in ("config.resolved.yaml", "env.json", "history.csv", "val_epochs.jsonl", "val_metrics.json",
                 "val_metrics_best.json", "summary.json"):
        assert (run_dir / name).is_file(), name
    assert not (run_dir / "test_metrics.json").exists()
    for name in ("best.pt", "last.pt", "SHA256SUMS"):
        assert (run_dir / "checkpoints" / name).is_file(), name

    env = json.loads((run_dir / "env.json").read_text())
    assert env["split"]["counts_used"] == {"train": 1248, "val": 624}
    assert env["val_highest_snr"] == 30 and env["val_n_highest_snr"] == 24
    assert env["dataset"]["storage"]["dtype"] == "float16"
    assert env["model"]["parameters"] > 0

    with open(run_dir / "history.csv") as f:
        assert len(list(csv.DictReader(f))) == 3
    epochs = [json.loads(line) for line in (run_dir / "val_epochs.jsonl").read_text().splitlines()]
    assert [e["epoch"] for e in epochs] == [1, 2, 3] and all(e["val"]["split"] == "val" for e in epochs)

    summary = json.loads((run_dir / "summary.json").read_text())
    for key in ("run_id", "git_sha", "seed", "dataset", "split_sha256", "model", "parameters", "optimizer",
                "scheduler", "batch_size", "epochs_configured", "amp", "augmentation", "best_epoch",
                "val_peak_accuracy_highest_snr", "val_overall_accuracy", "train_seconds_total", "best_checkpoint"):
        assert key in summary, key
    assert summary["val_highest_snr"] == 30 and summary["test_evaluated"] is False and not summary["smoke"]
    assert summary["target_peak_accuracy"] == 0.96

    rows = read_registry(setup["registry"])
    assert [r["event"] for r in rows] == ["train"] and rows[0]["dataset"] == "RML2018.01a"
    assert rows[0]["test_peak_accuracy_highest_snr"] == ""

    with pytest.raises(SystemExit, match="--final"):
        _evaluate(setup, run_dir)
    result = _evaluate(setup, run_dir, "--final")
    assert result["split"] == "test" and result["highest_snr"] == 30
    assert result["n_samples"] == 624 and result["n_samples_highest_snr"] == 24
    assert result["target_peak_accuracy"] == 0.96 and "meets_target" in result
    assert [r["event"] for r in read_registry(setup["registry"])] == ["train", "final_test"]
    with pytest.raises(SystemExit, match="once"):
        _evaluate(setup, run_dir, "--final")


def test_resume_matches_uninterrupted_run(setup, monkeypatch):
    full = _train(setup)

    calls = {"n": 0}
    real_eval = batched_module.evaluate_predictions

    def crash_on_third(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt("simulated preemption")
        return real_eval(*a, **k)

    monkeypatch.setattr(batched_module, "evaluate_predictions", crash_on_third)
    with pytest.raises(KeyboardInterrupt):
        _train(setup)
    monkeypatch.undo()
    interrupted = sorted((setup["exp"].parent / "experiments" / "rml2018.01a").iterdir())[-1]
    assert interrupted != full and not (interrupted / "summary.json").exists()

    resumed = _train(setup, "--resume", str(interrupted), config=False)
    assert resumed == interrupted
    wa = torch.load(full / "checkpoints" / "last.pt", weights_only=True)["model_state"]
    wb = torch.load(resumed / "checkpoints" / "last.pt", weights_only=True)["model_state"]
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
    ha = [r["train_loss"] for r in csv.DictReader(open(full / "history.csv"))]
    hb = [r["train_loss"] for r in csv.DictReader(open(resumed / "history.csv"))]
    assert ha == hb and len(hb) == 3
    assert (resumed / "env.resume1.json").is_file()
    assert json.loads((resumed / "summary.json").read_text())["resumed_from_epoch"] == 2

    with pytest.raises(SystemExit, match="already finished"):
        _train(setup, "--resume", str(resumed), config=False)


def test_smoke_run_is_not_registered(setup):
    run_dir = _train(setup, "--smoke")
    assert run_dir.parent.parent.name == "_smoke"
    assert json.loads((run_dir / "summary.json").read_text())["smoke"] is True
    assert read_registry(setup["registry"]) == []
    with pytest.raises(SystemExit, match="Smoke"):
        _evaluate(setup, run_dir, "--final")


def test_training_is_reproducible_on_cpu(setup):
    a, b = _train(setup), _train(setup)
    wa = torch.load(a / "checkpoints" / "best.pt", weights_only=True)["model_state"]
    wb = torch.load(b / "checkpoints" / "best.pt", weights_only=True)["model_state"]
    assert all(torch.equal(wa[k], wb[k]) for k in wa)


# --- Test-split isolation of the new training sources ---------------------------------


@pytest.mark.parametrize("name", ["scripts/train_rml2018.py", "src/rml/training/batched.py",
                                  "src/rml/training/torch_transforms.py"])
def test_rml2018_training_sources_do_not_touch_test_split(name):
    source = (PROJECT_ROOT / name).read_text(encoding="utf-8")
    for pattern, what in FORBIDDEN:
        assert not re.search(pattern, source), f"{name} contains {what}"


def test_fit_batched_has_no_test_parameters():
    tree = ast.parse((PROJECT_ROOT / "src" / "rml" / "training" / "batched.py").read_text(encoding="utf-8"))
    fit = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "fit_batched")
    names = [a.arg for a in fit.args.args + fit.args.kwonlyargs]
    assert names[:3] == ["model", "train", "val"]
    assert not [n for n in names if "test" in n]
