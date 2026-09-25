"""End-to-end smoke test of train.py / evaluate_final.py on a tiny synthetic dataset (CPU)."""

import csv
import importlib.util
import json
import pickle

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from conftest import make_groups  # noqa: E402
from rml.config import PROJECT_ROOT  # noqa: E402
from rml.data import export_split, from_groups, make_split  # noqa: E402
from rml.data import views as views_module  # noqa: E402
from rml.data.views import Subset  # noqa: E402
from rml.experiment.registry import read_registry  # noqa: E402
from rml.training.trainer import HISTORY_COLUMNS, fit  # noqa: E402

SNRS = [-10, 0, 6, 18]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup(tmp_path):
    groups = make_groups()
    data_file = tmp_path / "synthetic.pkl"
    data_file.write_bytes(pickle.dumps({(m.encode(), s): a for (m, s), a in groups.items()}, protocol=2))
    ds = from_groups(groups)
    split_file = export_split(make_split(ds.group_sizes, seed=42), tmp_path / "split.json", "synthetic")

    base = yaml.safe_load((PROJECT_ROOT / "configs" / "rml2016" / "base.yaml").read_text())
    base["dataset"]["split"]["file"] = str(split_file)
    base["dataset"]["expected"].update(
        num_classes=3, snrs=SNRS, samples_per_group=20, counts={"train": 192, "val": 24, "test": 24}
    )
    base_file = tmp_path / "base.yaml"
    base_file.write_text(yaml.safe_dump(base))

    exp = yaml.safe_load((PROJECT_ROOT / "configs" / "rml2016" / "exp001_cldnn.yaml").read_text())
    exp["base"] = str(base_file)
    exp["model"]["params"].update(lstm_hidden=8, attention_hidden=8, conv_channels=[8, 8, 8])
    exp["train"].update(epochs=3, batch_size=32, amp=False)
    exp["selection"]["early_stopping_patience"] = 50
    exp["output"] = {"root": str(tmp_path / "experiments"), "registry": str(tmp_path / "results" / "registry.csv")}
    exp_file = tmp_path / "exp.yaml"
    exp_file.write_text(yaml.safe_dump(exp))
    return {"tmp": tmp_path, "data": data_file, "exp": exp_file, "ds": ds, "registry": exp["output"]["registry"]}


def _train(setup, *extra):
    return _load_script("train").main(
        ["--config", str(setup["exp"]), "--data", str(setup["data"]), "--device", "cpu", *extra]
    )


def test_train_then_final_eval(setup, monkeypatch):
    # Training must never build the test subset.
    def forbidden(*a, **k):
        raise AssertionError("train.py called make_test")

    monkeypatch.setattr(views_module, "make_test", forbidden)
    run_dir = _train(setup)
    monkeypatch.undo()

    for name in ("config.resolved.yaml", "env.json", "history.csv", "val_metrics.json", "summary.json"):
        assert (run_dir / name).is_file(), name
    assert not (run_dir / "test_metrics.json").exists()
    for name in ("best.pt", "last.pt", "SHA256SUMS"):
        assert (run_dir / "checkpoints" / name).is_file(), name

    env = json.loads((run_dir / "env.json").read_text())
    assert env["split"]["counts_used"] == {"train": 192, "val": 24}
    assert env["split"]["seed"] == 42 and set(env["split"]["index_sha256"]) == {"train", "val", "test"}
    assert env["seeding"]["seed"] == 0 and env["val_highest_snr"] == 18
    assert len(env["dataset"]["file_sha256"]) == 64
    assert "git" in env and "environment" in env

    with open(run_dir / "history.csv") as f:
        history = list(csv.DictReader(f))
    assert tuple(history[0]) == HISTORY_COLUMNS and len(history) == 3

    val = json.loads((run_dir / "val_metrics.json").read_text())
    assert val["split"] == "val" and val["highest_snr"] == 18 and val["n_samples_highest_snr"] == 6

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["test_evaluated"] is False
    assert summary["selection_metric"] == "val_peak_accuracy_highest_snr"

    rows = read_registry(setup["registry"])
    assert len(rows) == 1 and rows[0]["event"] == "train"
    assert rows[0]["test_peak_accuracy_highest_snr"] == ""
    assert rows[0]["checkpoint_sha256"] == summary["checkpoint_sha256"]["best.pt"]

    evaluate = _load_script("evaluate_final")
    with pytest.raises(SystemExit, match="--final"):
        evaluate.main(["--run", str(run_dir), "--data", str(setup["data"]), "--device", "cpu"])

    result = evaluate.main(["--run", str(run_dir), "--data", str(setup["data"]), "--device", "cpu", "--final"])
    assert result["split"] == "test" and result["highest_snr"] == 18 and result["n_samples"] == 24
    assert result["n_samples_highest_snr"] == 6
    assert (run_dir / "test_metrics.json").is_file()
    rows = read_registry(setup["registry"])
    assert [r["event"] for r in rows] == ["train", "final_test"]

    with pytest.raises(SystemExit, match="once"):
        evaluate.main(["--run", str(run_dir), "--data", str(setup["data"]), "--device", "cpu", "--final"])


def test_runs_never_share_a_directory(setup, monkeypatch):
    a = _train(setup)
    b = _train(setup, "--seed", "1")
    assert a != b and a.parent == b.parent
    assert len(read_registry(setup["registry"])) == 2


def test_training_is_reproducible_on_cpu(setup):
    a = _train(setup)
    b = _train(setup)
    wa = torch.load(a / "checkpoints" / "best.pt", weights_only=True)["model_state"]
    wb = torch.load(b / "checkpoints" / "best.pt", weights_only=True)["model_state"]
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
    ha = [r["train_loss"] for r in csv.DictReader(open(a / "history.csv"))]
    hb = [r["train_loss"] for r in csv.DictReader(open(b / "history.csv"))]
    assert ha == hb


def test_final_eval_detects_tampered_checkpoint(setup):
    run_dir = _train(setup)
    ckpt = run_dir / "checkpoints" / "best.pt"
    ckpt.write_bytes(ckpt.read_bytes() + b"x")
    with pytest.raises(SystemExit, match="SHA-256"):
        _load_script("evaluate_final").main(["--run", str(run_dir), "--data", str(setup["data"]), "--final"])


def test_fit_rejects_test_subset(tmp_path):
    x = np.zeros((4, 4, 128), dtype=np.float32)
    s = Subset("test", x, np.zeros(4, dtype=np.int64), np.full(4, 18))
    v = Subset("val", x, np.zeros(4, dtype=np.int64), np.full(4, 18))
    model = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match="train and val"):
        fit(model, s, v, classes=["a"], train_cfg={}, selection_cfg={}, expected_highest_snr=18,
            run_dir=tmp_path, device=torch.device("cpu"), seed=0)
