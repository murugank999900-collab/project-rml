import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone

import numpy as np
import pytest

from rml.config import PROJECT_ROOT, load_experiment_config
from rml.experiment.metadata import environment_info, git_info, sha256_file, write_json
from rml.experiment.run_dir import create_run_dir, make_run_id
from rml.training.seeding import seed_everything

EXP001 = PROJECT_ROOT / "configs" / "rml2016" / "exp001_cldnn.yaml"


# --- run directories ---------------------------------------------------------

def test_run_id_format():
    now = datetime(2026, 9, 25, 12, 30, 5, tzinfo=timezone.utc)
    assert make_run_id("exp001 cldnn/x", 3, "abcdef1234", now) == "20260925-123005_exp001-cldnn-x_s3_abcdef1"
    assert make_run_id("e", 0, None, now).endswith("_nogit")


def test_run_dir_is_never_reused(tmp_path):
    d = create_run_dir(tmp_path / "experiments", "run1")
    assert (d / "checkpoints").is_dir()
    (d / "marker").write_text("keep")
    d2 = create_run_dir(tmp_path / "experiments", "run1")
    d3 = create_run_dir(tmp_path / "experiments", "run1")
    assert (d2.name, d3.name) == ("run1_r2", "run1_r3")
    assert (d / "marker").read_text() == "keep"
    assert not (d2 / "marker").exists()


# --- metadata ----------------------------------------------------------------

def test_sha256_file(tmp_path):
    p = tmp_path / "f.bin"
    data = bytes(range(256)) * 10000
    p.write_bytes(data)
    assert sha256_file(p, chunk_size=1000) == hashlib.sha256(data).hexdigest()


def test_write_json_is_exclusive(tmp_path):
    p = write_json(tmp_path / "a.json", {"x": np.float32(0.5), "y": np.arange(3), "z": tmp_path})
    assert json.loads(p.read_text())["y"] == [0, 1, 2]
    with pytest.raises(FileExistsError):
        write_json(p, {})


def test_git_info_records_sha():
    info = git_info(PROJECT_ROOT)
    if not info["available"]:
        pytest.skip("git not available")
    head = subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    assert info["sha"] == head.stdout.strip()
    assert isinstance(info["dirty"], bool)
    assert not any(f.startswith(("experiments/", "results/")) for f in info["dirty_files"])


def test_git_info_outside_repo(tmp_path):
    assert git_info(tmp_path)["available"] is False


def test_environment_info_without_gpu_requirements():
    info = environment_info()
    assert info["numpy"] == np.__version__ and "python" in info and "timestamp_utc" in info


# --- seeding -----------------------------------------------------------------

def test_seed_everything_reproducible():
    seed_everything(123)
    a = (random.random(), np.random.rand(3))
    seed_everything(123)
    b = (random.random(), np.random.rand(3))
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])


# --- experiment config -------------------------------------------------------

def test_exp001_config():
    cfg = load_experiment_config(EXP001)
    assert cfg["dataset"]["name"] == "RML2016.10a"
    assert cfg["dataset"]["split_seed"] == 42
    assert cfg["dataset"]["target_peak_accuracy"] == 0.90
    assert cfg["dataset"]["expected"]["highest_snr"] == 18
    assert cfg["dataset"]["split"]["file"] == "splits/rml2016.10a_seed42.json"
    assert cfg["selection"]["metric"] == "val_peak_accuracy_highest_snr"
    assert cfg["data"]["train_snrs"] == "all"
    assert cfg["model"]["name"] == "cldnn_v1"
    assert cfg["data"]["features"] == "iq_amp_phase"


def test_overrides(tmp_path):
    cfg = load_experiment_config(EXP001, ["train.epochs=3", "experiment.seed=7", "train.optimizer.lr=0.01"])
    assert (cfg["train"]["epochs"], cfg["experiment"]["seed"], cfg["train"]["optimizer"]["lr"]) == (3, 7, 0.01)
    with pytest.raises(KeyError):
        load_experiment_config(EXP001, ["train.epochz=3"])
    with pytest.raises(ValueError):
        load_experiment_config(EXP001, ["train.epochs"])


@pytest.mark.parametrize(
    "override, message",
    [
        ("selection.metric=test_peak_accuracy_highest_snr", "validation"),
        ("selection.tie_breaker=test_overall_accuracy", "validation"),
        ("data.train_snrs=[18]", "complete"),
        ("experiment.seed=abc", "integer"),
    ],
)
def test_config_validation(override, message):
    with pytest.raises(ValueError, match=message):
        load_experiment_config(EXP001, [override])


def test_official_split_file_matches_code():
    from rml.data import load_split, make_split

    split = load_split(PROJECT_ROOT / "splits" / "rml2016.10a_seed42.json")
    assert split.counts() == {"train": 176000, "val": 22000, "test": 22000}
    assert make_split(split.group_sizes, 42, 0.1, 0.1).same_indices(split)
    assert max(g.snr for g in split.groups) == 18
