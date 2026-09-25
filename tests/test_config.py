import pytest

from rml.config import PROJECT_ROOT, load_config, resolve_dataset_file

BASE = PROJECT_ROOT / "configs" / "rml2016" / "base.yaml"


def test_base_config_values():
    d = load_config(BASE)["dataset"]
    assert d["name"] == "RML2016.10a"
    assert d["target_peak_accuracy"] == 0.90
    assert d["split_seed"] == 42
    assert d["root"] is None  # no runtime path baked into the repo
    assert d["expected"]["highest_snr"] == 18
    assert max(d["expected"]["snrs"]) == 18


def test_resolve_from_env_directory(tmp_path):
    d = load_config(BASE)["dataset"]
    (tmp_path / d["filename"]).write_bytes(b"x")
    assert resolve_dataset_file(d, environ={d["root_env"]: str(tmp_path)}) == tmp_path / d["filename"]


def test_override_beats_env(tmp_path):
    d = load_config(BASE)["dataset"]
    f = tmp_path / "custom.pkl"
    f.write_bytes(b"x")
    assert resolve_dataset_file(d, override=f, environ={d["root_env"]: "/nonexistent"}) == f


def test_missing_configuration_raises():
    d = load_config(BASE)["dataset"]
    with pytest.raises(ValueError, match="not configured"):
        resolve_dataset_file(d, environ={})


def test_missing_file_raises(tmp_path):
    d = load_config(BASE)["dataset"]
    with pytest.raises(FileNotFoundError):
        resolve_dataset_file(d, environ={d["root_env"]: str(tmp_path)})
