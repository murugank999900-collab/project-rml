import pytest

from rml.experiment.registry import REGISTRY_COLUMNS, append_registry_row, best_by_validation, read_registry


def row(run_id, peak=None, event="train", **kw):
    return {"event": event, "run_id": run_id, "dataset": "RML2016.10a", "val_peak_accuracy_highest_snr": peak, **kw}


def test_append_only(tmp_path):
    path = tmp_path / "results" / "registry.csv"
    append_registry_row(path, row("a", 0.8))
    first = path.read_bytes()
    append_registry_row(path, row("b", 0.9))
    append_registry_row(path, row("a", event="final_test", test_peak_accuracy_highest_snr=0.85))
    content = path.read_bytes()
    assert content.startswith(first)
    rows = read_registry(path)
    assert [r["run_id"] for r in rows] == ["a", "b", "a"]
    assert tuple(rows[0]) == REGISTRY_COLUMNS
    assert rows[0]["val_peak_accuracy_highest_snr"] == "0.8"
    assert rows[0]["test_peak_accuracy_highest_snr"] == ""


def test_rejects_unknown_columns_and_events(tmp_path):
    path = tmp_path / "r.csv"
    with pytest.raises(ValueError, match="Unknown"):
        append_registry_row(path, {**row("a"), "bogus": 1})
    with pytest.raises(ValueError, match="event"):
        append_registry_row(path, row("a", event="retrain"))
    assert not path.exists()


def test_refuses_foreign_header(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("x,y\n1,2\n")
    with pytest.raises(ValueError, match="header"):
        append_registry_row(path, row("a"))
    assert path.read_text() == "x,y\n1,2\n"


def test_best_by_validation_ignores_test_results(tmp_path):
    path = tmp_path / "r.csv"
    append_registry_row(path, row("a", 0.80))
    append_registry_row(path, row("b", 0.85))
    append_registry_row(path, row("a", event="final_test", test_peak_accuracy_highest_snr=0.99))
    append_registry_row(path, row("c", 0.70, dataset="RML2018.01a"))
    rows = read_registry(path)
    assert best_by_validation(rows)["run_id"] == "b"
    assert best_by_validation(rows, dataset="RML2018.01a")["run_id"] == "c"
    assert best_by_validation([]) is None


@pytest.mark.parametrize("metric", ["test_peak_accuracy_highest_snr", "test_overall_accuracy"])
def test_best_by_validation_rejects_test_metrics(metric):
    with pytest.raises(ValueError, match="validation"):
        best_by_validation([], metric=metric)
