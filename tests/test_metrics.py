import numpy as np
import pytest

from rml.evaluation import (
    accuracy_by_snr,
    classification_report,
    confusion_matrix,
    evaluate_predictions,
    highest_snr,
    peak_accuracy_highest_snr,
    verify_highest_snr,
)

CLASSES = ["A", "B", "C"]


def test_highest_snr_detected_from_data():
    assert highest_snr([0, -20, 18, 4, 18, -2]) == 18
    assert highest_snr([-20, -10, -4]) == -4
    assert highest_snr(np.array([30, 10, 20])) == 30


def test_verify_highest_snr():
    assert verify_highest_snr([-20, 0, 18], expected=18) == 18
    with pytest.raises(ValueError, match="16"):
        verify_highest_snr([-20, 0, 16], expected=18)


def test_peak_accuracy_uses_only_highest_snr():
    snr = np.array([18, 18, 18, 18, 0, 0, -20, -20])
    y_true = np.array([0, 1, 2, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 2, 1, 0, 0, 0, 0])  # 3/4 right at 18 dB, 0/4 elsewhere
    assert peak_accuracy_highest_snr(y_true, y_pred, snr) == pytest.approx(0.75)


def test_peak_accuracy_perfect_and_zero():
    snr = np.array([10, 18, 18])
    assert peak_accuracy_highest_snr([0, 1, 2], [2, 1, 2], snr) == 1.0
    assert peak_accuracy_highest_snr([0, 1, 2], [0, 0, 0], snr) == 0.0


def test_accuracy_by_snr():
    by = accuracy_by_snr([0, 1, 2, 0], [0, 0, 2, 0], [0, 0, 18, 18])
    assert list(by) == [0, 18]
    assert by[0] == {"accuracy": 0.5, "n": 2, "correct": 1}
    assert by[18] == {"accuracy": 1.0, "n": 2, "correct": 2}


def test_confusion_matrix_and_report():
    cm = confusion_matrix([0, 0, 1, 2], [0, 1, 1, 2], num_classes=3)
    np.testing.assert_array_equal(cm, [[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    rep = classification_report(cm, CLASSES)
    assert rep["A"]["recall"] == pytest.approx(0.5)
    assert rep["B"]["precision"] == pytest.approx(0.5)
    assert rep["C"]["f1"] == pytest.approx(1.0)
    assert rep["accuracy"] == pytest.approx(0.75)


def test_confusion_matrix_rejects_out_of_range_labels():
    with pytest.raises(ValueError):
        confusion_matrix([0, 3], [0, 1], num_classes=3)


def test_evaluate_predictions_end_to_end():
    snr = np.array([18, 18, 18, 18, 0, 0])
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 2, 1, 0])
    res = evaluate_predictions(
        y_true, y_pred, snr, CLASSES, split="test", expected_highest_snr=18, target_peak_accuracy=0.9
    )
    assert res["highest_snr"] == 18
    assert res["n_samples_highest_snr"] == 4
    assert res["peak_accuracy_highest_snr"] == pytest.approx(0.75)
    assert res["overall_accuracy"] == pytest.approx(4 / 6)
    assert res["meets_target"] is False
    assert np.sum(res["confusion_matrix_highest_snr"]) == 4
    assert res["split"] == "test"


def test_evaluate_predictions_detects_unexpected_highest_snr():
    with pytest.raises(ValueError, match="Highest SNR"):
        evaluate_predictions([0], [0], [16], CLASSES, split="val", expected_highest_snr=18)


def test_input_validation():
    with pytest.raises(ValueError):
        peak_accuracy_highest_snr([0, 1], [0], [18, 18])
    with pytest.raises(ValueError):
        evaluate_predictions([0], [0], [18], CLASSES, split="holdout")
