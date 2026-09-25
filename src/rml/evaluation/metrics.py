"""Evaluation metrics for AMC, with accuracy at the highest SNR as the primary metric.

``peak_accuracy_highest_snr`` is the accuracy on the samples whose SNR equals
the maximum SNR present in the evaluated data. The highest SNR is always
derived from the data, never assumed; :func:`verify_highest_snr` checks it
against the expected value from configuration.

Model selection must use the validation split. Results from the test split
are for final reporting only.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _as_1d_int(name: str, a) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size and not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.equal(np.mod(arr, 1), 0)):
            raise ValueError(f"{name} must contain integers")
    return arr.astype(np.int64)


def _check_inputs(y_true, y_pred, snr=None):
    y_true = _as_1d_int("y_true", y_true)
    y_pred = _as_1d_int("y_pred", y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred lengths differ: {y_true.size} vs {y_pred.size}")
    if y_true.size == 0:
        raise ValueError("Cannot evaluate an empty set")
    if snr is None:
        return y_true, y_pred
    snr = _as_1d_int("snr", snr)
    if snr.shape != y_true.shape:
        raise ValueError(f"snr length {snr.size} does not match labels length {y_true.size}")
    return y_true, y_pred, snr


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = _check_inputs(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def highest_snr(snr) -> int:
    """Highest SNR present in the evaluated data."""
    snr = _as_1d_int("snr", snr)
    if snr.size == 0:
        raise ValueError("Cannot determine highest SNR of an empty set")
    return int(snr.max())


def verify_highest_snr(snr, expected: int) -> int:
    """Return the highest SNR in ``snr``; raise ValueError if it differs from ``expected``."""
    found = highest_snr(snr)
    if found != int(expected):
        raise ValueError(f"Highest SNR in data is {found} dB, expected {expected} dB")
    return found


def accuracy_by_snr(y_true, y_pred, snr) -> dict[int, dict[str, float | int]]:
    """Map each SNR (ascending) to its accuracy, sample count and correct count."""
    y_true, y_pred, snr = _check_inputs(y_true, y_pred, snr)
    out = {}
    for s in np.unique(snr):
        mask = snr == s
        correct = int(np.sum(y_true[mask] == y_pred[mask]))
        n = int(mask.sum())
        out[int(s)] = {"accuracy": correct / n, "n": n, "correct": correct}
    return out


def peak_accuracy_highest_snr(y_true, y_pred, snr) -> float:
    """Accuracy on the samples at the highest SNR present in ``snr``."""
    y_true, y_pred, snr = _check_inputs(y_true, y_pred, snr)
    mask = snr == snr.max()
    return float(np.mean(y_true[mask] == y_pred[mask]))


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    """Rows are true classes, columns are predicted classes."""
    y_true, y_pred = _check_inputs(y_true, y_pred)
    for name, arr in (("y_true", y_true), ("y_pred", y_pred)):
        if arr.min() < 0 or arr.max() >= num_classes:
            raise ValueError(f"{name} contains labels outside [0, {num_classes})")
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def classification_report(cm: np.ndarray, class_names: Sequence[str]) -> dict:
    """Per-class precision/recall/F1/support plus macro and weighted averages."""
    cm = np.asarray(cm)
    if cm.shape != (len(class_names), len(class_names)):
        raise ValueError(f"Confusion matrix shape {cm.shape} does not match {len(class_names)} classes")
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)

    report = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    }
    total = support.sum()
    weights = support / total if total else np.zeros_like(support)
    report["macro_avg"] = {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
        "support": int(total),
    }
    report["weighted_avg"] = {
        "precision": float((precision * weights).sum()),
        "recall": float((recall * weights).sum()),
        "f1": float((f1 * weights).sum()),
        "support": int(total),
    }
    report["accuracy"] = float(tp.sum() / total) if total else 0.0
    return report


def format_classification_report(report: dict) -> str:
    rows = [k for k in report if k not in ("accuracy",)]
    width = max(len(r) for r in rows)
    lines = [f"{'':<{width}}  precision  recall     f1  support"]
    for r in rows:
        m = report[r]
        lines.append(f"{r:<{width}}  {m['precision']:9.4f}  {m['recall']:6.4f}  {m['f1']:.4f}  {m['support']:7d}")
    lines.append(f"{'accuracy':<{width}}  {report['accuracy']:.4f}")
    return "\n".join(lines)


def evaluate_predictions(
    y_true,
    y_pred,
    snr,
    class_names: Sequence[str],
    *,
    split: str,
    expected_highest_snr: int | None = None,
    target_peak_accuracy: float | None = None,
) -> dict:
    """Compute the full metric set for one split.

    ``split`` ("val" or "test") is recorded in the result so downstream tooling
    can refuse to select models on test results.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")
    y_true, y_pred, snr = _check_inputs(y_true, y_pred, snr)

    top_snr = highest_snr(snr) if expected_highest_snr is None else verify_highest_snr(snr, expected_highest_snr)
    top_mask = snr == top_snr
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, num_classes)
    cm_top = confusion_matrix(y_true[top_mask], y_pred[top_mask], num_classes)
    peak = peak_accuracy_highest_snr(y_true, y_pred, snr)

    result = {
        "split": split,
        "n_samples": int(y_true.size),
        "overall_accuracy": accuracy(y_true, y_pred),
        "accuracy_by_snr": accuracy_by_snr(y_true, y_pred, snr),
        "highest_snr": top_snr,
        "n_samples_highest_snr": int(top_mask.sum()),
        "peak_accuracy_highest_snr": peak,
        "classification_report": classification_report(cm, class_names),
        "classification_report_highest_snr": classification_report(cm_top, class_names),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_highest_snr": cm_top.tolist(),
        "class_names": list(class_names),
    }
    if target_peak_accuracy is not None:
        result["target_peak_accuracy"] = float(target_peak_accuracy)
        result["meets_target"] = bool(peak >= target_peak_accuracy)
    return result
