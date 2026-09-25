import pytest

from rml.training.selection import BestTracker

M, TB = "val_peak_accuracy_highest_snr", "val_accuracy_high_snr"


def m(peak, high=0.0):
    return {M: peak, TB: high, "test_peak_accuracy_highest_snr": 1.0}


def test_selects_on_primary_metric():
    t = BestTracker(M, TB)
    assert t.update(1, m(0.80))
    assert not t.update(2, m(0.79, high=0.99))
    assert t.update(3, m(0.85))
    assert t.best_epoch == 3


def test_tie_breaker_only_on_exact_tie():
    t = BestTracker(M, TB)
    t.update(1, m(0.80, 0.70))
    assert not t.update(2, m(0.80, 0.70))  # equal: not an improvement
    assert t.update(3, m(0.80, 0.71))
    assert t.best_epoch == 3


def test_test_metrics_ignored_for_selection():
    t = BestTracker(M)
    t.update(1, {M: 0.9, "test_peak_accuracy_highest_snr": 0.1})
    assert not t.update(2, {M: 0.8, "test_peak_accuracy_highest_snr": 1.0})
    assert t.best_epoch == 1


@pytest.mark.parametrize("metric", ["test_peak_accuracy_highest_snr", "peak_accuracy_highest_snr", "train_loss"])
def test_rejects_non_validation_metrics(metric):
    with pytest.raises(ValueError, match="validation"):
        BestTracker(metric)
    with pytest.raises(ValueError, match="validation"):
        BestTracker(M, tie_breaker=metric)


def test_early_stopping_patience():
    t = BestTracker(M, patience=2)
    t.update(1, m(0.5))
    t.update(2, m(0.4))
    assert not t.should_stop
    t.update(3, m(0.4))
    assert t.should_stop
    t.update(4, m(0.6))
    assert not t.should_stop


def test_no_patience_never_stops():
    t = BestTracker(M)
    for e in range(1, 50):
        t.update(e, m(0.1))
    assert not t.should_stop
