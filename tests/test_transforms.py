import numpy as np
import pytest

from rml.data.transforms import feature_channels, normalize_per_sample_power, prepare_inputs


def iq(n=5, t=128, seed=0, scale=1.0):
    return (np.random.default_rng(seed).standard_normal((n, 2, t)) * scale).astype(np.float32)


def test_per_sample_power_gives_unit_power():
    X = iq() * np.array([0.01, 1, 5, 100, 3], dtype=np.float32)[:, None, None]
    Xn = normalize_per_sample_power(X)
    power = np.mean(Xn[:, 0] ** 2 + Xn[:, 1] ** 2, axis=1)
    np.testing.assert_allclose(power, 1.0, rtol=1e-5)
    assert Xn.dtype == np.float32 and Xn.shape == X.shape


def test_per_sample_power_is_per_sample():
    """A sample's output must not depend on other samples in the batch (no cross-split leakage)."""
    X = iq(n=4)
    alone = normalize_per_sample_power(X[:1])
    together = normalize_per_sample_power(np.concatenate([X[:1], X[1:] * 1000]))
    np.testing.assert_allclose(alone[0], together[0], rtol=1e-6)


def test_zero_sample_does_not_produce_nan():
    assert np.all(np.isfinite(normalize_per_sample_power(np.zeros((1, 2, 128), dtype=np.float32))))


def test_iq_amp_phase_features():
    X = np.zeros((1, 2, 4), dtype=np.float32)
    X[0, 0] = [1, 0, -1, 3]
    X[0, 1] = [0, 1, 0, 4]
    F = prepare_inputs(X, "none", "iq_amp_phase")
    assert F.shape == (1, 4, 4) and feature_channels("iq_amp_phase") == 4
    np.testing.assert_allclose(F[0, 0], X[0, 0])
    np.testing.assert_allclose(F[0, 1], X[0, 1])
    np.testing.assert_allclose(F[0, 2], [1, 1, 1, 5])
    np.testing.assert_allclose(F[0, 3], [0, 0.5, 1, np.arctan2(4, 3) / np.pi], rtol=1e-6)
    assert np.all(np.abs(F[0, 3]) <= 1)


def test_prepare_inputs_validates():
    with pytest.raises(ValueError, match="normalization"):
        prepare_inputs(iq(), "zscore", "iq")
    with pytest.raises(ValueError, match="features"):
        prepare_inputs(iq(), "none", "spectrogram")
    with pytest.raises(ValueError, match="IQ"):
        prepare_inputs(np.zeros((2, 3, 128), dtype=np.float32), "none", "iq")
