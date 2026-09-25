import numpy as np
import pytest

from rml.data.transforms import (
    feature_channels,
    features_iq_amp_phase,
    features_iq_amp_phase_if,
    instantaneous_frequency,
    normalize_per_sample_power,
    prepare_inputs,
)


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


# --- instantaneous frequency (iq_amp_phase_if) ---------------------------------


def tone(freq, t=128, amp=1.0, phase0=0.0):
    """Complex tone at ``freq`` cycles/sample as a (1, 2, T) IQ array."""
    z = amp * np.exp(1j * (2 * np.pi * freq * np.arange(t) + phase0))
    return np.stack([z.real, z.imag])[None].astype(np.float32)


@pytest.mark.parametrize("freq", [0.0, 0.05, -0.12, 0.3])
def test_inst_freq_constant_tone(freq):
    f = instantaneous_frequency(tone(freq, amp=2.5, phase0=1.0))
    assert f.shape == (1, 128) and f.dtype == np.float32
    assert f[0, 0] == 0.0
    np.testing.assert_allclose(f[0, 1:], 2 * freq, atol=1e-5)  # delta-phase 2*pi*f, divided by pi


@pytest.mark.parametrize("freq", [0.45, -0.45, 0.49])
def test_inst_freq_is_continuous_across_phase_wrap(freq):
    X = tone(freq)
    phase = features_iq_amp_phase(X)[0, 3]
    assert np.abs(np.diff(phase)).max() > 1.0  # the wrapped phase channel jumps at +/-pi
    np.testing.assert_allclose(instantaneous_frequency(X)[0, 1:], 2 * freq, atol=1e-4)


def test_inst_freq_matches_definition_on_random_signals():
    X = iq(n=6, seed=3, scale=7.0)
    z = X[:, 0].astype(np.float64) + 1j * X[:, 1].astype(np.float64)
    expected = np.angle(z[:, 1:] * np.conj(z[:, :-1])) / np.pi
    f = instantaneous_frequency(X)
    np.testing.assert_allclose(f[:, 1:], expected, atol=1e-6)
    assert np.all(f[:, 0] == 0)


def test_inst_freq_range_and_finite():
    X = np.concatenate([iq(n=50, seed=4, scale=1e3), np.zeros((1, 2, 128), dtype=np.float32), tone(0.5)])
    f = instantaneous_frequency(X)
    assert np.all(np.isfinite(f))
    assert f.min() >= -1.0 and f.max() <= 1.0


def test_iq_amp_phase_if_first_four_channels_unchanged():
    X = normalize_per_sample_power(iq(n=8, seed=5))
    F5 = features_iq_amp_phase_if(X)
    assert F5.shape == (8, 5, 128) and F5.dtype == np.float32 and feature_channels("iq_amp_phase_if") == 5
    np.testing.assert_array_equal(F5[:, :4], features_iq_amp_phase(X))
    np.testing.assert_array_equal(F5[:, 4], instantaneous_frequency(X))
    np.testing.assert_array_equal(
        prepare_inputs(X, "per_sample_power", "iq_amp_phase_if")[:, :4],
        prepare_inputs(X, "per_sample_power", "iq_amp_phase"),
    )


def test_iq_amp_phase_if_sample_independence():
    X = iq(n=5, seed=6)
    alone = prepare_inputs(X[2:3], "per_sample_power", "iq_amp_phase_if")
    batch = prepare_inputs(np.concatenate([X[:2] * 50, X[2:3], X[3:] * 0.01]), "per_sample_power", "iq_amp_phase_if")
    np.testing.assert_array_equal(alone[0], batch[2])
