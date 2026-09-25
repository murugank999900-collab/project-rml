"""Stateless per-sample input transforms (NumPy only).

Every transform depends only on the sample itself, so applying it to any split
cannot leak information between splits.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def normalize_none(X: np.ndarray) -> np.ndarray:
    return X.astype(np.float32, copy=False)


def normalize_per_sample_power(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale each (2, T) IQ sample to unit average power: mean(I^2 + Q^2) = 1."""
    power = np.mean(X[:, 0, :] ** 2 + X[:, 1, :] ** 2, axis=1, dtype=np.float64)
    scale = 1.0 / np.sqrt(np.maximum(power, eps))
    return (X * scale[:, None, None]).astype(np.float32)


def features_iq(X: np.ndarray) -> np.ndarray:
    return X.astype(np.float32, copy=False)


def features_iq_amp_phase(X: np.ndarray) -> np.ndarray:
    """(N, 2, T) IQ -> (N, 4, T): I, Q, amplitude, phase / pi (in [-1, 1])."""
    i, q = X[:, 0, :], X[:, 1, :]
    amp = np.sqrt(i**2 + q**2)
    phase = np.arctan2(q, i) / np.pi
    return np.stack([i, q, amp, phase], axis=1).astype(np.float32)


NORMALIZERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "none": normalize_none,
    "per_sample_power": normalize_per_sample_power,
}

FEATURES: dict[str, tuple[Callable[[np.ndarray], np.ndarray], int]] = {
    "iq": (features_iq, 2),
    "iq_amp_phase": (features_iq_amp_phase, 4),
}


def feature_channels(features: str) -> int:
    return FEATURES[features][1]


def prepare_inputs(X: np.ndarray, normalize: str, features: str) -> np.ndarray:
    """Apply the configured normalization then feature transform to (N, 2, T) IQ samples."""
    if normalize not in NORMALIZERS:
        raise ValueError(f"Unknown normalization {normalize!r}; options: {sorted(NORMALIZERS)}")
    if features not in FEATURES:
        raise ValueError(f"Unknown features {features!r}; options: {sorted(FEATURES)}")
    if X.ndim != 3 or X.shape[1] != 2:
        raise ValueError(f"Expected (N, 2, T) IQ samples, got {X.shape}")
    return FEATURES[features][0](NORMALIZERS[normalize](X))
