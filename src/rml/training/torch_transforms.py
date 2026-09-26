"""PyTorch versions of :mod:`rml.data.transforms`, applied per batch on the device.

Used when inputs are too large to transform ahead of time (RML2018.01a). Each
function matches its NumPy counterpart (checked in tests) and, like it, depends
only on the sample itself.
"""

from __future__ import annotations

import math

import torch

from rml.data.transforms import FEATURES, NORMALIZERS


def normalize_per_sample_power(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    power = (x[:, 0, :] ** 2 + x[:, 1, :] ** 2).mean(dim=1)
    return x * torch.rsqrt(power.clamp_min(eps))[:, None, None]


def instantaneous_frequency(x: torch.Tensor) -> torch.Tensor:
    i, q = x[:, 0, :], x[:, 1, :]
    re = i[:, 1:] * i[:, :-1] + q[:, 1:] * q[:, :-1]
    im = q[:, 1:] * i[:, :-1] - i[:, 1:] * q[:, :-1]
    return torch.nn.functional.pad(torch.atan2(im, re) / math.pi, (1, 0))


def features_iq_amp_phase(x: torch.Tensor) -> torch.Tensor:
    i, q = x[:, 0, :], x[:, 1, :]
    return torch.stack([i, q, torch.sqrt(i**2 + q**2), torch.atan2(q, i) / math.pi], dim=1)


_NORMALIZERS = {"none": lambda x: x, "per_sample_power": normalize_per_sample_power}
_FEATURES = {
    "iq": lambda x: x,
    "iq_amp_phase": features_iq_amp_phase,
    "iq_amp_phase_if": lambda x: torch.cat([features_iq_amp_phase(x), instantaneous_frequency(x)[:, None, :]], dim=1),
}
assert set(_NORMALIZERS) == set(NORMALIZERS) and set(_FEATURES) == set(FEATURES), "keep in sync with rml.data.transforms"


def prepare_inputs_torch(x: torch.Tensor, normalize: str, features: str) -> torch.Tensor:
    """float32 normalization then features for a (B, 2, T) batch (any float dtype in)."""
    if normalize not in _NORMALIZERS:
        raise ValueError(f"Unknown normalization {normalize!r}; options: {sorted(_NORMALIZERS)}")
    if features not in _FEATURES:
        raise ValueError(f"Unknown features {features!r}; options: {sorted(_FEATURES)}")
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"Expected (B, 2, T) IQ samples, got {tuple(x.shape)}")
    return _FEATURES[features](_NORMALIZERS[normalize](x.float()))
