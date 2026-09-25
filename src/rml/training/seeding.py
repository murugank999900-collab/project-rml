"""Reproducible seeding. PyTorch is optional."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> dict:
    """Seed Python, NumPy and (if installed) PyTorch; return the settings applied.

    Deterministic mode makes GPU runs repeatable where PyTorch supports it, but
    bit-exact results across GPUs, drivers or library versions are not guaranteed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects subprocesses only
    info: dict = {"seed": seed, "deterministic": deterministic, "torch_seeded": False}
    try:
        import torch
    except ImportError:
        return info

    if deterministic:
        # Required by cuBLAS for deterministic results; must be set before CUDA work starts.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    info.update(
        torch_seeded=True,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )
    return info
