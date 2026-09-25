"""Reproducibility metadata: file hashes, Git state and runtime environment."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Run outputs live under these paths, so their changes do not make the source tree "dirty".
_OUTPUT_PREFIXES = ("experiments/", "results/")


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def git_info(repo: str | Path) -> dict:
    """Commit SHA, branch and uncommitted source changes (run outputs excluded)."""
    repo = Path(repo)
    try:
        sha = _git(repo, "rev-parse", "HEAD")
        branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
        status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError):
        return {"available": False, "sha": None, "branch": None, "dirty": None, "dirty_files": []}
    dirty_files = [line[3:] for line in status.splitlines() if line and not line[3:].startswith(_OUTPUT_PREFIXES)]
    return {"available": True, "sha": sha, "branch": branch, "dirty": bool(dirty_files), "dirty_files": dirty_files}


def environment_info() -> dict:
    """Python/NumPy/PyTorch/CUDA/GPU details. Works without PyTorch installed."""
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "kaggle": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")),
        "colab": "COLAB_RELEASE_TAG" in os.environ,
        "torch": None,
    }
    try:
        import torch
    except ImportError:
        return info
    cuda = torch.cuda.is_available()
    info["torch"] = {
        "version": torch.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda else None,
        "gpus": [
            {
                "name": torch.cuda.get_device_name(i),
                "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
            }
            for i in range(torch.cuda.device_count())
        ]
        if cuda
        else [],
    }
    return info


def write_json(path: str | Path, obj, exclusive: bool = True) -> Path:
    """Write JSON; by default refuse to overwrite an existing file."""
    path = Path(path)
    with open(path, "x" if exclusive else "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=_json_default)
        f.write("\n")
    return path


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o).__name__}")
