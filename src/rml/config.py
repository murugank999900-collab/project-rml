"""Configuration loading and runtime dataset path resolution.

The dataset location is never hard-coded. It is resolved, in order, from:
  1. an explicit override (e.g. a ``--data`` CLI argument),
  2. the environment variable named by ``dataset.root_env``,
  3. ``dataset.root`` in the config file.
The value may be a directory (``dataset.filename`` is appended) or a file path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "dataset" not in cfg:
        raise ValueError(f"Config {path} must contain a 'dataset' section")
    return cfg


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a repo-relative path (e.g. the split file) against the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def resolve_dataset_file(
    dataset_cfg: Mapping,
    override: str | Path | None = None,
    environ: Mapping[str, str] = os.environ,
) -> Path:
    env_name = dataset_cfg.get("root_env")
    candidates = [
        ("override", override),
        (f"${env_name}" if env_name else "environment", environ.get(env_name) if env_name else None),
        ("dataset.root", dataset_cfg.get("root")),
    ]
    source, root = next(((s, v) for s, v in candidates if v), (None, None))
    if root is None:
        hint = f"set ${env_name}, " if env_name else ""
        raise ValueError(f"Dataset location not configured: {hint}pass --data, or set dataset.root")

    p = Path(root).expanduser()
    if p.is_dir():
        p = p / dataset_cfg["filename"]
    if not p.is_file():
        raise FileNotFoundError(f"Dataset file not found: {p} (from {source})")
    return p
