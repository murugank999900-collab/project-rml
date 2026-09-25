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


# --- Experiment configs -------------------------------------------------------

REQUIRED_SECTIONS = ("dataset", "experiment", "model", "data", "train", "selection", "output")


def deep_merge(base: Mapping, override: Mapping) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, Mapping) and isinstance(out.get(k), Mapping) else v
    return out


def apply_override(cfg: dict, assignment: str) -> None:
    """Apply a ``dotted.key=value`` override in place; ``value`` is parsed as YAML."""
    key, sep, raw = assignment.partition("=")
    if not sep or not key:
        raise ValueError(f"Override must look like key.path=value, got {assignment!r}")
    *parents, leaf = key.strip().split(".")
    node = cfg
    for p in parents:
        if not isinstance(node.get(p), dict):
            raise KeyError(f"Override {assignment!r}: '{p}' is not a config section")
        node = node[p]
    if leaf not in node:
        raise KeyError(f"Override {assignment!r}: unknown key '{leaf}'")
    node[leaf] = yaml.safe_load(raw)


def load_experiment_config(path: str | Path, overrides: tuple[str, ...] | list[str] = ()) -> dict:
    """Load an experiment config, merged over the file named by its ``base`` key."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base_path = raw.pop("base", None)
    cfg = deep_merge(load_config(resolve_project_path(base_path)), raw) if base_path else raw
    for o in overrides:
        apply_override(cfg, o)
    validate_experiment_config(cfg)
    return cfg


def validate_experiment_config(cfg: Mapping) -> None:
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"Experiment config missing sections: {missing}")
    sel = cfg["selection"]
    for key in ("metric", "tie_breaker"):
        name = sel.get(key)
        if name is not None and not str(name).startswith("val_"):
            raise ValueError(f"selection.{key} must be a validation metric (val_*), got {name!r}")
    if cfg["data"].get("train_snrs", "all") != "all":
        raise ValueError("data.train_snrs must be 'all': training uses the complete official training split")
    if not isinstance(cfg["experiment"].get("seed"), int):
        raise ValueError("experiment.seed must be an integer")
