"""Unique, never-reused run directories."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")
    if not slug:
        raise ValueError(f"Cannot build a run id from {text!r}")
    return slug


def make_run_id(experiment: str, seed: int, git_sha: str | None, now: datetime | None = None) -> str:
    """``<UTC timestamp>_<experiment>_s<seed>_<short git sha>``."""
    now = now or datetime.now(timezone.utc)
    sha = (git_sha or "nogit")[:7]
    return f"{now:%Y%m%d-%H%M%S}_{_slug(experiment)}_s{seed}_{sha}"


def create_run_dir(root: str | Path, run_id: str, max_attempts: int = 100) -> Path:
    """Create a new ``root/run_id`` directory (with ``checkpoints/``) and return it.

    An existing directory is never reused: if ``run_id`` is taken, ``_r2``,
    ``_r3``, ... are appended. The run id actually used is the directory name.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_attempts + 1):
        path = root / (run_id if attempt == 1 else f"{run_id}_r{attempt}")
        try:
            path.mkdir()
        except FileExistsError:
            continue
        (path / "checkpoints").mkdir()
        return path
    raise FileExistsError(f"Could not create a unique run directory for {run_id!r} in {root}")
