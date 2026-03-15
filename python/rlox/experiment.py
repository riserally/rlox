"""Experiment metadata capture and persistence."""

from __future__ import annotations

import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def capture_experiment_metadata(
    config: dict[str, Any] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Capture experiment metadata: git hash, versions, platform, timestamp.

    Robust against missing git, missing torch, and other failures.
    All fields are populated on a best-effort basis.
    """
    meta: dict[str, Any] = {
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
    }

    # Git hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        meta["git_hash"] = result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        meta["git_hash"] = None

    # Git dirty status
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            meta["git_dirty"] = bool(result.stdout.strip())
        else:
            meta["git_dirty"] = None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        meta["git_dirty"] = None

    # rlox version
    try:
        import rlox

        meta["rlox_version"] = rlox.__version__
    except (ImportError, AttributeError):
        pass

    # PyTorch version
    try:
        import torch

        meta["torch_version"] = torch.__version__
    except ImportError:
        pass

    # NumPy version
    try:
        import numpy

        meta["numpy_version"] = numpy.__version__
    except ImportError:
        pass

    if seed is not None:
        meta["seed"] = seed
    if config is not None:
        meta["config"] = config

    return meta


def save_experiment(
    path: str | Path,
    config: dict[str, Any],
    seed: int | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Save experiment metadata + results as a JSON file.

    Parameters
    ----------
    path : file path to write (will be created/overwritten)
    config : experiment configuration dict
    seed : random seed used
    metrics : training results / evaluation metrics

    Returns
    -------
    The resolved Path where the file was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = capture_experiment_metadata(config=config, seed=seed)
    if metrics is not None:
        data["metrics"] = metrics

    path.write_text(json.dumps(data, indent=2, default=str))
    return path
