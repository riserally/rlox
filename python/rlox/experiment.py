"""Experiment metadata capture."""

from __future__ import annotations

import datetime
import platform
import subprocess
import sys
from typing import Any


def capture_experiment_metadata(
    config: dict[str, Any] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Capture experiment metadata: git hash, versions, platform, timestamp."""
    meta: dict[str, Any] = {
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
    }

    # Try to capture git hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            meta["git_hash"] = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        meta["git_hash"] = None

    # Try to get torch version
    try:
        import torch
        meta["torch_version"] = torch.__version__
    except ImportError:
        pass

    if seed is not None:
        meta["seed"] = seed
    if config is not None:
        meta["config"] = config

    return meta
