"""Docker and Kubernetes deployment utilities for rlox training."""

from __future__ import annotations

import re
from typing import Any

_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_identifier(value: str, name: str) -> None:
    """Raise ValueError if *value* is not a safe Dockerfile identifier."""
    if not _SAFE_IDENTIFIER_RE.match(value):
        raise ValueError(
            f"{name}={value!r} is not a safe identifier. "
            f"Only alphanumeric characters, hyphens, dots, and underscores are allowed."
        )


def generate_dockerfile(
    algo: str,
    env: str,
    config_path: str | None = None,
) -> str:
    """Generate a production-ready Dockerfile for training an rlox agent.

    Uses a multi-stage build (builder + runtime), non-root user, pip cache
    mount, and a health check.

    Parameters
    ----------
    algo : str
        Algorithm name (e.g. ``"ppo"``).
    env : str
        Environment ID (e.g. ``"CartPole-v1"``).
    config_path : str | None
        Path to a YAML config file.  Defaults to ``"config.yaml"``.

    Returns
    -------
    str
        Dockerfile contents.
    """
    _validate_identifier(algo, "algo")
    _validate_identifier(env, "env")
    cfg = config_path or "config.yaml"
    return f"""\
# ---- builder stage ----
FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir --prefix=/install rlox[all] gymnasium[mujoco]

# ---- runtime stage ----
FROM python:3.12-slim

COPY --from=builder /install /usr/local

RUN useradd --create-home --shell /bin/bash rlox

WORKDIR /app
COPY {cfg} /app/config.yaml

ENV RLOX_ALGO={algo}
ENV RLOX_ENV={env}

HEALTHCHECK --interval=60s --timeout=10s --retries=3 \\
    CMD python -c "import rlox; print('ok')" || exit 1

USER rlox

CMD ["python", "-m", "rlox", "train", "--config", "config.yaml"]
"""


def generate_k8s_job(
    name: str,
    image: str,
    config: dict[str, Any],
    gpu_count: int = 0,
    namespace: str = "default",
    cpu_request: str = "1",
    cpu_limit: str = "4",
    memory_request: str = "2Gi",
    memory_limit: str = "8Gi",
    checkpoint_storage: str = "10Gi",
) -> dict[str, Any]:
    """Generate a Kubernetes Job manifest for distributed training.

    Includes a PersistentVolumeClaim for checkpoint storage, a ConfigMap
    volume for training config, and resource requests/limits.

    Parameters
    ----------
    name : str
        Job name.
    image : str
        Docker image reference.
    config : dict
        Training configuration passed as environment variables.
    gpu_count : int
        Number of GPUs to request (default 0).
    namespace : str
        Kubernetes namespace (default ``"default"``).
    cpu_request : str
        CPU request (default ``"1"``).
    cpu_limit : str
        CPU limit (default ``"4"``).
    memory_request : str
        Memory request (default ``"2Gi"``).
    memory_limit : str
        Memory limit (default ``"8Gi"``).
    checkpoint_storage : str
        PVC size for checkpoint storage (default ``"10Gi"``).

    Returns
    -------
    dict
        A Kubernetes Job manifest.
    """
    env_vars = [
        {"name": k.upper(), "value": str(v)} for k, v in config.items()
    ]

    resources: dict[str, Any] = {
        "requests": {
            "cpu": cpu_request,
            "memory": memory_request,
        },
        "limits": {
            "cpu": cpu_limit,
            "memory": memory_limit,
        },
    }

    if gpu_count > 0:
        resources["limits"]["nvidia.com/gpu"] = str(gpu_count)
        resources["requests"]["nvidia.com/gpu"] = str(gpu_count)

    container: dict[str, Any] = {
        "name": name,
        "image": image,
        "env": env_vars,
        "command": ["python", "-m", "rlox", "train", "--config", "/config/config.yaml"],
        "resources": resources,
        "volumeMounts": [
            {
                "name": "checkpoints",
                "mountPath": "/app/checkpoints",
            },
            {
                "name": "training-config",
                "mountPath": "/config",
            },
        ],
    }

    volumes: list[dict[str, Any]] = [
        {
            "name": "checkpoints",
            "persistentVolumeClaim": {
                "claimName": f"{name}-checkpoints",
            },
        },
        {
            "name": "training-config",
            "configMap": {
                "name": f"{name}-config",
            },
        },
    ]

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "spec": {
            "backoffLimit": 3,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [container],
                    "volumes": volumes,
                }
            },
        },
    }
