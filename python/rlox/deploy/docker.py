"""Docker and Kubernetes deployment utilities for rlox training."""

from __future__ import annotations


def generate_dockerfile(
    algo: str,
    env: str,
    config_path: str | None = None,
) -> str:
    """Generate a Dockerfile for training an rlox agent.

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
    cfg = config_path or "config.yaml"
    return f"""\
FROM python:3.12-slim

RUN pip install --no-cache-dir rlox[all] gymnasium[mujoco]

WORKDIR /app
COPY {cfg} /app/config.yaml

ENV RLOX_ALGO={algo}
ENV RLOX_ENV={env}

CMD ["python", "-m", "rlox", "train", "--config", "config.yaml"]
"""


def generate_k8s_job(
    name: str,
    image: str,
    config: dict,
    gpu_count: int = 0,
    namespace: str = "default",
) -> dict:
    """Generate a Kubernetes Job manifest for distributed training.

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

    Returns
    -------
    dict
        A Kubernetes Job manifest.
    """
    env_vars = [
        {"name": k.upper(), "value": str(v)} for k, v in config.items()
    ]

    container: dict = {
        "name": name,
        "image": image,
        "env": env_vars,
        "command": ["python", "-m", "rlox", "train", "--config", "config.yaml"],
    }

    if gpu_count > 0:
        container["resources"] = {
            "limits": {"nvidia.com/gpu": str(gpu_count)},
            "requests": {"nvidia.com/gpu": str(gpu_count)},
        }

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
                }
            },
        },
    }
