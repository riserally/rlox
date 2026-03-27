"""Shared utilities for convergence benchmarks."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml


@dataclass
class EvalRecord:
    """Single evaluation checkpoint."""

    step: int
    wall_clock_s: float
    mean_return: float
    std_return: float
    ep_length: float
    sps: float
    training_sps: float = 0.0  # SPS excluding eval time


@dataclass
class ExperimentLog:
    """Full experiment log matching the plan's JSON schema."""

    framework: str
    algorithm: str
    environment: str
    seed: int
    hyperparameters: dict[str, Any]
    hardware: dict[str, str]
    evaluations: list[EvalRecord] = field(default_factory=list)
    total_wall_clock_s: float = 0.0
    total_steps: int = 0
    mean_sps: float = 0.0
    peak_memory_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "algorithm": self.algorithm,
            "environment": self.environment,
            "seed": self.seed,
            "hyperparameters": self.hyperparameters,
            "hardware": self.hardware,
            "evaluations": [
                {
                    "step": e.step,
                    "wall_clock_s": e.wall_clock_s,
                    "mean_return": e.mean_return,
                    "std_return": e.std_return,
                    "ep_length": e.ep_length,
                    "sps": e.sps,
                }
                for e in self.evaluations
            ],
            "training_metrics": {
                "total_wall_clock_s": self.total_wall_clock_s,
                "total_steps": self.total_steps,
                "mean_sps": self.mean_sps,
                "peak_memory_mb": self.peak_memory_mb,
            },
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_hardware_info() -> dict[str, str]:
    """Collect hardware information for reproducibility."""
    info = {
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda or "N/A"
    elif torch.backends.mps.is_available():
        info["gpu"] = "Apple MPS"
    else:
        info["gpu"] = "None"
    return info


def evaluate_policy_gym(
    env_id: str,
    get_action_fn,
    n_episodes: int = 10,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Evaluate a policy by running n_episodes in a fresh gym env.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID.
    get_action_fn : callable
        Takes np.ndarray observation, returns np.ndarray action.
    n_episodes : int
        Number of evaluation episodes.
    seed : int
        Seed for the evaluation environment.

    Returns
    -------
    mean_return, std_return, mean_ep_length
    """
    env = gym.make(env_id)
    returns = []
    lengths = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        ep_len = 0
        while not done:
            action = get_action_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            ep_len += 1
            done = terminated or truncated
        returns.append(ep_return)
        lengths.append(ep_len)

    env.close()
    return float(np.mean(returns)), float(np.std(returns)), float(np.mean(lengths))


def result_path(
    results_dir: Path,
    framework: str,
    algorithm: str,
    environment: str,
    seed: int,
) -> Path:
    """Construct a standardized result file path."""
    env_clean = environment.replace("/", "_")
    return results_dir / f"{framework}_{algorithm}_{env_clean}_seed{seed}.json"
