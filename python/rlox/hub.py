"""HuggingFace Hub integration for saving and loading rlox models.

Usage::

    from rlox.hub import push_to_hub, load_from_hub

    trainer = PPOTrainer(env="CartPole-v1")
    trainer.train(50_000)
    push_to_hub(trainer, repo_id="username/ppo-cartpole", commit_message="PPO CartPole 50K")

    loaded = load_from_hub("username/ppo-cartpole", PPOTrainer)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

_CHECKPOINT_FILENAME = "model.pt"
_CONFIG_FILENAME = "config.json"
_README_FILENAME = "README.md"


def _require_huggingface_hub() -> Any:
    """Lazy import huggingface_hub, raising a clear error if missing."""
    try:
        import huggingface_hub
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for Hub integration. "
            "Install it with: pip install huggingface_hub"
        ) from exc
    return huggingface_hub


def _extract_info(trainer_or_algo: Any) -> dict[str, Any]:
    """Extract algorithm name, env_id, and config from a trainer or algorithm."""
    algo = getattr(trainer_or_algo, "algo", trainer_or_algo)
    algo_name = type(algo).__name__
    env_id = getattr(trainer_or_algo, "env", None) or getattr(algo, "env_id", "unknown")
    config = {}
    if hasattr(algo, "config") and hasattr(algo.config, "to_dict"):
        config = algo.config.to_dict()
    return {"algo_name": algo_name, "env_id": env_id, "config": config}


def create_model_card(
    algo_name: str,
    env_id: str,
    metrics: dict[str, float] | None = None,
    library_name: str = "rlox",
) -> str:
    """Create a HuggingFace model card in markdown format.

    Parameters
    ----------
    algo_name : str
        Algorithm name (e.g. "PPO", "SAC", "DQN").
    env_id : str
        Gymnasium environment ID.
    metrics : dict, optional
        Training metrics to display.
    library_name : str
        Library identifier for the model card metadata.

    Returns
    -------
    str
        Markdown model card string.
    """
    metrics = metrics or {}

    metrics_table = ""
    if metrics:
        rows = "\n".join(f"| {k} | {v:.4f} |" for k, v in metrics.items())
        metrics_table = f"""
## Metrics

| Metric | Value |
|--------|-------|
{rows}
"""

    return f"""---
library_name: {library_name}
tags:
- reinforcement-learning
- {algo_name.lower()}
- {env_id}
---

# {algo_name} on {env_id}

Trained with [rlox](https://github.com/rlox-ai/rlox) -- Rust-accelerated reinforcement learning.

## Algorithm

- **Algorithm**: {algo_name}
- **Environment**: {env_id}
{metrics_table}
## Usage

```python
from rlox.hub import load_from_hub
from rlox.trainers import {algo_name}Trainer

trainer = load_from_hub("{algo_name.lower()}-{env_id.lower()}", {algo_name}Trainer)
```
"""


def push_to_hub(
    trainer_or_algo: Any,
    repo_id: str,
    commit_message: str | None = None,
    private: bool = False,
    token: str | None = None,
    metrics: dict[str, float] | None = None,
) -> str:
    """Push a trained model to the HuggingFace Hub.

    Parameters
    ----------
    trainer_or_algo
        A trainer (PPOTrainer, SACTrainer, DQNTrainer) or algorithm instance.
    repo_id : str
        Hub repository ID (e.g. "username/ppo-cartpole").
    commit_message : str, optional
        Git commit message for the upload.
    private : bool
        Whether the repo should be private.
    token : str, optional
        HuggingFace API token. If None, uses cached token.
    metrics : dict, optional
        Training metrics to include in the model card.

    Returns
    -------
    str
        URL of the uploaded model.
    """
    hf_hub = _require_huggingface_hub()
    api = hf_hub.HfApi(token=token)

    info = _extract_info(trainer_or_algo)
    algo = getattr(trainer_or_algo, "algo", trainer_or_algo)

    if commit_message is None:
        commit_message = f"Upload {info['algo_name']} trained on {info['env_id']}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Save checkpoint
        checkpoint_path = str(tmppath / _CHECKPOINT_FILENAME)
        algo.save(checkpoint_path)

        # Save config
        config_data = {
            "algo_name": info["algo_name"],
            "env_id": info["env_id"],
            "config": info["config"],
        }
        (tmppath / _CONFIG_FILENAME).write_text(json.dumps(config_data, indent=2))

        # Save model card
        card = create_model_card(
            algo_name=info["algo_name"],
            env_id=info["env_id"],
            metrics=metrics,
        )
        (tmppath / _README_FILENAME).write_text(card)

        # Create repo and upload
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        url = api.upload_folder(
            folder_path=tmpdir,
            repo_id=repo_id,
            commit_message=commit_message,
        )

    return url


def load_from_hub(
    repo_id: str,
    trainer_cls: type[T] | None = None,
    env_id: str | None = None,
    token: str | None = None,
) -> T:
    """Load a model from the HuggingFace Hub.

    Parameters
    ----------
    repo_id : str
        Hub repository ID (e.g. "username/ppo-cartpole").
    trainer_cls : type, optional
        Trainer class to instantiate (PPOTrainer, SACTrainer, DQNTrainer).
        If None, auto-detects from config.json.
    env_id : str, optional
        Override the environment ID from the config.
    token : str, optional
        HuggingFace API token.

    Returns
    -------
    Trainer or algorithm instance.
    """
    hf_hub = _require_huggingface_hub()

    # Download checkpoint and config
    checkpoint_path = hf_hub.hf_hub_download(
        repo_id=repo_id, filename=_CHECKPOINT_FILENAME, token=token
    )
    config_path = hf_hub.hf_hub_download(
        repo_id=repo_id, filename=_CONFIG_FILENAME, token=token
    )

    config_data = json.loads(Path(config_path).read_text())
    resolved_env = env_id or config_data.get("env_id")

    # Auto-detect trainer class if not provided
    if trainer_cls is None:
        trainer_cls = _resolve_trainer_cls(config_data.get("algo_name", ""))

    return trainer_cls.from_checkpoint(checkpoint_path, env=resolved_env)


def _resolve_trainer_cls(algo_name: str) -> type:
    """Map algorithm name to trainer class."""
    from rlox.trainers import DQNTrainer, PPOTrainer, SACTrainer

    mapping: dict[str, type] = {
        "PPO": PPOTrainer,
        "SAC": SACTrainer,
        "DQN": DQNTrainer,
    }
    cls = mapping.get(algo_name)
    if cls is None:
        raise ValueError(
            f"Unknown algorithm '{algo_name}'. "
            f"Supported: {list(mapping.keys())}. "
            f"Pass trainer_cls explicitly."
        )
    return cls
