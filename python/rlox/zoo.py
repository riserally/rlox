"""Model zoo -- registry of pretrained models with metadata.

Provides :class:`ModelZoo` for discovering and loading pretrained RL models,
and :class:`ModelCard` for structured model metadata.

Example
-------
>>> from rlox.zoo import ModelZoo, ModelCard
>>>
>>> ModelZoo.register("ppo-cartpole", "ppo", "CartPole-v1", "/models/ppo.pt", 475.0)
>>> models = ModelZoo.search(env="CartPole-v1")
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import rlox


# ---------------------------------------------------------------------------
# ModelZoo
# ---------------------------------------------------------------------------

class ModelZoo:
    """Registry of pretrained models with metadata."""

    PRETRAINED: dict[str, dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        algo: str,
        env: str,
        checkpoint_url: str,
        mean_return: float,
        **metadata: Any,
    ) -> None:
        """Register a pretrained model.

        Parameters
        ----------
        name : str
            Unique name for the model (e.g. ``"ppo-cartpole-v1"``).
        algo : str
            Algorithm name.
        env : str
            Environment ID.
        checkpoint_url : str
            Path or URL to the checkpoint file.
        mean_return : float
            Mean evaluation return.
        **metadata
            Additional key-value metadata (e.g. ``training_steps``, ``author``).
        """
        cls.PRETRAINED[name] = {
            "algo": algo,
            "env": env,
            "url": checkpoint_url,
            "mean_return": mean_return,
            **metadata,
        }

    @classmethod
    def load(cls, name: str, device: str = "cpu") -> Any:
        """Load a pretrained model by name.

        Parameters
        ----------
        name : str
            Registered model name.
        device : str
            Device to load to (default ``"cpu"``).

        Returns
        -------
        Trainer
            A :class:`~rlox.trainer.Trainer` restored from the checkpoint.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        info = cls.PRETRAINED[name]
        from rlox.trainer import Trainer

        return Trainer.from_checkpoint(
            info["url"], algorithm=info["algo"], env=info["env"]
        )

    @classmethod
    def list(cls) -> list[dict[str, Any]]:
        """List all available pretrained models."""
        return [{"name": k, **v} for k, v in cls.PRETRAINED.items()]

    @classmethod
    def search(
        cls,
        env: str | None = None,
        algo: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search pretrained models by environment and/or algorithm.

        Parameters
        ----------
        env : str, optional
            Filter by environment ID.
        algo : str, optional
            Filter by algorithm name.

        Returns
        -------
        list[dict]
            Matching model entries.
        """
        results: list[dict[str, Any]] = []
        for name, info in cls.PRETRAINED.items():
            if env is not None and info["env"] != env:
                continue
            if algo is not None and info["algo"] != algo:
                continue
            results.append({"name": name, **info})
        return results


# ---------------------------------------------------------------------------
# ModelCard
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ModelCard:
    """Metadata card for a trained model."""

    algorithm: str
    environment: str
    mean_return: float
    std_return: float
    total_timesteps: int
    training_time_s: float
    hyperparameters: dict[str, Any]
    framework_version: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (round-trippable via ``ModelCard(**d)``)."""
        return asdict(self)

    def to_markdown(self) -> str:
        """Generate a human-readable markdown model card."""
        lines = [
            f"# {self.algorithm} on {self.environment}",
            "",
            "## Performance",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Mean Return | {self.mean_return} |",
            f"| Std Return | {self.std_return} |",
            f"| Total Timesteps | {self.total_timesteps} |",
            f"| Training Time (s) | {self.training_time_s} |",
            "",
            "## Hyperparameters",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
        ]
        for k, v in self.hyperparameters.items():
            lines.append(f"| {k} | {v} |")

        lines.extend([
            "",
            f"*Framework version: {self.framework_version}*",
        ])
        return "\n".join(lines)

    @classmethod
    def from_trainer(cls, trainer: Any, metrics: dict[str, Any]) -> ModelCard:
        """Create a ModelCard from a Trainer and training metrics.

        Parameters
        ----------
        trainer
            A :class:`~rlox.trainer.Trainer` (or similar) instance.
        metrics : dict
            Must contain ``mean_return``, ``std_return``, ``total_timesteps``,
            and ``training_time_s``.

        Returns
        -------
        ModelCard
        """
        algo = trainer.algo
        algo_name = type(algo).__name__
        env_id = getattr(trainer, "env", "unknown")

        # Extract hyperparameters from config if available
        hyperparameters: dict[str, Any] = {}
        if hasattr(algo, "config") and hasattr(algo.config, "to_dict"):
            hyperparameters = algo.config.to_dict()

        return cls(
            algorithm=algo_name,
            environment=env_id,
            mean_return=metrics["mean_return"],
            std_return=metrics["std_return"],
            total_timesteps=metrics["total_timesteps"],
            training_time_s=metrics["training_time_s"],
            hyperparameters=hyperparameters,
            framework_version=rlox.__version__,
        )
