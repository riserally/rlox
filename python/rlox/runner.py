"""Config-driven training runner.

Provides :func:`train_from_config` which accepts a :class:`TrainingConfig`
(or a path to a YAML/TOML file) and runs the full training loop, wiring up
the algorithm, callbacks, and logger automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rlox.callbacks import (
    Callback,
    CheckpointCallback,
    EvalCallback,
    ProgressBarCallback,
    TimingCallback,
)
from rlox.config import TrainingConfig


_ALGO_TRAINER_MAP: dict[str, str] = {
    "ppo": "rlox.trainers:PPOTrainer",
    "sac": "rlox.trainers:SACTrainer",
    "dqn": "rlox.trainers:DQNTrainer",
    "a2c": "rlox.trainers:A2CTrainer",
    "td3": "rlox.trainers:TD3Trainer",
    "mappo": "rlox.trainers:MAPPOTrainer",
    "dreamer": "rlox.trainers:DreamerV3Trainer",
    "impala": "rlox.trainers:IMPALATrainer",
}


def _import_class(dotted: str) -> type:
    module_path, cls_name = dotted.rsplit(":", 1)
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def _build_callbacks(config: TrainingConfig) -> list[Callback]:
    """Instantiate callbacks from string names in the config."""
    cbs: list[Callback] = []
    for name in config.callbacks:
        match name:
            case "eval":
                cbs.append(
                    EvalCallback(
                        eval_freq=config.eval_freq,
                        n_eval_episodes=config.eval_episodes,
                    )
                )
            case "checkpoint":
                cbs.append(
                    CheckpointCallback(
                        save_freq=config.checkpoint_freq,
                        save_path=config.checkpoint_dir,
                    )
                )
            case "progress":
                cbs.append(ProgressBarCallback())
            case "timing":
                cbs.append(TimingCallback())
            case _:
                raise ValueError(f"Unknown callback name: {name!r}")
    return cbs


def _build_logger(config: TrainingConfig) -> Any:
    """Instantiate a logger from the config string."""
    if config.logger is None:
        return None

    match config.logger:
        case "console":
            from rlox.logging import ConsoleLogger

            return ConsoleLogger()
        case "tensorboard":
            from rlox.logging import TensorBoardLogger

            return TensorBoardLogger(log_dir=config.log_dir or "runs")
        case "wandb":
            from rlox.logging import WandbLogger

            return WandbLogger(project="rlox")
        case _:
            raise ValueError(f"Unknown logger: {config.logger!r}")


def train_from_config(config: TrainingConfig | str | Path) -> dict[str, float]:
    """Run training from a TrainingConfig or a path to a YAML/TOML file.

    Parameters
    ----------
    config : TrainingConfig | str | Path
        Either a config object or a file path (YAML if ``.yaml``/``.yml``,
        TOML if ``.toml``).

    Returns
    -------
    dict[str, float]
        Training metrics from the final run.

    Raises
    ------
    ValueError
        If the algorithm is not recognised or the file extension is
        unsupported.
    """
    if isinstance(config, (str, Path)):
        path = Path(config)
        if path.suffix in (".yaml", ".yml"):
            config = TrainingConfig.from_yaml(path)
        elif path.suffix == ".toml":
            config = TrainingConfig.from_toml(path)
        else:
            raise ValueError(
                f"Unsupported config file extension: {path.suffix!r}. "
                "Use .yaml, .yml, or .toml."
            )

    callbacks = _build_callbacks(config)
    logger = _build_logger(config)

    hp = dict(config.hyperparameters)
    hp["n_envs"] = config.n_envs
    if config.normalize_obs:
        hp["normalize_obs"] = True
    if config.normalize_rewards:
        hp["normalize_rewards"] = True

    from rlox.trainer import Trainer

    trainer = Trainer(
        algorithm=config.algorithm,
        env=config.env_id,
        config=hp,
        callbacks=callbacks or None,
        logger=logger,
        seed=config.seed,
    )
    return trainer.train(total_timesteps=config.total_timesteps)
