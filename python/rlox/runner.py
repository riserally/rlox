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
    # a2c and td3 use the raw algorithm classes (no dedicated trainer)
    "a2c": "rlox.algorithms.a2c:A2C",
    "td3": "rlox.algorithms.td3:TD3",
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

    if config.algorithm not in _ALGO_TRAINER_MAP:
        raise ValueError(
            f"Unknown algorithm: {config.algorithm!r}. "
            f"Supported: {sorted(_ALGO_TRAINER_MAP)}"
        )

    callbacks = _build_callbacks(config)
    logger = _build_logger(config)

    trainer_dotted = _ALGO_TRAINER_MAP[config.algorithm]
    trainer_cls = _import_class(trainer_dotted)

    # Build kwargs depending on whether this is a Trainer or raw algo class
    hp = dict(config.hyperparameters)
    hp["n_envs"] = config.n_envs
    if config.normalize_obs:
        hp["normalize_obs"] = True
    if config.normalize_rewards:
        hp["normalize_rewards"] = True

    if config.algorithm in ("ppo",):
        # PPOTrainer takes config dict, callbacks, logger, seed
        trainer = trainer_cls(
            env=config.env_id,
            config=hp,
            callbacks=callbacks or None,
            logger=logger,
            seed=config.seed,
        )
    elif config.algorithm in ("sac", "dqn"):
        trainer = trainer_cls(
            env=config.env_id,
            config=hp,
            callbacks=callbacks or None,
            seed=config.seed,
        )
        # Attach logger to inner algo
        if logger is not None:
            algo_inner = getattr(trainer, "algo", trainer)
            if hasattr(algo_inner, "logger"):
                algo_inner.logger = logger
    else:
        # a2c, td3 — raw algorithm class
        trainer = trainer_cls(
            env_id=config.env_id,
            seed=config.seed,
            callbacks=callbacks or None,
            **hp,
        )
        if logger is not None and hasattr(trainer, "logger"):
            trainer.logger = logger

    metrics = trainer.train(total_timesteps=config.total_timesteps)
    return metrics
