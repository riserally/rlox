"""Tests for TrainingConfig and train_from_config (Layer 2 config-driven training)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from rlox.config import (
    TrainingConfig,
    PPOConfig,
    SACConfig,
    DQNConfig,
    A2CConfig,
    TD3Config,
)


# ---------------------------------------------------------------------------
# TrainingConfig serialization roundtrips
# ---------------------------------------------------------------------------


class TestTrainingConfigYAML:
    def test_roundtrip(self, tmp_path: Path) -> None:
        cfg = TrainingConfig(
            algorithm="ppo",
            env_id="CartPole-v1",
            total_timesteps=50_000,
            seed=123,
            n_envs=4,
            hyperparameters={"learning_rate": 0.001, "n_steps": 64},
            callbacks=["eval", "progress"],
            logger="console",
            eval_freq=5000,
        )
        path = tmp_path / "cfg.yaml"
        cfg.to_yaml(path)

        loaded = TrainingConfig.from_yaml(path)
        assert loaded.algorithm == "ppo"
        assert loaded.env_id == "CartPole-v1"
        assert loaded.total_timesteps == 50_000
        assert loaded.seed == 123
        assert loaded.n_envs == 4
        assert loaded.hyperparameters["learning_rate"] == 0.001
        assert loaded.callbacks == ["eval", "progress"]
        assert loaded.logger == "console"

    def test_yaml_file_contents_are_valid(self, tmp_path: Path) -> None:
        cfg = TrainingConfig(algorithm="sac", env_id="Pendulum-v1")
        path = tmp_path / "cfg.yaml"
        cfg.to_yaml(path)

        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f)
        assert raw["algorithm"] == "sac"
        assert raw["env_id"] == "Pendulum-v1"


class TestTrainingConfigTOML:
    def test_roundtrip(self, tmp_path: Path) -> None:
        cfg = TrainingConfig(
            algorithm="dqn",
            env_id="CartPole-v1",
            total_timesteps=20_000,
            seed=7,
            hyperparameters={"double_dqn": True, "learning_rate": 0.0001},
            callbacks=["checkpoint"],
            checkpoint_freq=5000,
        )
        path = tmp_path / "cfg.toml"
        cfg.to_toml(path)

        loaded = TrainingConfig.from_toml(path)
        assert loaded.algorithm == "dqn"
        assert loaded.total_timesteps == 20_000
        assert loaded.seed == 7
        assert loaded.hyperparameters["double_dqn"] is True
        assert loaded.callbacks == ["checkpoint"]

    def test_toml_file_is_parseable(self, tmp_path: Path) -> None:
        tomllib = pytest.importorskip("tomllib", reason="requires Python 3.11+")

        cfg = TrainingConfig(algorithm="ppo", env_id="CartPole-v1", n_envs=4)
        path = tmp_path / "cfg.toml"
        cfg.to_toml(path)

        with open(path, "rb") as f:
            raw = tomllib.load(f)
        assert raw["algorithm"] == "ppo"
        assert raw["n_envs"] == 4


class TestTrainingConfigDict:
    def test_roundtrip(self) -> None:
        cfg = TrainingConfig(
            algorithm="td3",
            env_id="Pendulum-v1",
            total_timesteps=10_000,
            hyperparameters={"tau": 0.01},
        )
        d = cfg.to_dict()
        loaded = TrainingConfig.from_dict(d)
        assert loaded.algorithm == "td3"
        assert loaded.hyperparameters["tau"] == 0.01
        assert loaded.to_dict() == d

    def test_from_dict_ignores_unknown_keys(self) -> None:
        d = {"algorithm": "ppo", "env_id": "CartPole-v1", "unknown_key": 99}
        cfg = TrainingConfig.from_dict(d)
        assert cfg.algorithm == "ppo"
        assert not hasattr(cfg, "unknown_key")

    def test_defaults(self) -> None:
        cfg = TrainingConfig(algorithm="ppo", env_id="CartPole-v1")
        assert cfg.total_timesteps == 100_000
        assert cfg.seed == 42
        assert cfg.n_envs == 1
        assert cfg.hyperparameters == {}
        assert cfg.callbacks == []
        assert cfg.logger is None
        assert cfg.normalize_obs is False
        assert cfg.normalize_rewards is False


# ---------------------------------------------------------------------------
# TOML support on algorithm configs
# ---------------------------------------------------------------------------


class TestAlgoConfigTOML:
    @pytest.mark.parametrize(
        "config_cls",
        [PPOConfig, SACConfig, DQNConfig, A2CConfig, TD3Config],
    )
    def test_toml_roundtrip(self, config_cls, tmp_path: Path) -> None:
        cfg = config_cls()
        path = tmp_path / "algo.toml"
        cfg.to_toml(path)

        loaded = config_cls.from_toml(path)
        assert loaded.to_dict() == cfg.to_dict()


# ---------------------------------------------------------------------------
# train_from_config
# ---------------------------------------------------------------------------


class TestTrainFromConfig:
    def test_ppo_cartpole_short_run(self) -> None:
        from rlox.runner import train_from_config

        cfg = TrainingConfig(
            algorithm="ppo",
            env_id="CartPole-v1",
            total_timesteps=5_000,
            seed=42,
            n_envs=2,
            hyperparameters={"n_steps": 64, "n_epochs": 2, "batch_size": 64},
        )
        metrics = train_from_config(cfg)
        assert isinstance(metrics, dict)
        assert "mean_reward" in metrics

    def test_from_yaml_path(self, tmp_path: Path) -> None:
        from rlox.runner import train_from_config

        cfg = TrainingConfig(
            algorithm="ppo",
            env_id="CartPole-v1",
            total_timesteps=5_000,
            n_envs=2,
            hyperparameters={"n_steps": 64, "n_epochs": 2, "batch_size": 64},
        )
        path = tmp_path / "run.yaml"
        cfg.to_yaml(path)

        metrics = train_from_config(str(path))
        assert isinstance(metrics, dict)
        assert "mean_reward" in metrics

    def test_from_toml_path(self, tmp_path: Path) -> None:
        from rlox.runner import train_from_config

        cfg = TrainingConfig(
            algorithm="ppo",
            env_id="CartPole-v1",
            total_timesteps=5_000,
            n_envs=2,
            hyperparameters={"n_steps": 64, "n_epochs": 2, "batch_size": 64},
        )
        path = tmp_path / "run.toml"
        cfg.to_toml(path)

        metrics = train_from_config(path)
        assert isinstance(metrics, dict)
        assert "mean_reward" in metrics

    def test_with_console_logger(self) -> None:
        from rlox.runner import train_from_config

        cfg = TrainingConfig(
            algorithm="ppo",
            env_id="CartPole-v1",
            total_timesteps=5_000,
            n_envs=2,
            logger="console",
            hyperparameters={"n_steps": 64, "n_epochs": 2, "batch_size": 64},
        )
        metrics = train_from_config(cfg)
        assert "mean_reward" in metrics

    def test_with_eval_callback(self) -> None:
        from rlox.runner import train_from_config

        cfg = TrainingConfig(
            algorithm="ppo",
            env_id="CartPole-v1",
            total_timesteps=5_000,
            n_envs=2,
            callbacks=["eval"],
            eval_freq=2000,
            eval_episodes=2,
            hyperparameters={"n_steps": 64, "n_epochs": 2, "batch_size": 64},
        )
        metrics = train_from_config(cfg)
        assert "mean_reward" in metrics

    def test_invalid_algorithm_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown algorithm"):
            TrainingConfig(algorithm="invalid_algo", env_id="CartPole-v1")


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLIConfigDriven:
    def test_train_with_yaml_config(self, tmp_path: Path) -> None:
        cfg = TrainingConfig(
            algorithm="ppo",
            env_id="CartPole-v1",
            total_timesteps=2_000,
            n_envs=2,
            hyperparameters={"n_steps": 64, "n_epochs": 2, "batch_size": 64},
        )
        path = tmp_path / "cli_test.yaml"
        cfg.to_yaml(path)

        result = subprocess.run(
            [sys.executable, "-m", "rlox", "train", "--config", str(path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Training complete" in result.stdout or "mean_reward" in result.stdout
