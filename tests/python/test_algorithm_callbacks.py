"""Tests for callback, logger, and checkpoint support in PPO, SAC, DQN, TD3."""

from __future__ import annotations

import os
import tempfile
from typing import Any
from unittest.mock import MagicMock

import pytest

from rlox.callbacks import Callback, CallbackList, EarlyStoppingCallback


class SpyCallback(Callback):
    """Records which hooks were called and how many times."""

    def __init__(self) -> None:
        self.training_started = False
        self.training_ended = False
        self.step_count = 0
        self.train_batch_count = 0

    def on_training_start(self, **kwargs: Any) -> None:
        self.training_started = True

    def on_step(self, **kwargs: Any) -> bool:
        self.step_count += 1
        return True

    def on_train_batch(self, **kwargs: Any) -> None:
        self.train_batch_count += 1

    def on_training_end(self, **kwargs: Any) -> None:
        self.training_ended = True


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------


class TestPPOCallbacks:
    def test_ppo_callbacks_called(self) -> None:
        from rlox.algorithms.ppo import PPO

        spy = SpyCallback()
        ppo = PPO("CartPole-v1", n_envs=2, n_steps=32, callbacks=[spy])
        ppo.train(total_timesteps=64)

        assert spy.training_started
        assert spy.training_ended
        assert spy.step_count > 0

    def test_ppo_save_load_roundtrip(self, tmp_path) -> None:
        from rlox.algorithms.ppo import PPO

        ppo = PPO("CartPole-v1", n_envs=2, n_steps=32, learning_rate=1e-3)
        ppo.train(total_timesteps=64)

        ckpt_path = str(tmp_path / "ppo.pt")
        ppo.save(ckpt_path)

        loaded = PPO.from_checkpoint(ckpt_path, env_id="CartPole-v1")
        assert loaded.config.learning_rate == ppo.config.learning_rate
        assert loaded.config.n_envs == ppo.config.n_envs
        assert loaded._global_step == ppo._global_step


# ---------------------------------------------------------------------------
# SAC
# ---------------------------------------------------------------------------


class TestSACCallbacks:
    def test_sac_callbacks_called(self) -> None:
        from rlox.algorithms.sac import SAC

        spy = SpyCallback()
        sac = SAC("Pendulum-v1", learning_starts=50, callbacks=[spy])
        sac.train(total_timesteps=100)

        assert spy.training_started
        assert spy.training_ended
        assert spy.step_count > 0

    def test_sac_early_stopping(self) -> None:
        from rlox.algorithms.sac import SAC

        # Patience=0 means stop immediately on first step
        early_stop = EarlyStoppingCallback(patience=0, min_delta=1e9)
        sac = SAC("Pendulum-v1", learning_starts=5, callbacks=[early_stop])
        sac.train(total_timesteps=10000)

        # Should have stopped well before 10000 steps
        assert sac._global_step < 10000

    def test_sac_save_load_roundtrip(self, tmp_path) -> None:
        from rlox.algorithms.sac import SAC

        sac = SAC("Pendulum-v1", learning_starts=10, learning_rate=1e-3)
        sac.train(total_timesteps=20)

        ckpt_path = str(tmp_path / "sac.pt")
        sac.save(ckpt_path)

        loaded = SAC.from_checkpoint(ckpt_path, env_id="Pendulum-v1")
        assert loaded.config.learning_rate == sac.config.learning_rate
        assert loaded.config.batch_size == sac.config.batch_size
        assert loaded._global_step == sac._global_step

    def test_sac_logger_called(self) -> None:
        from rlox.algorithms.sac import SAC
        from rlox.logging import LoggerCallback

        logger = MagicMock(spec=LoggerCallback)
        sac = SAC("Pendulum-v1", learning_starts=50, logger=logger)
        sac.train(total_timesteps=2000)

        # Logger should have been called at least once (every 1000 steps)
        assert logger.on_train_step.call_count >= 1


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------


class TestDQNCallbacks:
    def test_dqn_callbacks_called(self) -> None:
        from rlox.algorithms.dqn import DQN

        spy = SpyCallback()
        dqn = DQN("CartPole-v1", learning_starts=50, callbacks=[spy])
        dqn.train(total_timesteps=100)

        assert spy.training_started
        assert spy.training_ended
        assert spy.step_count > 0

    def test_dqn_save_load_roundtrip(self, tmp_path) -> None:
        from rlox.algorithms.dqn import DQN

        dqn = DQN("CartPole-v1", learning_starts=10, learning_rate=1e-3)
        dqn.train(total_timesteps=20)

        ckpt_path = str(tmp_path / "dqn.pt")
        dqn.save(ckpt_path)

        loaded = DQN.from_checkpoint(ckpt_path, env_id="CartPole-v1")
        assert loaded.config.learning_rate == dqn.config.learning_rate
        assert loaded.config.batch_size == dqn.config.batch_size
        assert loaded._global_step == dqn._global_step

    def test_dqn_early_stopping(self) -> None:
        from rlox.algorithms.dqn import DQN

        early_stop = EarlyStoppingCallback(patience=0, min_delta=1e9)
        dqn = DQN("CartPole-v1", learning_starts=5, callbacks=[early_stop])
        dqn.train(total_timesteps=10000)

        assert dqn._global_step < 10000


# ---------------------------------------------------------------------------
# TD3
# ---------------------------------------------------------------------------


class TestTD3Callbacks:
    def test_td3_callbacks_called(self) -> None:
        from rlox.algorithms.td3 import TD3

        spy = SpyCallback()
        td3 = TD3("Pendulum-v1", learning_starts=50, callbacks=[spy])
        td3.train(total_timesteps=100)

        assert spy.training_started
        assert spy.training_ended
        assert spy.step_count > 0

    def test_td3_save_load_roundtrip(self, tmp_path) -> None:
        from rlox.algorithms.td3 import TD3

        td3 = TD3("Pendulum-v1", learning_starts=10, learning_rate=1e-3)
        td3.train(total_timesteps=20)

        ckpt_path = str(tmp_path / "td3.pt")
        td3.save(ckpt_path)

        loaded = TD3.from_checkpoint(ckpt_path, env_id="Pendulum-v1")
        assert loaded.learning_rate == td3.learning_rate
        assert loaded.batch_size == td3.batch_size
        assert loaded._global_step == td3._global_step


# ---------------------------------------------------------------------------
# Config YAML round-trip
# ---------------------------------------------------------------------------


class TestConfigYaml:
    def test_sac_config_yaml_roundtrip(self, tmp_path) -> None:
        from rlox.config import SACConfig

        cfg = SACConfig(learning_rate=1e-3, batch_size=128, gamma=0.95)
        yaml_path = str(tmp_path / "sac.yaml")
        cfg.to_yaml(yaml_path)

        loaded = SACConfig.from_yaml(yaml_path)
        assert loaded.learning_rate == cfg.learning_rate
        assert loaded.batch_size == cfg.batch_size
        assert loaded.gamma == cfg.gamma

    def test_dqn_config_yaml_roundtrip(self, tmp_path) -> None:
        from rlox.config import DQNConfig

        cfg = DQNConfig(learning_rate=5e-4, batch_size=32, double_dqn=False)
        yaml_path = str(tmp_path / "dqn.yaml")
        cfg.to_yaml(yaml_path)

        loaded = DQNConfig.from_yaml(yaml_path)
        assert loaded.learning_rate == cfg.learning_rate
        assert loaded.batch_size == cfg.batch_size
        assert loaded.double_dqn == cfg.double_dqn
