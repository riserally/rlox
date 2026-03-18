"""Tests for the SB3 compatibility layer (rlox.compat.sb3)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from rlox.compat.sb3 import PPO, SAC, DQN


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

class TestPPO:
    def test_ppo_sb3_api_basic(self):
        """PPO('MlpPolicy', 'CartPole-v1').learn(1000) runs without error."""
        model = PPO("MlpPolicy", "CartPole-v1")
        result = model.learn(total_timesteps=1024)
        assert result is model  # learn() returns self for chaining

    def test_ppo_sb3_predict(self):
        """model.predict(obs) returns (action, None)."""
        model = PPO("MlpPolicy", "CartPole-v1")
        model.learn(total_timesteps=1024)

        obs = np.zeros(4, dtype=np.float32)
        action, state = model.predict(obs)
        assert state is None
        assert isinstance(action, np.ndarray)

    def test_ppo_sb3_predict_deterministic(self):
        """Deterministic predict returns consistent actions."""
        model = PPO("MlpPolicy", "CartPole-v1")
        obs = np.zeros(4, dtype=np.float32)

        action1, _ = model.predict(obs, deterministic=True)
        action2, _ = model.predict(obs, deterministic=True)
        np.testing.assert_array_equal(action1, action2)

    def test_ppo_sb3_save_load(self):
        """Save and load roundtrip preserves the model."""
        model = PPO("MlpPolicy", "CartPole-v1")
        model.learn(total_timesteps=1024)

        obs = np.zeros(4, dtype=np.float32)
        action_before, _ = model.predict(obs, deterministic=True)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            model.save(path)
            loaded = PPO.load(path, env="CartPole-v1")
            action_after, _ = loaded.predict(obs, deterministic=True)
            np.testing.assert_array_equal(action_before, action_after)
        finally:
            os.unlink(path)

    def test_ppo_sb3_learn_returns_self(self):
        """learn() returns self so chaining works: PPO(...).learn(1000).save(...)."""
        model = PPO("MlpPolicy", "CartPole-v1")
        ret = model.learn(total_timesteps=1024)
        assert ret is model


# ---------------------------------------------------------------------------
# SAC
# ---------------------------------------------------------------------------

class TestSAC:
    def test_sac_sb3_api_basic(self):
        """SAC('MlpPolicy', 'Pendulum-v1').learn(500) runs without error."""
        model = SAC("MlpPolicy", "Pendulum-v1")
        result = model.learn(total_timesteps=500)
        assert result is model

    def test_sac_sb3_predict(self):
        """SAC predict returns (action, None) with correct shape."""
        model = SAC("MlpPolicy", "Pendulum-v1")
        obs = np.zeros(3, dtype=np.float32)
        action, state = model.predict(obs)
        assert state is None
        assert isinstance(action, np.ndarray)

    def test_sac_sb3_save_load(self):
        """SAC save/load roundtrip."""
        model = SAC("MlpPolicy", "Pendulum-v1")
        model.learn(total_timesteps=500)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            model.save(path)
            loaded = SAC.load(path, env="Pendulum-v1")
            obs = np.zeros(3, dtype=np.float32)
            action, _ = loaded.predict(obs, deterministic=True)
            assert isinstance(action, np.ndarray)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------

class TestDQN:
    def test_dqn_sb3_api_basic(self):
        """DQN('MlpPolicy', 'CartPole-v1').learn(500) runs without error."""
        model = DQN("MlpPolicy", "CartPole-v1")
        result = model.learn(total_timesteps=500)
        assert result is model

    def test_dqn_sb3_predict(self):
        """DQN predict returns (action, None)."""
        model = DQN("MlpPolicy", "CartPole-v1")
        obs = np.zeros(4, dtype=np.float32)
        action, state = model.predict(obs)
        assert state is None
        assert isinstance(action, np.ndarray)

    def test_dqn_sb3_save_load(self):
        """DQN save/load roundtrip."""
        model = DQN("MlpPolicy", "CartPole-v1")
        model.learn(total_timesteps=500)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            model.save(path)
            loaded = DQN.load(path, env="CartPole-v1")
            obs = np.zeros(4, dtype=np.float32)
            action, _ = loaded.predict(obs, deterministic=True)
            assert isinstance(action, np.ndarray)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Cross-cutting concerns
# ---------------------------------------------------------------------------

class TestSB3Compat:
    def test_sb3_verbose_ignored(self):
        """verbose=1 does not crash any wrapper."""
        PPO("MlpPolicy", "CartPole-v1", verbose=1)
        SAC("MlpPolicy", "Pendulum-v1", verbose=1)
        DQN("MlpPolicy", "CartPole-v1", verbose=1)

    def test_sb3_kwargs_mapping_ppo(self):
        """SB3 kwargs map correctly to rlox PPOConfig fields."""
        model = PPO(
            "MlpPolicy",
            "CartPole-v1",
            learning_rate=1e-3,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            gamma=0.98,
            gae_lambda=0.9,
            clip_range=0.1,
            ent_coef=0.02,
            vf_coef=0.4,
            max_grad_norm=1.0,
        )
        cfg = model._algo.config
        assert cfg.learning_rate == 1e-3
        assert cfg.n_steps == 64
        assert cfg.batch_size == 32
        assert cfg.n_epochs == 2
        assert cfg.gamma == 0.98
        assert cfg.gae_lambda == 0.9
        assert cfg.clip_eps == 0.1
        assert cfg.ent_coef == 0.02
        assert cfg.vf_coef == 0.4
        assert cfg.max_grad_norm == 1.0

    def test_sb3_kwargs_mapping_sac(self):
        """SAC kwargs mapping."""
        model = SAC(
            "MlpPolicy",
            "Pendulum-v1",
            learning_rate=1e-3,
            buffer_size=5000,
            batch_size=64,
            tau=0.01,
            gamma=0.95,
            learning_starts=100,
        )
        cfg = model._algo.config
        assert cfg.learning_rate == 1e-3
        assert cfg.buffer_size == 5000
        assert cfg.batch_size == 64
        assert cfg.tau == 0.01
        assert cfg.gamma == 0.95
        assert cfg.learning_starts == 100

    def test_sb3_kwargs_mapping_dqn(self):
        """DQN kwargs mapping."""
        model = DQN(
            "MlpPolicy",
            "CartPole-v1",
            learning_rate=5e-4,
            buffer_size=10000,
            batch_size=32,
            gamma=0.95,
            learning_starts=200,
            exploration_fraction=0.2,
            exploration_initial_eps=0.8,
            exploration_final_eps=0.01,
        )
        cfg = model._algo.config
        assert cfg.learning_rate == 5e-4
        assert cfg.buffer_size == 10000
        assert cfg.batch_size == 32
        assert cfg.gamma == 0.95
        assert cfg.learning_starts == 200
        assert cfg.exploration_fraction == 0.2
        assert cfg.exploration_initial_eps == 0.8
        assert cfg.exploration_final_eps == 0.01

    def test_sb3_unknown_kwarg_warns(self):
        """Unknown kwargs produce a warning."""
        with pytest.warns(UserWarning, match="no rlox mapping"):
            PPO("MlpPolicy", "CartPole-v1", totally_fake_param=42)

    def test_sb3_seed_passthrough(self):
        """seed= is forwarded to the underlying algorithm."""
        model = PPO("MlpPolicy", "CartPole-v1", seed=123)
        assert model._seed == 123

    def test_ppo_policy_property(self):
        """The .policy property exposes the underlying nn.Module."""
        model = PPO("MlpPolicy", "CartPole-v1")
        assert isinstance(model.policy, torch.nn.Module)
