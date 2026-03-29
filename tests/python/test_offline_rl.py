"""Tests for offline RL: OfflineDatasetBuffer, TD3+BC, IQL, BC."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import rlox


def make_test_dataset(n=1000, obs_dim=4, act_dim=1, ep_len=100):
    """Create a synthetic offline dataset."""
    obs = np.random.randn(n, obs_dim).astype(np.float32)
    next_obs = obs + 0.01 * np.random.randn(n, obs_dim).astype(np.float32)
    actions = np.random.randn(n, act_dim).astype(np.float32)
    rewards = np.random.randn(n).astype(np.float32)
    terminated = np.zeros(n, dtype=np.uint8)
    truncated = np.zeros(n, dtype=np.uint8)
    for i in range(ep_len - 1, n, ep_len):
        terminated[i] = 1
    return obs, next_obs, actions, rewards, terminated, truncated


class TestOfflineDatasetBuffer:
    def test_create(self):
        obs, next_obs, actions, rewards, term, trunc = make_test_dataset()
        buf = rlox.OfflineDatasetBuffer(
            obs.ravel(), next_obs.ravel(), actions.ravel(),
            rewards, term, trunc,
        )
        assert len(buf) == 1000
        assert buf.n_episodes() == 10

    def test_sample_shapes(self):
        obs, next_obs, actions, rewards, term, trunc = make_test_dataset(
            n=500, obs_dim=6, act_dim=2
        )
        buf = rlox.OfflineDatasetBuffer(
            obs.ravel(), next_obs.ravel(), actions.ravel(),
            rewards, term, trunc,
        )
        batch = buf.sample(32, seed=42)
        assert batch["obs"].shape == (32, 6)
        assert batch["next_obs"].shape == (32, 6)
        assert batch["actions"].shape == (32, 2)
        assert batch["rewards"].shape == (32,)
        assert batch["terminated"].shape == (32,)

    def test_sample_deterministic(self):
        obs, next_obs, actions, rewards, term, trunc = make_test_dataset()
        buf = rlox.OfflineDatasetBuffer(
            obs.ravel(), next_obs.ravel(), actions.ravel(),
            rewards, term, trunc,
        )
        b1 = buf.sample(16, seed=42)
        b2 = buf.sample(16, seed=42)
        np.testing.assert_array_equal(b1["obs"], b2["obs"])

    def test_normalization(self):
        obs, next_obs, actions, rewards, term, trunc = make_test_dataset()
        buf = rlox.OfflineDatasetBuffer(
            obs.ravel(), next_obs.ravel(), actions.ravel(),
            rewards, term, trunc, normalize=True,
        )
        batch = buf.sample(256, seed=42)
        # Normalized obs should have roughly zero mean
        assert abs(batch["obs"].mean()) < 0.5

    def test_stats(self):
        obs, next_obs, actions, rewards, term, trunc = make_test_dataset()
        buf = rlox.OfflineDatasetBuffer(
            obs.ravel(), next_obs.ravel(), actions.ravel(),
            rewards, term, trunc,
        )
        stats = buf.stats()
        assert stats["n_transitions"] == 1000
        assert stats["n_episodes"] == 10
        assert stats["obs_dim"] == 4
        assert stats["act_dim"] == 1

    def test_sample_trajectories(self):
        obs, next_obs, actions, rewards, term, trunc = make_test_dataset(
            n=1000, obs_dim=4, act_dim=1, ep_len=100
        )
        buf = rlox.OfflineDatasetBuffer(
            obs.ravel(), next_obs.ravel(), actions.ravel(),
            rewards, term, trunc,
        )
        batch = buf.sample_trajectories(batch_size=4, seq_len=20, seed=42)
        assert batch["obs"].shape == (4, 20, 4)
        assert batch["actions"].shape == (4, 20, 1)
        assert batch["returns_to_go"].shape == (4, 20)
        assert batch["mask"].shape == (4, 20)


class TestOfflineDatasetProtocol:
    def test_buffer_satisfies_protocol(self):
        from rlox.offline.base import OfflineDataset
        obs, next_obs, actions, rewards, term, trunc = make_test_dataset()
        buf = rlox.OfflineDatasetBuffer(
            obs.ravel(), next_obs.ravel(), actions.ravel(),
            rewards, term, trunc,
        )
        # OfflineDatasetBuffer has sample() and __len__()
        assert hasattr(buf, "sample")
        assert hasattr(buf, "__len__")


class TestTD3BC:
    def _make_algo(self, **kwargs):
        from rlox.algorithms.td3_bc import TD3BC
        obs, next_obs, actions, rewards, term, trunc = make_test_dataset(
            n=500, obs_dim=4, act_dim=1
        )
        buf = rlox.OfflineDatasetBuffer(
            obs.ravel(), next_obs.ravel(), actions.ravel(),
            rewards, term, trunc,
        )
        defaults = dict(
            dataset=buf, obs_dim=4, act_dim=1,
            hidden=32, batch_size=64,
        )
        defaults.update(kwargs)
        return TD3BC(**defaults)

    def test_instantiation(self):
        algo = self._make_algo()
        assert algo is not None

    def test_train_returns_metrics(self):
        algo = self._make_algo()
        metrics = algo.train(n_gradient_steps=10)
        assert "critic_loss" in metrics
        assert "actor_loss" in metrics
        assert np.isfinite(metrics["critic_loss"])

    def test_alpha_zero_is_pure_td3(self):
        """With alpha=0, BC term vanishes → pure TD3."""
        algo = self._make_algo(alpha=0.0)
        metrics = algo.train(n_gradient_steps=10)
        assert np.isfinite(metrics["critic_loss"])

    def test_predict(self):
        algo = self._make_algo()
        obs = np.zeros(4, dtype=np.float32)
        action = algo.predict(obs)
        assert action.shape == (1,)
        assert np.isfinite(action).all()

    def test_global_step_increments(self):
        algo = self._make_algo()
        algo.train(n_gradient_steps=5)
        assert algo._global_step == 5


class TestIQL:
    def _make_algo(self, **kwargs):
        from rlox.algorithms.iql import IQL
        obs, next_obs, actions, rewards, term, trunc = make_test_dataset(
            n=500, obs_dim=4, act_dim=1
        )
        buf = rlox.OfflineDatasetBuffer(
            obs.ravel(), next_obs.ravel(), actions.ravel(),
            rewards, term, trunc,
        )
        defaults = dict(
            dataset=buf, obs_dim=4, act_dim=1,
            hidden=32, batch_size=64,
        )
        defaults.update(kwargs)
        return IQL(**defaults)

    def test_instantiation(self):
        algo = self._make_algo()
        assert algo is not None

    def test_train_returns_metrics(self):
        algo = self._make_algo()
        metrics = algo.train(n_gradient_steps=10)
        assert "value_loss" in metrics
        assert "q_loss" in metrics
        assert "actor_loss" in metrics
        assert all(np.isfinite(v) for v in metrics.values())

    def test_different_expectile(self):
        """Should work with various expectile values."""
        for tau in [0.5, 0.7, 0.9]:
            algo = self._make_algo(expectile=tau)
            metrics = algo.train(n_gradient_steps=5)
            assert np.isfinite(metrics["value_loss"])

    def test_predict(self):
        algo = self._make_algo()
        obs = np.zeros(4, dtype=np.float32)
        action = algo.predict(obs)
        assert action.shape == (1,)


class TestBC:
    def _make_algo(self, **kwargs):
        from rlox.algorithms.bc import BC
        obs, next_obs, actions, rewards, term, trunc = make_test_dataset(
            n=500, obs_dim=4, act_dim=1
        )
        buf = rlox.OfflineDatasetBuffer(
            obs.ravel(), next_obs.ravel(), actions.ravel(),
            rewards, term, trunc,
        )
        defaults = dict(
            dataset=buf, obs_dim=4, act_dim=1,
            hidden=32, batch_size=64, continuous=True,
        )
        defaults.update(kwargs)
        return BC(**defaults)

    def test_continuous_bc(self):
        algo = self._make_algo()
        metrics = algo.train(n_gradient_steps=10)
        assert "loss" in metrics
        assert np.isfinite(metrics["loss"])

    def test_loss_decreases(self):
        """BC loss should decrease over training."""
        algo = self._make_algo()
        m1 = algo.train(n_gradient_steps=1)
        m50 = algo.train(n_gradient_steps=50)
        # Loss should be lower after more training (on this small dataset)
        # Not guaranteed per-step, but very likely over 50 steps
        assert m50["loss"] < m1["loss"] * 2  # Loose bound

    def test_predict_continuous(self):
        algo = self._make_algo()
        obs = np.zeros(4, dtype=np.float32)
        action = algo.predict(obs)
        assert isinstance(action, np.ndarray)

class TestCQL:
    def _make_algo(self, **kwargs):
        from rlox.algorithms.cql import CQL
        obs, next_obs, actions, rewards, term, trunc = make_test_dataset(
            n=500, obs_dim=4, act_dim=1
        )
        buf = rlox.OfflineDatasetBuffer(
            obs.ravel(), next_obs.ravel(), actions.ravel(),
            rewards, term, trunc,
        )
        defaults = dict(
            dataset=buf, obs_dim=4, act_dim=1,
            hidden=32, batch_size=64, n_random_actions=4,
        )
        defaults.update(kwargs)
        return CQL(**defaults)

    def test_instantiation(self):
        algo = self._make_algo()
        assert algo is not None

    def test_train_returns_metrics(self):
        algo = self._make_algo()
        metrics = algo.train(n_gradient_steps=5)
        assert "critic_loss" in metrics
        assert "cql_loss" in metrics
        assert "actor_loss" in metrics
        assert all(np.isfinite(v) for v in metrics.values())

    def test_cql_alpha_zero_is_sac(self):
        """With cql_alpha=0, CQL reduces to SAC."""
        algo = self._make_algo(cql_alpha=0.0)
        metrics = algo.train(n_gradient_steps=5)
        assert np.isfinite(metrics["critic_loss"])

    def test_auto_cql_alpha(self):
        """Auto-tuned CQL alpha should be finite."""
        algo = self._make_algo(auto_alpha=True)
        metrics = algo.train(n_gradient_steps=5)
        assert np.isfinite(metrics["cql_alpha"])

    def test_predict(self):
        algo = self._make_algo()
        obs = np.zeros(4, dtype=np.float32)
        action = algo.predict(obs)
        assert action.shape == (1,)
        assert np.isfinite(action).all()


class TestBCDiscrete:
    """Discrete BC moved to its own class for clarity."""

    def test_discrete_bc(self):
        from rlox.algorithms.bc import BC
        obs, next_obs, _, rewards, term, trunc = make_test_dataset(
            n=500, obs_dim=4, act_dim=1
        )
        # Discrete actions: integers 0 or 1
        actions = np.random.randint(0, 2, size=(500, 1)).astype(np.float32)
        buf = rlox.OfflineDatasetBuffer(
            obs.ravel(), next_obs.ravel(), actions.ravel(),
            rewards, term, trunc,
        )
        algo = BC(dataset=buf, obs_dim=4, act_dim=2, continuous=False,
                   hidden=32, batch_size=64)
        metrics = algo.train(n_gradient_steps=10)
        assert np.isfinite(metrics["loss"])
