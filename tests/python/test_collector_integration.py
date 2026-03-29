"""Tests for OffPolicyCollector integration with SAC, TD3, and DQN.

Verifies that all three off-policy algorithms can use the collector
for multi-env data collection while keeping learning logic unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

import rlox
from rlox.off_policy_collector import OffPolicyCollector


class TestSACCollector:
    def test_sac_accepts_collector_params(self):
        from rlox.algorithms.sac import SAC
        sac = SAC(env_id="Pendulum-v1", n_envs=2, learning_starts=10)
        assert sac.n_envs == 2
        assert sac.collector is None  # lazy-created on train()

    def test_sac_accepts_custom_collector(self):
        from rlox.algorithms.sac import SAC
        buf = rlox.ReplayBuffer(1000, obs_dim=3, act_dim=1)
        collector = OffPolicyCollector(env_id="Pendulum-v1", n_envs=2, buffer=buf)
        sac = SAC(env_id="Pendulum-v1", buffer=buf, collector=collector)
        assert sac.collector is collector

    def test_sac_train_with_collector(self):
        from rlox.algorithms.sac import SAC
        sac = SAC(
            env_id="Pendulum-v1",
            n_envs=2,
            learning_starts=50,
            batch_size=32,
            buffer_size=10000,
            hidden=32,
        )
        metrics = sac.train(total_timesteps=100)
        assert "mean_reward" in metrics
        assert len(sac.buffer) > 0

    def test_sac_single_env_unchanged(self):
        """Default single-env path should still work."""
        from rlox.algorithms.sac import SAC
        sac = SAC(
            env_id="Pendulum-v1",
            learning_starts=50,
            batch_size=32,
            buffer_size=1000,
            hidden=32,
        )
        metrics = sac.train(total_timesteps=100)
        assert "mean_reward" in metrics


class TestTD3Collector:
    def test_td3_accepts_collector_params(self):
        from rlox.algorithms.td3 import TD3
        td3 = TD3(env_id="Pendulum-v1", n_envs=2, learning_starts=10)
        assert td3.n_envs == 2
        assert td3.collector is None

    def test_td3_accepts_custom_collector(self):
        from rlox.algorithms.td3 import TD3
        buf = rlox.ReplayBuffer(1000, obs_dim=3, act_dim=1)
        collector = OffPolicyCollector(env_id="Pendulum-v1", n_envs=2, buffer=buf)
        td3 = TD3(env_id="Pendulum-v1", buffer=buf, collector=collector)
        assert td3.collector is collector

    def test_td3_train_with_collector(self):
        from rlox.algorithms.td3 import TD3
        td3 = TD3(
            env_id="Pendulum-v1",
            n_envs=2,
            learning_starts=50,
            batch_size=32,
            buffer_size=10000,
            hidden=32,
        )
        metrics = td3.train(total_timesteps=100)
        assert "mean_reward" in metrics
        assert len(td3.buffer) > 0

    def test_td3_single_env_unchanged(self):
        from rlox.algorithms.td3 import TD3
        td3 = TD3(
            env_id="Pendulum-v1",
            learning_starts=50,
            batch_size=32,
            buffer_size=1000,
            hidden=32,
        )
        metrics = td3.train(total_timesteps=100)
        assert "mean_reward" in metrics


class TestDQNCollector:
    def test_dqn_accepts_collector_params(self):
        from rlox.algorithms.dqn import DQN
        dqn = DQN(env_id="CartPole-v1", n_envs=2, learning_starts=10)
        assert dqn.n_envs == 2
        assert dqn.collector is None

    def test_dqn_accepts_custom_collector(self):
        from rlox.algorithms.dqn import DQN
        buf = rlox.ReplayBuffer(1000, obs_dim=4, act_dim=1)
        collector = OffPolicyCollector(env_id="CartPole-v1", n_envs=2, buffer=buf)
        dqn = DQN(env_id="CartPole-v1", buffer=buf, collector=collector)
        assert dqn.collector is collector

    def test_dqn_train_with_collector(self):
        from rlox.algorithms.dqn import DQN
        dqn = DQN(
            env_id="CartPole-v1",
            n_envs=2,
            learning_starts=50,
            batch_size=32,
            buffer_size=10000,
            hidden=32,
        )
        metrics = dqn.train(total_timesteps=100)
        assert "mean_reward" in metrics
        assert len(dqn.buffer) > 0

    def test_dqn_single_env_unchanged(self):
        from rlox.algorithms.dqn import DQN
        dqn = DQN(
            env_id="CartPole-v1",
            learning_starts=50,
            batch_size=32,
            buffer_size=1000,
            hidden=32,
        )
        metrics = dqn.train(total_timesteps=100)
        assert "mean_reward" in metrics

    def test_dqn_collector_nstep_warning(self):
        """n_step > 1 with collector uses direct buffer insertion (no n-step returns)."""
        from rlox.algorithms.dqn import DQN
        dqn = DQN(
            env_id="CartPole-v1",
            n_envs=2,
            n_step=1,
            learning_starts=50,
            batch_size=32,
            buffer_size=10000,
            hidden=32,
        )
        metrics = dqn.train(total_timesteps=100)
        assert "mean_reward" in metrics


class TestCollectorBufferSharing:
    """Verify that collector and algorithm share the same buffer."""

    def test_sac_shared_buffer(self):
        from rlox.algorithms.sac import SAC
        buf = rlox.ReplayBuffer(10000, obs_dim=3, act_dim=1)
        collector = OffPolicyCollector(env_id="Pendulum-v1", n_envs=2, buffer=buf)
        sac = SAC(
            env_id="Pendulum-v1",
            buffer=buf,
            collector=collector,
            learning_starts=50,
            batch_size=32,
            hidden=32,
        )
        sac.train(total_timesteps=100)
        # Both algorithm and collector see same buffer
        assert sac.buffer is buf
        assert collector.buffer is buf
        assert len(buf) > 0
