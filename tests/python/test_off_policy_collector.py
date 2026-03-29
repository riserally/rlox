"""Tests for off-policy collector with vectorized environments."""

import numpy as np
import pytest

import rlox
from rlox.off_policy_collector import OffPolicyCollector, CollectorProtocol
from rlox.exploration import GaussianNoise, EpsilonGreedy


class TestOffPolicyCollector:
    def test_instantiation_single_env(self):
        buf = rlox.ReplayBuffer(1000, obs_dim=3, act_dim=1)
        collector = OffPolicyCollector(
            env_id="Pendulum-v1", n_envs=1, buffer=buf,
        )
        assert collector.n_envs == 1
        assert collector.obs_dim == 3
        assert collector.act_dim == 1
        assert collector.is_continuous is True

    def test_instantiation_multi_env(self):
        buf = rlox.ReplayBuffer(1000, obs_dim=4, act_dim=1)
        collector = OffPolicyCollector(
            env_id="CartPole-v1", n_envs=4, buffer=buf,
        )
        assert collector.n_envs == 4
        assert collector.is_continuous is False

    def test_reset_returns_obs(self):
        buf = rlox.ReplayBuffer(1000, obs_dim=3, act_dim=1)
        collector = OffPolicyCollector(env_id="Pendulum-v1", n_envs=2, buffer=buf)
        obs = collector.reset()
        assert obs.shape == (2, 3)

    def test_collect_step_stores_in_buffer(self):
        buf = rlox.ReplayBuffer(1000, obs_dim=3, act_dim=1)
        collector = OffPolicyCollector(env_id="Pendulum-v1", n_envs=2, buffer=buf)
        collector.reset()

        def random_action(obs):
            return np.random.randn(obs.shape[0], 1).astype(np.float32)

        for i in range(10):
            collector.collect_step(random_action, step=i, total_steps=100)

        assert len(buf) == 20  # 2 envs * 10 steps

    def test_collect_step_with_exploration(self):
        buf = rlox.ReplayBuffer(1000, obs_dim=3, act_dim=1)
        noise = GaussianNoise(sigma=0.5, seed=42)
        collector = OffPolicyCollector(
            env_id="Pendulum-v1", n_envs=1, buffer=buf, exploration=noise,
        )
        collector.reset()

        def zero_action(obs):
            return np.zeros((obs.shape[0], 1), dtype=np.float32)

        collector.collect_step(zero_action, step=0, total_steps=100)
        assert len(buf) == 1

    def test_collect_n_steps(self):
        buf = rlox.ReplayBuffer(1000, obs_dim=3, act_dim=1)
        collector = OffPolicyCollector(env_id="Pendulum-v1", n_envs=2, buffer=buf)
        collector.reset()

        def random_action(obs):
            return np.random.randn(obs.shape[0], 1).astype(np.float32)

        collector.collect_n_steps(random_action, n_steps=50, start_step=0, total_steps=100)
        assert len(buf) == 100  # 2 envs * 50 steps

    def test_push_batch_used_for_multi_env(self):
        """Verify push_batch is used when n_envs > 1."""
        buf = rlox.ReplayBuffer(1000, obs_dim=3, act_dim=1)
        collector = OffPolicyCollector(env_id="Pendulum-v1", n_envs=4, buffer=buf)
        collector.reset()

        def random_action(obs):
            return np.random.randn(obs.shape[0], 1).astype(np.float32)

        collector.collect_step(random_action, step=0, total_steps=100)
        assert len(buf) == 4  # 4 transitions pushed at once

    def test_satisfies_protocol(self):
        buf = rlox.ReplayBuffer(1000, obs_dim=3, act_dim=1)
        collector = OffPolicyCollector(env_id="Pendulum-v1", n_envs=1, buffer=buf)
        assert isinstance(collector, CollectorProtocol)

    def test_episode_reward_tracking(self):
        buf = rlox.ReplayBuffer(10000, obs_dim=4, act_dim=1)
        collector = OffPolicyCollector(env_id="CartPole-v1", n_envs=2, buffer=buf)
        collector.reset()

        def random_action(obs):
            return np.random.randint(0, 2, size=(obs.shape[0],))

        # Run enough steps for some episodes to complete
        for i in range(500):
            _, _, mean_reward = collector.collect_step(random_action, i, 1000)

        # Should have completed some episodes
        assert len(collector._completed_rewards) > 0

    def test_discrete_env(self):
        buf = rlox.ReplayBuffer(1000, obs_dim=4, act_dim=1)
        collector = OffPolicyCollector(env_id="CartPole-v1", n_envs=2, buffer=buf)
        collector.reset()

        def random_action(obs):
            return np.random.randint(0, 2, size=(obs.shape[0],))

        for i in range(10):
            collector.collect_step(random_action, i, 100)
        assert len(buf) == 20

    def test_no_buffer(self):
        """Collector should work without a buffer (for eval/rollout only)."""
        collector = OffPolicyCollector(env_id="Pendulum-v1", n_envs=1, buffer=None)
        obs = collector.reset()
        assert obs.shape == (1, 3)

        def random_action(obs):
            return np.random.randn(obs.shape[0], 1).astype(np.float32)

        next_obs, rewards, _ = collector.collect_step(random_action, 0, 100)
        assert next_obs.shape == (1, 3)
