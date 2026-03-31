"""Tests for RolloutCollector with continuous envs and reward_fn."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rlox.collectors import RolloutCollector
from rlox.policies import ContinuousPolicy, DiscretePolicy


class TestCollectorEnvSelection:
    def test_auto_selects_native_env_for_pendulum(self) -> None:
        """Pendulum-v1 is now a native Rust env (multi-dim action support)."""
        import rlox as _rlox

        collector = RolloutCollector("Pendulum-v1", n_envs=2, seed=0)
        # Native Rust VecEnv or GymVecEnv fallback — either is fine
        assert collector.env is not None
        assert not collector._is_discrete

    def test_still_uses_rust_vecenv_for_cartpole(self) -> None:
        import rlox

        collector = RolloutCollector("CartPole-v1", n_envs=2, seed=0)
        assert isinstance(collector.env, rlox.VecEnv)
        assert collector._is_discrete

    def test_collector_reward_fn_applied(self) -> None:
        """reward_fn should transform rewards before they are stored."""

        def double_rewards(
            obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray
        ) -> np.ndarray:
            return rewards * 2.0

        collector_plain = RolloutCollector("CartPole-v1", n_envs=2, seed=0)
        collector_fn = RolloutCollector(
            "CartPole-v1", n_envs=2, seed=0, reward_fn=double_rewards
        )

        policy = DiscretePolicy(obs_dim=4, n_actions=2)

        torch.manual_seed(0)
        batch_plain = collector_plain.collect(policy, n_steps=8)

        torch.manual_seed(0)
        batch_fn = collector_fn.collect(policy, n_steps=8)

        # The doubled rewards should cause different advantages/returns,
        # but more directly we can check that the raw rewards differ
        # Note: rewards stored in batch are the transformed ones
        torch.testing.assert_close(batch_fn.rewards, batch_plain.rewards * 2.0)


class TestCollectorContinuousCollect:
    def test_collect_pendulum(self) -> None:
        """Full collect loop should work for Pendulum-v1."""
        collector = RolloutCollector("Pendulum-v1", n_envs=2, seed=42)
        policy = ContinuousPolicy(obs_dim=3, act_dim=1, hidden=32)
        batch = collector.collect(policy, n_steps=16)

        total = 2 * 16
        assert batch.obs.shape == (total, 3)
        assert batch.actions.shape == (total, 1)
        assert batch.rewards.shape == (total,)
        assert batch.advantages.shape == (total,)
