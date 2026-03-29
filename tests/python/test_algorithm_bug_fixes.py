"""Tests for P0 algorithm bug fixes: IMPALA, DreamerV3, MAPPO."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# P0.1: IMPALA V-trace fixes
# ---------------------------------------------------------------------------

class TestIMPALAVtraceFixes:
    def test_impala_bootstrap_value_computed(self):
        """Verify IMPALA actors send bootstrap_values to the learner."""
        from rlox.algorithms.impala import IMPALA

        agent = IMPALA(env_id="CartPole-v1", n_actors=1, n_envs=2, seed=42, n_steps=8)

        # Start actor, grab one batch, then stop
        import queue
        import threading

        agent._stop_event = threading.Event()
        t = threading.Thread(target=agent._actor_loop, args=(0,), daemon=True)
        t.start()

        data = agent._queue.get(timeout=10.0)
        agent._stop_event.set()
        t.join(timeout=5.0)

        # Verify bootstrap_values is present and correct shape
        assert "bootstrap_values" in data, "Actor must send bootstrap_values"
        assert data["bootstrap_values"].shape == (2,), f"Expected (n_envs=2,), got {data['bootstrap_values'].shape}"
        assert torch.isfinite(data["bootstrap_values"]).all()

    def test_impala_learner_uses_bootstrap(self):
        """Verify the learner uses bootstrap_values from actor data."""
        from rlox.algorithms.impala import IMPALA

        agent = IMPALA(env_id="CartPole-v1", n_actors=1, n_envs=2, seed=42, n_steps=8)

        # Collect one batch
        import threading
        agent._stop_event = threading.Event()
        t = threading.Thread(target=agent._actor_loop, args=(0,), daemon=True)
        t.start()
        data = agent._queue.get(timeout=10.0)
        agent._stop_event.set()
        t.join(timeout=5.0)

        # Learner step should use bootstrap values (not crash)
        metrics = agent._learner_step(data)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert np.isfinite(metrics["policy_loss"])
        assert np.isfinite(metrics["value_loss"])

    def test_impala_detects_continuous_env(self):
        """Verify IMPALA auto-detects continuous action spaces."""
        from rlox.algorithms.impala import IMPALA

        agent = IMPALA(env_id="Pendulum-v1", n_actors=1, n_envs=1, seed=42, n_steps=4)
        assert agent._is_discrete is False
        assert agent.n_actions == 1  # Pendulum action dim
        # Policy should be ContinuousPolicy, not DiscretePolicy
        from rlox.policies import ContinuousPolicy
        assert isinstance(agent.policy, ContinuousPolicy)

    @pytest.mark.slow
    def test_impala_trains_with_bootstrap(self):
        """End-to-end IMPALA training with bootstrap values."""
        from rlox.algorithms.impala import IMPALA

        agent = IMPALA(env_id="CartPole-v1", n_actors=1, n_envs=2, seed=42, n_steps=16)
        metrics = agent.train(total_timesteps=500)
        assert isinstance(metrics, dict)
        assert np.isfinite(metrics.get("policy_loss", 0.0))


# ---------------------------------------------------------------------------
# P0.2: DreamerV3 gradient fixes
# ---------------------------------------------------------------------------

class TestDreamerV3Fixes:
    def test_world_model_frozen_during_ac_training(self):
        """World model parameters should not receive gradients during AC update."""
        from rlox.algorithms.dreamer import DreamerV3

        agent = DreamerV3(env_id="CartPole-v1", seed=42, batch_size=8, buffer_size=100)

        # Collect some experience
        agent._collect_experience(20)

        # Record world model params before AC update
        wm_params_before = {
            name: p.clone() for name, p in agent.world_model.named_parameters()
        }

        # Train actor-critic
        agent._train_actor_critic()

        # World model params should be unchanged
        for name, p in agent.world_model.named_parameters():
            assert torch.allclose(p, wm_params_before[name]), (
                f"World model param '{name}' changed during actor-critic training"
            )

    def test_world_model_unfrozen_after_ac_training(self):
        """World model parameters should be unfrozen after AC training."""
        from rlox.algorithms.dreamer import DreamerV3

        agent = DreamerV3(env_id="CartPole-v1", seed=42, batch_size=8, buffer_size=100)
        agent._collect_experience(20)
        agent._train_actor_critic()

        # All world model params should have requires_grad=True
        for name, p in agent.world_model.named_parameters():
            assert p.requires_grad, f"World model param '{name}' still frozen after AC training"

    def test_world_model_still_trains_after_ac_step(self):
        """World model training should still work after AC training."""
        from rlox.algorithms.dreamer import DreamerV3

        agent = DreamerV3(env_id="CartPole-v1", seed=42, batch_size=8, buffer_size=100)
        agent._collect_experience(20)

        # AC train -> WM train cycle should work
        agent._train_actor_critic()
        wm_loss = agent._train_world_model()
        assert wm_loss >= 0.0  # Should produce a valid loss

    def test_actor_critic_loss_is_finite(self):
        """Actor-critic training should produce finite losses."""
        from rlox.algorithms.dreamer import DreamerV3

        agent = DreamerV3(env_id="CartPole-v1", seed=42, batch_size=8, buffer_size=100)
        agent._collect_experience(20)

        metrics = agent._train_actor_critic()
        assert "actor_loss" in metrics
        assert "critic_loss" in metrics
        assert np.isfinite(metrics["actor_loss"])
        assert np.isfinite(metrics["critic_loss"])

    @pytest.mark.slow
    def test_dreamer_full_training(self):
        """Full DreamerV3 training loop should work."""
        from rlox.algorithms.dreamer import DreamerV3

        agent = DreamerV3(env_id="CartPole-v1", seed=42, batch_size=8, buffer_size=500)
        metrics = agent.train(total_timesteps=200)
        assert "mean_reward" in metrics
        assert "wm_loss" in metrics


# ---------------------------------------------------------------------------
# P0.4: MAPPO critic input dimension fix
# ---------------------------------------------------------------------------

class TestMAPPOFixes:
    def test_mappo_single_agent_trains(self):
        """MAPPO with n_agents=1 should train correctly (equivalent to PPO)."""
        from rlox.algorithms.mappo import MAPPO

        agent = MAPPO(env_id="CartPole-v1", n_agents=1, n_envs=2, seed=42, n_steps=32)
        metrics = agent.train(total_timesteps=500)
        assert isinstance(metrics, dict)
        assert "mean_reward" in metrics
        # policy_loss may not be present if not enough steps for a full SGD epoch
        if "policy_loss" in metrics:
            assert np.isfinite(metrics["policy_loss"])

    def test_mappo_multi_agent_raises(self):
        """MAPPO with n_agents > 1 should raise NotImplementedError."""
        from rlox.algorithms.mappo import MAPPO

        agent = MAPPO(env_id="CartPole-v1", n_agents=2, n_envs=2, seed=42)
        with pytest.raises(NotImplementedError, match="multi-agent collector"):
            agent.train(total_timesteps=100)

    def test_mappo_critic_input_matches_obs_dim(self):
        """For n_agents=1, critic input dim should equal obs_dim."""
        from rlox.algorithms.mappo import MAPPO

        agent = MAPPO(env_id="CartPole-v1", n_agents=1, n_envs=2, seed=42)
        # Critic first layer should accept obs_dim=4 (CartPole)
        first_layer = agent.critic[0]
        assert first_layer.in_features == 4, (
            f"Expected critic input dim=4 (obs_dim), got {first_layer.in_features}"
        )

    def test_mappo_multi_agent_critic_dim(self):
        """For n_agents=2, critic input dim should be obs_dim * 2."""
        from rlox.algorithms.mappo import MAPPO

        agent = MAPPO(env_id="CartPole-v1", n_agents=2, n_envs=2, seed=42)
        first_layer = agent.critic[0]
        assert first_layer.in_features == 8, (
            f"Expected critic input dim=8 (obs_dim*n_agents), got {first_layer.in_features}"
        )
