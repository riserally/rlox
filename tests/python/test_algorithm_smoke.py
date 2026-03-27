"""Smoke tests for algorithms that previously had zero test coverage.

Tests verify that each algorithm can:
1. Be instantiated without errors
2. Run a few training steps without crashing
3. Produce finite loss values
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# IMPALA
# ---------------------------------------------------------------------------

class TestIMPALA:
    def test_impala_instantiation(self):
        from rlox.algorithms.impala import IMPALA
        agent = IMPALA(env_id="CartPole-v1", n_actors=1, n_envs=2, seed=42)
        assert agent is not None

    @pytest.mark.slow
    def test_impala_trains(self):
        from rlox.algorithms.impala import IMPALA
        agent = IMPALA(env_id="CartPole-v1", n_actors=1, n_envs=2, seed=42)
        metrics = agent.train(total_timesteps=500)
        assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# MAPPO
# ---------------------------------------------------------------------------

class TestMAPPO:
    def test_mappo_instantiation(self):
        from rlox.algorithms.mappo import MAPPO
        agent = MAPPO(env_id="CartPole-v1", n_agents=1, n_envs=2, seed=42)
        assert agent is not None
        assert agent.n_agents == 1

    @pytest.mark.slow
    def test_mappo_trains(self):
        from rlox.algorithms.mappo import MAPPO
        agent = MAPPO(env_id="CartPole-v1", n_agents=1, n_envs=2, seed=42)
        metrics = agent.train(total_timesteps=500)
        assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# DreamerV3
# ---------------------------------------------------------------------------

class TestDreamerV3:
    def test_dreamer_instantiation(self):
        from rlox.algorithms.dreamer import DreamerV3
        agent = DreamerV3(env_id="CartPole-v1", seed=42)
        assert agent is not None

    @pytest.mark.slow
    def test_dreamer_trains(self):
        from rlox.algorithms.dreamer import DreamerV3
        agent = DreamerV3(env_id="CartPole-v1", seed=42)
        metrics = agent.train(total_timesteps=200)
        assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------

class TestDPO:
    def _make_tiny_model(self, vocab_size=10, hidden=16):
        return nn.Sequential(
            nn.Embedding(vocab_size, hidden),
            nn.Linear(hidden, vocab_size),
        )

    def test_dpo_instantiation(self):
        from rlox.algorithms.dpo import DPO
        model = self._make_tiny_model()
        ref_model = self._make_tiny_model()
        ref_model.load_state_dict(model.state_dict())
        dpo = DPO(model=model, ref_model=ref_model)
        assert dpo is not None
        assert dpo.max_grad_norm == 1.0

    def test_dpo_loss_is_finite(self):
        from rlox.algorithms.dpo import DPO
        model = self._make_tiny_model()
        ref_model = self._make_tiny_model()
        ref_model.load_state_dict(model.state_dict())
        dpo = DPO(model=model, ref_model=ref_model)

        prompt = torch.randint(0, 10, (1, 3))
        chosen = torch.randint(0, 10, (1, 5))
        rejected = torch.randint(0, 10, (1, 5))

        loss, metrics = dpo.compute_loss(prompt, chosen, rejected)
        assert loss.isfinite()

    def test_dpo_train_step(self):
        from rlox.algorithms.dpo import DPO
        model = self._make_tiny_model()
        ref_model = self._make_tiny_model()
        ref_model.load_state_dict(model.state_dict())
        dpo = DPO(model=model, ref_model=ref_model)

        prompt = torch.randint(0, 10, (1, 3))
        chosen = torch.randint(0, 10, (1, 5))
        rejected = torch.randint(0, 10, (1, 5))

        metrics = dpo.train_step(prompt, chosen, rejected)
        assert "loss" in metrics or "chosen_reward" in metrics


# ---------------------------------------------------------------------------
# OnlineDPO
# ---------------------------------------------------------------------------

class TestOnlineDPO:
    def _make_tiny_model(self, vocab_size=10, hidden=16):
        return nn.Sequential(
            nn.Embedding(vocab_size, hidden),
            nn.Linear(hidden, vocab_size),
        )

    def test_online_dpo_instantiation(self):
        from rlox.algorithms.online_dpo import OnlineDPO
        model = self._make_tiny_model()
        ref_model = self._make_tiny_model()
        ref_model.load_state_dict(model.state_dict())

        def pref_fn(pairs):
            return [0] * len(pairs)

        odpo = OnlineDPO(model=model, ref_model=ref_model, preference_fn=pref_fn)
        assert odpo is not None
        assert odpo.max_grad_norm == 1.0


# ---------------------------------------------------------------------------
# BestOfN
# ---------------------------------------------------------------------------

class TestBestOfN:
    def test_best_of_n_instantiation(self):
        from rlox.algorithms.best_of_n import BestOfN

        model = nn.Sequential(
            nn.Embedding(10, 16),
            nn.Linear(16, 10),
        )
        reward_fn = lambda completions: [float(c.sum()) for c in completions]

        bon = BestOfN(model=model, reward_fn=reward_fn, n=4)
        assert bon.n == 4


# ---------------------------------------------------------------------------
# predict() methods
# ---------------------------------------------------------------------------

class TestPredict:
    def test_sac_predict(self):
        from rlox.algorithms.sac import SAC
        sac = SAC(env_id="Pendulum-v1", learning_starts=0, seed=42)
        obs = np.zeros(3, dtype=np.float32)
        action = sac.predict(obs, deterministic=True)
        assert action.shape == (1,)
        assert np.isfinite(action).all()

    def test_td3_predict(self):
        from rlox.algorithms.td3 import TD3
        td3 = TD3(env_id="Pendulum-v1", learning_starts=0, seed=42)
        obs = np.zeros(3, dtype=np.float32)
        action = td3.predict(obs)
        assert action.shape == (1,)
        assert np.isfinite(action).all()

    def test_dqn_predict(self):
        from rlox.algorithms.dqn import DQN
        dqn = DQN(env_id="CartPole-v1", learning_starts=0, seed=42)
        obs = np.zeros(4, dtype=np.float32)
        action = dqn.predict(obs)
        assert isinstance(action, int)
        assert 0 <= action <= 1


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class TestNewCallbacks:
    def test_progress_bar_callback(self):
        from rlox.callbacks import ProgressBarCallback
        cb = ProgressBarCallback()
        cb.on_training_start(total_timesteps=100)
        for i in range(10):
            cb.on_step(step=i, reward=1.0)
        cb.on_training_end()

    def test_timing_callback(self):
        from rlox.callbacks import TimingCallback
        import time
        cb = TimingCallback()
        cb.on_training_start()
        time.sleep(0.01)
        cb.on_step()
        time.sleep(0.01)
        cb.on_rollout_end()
        time.sleep(0.01)
        cb.on_train_batch()
        time.sleep(0.01)
        cb.on_training_end()
        summary = cb.summary()
        assert len(summary) > 0
        assert abs(sum(summary.values()) - 100.0) < 1.0  # should sum to ~100%


# ---------------------------------------------------------------------------
# Custom env support
# ---------------------------------------------------------------------------

class TestCustomEnv:
    def test_sac_accepts_env_instance(self):
        import gymnasium as gym
        from rlox.algorithms.sac import SAC
        env = gym.make("Pendulum-v1")
        sac = SAC(env_id=env, learning_starts=0, seed=42)
        assert sac.env_id == "Pendulum-v1"
        env.close()

    def test_dqn_accepts_env_instance(self):
        import gymnasium as gym
        from rlox.algorithms.dqn import DQN
        env = gym.make("CartPole-v1")
        dqn = DQN(env_id=env, learning_starts=0, seed=42)
        assert dqn.env_id == "CartPole-v1"
        env.close()
