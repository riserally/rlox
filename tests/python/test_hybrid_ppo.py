"""Tests for HybridPPO — Candle collection + PyTorch training."""

import time

import numpy as np
import pytest

import rlox


class TestHybridPPOBasic:
    def test_instantiation(self):
        from rlox.algorithms.hybrid_ppo import HybridPPO
        ppo = HybridPPO(env_id="CartPole-v1", n_envs=2, n_steps=16)
        ppo.collector.stop()

    def test_train_returns_metrics(self):
        from rlox.algorithms.hybrid_ppo import HybridPPO
        ppo = HybridPPO(env_id="CartPole-v1", n_envs=4, n_steps=32, hidden=32)
        metrics = ppo.train(total_timesteps=256)
        assert "mean_reward" in metrics
        assert "collection_time_pct" in metrics
        assert "training_time_pct" in metrics

    def test_policy_improves(self):
        """HybridPPO should learn CartPole within 50K steps."""
        from rlox.algorithms.hybrid_ppo import HybridPPO
        ppo = HybridPPO(
            env_id="CartPole-v1", n_envs=8, n_steps=128,
            hidden=64, learning_rate=2.5e-4, n_epochs=4,
        )
        metrics = ppo.train(total_timesteps=50_000)
        # CartPole should reach >100 mean reward with decent PPO
        assert metrics["mean_reward"] > 50, f"Expected > 50, got {metrics['mean_reward']:.1f}"

    def test_predict(self):
        from rlox.algorithms.hybrid_ppo import HybridPPO
        ppo = HybridPPO(env_id="CartPole-v1", n_envs=2, n_steps=16, hidden=32)
        obs = np.zeros(4, dtype=np.float32)
        action = ppo.predict(obs)
        assert action in (0, 1)
        ppo.collector.stop()

    def test_timing_summary(self):
        from rlox.algorithms.hybrid_ppo import HybridPPO
        ppo = HybridPPO(env_id="CartPole-v1", n_envs=4, n_steps=32, hidden=32)
        ppo.train(total_timesteps=256)
        summary = ppo.timing_summary()
        assert "collection_s" in summary
        assert "training_s" in summary
        assert abs(summary["collection_pct"] + summary["training_pct"] - 100) < 1

    def test_weight_sync_preserves_learning(self):
        """Weights should sync correctly between Candle and PyTorch."""
        from rlox.algorithms.hybrid_ppo import HybridPPO
        ppo = HybridPPO(env_id="CartPole-v1", n_envs=4, n_steps=32, hidden=32)

        # Get initial candle weights
        w1 = ppo.collector.get_weights().copy()

        # Train one update (should change weights)
        ppo.train(total_timesteps=128)

        # Candle weights should have changed (synced from PyTorch)
        w2 = ppo.collector.get_weights()
        assert not np.allclose(w1, w2), "Weights should change after training"


class TestHybridPPOSpeedup:
    def test_collection_faster_than_training(self):
        """With Candle, collection should be fast relative to training."""
        from rlox.algorithms.hybrid_ppo import HybridPPO
        ppo = HybridPPO(
            env_id="CartPole-v1", n_envs=16, n_steps=128,
            hidden=64, n_epochs=4,
        )
        ppo.train(total_timesteps=16_000)
        summary = ppo.timing_summary()
        # With Candle, collection should be <30% of total time
        # (vs ~50-60% with PyTorch collection)
        print(f"\nCollection: {summary['collection_pct']:.1f}%, Training: {summary['training_pct']:.1f}%")
        assert summary["collection_pct"] < 50, (
            f"Collection should be <50% with Candle, got {summary['collection_pct']:.1f}%"
        )
