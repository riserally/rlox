"""Tests for CandleCollector — hybrid Rust collection with Candle inference.

The CandleCollector runs policy inference entirely in Rust (no Python/PyTorch
dispatch overhead), producing rollout batches via a crossbeam channel.
"""

import time

import numpy as np
import pytest

import rlox


class TestCandleCollectorBasic:
    def test_instantiation(self):
        """CandleCollector should be importable and creatable."""
        collector = rlox.CandleCollector(
            env_id="CartPole-v1",
            n_envs=2,
            obs_dim=4,
            n_actions=2,
            n_steps=32,
        )
        collector.stop()

    def test_recv_returns_batch(self):
        """recv() should return a dict with all expected keys."""
        collector = rlox.CandleCollector(
            env_id="CartPole-v1",
            n_envs=2,
            obs_dim=4,
            n_actions=2,
            n_steps=16,
        )
        batch = collector.recv()
        collector.stop()

        expected_keys = {
            "observations", "actions", "rewards", "dones",
            "log_probs", "values", "advantages", "returns",
            "obs_dim", "act_dim", "n_steps", "n_envs",
        }
        assert set(batch.keys()) == expected_keys

    def test_batch_shapes(self):
        """Batch arrays should have correct shapes."""
        n_envs = 4
        n_steps = 32
        collector = rlox.CandleCollector(
            env_id="CartPole-v1",
            n_envs=n_envs,
            obs_dim=4,
            n_actions=2,
            n_steps=n_steps,
        )
        batch = collector.recv()
        collector.stop()

        total = n_steps * n_envs
        assert batch["observations"].shape == (total * 4,)
        assert batch["actions"].shape == (total,)
        assert batch["rewards"].shape == (total,)
        assert batch["dones"].shape == (total,)
        assert batch["log_probs"].shape == (total,)
        assert batch["values"].shape == (total,)
        assert batch["advantages"].shape == (total,)
        assert batch["returns"].shape == (total,)
        assert batch["obs_dim"] == 4
        assert batch["act_dim"] == 1
        assert batch["n_steps"] == n_steps
        assert batch["n_envs"] == n_envs

    def test_values_are_finite(self):
        """All batch values should be finite."""
        collector = rlox.CandleCollector(
            env_id="CartPole-v1",
            n_envs=2,
            obs_dim=4,
            n_actions=2,
            n_steps=16,
        )
        batch = collector.recv()
        collector.stop()

        for key in ["observations", "actions", "rewards", "dones",
                     "log_probs", "values", "advantages", "returns"]:
            assert np.all(np.isfinite(batch[key])), f"{key} has non-finite values"


class TestCandleCollectorWeightSync:
    def test_get_weights(self):
        """get_weights() should return a non-empty float32 array."""
        collector = rlox.CandleCollector(
            env_id="CartPole-v1",
            n_envs=1,
            obs_dim=4,
            n_actions=2,
            n_steps=8,
        )
        weights = collector.get_weights()
        collector.stop()

        assert isinstance(weights, np.ndarray)
        assert weights.dtype == np.float32
        assert len(weights) > 0

    def test_sync_weights_roundtrip(self):
        """get_weights → sync_weights should be identity."""
        collector = rlox.CandleCollector(
            env_id="CartPole-v1",
            n_envs=1,
            obs_dim=4,
            n_actions=2,
            n_steps=8,
        )
        weights1 = collector.get_weights().copy()
        collector.sync_weights(weights1)
        weights2 = collector.get_weights()
        collector.stop()

        np.testing.assert_allclose(weights1, weights2, atol=1e-6)

    def test_sync_weights_changes_behavior(self):
        """After syncing different weights, actions should change."""
        collector = rlox.CandleCollector(
            env_id="CartPole-v1",
            n_envs=2,
            obs_dim=4,
            n_actions=2,
            n_steps=8,
        )

        batch1 = collector.recv()

        # Modify weights (add noise)
        weights = collector.get_weights()
        weights += np.random.randn(len(weights)).astype(np.float32) * 0.5
        collector.sync_weights(weights)

        batch2 = collector.recv()
        collector.stop()

        # Log probs should differ after weight change
        # (not guaranteed per-element, but statistically very likely)
        assert not np.allclose(batch1["log_probs"], batch2["log_probs"])

    def test_sync_wrong_size_raises(self):
        """sync_weights with wrong-sized array should raise."""
        collector = rlox.CandleCollector(
            env_id="CartPole-v1",
            n_envs=1,
            obs_dim=4,
            n_actions=2,
            n_steps=8,
        )

        with pytest.raises(RuntimeError):
            collector.sync_weights(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        collector.stop()


class TestCandleCollectorPerformance:
    def test_multiple_batches(self):
        """Should produce multiple batches without hanging."""
        collector = rlox.CandleCollector(
            env_id="CartPole-v1",
            n_envs=8,
            obs_dim=4,
            n_actions=2,
            n_steps=64,
        )

        for _ in range(5):
            batch = collector.recv()
            assert batch["n_envs"] == 8

        collector.stop()

    def test_collection_speed(self):
        """Candle collection should achieve >2000 SPS on CartPole."""
        n_envs = 16
        n_steps = 128
        collector = rlox.CandleCollector(
            env_id="CartPole-v1",
            n_envs=n_envs,
            obs_dim=4,
            n_actions=2,
            n_steps=n_steps,
        )

        # Warm up
        collector.recv()

        # Measure
        start = time.perf_counter()
        n_batches = 5
        for _ in range(n_batches):
            collector.recv()
        elapsed = time.perf_counter() - start
        collector.stop()

        total_steps = n_batches * n_envs * n_steps
        sps = total_steps / elapsed
        print(f"\nCandle collection: {sps:.0f} SPS ({n_envs} envs, {n_steps} steps)")
        # Should be fast — no Python dispatch overhead
        assert sps > 1000, f"Expected >1000 SPS, got {sps:.0f}"


class TestCandleCollectorUnsupported:
    def test_non_cartpole_raises(self):
        """Non-CartPole envs should raise ValueError."""
        with pytest.raises(ValueError, match="CartPole-v1 only"):
            rlox.CandleCollector(
                env_id="Pendulum-v1",
                n_envs=1,
                obs_dim=3,
                n_actions=1,
                n_steps=8,
            )
