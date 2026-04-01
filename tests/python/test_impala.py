"""Tests for IMPALA: vectorized V-trace, config, and distributed wiring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Vectorized V-trace
# ---------------------------------------------------------------------------


class TestVectorizedVtrace:
    def test_vtrace_batched_matches_per_env(self):
        """Vectorized V-trace should produce same results as per-env loop."""
        import rlox
        from rlox.algorithms.impala import _compute_vtrace_batched

        n_steps = 10
        n_envs = 4
        gamma = 0.99
        rho_bar = 1.0
        c_bar = 1.0

        rng = np.random.default_rng(42)
        log_rhos = rng.standard_normal((n_steps, n_envs)).astype(np.float32)
        rewards = rng.standard_normal((n_steps, n_envs)).astype(np.float32)
        values = rng.standard_normal((n_steps, n_envs)).astype(np.float32)
        dones = (rng.random((n_steps, n_envs)) > 0.9).astype(np.float32)
        bootstrap_values = rng.standard_normal(n_envs).astype(np.float32)

        # Per-env loop (old approach)
        vs_list, pg_list = [], []
        for env_idx in range(n_envs):
            vs, pg = rlox.compute_vtrace(
                np.ascontiguousarray(log_rhos[:, env_idx]),
                np.ascontiguousarray(rewards[:, env_idx]),
                np.ascontiguousarray(values[:, env_idx]),
                np.ascontiguousarray(dones[:, env_idx]),
                bootstrap_value=float(bootstrap_values[env_idx]),
                gamma=gamma,
                rho_bar=rho_bar,
                c_bar=c_bar,
            )
            vs_list.append(vs)
            pg_list.append(pg)

        vs_expected = np.stack(vs_list, axis=1)  # (n_steps, n_envs)
        pg_expected = np.stack(pg_list, axis=1)

        # Batched approach
        vs_batched, pg_batched = _compute_vtrace_batched(
            log_rhos, rewards, values, dones, bootstrap_values,
            gamma=gamma, rho_bar=rho_bar, c_bar=c_bar,
        )

        np.testing.assert_allclose(vs_batched, vs_expected, atol=1e-5)
        np.testing.assert_allclose(pg_batched, pg_expected, atol=1e-5)

    def test_vtrace_batched_single_env(self):
        """Batched V-trace works with a single environment."""
        from rlox.algorithms.impala import _compute_vtrace_batched

        n_steps = 5
        rng = np.random.default_rng(123)
        log_rhos = rng.standard_normal((n_steps, 1)).astype(np.float32)
        rewards = rng.standard_normal((n_steps, 1)).astype(np.float32)
        values = rng.standard_normal((n_steps, 1)).astype(np.float32)
        dones = np.zeros((n_steps, 1), dtype=np.float32)
        bootstrap = rng.standard_normal(1).astype(np.float32)

        vs, pg = _compute_vtrace_batched(
            log_rhos, rewards, values, dones, bootstrap,
            gamma=0.99, rho_bar=1.0, c_bar=1.0,
        )
        assert vs.shape == (n_steps, 1)
        assert pg.shape == (n_steps, 1)
        assert np.all(np.isfinite(vs))
        assert np.all(np.isfinite(pg))


# ---------------------------------------------------------------------------
# IMPALA instantiation and short training
# ---------------------------------------------------------------------------


class TestIMPALA:
    def test_instantiation(self):
        from rlox.algorithms.impala import IMPALA

        agent = IMPALA(env_id="CartPole-v1", n_actors=1, n_envs=2, seed=42)
        assert agent is not None
        assert agent.n_actors == 1

    @pytest.mark.slow
    def test_trains(self):
        from rlox.algorithms.impala import IMPALA

        agent = IMPALA(env_id="CartPole-v1", n_actors=1, n_envs=2, seed=42)
        metrics = agent.train(total_timesteps=500)
        assert isinstance(metrics, dict)
        assert "mean_reward" in metrics


# ---------------------------------------------------------------------------
# IMPALAConfig
# ---------------------------------------------------------------------------


class TestIMPALAConfig:
    def test_defaults(self):
        from rlox.config import IMPALAConfig

        cfg = IMPALAConfig()
        assert cfg.n_actors == 4
        assert cfg.learning_rate == 4e-4
        assert cfg.rho_clip == 1.0
        assert cfg.c_clip == 1.0
        assert cfg.queue_size == 16
        assert cfg.n_envs_per_actor == 1

    def test_validation_rejects_bad_values(self):
        from rlox.config import IMPALAConfig

        with pytest.raises(ValueError):
            IMPALAConfig(learning_rate=-1.0)
        with pytest.raises(ValueError):
            IMPALAConfig(n_actors=0)
        with pytest.raises(ValueError):
            IMPALAConfig(n_steps=0)

    def test_from_dict_ignores_unknown(self):
        from rlox.config import IMPALAConfig

        cfg = IMPALAConfig.from_dict({"n_actors": 8, "bogus": True})
        assert cfg.n_actors == 8

    def test_roundtrip_dict(self):
        from rlox.config import IMPALAConfig

        cfg = IMPALAConfig(n_actors=8, hidden=512)
        d = cfg.to_dict()
        loaded = IMPALAConfig.from_dict(d)
        assert loaded.to_dict() == d

    def test_yaml_roundtrip(self, tmp_path):
        from rlox.config import IMPALAConfig

        cfg = IMPALAConfig(n_actors=2, queue_size=32)
        path = tmp_path / "impala.yaml"
        cfg.to_yaml(path)
        loaded = IMPALAConfig.from_yaml(path)
        assert loaded.n_actors == 2
        assert loaded.queue_size == 32

    def test_toml_roundtrip(self, tmp_path):
        from rlox.config import IMPALAConfig

        cfg = IMPALAConfig(rho_clip=0.5, c_clip=0.5)
        path = tmp_path / "impala.toml"
        cfg.to_toml(path)
        loaded = IMPALAConfig.from_toml(path)
        assert loaded.rho_clip == 0.5
        assert loaded.c_clip == 0.5


# ---------------------------------------------------------------------------
# Distributed IMPALA wiring
# ---------------------------------------------------------------------------


class TestIMPALADistributed:
    """Tests for IMPALA remote env (distributed actor) support."""

    def test_impala_accepts_worker_addresses(self) -> None:
        """IMPALA.__init__ should accept worker_addresses without crashing."""
        from rlox.algorithms.impala import IMPALA

        addresses = ["gpu-node-1:50051", "gpu-node-2:50051"]
        agent = IMPALA(
            env_id="CartPole-v1",
            n_actors=2,
            n_envs=2,
            seed=42,
            worker_addresses=addresses,
        )
        assert agent._worker_addresses == addresses

    def test_impala_without_addresses_uses_local_env(self) -> None:
        """When worker_addresses is None, IMPALA should use local envs (backward compat)."""
        from rlox.algorithms.impala import IMPALA

        agent = IMPALA(env_id="CartPole-v1", n_actors=1, n_envs=2, seed=42)
        assert agent._worker_addresses is None

    def test_impala_partitions_addresses_across_actors(self) -> None:
        """Addresses should be split across actors."""
        from rlox.algorithms.impala import IMPALA

        addresses = [
            "node-1:50051", "node-2:50051",
            "node-3:50051", "node-4:50051",
        ]
        agent = IMPALA(
            env_id="CartPole-v1",
            n_actors=2,
            n_envs=2,
            seed=42,
            worker_addresses=addresses,
        )
        partitions = agent._partition_addresses()
        assert len(partitions) == 2
        # All addresses accounted for
        all_addrs = [a for p in partitions for a in p]
        assert sorted(all_addrs) == sorted(addresses)

    def test_impala_single_actor_gets_all_addresses(self) -> None:
        """With one actor, it should receive all addresses."""
        from rlox.algorithms.impala import IMPALA

        addresses = ["node-1:50051", "node-2:50051"]
        agent = IMPALA(
            env_id="CartPole-v1",
            n_actors=1,
            n_envs=2,
            seed=42,
            worker_addresses=addresses,
        )
        partitions = agent._partition_addresses()
        assert len(partitions) == 1
        assert partitions[0] == addresses

    def test_impala_more_actors_than_addresses_raises(self) -> None:
        """Having more actors than addresses should raise ValueError."""
        from rlox.algorithms.impala import IMPALA

        with pytest.raises(ValueError, match="more actors.*than worker addresses"):
            IMPALA(
                env_id="CartPole-v1",
                n_actors=4,
                n_envs=2,
                seed=42,
                worker_addresses=["node-1:50051", "node-2:50051"],
            )

    def test_actor_loop_creates_remote_env_pool_when_distributed(self) -> None:
        """When worker_addresses are provided, _actor_loop should create a RemoteEnvPool."""
        from rlox.algorithms.impala import IMPALA

        addresses = ["node-1:50051", "node-2:50051"]
        agent = IMPALA(
            env_id="CartPole-v1",
            n_actors=2,
            n_envs=2,
            seed=42,
            worker_addresses=addresses,
        )

        # Patch RemoteEnvPool to capture construction
        mock_pool = MagicMock()
        mock_pool.reset_all.return_value = np.zeros((2, 4), dtype=np.float32)

        with patch(
            "rlox.algorithms.impala.RemoteEnvPool", return_value=mock_pool
        ) as mock_cls:
            # Set stop event immediately so the actor loop exits after one check
            agent._stop_event.set()
            agent._actor_loop(0)

            mock_cls.assert_called_once_with(addresses=["node-1:50051"])
            mock_pool.connect.assert_called_once()


class TestDistributedIMPALA:
    """Tests for the DistributedIMPALA convenience wrapper."""

    def test_distributed_impala_class_exists(self) -> None:
        """DistributedIMPALA should be importable."""
        from rlox.algorithms.impala import DistributedIMPALA

        assert DistributedIMPALA is not None

    def test_distributed_impala_is_subclass_of_impala(self) -> None:
        from rlox.algorithms.impala import IMPALA, DistributedIMPALA

        assert issubclass(DistributedIMPALA, IMPALA)

    def test_distributed_impala_requires_worker_addresses(self) -> None:
        """DistributedIMPALA must receive worker_addresses."""
        from rlox.algorithms.impala import DistributedIMPALA

        agent = DistributedIMPALA(
            env_id="CartPole-v1",
            worker_addresses=["node-1:50051", "node-2:50051"],
            n_actors=2,
        )
        assert agent._worker_addresses is not None
        assert len(agent._worker_addresses) == 2

    def test_distributed_impala_rejects_missing_addresses(self) -> None:
        """DistributedIMPALA should reject construction without addresses."""
        from rlox.algorithms.impala import DistributedIMPALA

        with pytest.raises((TypeError, ValueError)):
            DistributedIMPALA(env_id="CartPole-v1")
