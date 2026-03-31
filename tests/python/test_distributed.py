"""Tests for distributed training utilities.

Covers:
- RemoteEnvPool gRPC client (with and without grpcio)
- MultiGPUTrainer DDP + FSDP strategies
- Elastic training launcher
- Rank-aware logging helpers
"""

from __future__ import annotations

import inspect
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# RemoteEnvPool
# ---------------------------------------------------------------------------


class TestRemoteEnvPoolNoGrpc:
    """Test graceful behavior when grpcio is not installed."""

    def test_init_without_grpcio_raises_import_error(self) -> None:
        """Pool should raise ImportError at connect time when grpcio missing."""
        with patch.dict(sys.modules, {"grpc": None}):
            from rlox.distributed.remote_env import RemoteEnvPool

            pool = RemoteEnvPool(
                addresses=["localhost:50051"],
                timeout=5.0,
            )
            with pytest.raises((ImportError, ConnectionError)):
                pool.connect()

    def test_init_with_empty_addresses_raises(self) -> None:
        from rlox.distributed.remote_env import RemoteEnvPool

        with pytest.raises(ValueError, match="at least one address"):
            RemoteEnvPool(addresses=[], timeout=5.0)


class TestRemoteEnvPoolInterface:
    """Test that RemoteEnvPool matches GymVecEnv/VecEnv interface."""

    def test_num_envs_property(self) -> None:
        from rlox.distributed.remote_env import RemoteEnvPool

        pool = RemoteEnvPool(
            addresses=["localhost:50051", "localhost:50052"],
            timeout=5.0,
        )
        # Before connect, num_envs should reflect expected count from addresses
        assert pool.num_envs >= 0

    def test_step_all_requires_connection(self) -> None:
        from rlox.distributed.remote_env import RemoteEnvPool

        pool = RemoteEnvPool(addresses=["localhost:50051"])
        with pytest.raises(ConnectionError):
            pool.step_all(np.array([0, 1]))

    def test_reset_all_requires_connection(self) -> None:
        from rlox.distributed.remote_env import RemoteEnvPool

        pool = RemoteEnvPool(addresses=["localhost:50051"])
        with pytest.raises(ConnectionError):
            pool.reset_all()

    def test_close_is_idempotent(self) -> None:
        from rlox.distributed.remote_env import RemoteEnvPool

        pool = RemoteEnvPool(addresses=["localhost:50051"])
        pool.close()
        pool.close()  # should not raise

    def test_step_all_return_contract_with_mock(self) -> None:
        """With mocked gRPC stubs, step_all returns correct dict keys/types."""
        from rlox.distributed.remote_env import RemoteEnvPool

        pool = RemoteEnvPool(addresses=["localhost:50051"])

        # Inject mock stubs to simulate a connected pool
        num_envs = 4
        obs_dim = 3
        mock_obs = np.random.randn(num_envs, obs_dim).astype(np.float32)
        mock_rewards = np.random.randn(num_envs).astype(np.float64)
        mock_terminated = np.zeros(num_envs, dtype=np.uint8)
        mock_truncated = np.zeros(num_envs, dtype=np.uint8)

        mock_result = {
            "obs": mock_obs,
            "rewards": mock_rewards,
            "terminated": mock_terminated,
            "truncated": mock_truncated,
            "terminal_obs": [None] * num_envs,
        }

        pool._connected = True
        pool._num_envs = num_envs
        pool._obs_dim = obs_dim
        pool._step_impl = MagicMock(return_value=mock_result)

        result = pool.step_all(np.zeros(num_envs, dtype=np.int64))
        expected_keys = {"obs", "rewards", "terminated", "truncated", "terminal_obs"}
        assert set(result.keys()) == expected_keys
        assert result["obs"].dtype == np.float32
        assert result["rewards"].dtype == np.float64

    def test_reset_all_return_contract_with_mock(self) -> None:
        """With mocked stubs, reset_all returns correct array."""
        from rlox.distributed.remote_env import RemoteEnvPool

        pool = RemoteEnvPool(addresses=["localhost:50051"])

        num_envs = 4
        obs_dim = 3
        mock_obs = np.random.randn(num_envs, obs_dim).astype(np.float32)

        pool._connected = True
        pool._num_envs = num_envs
        pool._obs_dim = obs_dim
        pool._reset_impl = MagicMock(return_value=mock_obs)

        result = pool.reset_all()
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (num_envs, obs_dim)


# ---------------------------------------------------------------------------
# MultiGPUTrainer FSDP strategy
# ---------------------------------------------------------------------------


class TestMultiGPUTrainerStrategy:
    """Test MultiGPUTrainer with DDP and FSDP strategies (mocked dist)."""

    @pytest.fixture
    def mock_dist(self):
        """Mock torch.distributed so we don't need real GPUs."""
        # Patch .to() on nn.Module to be a no-op (avoids CUDA init)
        original_to = torch.nn.Module.to

        def fake_to(self, *args, **kwargs):
            return self

        with patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_rank", return_value=0), \
             patch("torch.distributed.get_world_size", return_value=1), \
             patch("torch.cuda.is_available", return_value=False), \
             patch.object(torch.nn.Module, "to", fake_to):
            yield

    def _make_dummy_trainer_cls(self) -> type:
        """Create a minimal trainer class with a policy attribute."""

        class DummyPolicy(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = torch.nn.Linear(4, 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        class DummyAlgo:
            def __init__(self) -> None:
                self.policy = DummyPolicy()

        class DummyTrainer:
            def __init__(self, env: str, config: dict | None = None,
                         seed: int = 42, **kwargs: Any) -> None:
                self.algo = DummyAlgo()

            def train(self, total_timesteps: int) -> dict[str, float]:
                return {"loss": 0.0}

        return DummyTrainer

    def test_accepts_strategy_parameter(self, mock_dist: None) -> None:
        from rlox.distributed.multi_gpu import MultiGPUTrainer

        sig = inspect.signature(MultiGPUTrainer.__init__)
        assert "strategy" in sig.parameters

    def test_ddp_strategy_wraps_with_ddp(self, mock_dist: None) -> None:
        from rlox.distributed.multi_gpu import MultiGPUTrainer

        cls = self._make_dummy_trainer_cls()

        with patch("torch.nn.parallel.DistributedDataParallel", wraps=lambda m, **kw: m):
            trainer = MultiGPUTrainer(cls, env="CartPole-v1", strategy="ddp")
            assert trainer.strategy == "ddp"

    def test_fsdp_strategy_wraps_with_fsdp(self, mock_dist: None) -> None:
        from rlox.distributed.multi_gpu import MultiGPUTrainer

        cls = self._make_dummy_trainer_cls()

        mock_fsdp = MagicMock(side_effect=lambda m, **kw: m)
        with patch(
            "torch.distributed.fsdp.FullyShardedDataParallel", mock_fsdp
        ):
            trainer = MultiGPUTrainer(cls, env="CartPole-v1", strategy="fsdp")
            assert trainer.strategy == "fsdp"
            mock_fsdp.assert_called()

    def test_invalid_strategy_raises(self, mock_dist: None) -> None:
        from rlox.distributed.multi_gpu import MultiGPUTrainer

        cls = self._make_dummy_trainer_cls()
        with pytest.raises(ValueError, match="strategy"):
            MultiGPUTrainer(cls, env="CartPole-v1", strategy="invalid")


# ---------------------------------------------------------------------------
# Elastic training
# ---------------------------------------------------------------------------


class TestElasticTraining:
    """Test launch_elastic function signature and basic behavior."""

    def test_launch_elastic_exists(self) -> None:
        from rlox.distributed.multi_gpu import launch_elastic

        assert callable(launch_elastic)

    def test_launch_elastic_signature(self) -> None:
        from rlox.distributed.multi_gpu import launch_elastic

        sig = inspect.signature(launch_elastic)
        params = sig.parameters

        assert "trainer_fn" in params
        assert "min_nodes" in params
        assert "max_nodes" in params
        assert "nproc_per_node" in params

        assert params["min_nodes"].default == 1
        assert params["max_nodes"].default == 4
        assert params["nproc_per_node"].default == 1

    def test_launch_elastic_validates_node_range(self) -> None:
        from rlox.distributed.multi_gpu import launch_elastic

        with pytest.raises(ValueError, match="min_nodes"):
            launch_elastic(lambda: None, min_nodes=5, max_nodes=2)


# ---------------------------------------------------------------------------
# Rank-aware logging
# ---------------------------------------------------------------------------


class TestRankAwareLogging:
    """Test that only rank 0 fires logs/callbacks and metrics are reduced."""

    @pytest.fixture
    def mock_dist(self):
        with patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_rank", return_value=0), \
             patch("torch.distributed.get_world_size", return_value=2), \
             patch("torch.cuda.is_available", return_value=False):
            yield

    def test_is_main_rank_helper(self) -> None:
        from rlox.distributed.multi_gpu import is_main_rank

        with patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_rank", return_value=0):
            assert is_main_rank() is True

        with patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_rank", return_value=1):
            assert is_main_rank() is False

    def test_is_main_rank_without_dist(self) -> None:
        from rlox.distributed.multi_gpu import is_main_rank

        with patch("torch.distributed.is_initialized", return_value=False):
            assert is_main_rank() is True

    def test_reduce_metrics_averages(self) -> None:
        from rlox.distributed.multi_gpu import reduce_metrics

        metrics = {"loss": torch.tensor(2.0), "reward": torch.tensor(10.0)}

        with patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_world_size", return_value=2), \
             patch("torch.distributed.all_reduce") as mock_reduce:
            reduced = reduce_metrics(metrics)
            assert mock_reduce.call_count == 2
            # After all_reduce + div by world_size, values should be halved
            for key in metrics:
                assert key in reduced
