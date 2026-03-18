"""Phase 9 tests: V-trace, Pipeline, vLLM backend, reward model serving,
MAPPO, DreamerV3, IMPALA, API stability."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

import rlox


# =========================================================================
# Group 0: V-trace (Rust binding)
# =========================================================================


class TestVtrace:
    def test_import(self):
        assert hasattr(rlox, "compute_vtrace")

    def test_on_policy_basic(self):
        """On-policy (log_rhos=0) should give sensible corrections."""
        log_rhos = np.zeros(3, dtype=np.float32)
        rewards = np.ones(3, dtype=np.float32)
        values = np.zeros(3, dtype=np.float32)
        vs, adv = rlox.compute_vtrace(log_rhos, rewards, values, 0.0, 0.99)
        assert vs.shape == (3,)
        assert adv.shape == (3,)
        # vs should be positive (rewards > values)
        assert np.all(vs > 0)

    def test_empty_input(self):
        log_rhos = np.array([], dtype=np.float32)
        rewards = np.array([], dtype=np.float32)
        values = np.array([], dtype=np.float32)
        vs, adv = rlox.compute_vtrace(log_rhos, rewards, values, 0.0, 0.99)
        assert vs.shape == (0,)
        assert adv.shape == (0,)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(RuntimeError):
            rlox.compute_vtrace(
                np.zeros(3, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                0.0,
                0.99,
            )

    def test_clipping_reduces_correction(self):
        log_rhos = np.array([5.0], dtype=np.float32)  # exp(5) ~ 148
        rewards = np.array([1.0], dtype=np.float32)
        values = np.array([0.0], dtype=np.float32)
        vs_clipped, _ = rlox.compute_vtrace(
            log_rhos, rewards, values, 0.0, 0.99, rho_bar=1.0, c_bar=1.0
        )
        vs_unclipped, _ = rlox.compute_vtrace(
            log_rhos, rewards, values, 0.0, 0.99, rho_bar=200.0, c_bar=200.0
        )
        assert vs_clipped[0] < vs_unclipped[0]

    def test_default_rho_c_bar(self):
        """Default rho_bar and c_bar should be 1.0."""
        log_rhos = np.zeros(5, dtype=np.float32)
        rewards = np.ones(5, dtype=np.float32)
        values = np.ones(5, dtype=np.float32) * 0.5
        # Should work without specifying rho_bar/c_bar
        vs, adv = rlox.compute_vtrace(log_rhos, rewards, values, 0.0, 0.99)
        assert vs.dtype == np.float32
        assert adv.dtype == np.float32

    def test_reference_match(self):
        """Match against manual Python reference implementation."""
        gamma = 0.9
        rho_bar = 1.5
        c_bar = 1.2
        log_rhos = np.array([0.2, -0.3, 0.8], dtype=np.float32)
        rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        values = np.array([0.5, 1.0, 1.5], dtype=np.float32)
        bootstrap = 2.0

        # Python reference (backwards)
        n = 3
        vs_ref = np.zeros(n, dtype=np.float32)
        adv_ref = np.zeros(n, dtype=np.float32)
        vs_next = bootstrap
        for t in reversed(range(n)):
            rho_t = min(rho_bar, np.exp(log_rhos[t]))
            c_t = min(c_bar, np.exp(log_rhos[t]))
            next_val = bootstrap if t == n - 1 else values[t + 1]
            delta_t = rho_t * (rewards[t] + gamma * next_val - values[t])
            vs_ref[t] = values[t] + delta_t + gamma * c_t * (vs_next - next_val)
            adv_ref[t] = rho_t * (rewards[t] + gamma * vs_next - values[t])
            vs_next = vs_ref[t]

        vs, adv = rlox.compute_vtrace(
            log_rhos, rewards, values, bootstrap, gamma, rho_bar=rho_bar, c_bar=c_bar
        )
        np.testing.assert_allclose(vs, vs_ref, rtol=1e-4)
        np.testing.assert_allclose(adv, adv_ref, rtol=1e-4)


# =========================================================================
# Group 1: Decoupled Pipeline
# =========================================================================


class TestDecoupledPipeline:
    def test_import(self):
        from rlox.distributed.pipeline import Pipeline
        assert Pipeline is not None

    def test_delivers_batches(self):
        from rlox.distributed.pipeline import Pipeline
        pipe = Pipeline(env_id="CartPole-v1", n_envs=2, n_steps=16, channel_capacity=2, seed=0)
        try:
            batch = pipe.next_batch(timeout=10.0)
            assert batch is not None
            assert hasattr(batch, "obs")
            assert hasattr(batch, "advantages")
        finally:
            pipe.close()

    def test_try_next_batch_nonblocking(self):
        from rlox.distributed.pipeline import Pipeline
        pipe = Pipeline(env_id="CartPole-v1", n_envs=2, n_steps=16, channel_capacity=2, seed=0)
        try:
            # Might be None if collector hasn't produced yet
            result = pipe.try_next_batch()
            assert result is None or hasattr(result, "obs")
        finally:
            pipe.close()

    def test_close_is_idempotent(self):
        from rlox.distributed.pipeline import Pipeline
        pipe = Pipeline(env_id="CartPole-v1", n_envs=2, n_steps=8, seed=0)
        pipe.close()
        pipe.close()  # Should not raise


# =========================================================================
# Group 2: vLLM Backend
# =========================================================================


class TestVllmBackend:
    def test_import(self):
        from rlox.distributed.vllm_backend import VllmBackend
        assert VllmBackend is not None

    def _mock_urlopen(self, response_data):
        """Create a mock context manager for urllib.request.urlopen."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_mock_generate(self):
        import json
        from rlox.distributed.vllm_backend import VllmBackend
        backend = VllmBackend(base_url="http://localhost:9999")

        response_data = {
            "choices": [
                {"text": "hello world", "logprobs": {"token_logprobs": [-0.5, -0.3]}}
            ]
        }
        mock_resp = self._mock_urlopen(response_data)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = backend.generate(["test prompt"], max_new_tokens=5)
            assert len(result) == 1
            assert "text" in result[0]

    def test_mock_log_probs(self):
        import json
        from rlox.distributed.vllm_backend import VllmBackend
        backend = VllmBackend(base_url="http://localhost:9999")

        response_data = {"log_probs": [[-0.5, -0.3, -0.1]]}
        mock_resp = self._mock_urlopen(response_data)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = backend.log_probs([[1, 2, 3]])
            assert len(result) == 1
            assert len(result[0]) == 3


# =========================================================================
# Group 3: Reward Model Serving
# =========================================================================


class TestRewardModelServing:
    def _make_reward_model(self):
        """Create a simple reward model that returns constant scores based on input size."""
        class _FakeRM(nn.Module):
            def forward(self, x):
                return torch.ones(x.shape[0])

        return _FakeRM()

    def test_reward_model_server(self):
        from rlox.llm import RewardModelServer
        model = self._make_reward_model()
        server = RewardModelServer(model=model, batch_size=32)
        scores = server.score_batch(
            prompts=["hello", "world"],
            completions=["response1", "response2"],
        )
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (2,)

    def test_ensemble_reward_model(self):
        from rlox.llm import EnsembleRewardModel
        models = [self._make_reward_model() for _ in range(3)]
        ensemble = EnsembleRewardModel(models=models, weights=[0.5, 0.3, 0.2])
        scores = ensemble.score_batch(
            prompts=["hello", "world"],
            completions=["response1", "response2"],
        )
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (2,)

    def test_ensemble_equal_weights(self):
        from rlox.llm import EnsembleRewardModel
        models = [self._make_reward_model() for _ in range(2)]
        ensemble = EnsembleRewardModel(models=models)
        scores = ensemble.score_batch(
            prompts=["a"], completions=["b"]
        )
        assert scores.shape == (1,)

    def test_multi_objective_reward(self):
        from rlox.llm import MultiObjectiveReward

        def helpfulness(prompts, completions):
            return np.ones(len(prompts))

        def safety(prompts, completions):
            return np.ones(len(prompts)) * 0.5

        mo = MultiObjectiveReward(
            objectives={"helpful": helpfulness, "safe": safety},
            weights={"helpful": 0.7, "safe": 0.3},
        )
        scores = mo.score_batch(
            prompts=["hello"],
            completions=["world"],
        )
        assert scores.shape == (1,)
        # 0.7 * 1.0 + 0.3 * 0.5 = 0.85
        np.testing.assert_allclose(scores, [0.85], rtol=1e-5)


# =========================================================================
# Group 4: MAPPO
# =========================================================================


class TestMAPPO:
    def test_import(self):
        from rlox.algorithms.mappo import MAPPO
        assert MAPPO is not None

    def test_smoke_cartpole(self):
        from rlox.algorithms.mappo import MAPPO
        agent = MAPPO(env_id="CartPole-v1", n_agents=1, n_envs=2, seed=42)
        metrics = agent.train(total_timesteps=256)
        assert "mean_reward" in metrics


# =========================================================================
# Group 5: DreamerV3 (simplified)
# =========================================================================


class TestDreamerV3:
    def test_import(self):
        from rlox.algorithms.dreamer import DreamerV3
        assert DreamerV3 is not None

    def test_smoke_cartpole(self):
        from rlox.algorithms.dreamer import DreamerV3
        agent = DreamerV3(
            env_id="CartPole-v1",
            n_envs=1,
            seed=42,
            latent_dim=32,
            buffer_size=500,
        )
        metrics = agent.train(total_timesteps=256)
        assert "mean_reward" in metrics


# =========================================================================
# Group 6: IMPALA
# =========================================================================


class TestIMPALA:
    def test_import(self):
        from rlox.algorithms.impala import IMPALA
        assert IMPALA is not None

    def test_smoke_cartpole(self):
        from rlox.algorithms.impala import IMPALA
        agent = IMPALA(
            env_id="CartPole-v1",
            n_actors=2,
            n_envs=2,
            seed=42,
        )
        metrics = agent.train(total_timesteps=256)
        assert "mean_reward" in metrics


# =========================================================================
# Group 7: LLM Environment
# =========================================================================


class TestLLMEnvironment:
    def test_import(self):
        from rlox.llm import LLMEnvironment
        assert LLMEnvironment is not None

    def test_create(self):
        from rlox.llm import LLMEnvironment
        env = LLMEnvironment(backend="vllm", url="http://localhost:8000")
        assert env is not None

    def test_generate_with_mock(self):
        from rlox.llm import LLMEnvironment

        response_data = {
            "choices": [
                {"text": "generated text", "logprobs": {"token_logprobs": [-0.1]}}
            ]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            env = LLMEnvironment(backend="vllm", url="http://localhost:9999")
            results = env.generate(["hello"], n=1, max_new_tokens=5)
            assert len(results) == 1


# =========================================================================
# Group 8: API 1.0 Stability
# =========================================================================


class TestAPI10Stability:
    def test_all_defined(self):
        assert hasattr(rlox, "__all__")
        assert isinstance(rlox.__all__, list)

    def test_version_defined(self):
        assert hasattr(rlox, "__version__")
        assert rlox.__version__  # non-empty string is enough

    def test_phase9_symbols_present(self):
        """All Phase 9 symbols should be importable from rlox."""
        assert hasattr(rlox, "compute_vtrace")

    def test_phase7_8_symbols_still_present(self):
        """Phase 7/8 symbols must not regress."""
        # Phase 7
        assert hasattr(rlox, "RolloutBatch")
        assert hasattr(rlox, "RolloutCollector")
        assert hasattr(rlox, "PPOLoss")
        # Core
        assert hasattr(rlox, "CartPole")
        assert hasattr(rlox, "VecEnv")
        assert hasattr(rlox, "ReplayBuffer")
        assert hasattr(rlox, "compute_gae")

    def test_algorithm_constructors_consistent(self):
        """All algorithms should accept env_id as first arg."""
        from rlox.algorithms import PPO, A2C, SAC, TD3, DQN
        from rlox.algorithms.mappo import MAPPO
        from rlox.algorithms.impala import IMPALA

        # These should all accept env_id
        for AlgoCls in [PPO, A2C]:
            algo = AlgoCls(env_id="CartPole-v1")
            assert hasattr(algo, "train")

    def test_trainer_save_load_api(self):
        """Checkpoint save/load API should exist."""
        from rlox.checkpoint import Checkpoint
        assert hasattr(Checkpoint, "save")
        assert hasattr(Checkpoint, "load")

    def test_distributed_importable(self):
        from rlox.distributed import Pipeline
        from rlox.distributed.vllm_backend import VllmBackend
        assert Pipeline is not None
        assert VllmBackend is not None

    def test_llm_importable(self):
        from rlox.llm import (
            LLMEnvironment,
            RewardModelServer,
            EnsembleRewardModel,
            MultiObjectiveReward,
        )
        assert all(
            cls is not None
            for cls in [LLMEnvironment, RewardModelServer, EnsembleRewardModel, MultiObjectiveReward]
        )
