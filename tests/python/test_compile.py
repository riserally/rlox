"""Tests for rlox.compile -- torch.compile integration."""

from __future__ import annotations

import sys
import warnings

import pytest
import torch
import torch.nn as nn

from rlox.compile import compile_policy

_TORCH_COMPILE_AVAILABLE = hasattr(torch, "compile") and sys.platform != "win32"

requires_compile = pytest.mark.skipif(
    not _TORCH_COMPILE_AVAILABLE,
    reason="torch.compile not available (torch < 2.0 or unsupported platform)",
)


# ---------------------------------------------------------------------------
# compile_policy on real algorithm objects
# ---------------------------------------------------------------------------


@requires_compile
class TestCompilePolicyPPO:
    def test_compile_policy_ppo_warns_no_forward(self) -> None:
        """PPO's DiscretePolicy has no forward() — compile_policy should warn."""
        from rlox.algorithms.ppo import PPO

        ppo = PPO(env_id="CartPole-v1", n_envs=1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compile_policy(ppo)
            # Should warn about no forward()
            assert any("no forward()" in str(warning.message) for warning in w)

        # Policy should still work
        obs = torch.randn(2, 4)
        actions, log_probs = ppo.policy.get_action_and_logprob(obs)
        assert actions.shape == (2,)


@requires_compile
class TestCompilePolicySAC:
    def test_compile_policy_sac(self) -> None:
        """SAC networks have forward() — should compile successfully."""
        from rlox.algorithms.sac import SAC

        sac = SAC(env_id="Pendulum-v1")
        original_actor_type = type(sac.actor)

        compile_policy(sac)

        # SquashedGaussianPolicy has forward(), should be compiled
        assert type(sac.actor) is not original_actor_type or hasattr(
            sac.actor, "_torchdynamo_orig_callable"
        )


@requires_compile
class TestCompilePolicyDQN:
    def test_compile_policy_dqn(self) -> None:
        """DQN's QNetwork has forward() — should compile successfully."""
        from rlox.algorithms.dqn import DQN

        dqn = DQN(env_id="CartPole-v1")
        original_type = type(dqn.q_network)

        compile_policy(dqn)

        assert type(dqn.q_network) is not original_type or hasattr(
            dqn.q_network, "_torchdynamo_orig_callable"
        )


@requires_compile
class TestCompileFlagOnTrainer:
    def test_ppo_trainer_compile_flag(self) -> None:
        """PPOTrainer(compile=True) should not crash."""
        from rlox.trainers import PPOTrainer

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            trainer = PPOTrainer(env="CartPole-v1", compile=True)

        # Should still train
        metrics = trainer.train(total_timesteps=128)
        assert "mean_reward" in metrics


# ---------------------------------------------------------------------------
# compile_policy with mock objects (no real envs)
# ---------------------------------------------------------------------------


class _SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


class _NoForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def custom_method(self, x):
        return self.linear(x)


@requires_compile
class TestCompilePolicyMock:
    def test_compile_net_with_forward(self) -> None:
        """Module with forward() should compile."""

        class MockAlgo:
            policy = _SimpleNet()

        algo = MockAlgo()
        compile_policy(algo)
        # Should be compiled (type changes or dynamo attribute added)
        result = algo.policy(torch.randn(1, 4))
        assert result.shape == (1, 2)

    def test_skip_net_without_forward(self) -> None:
        """Module without forward() should be skipped with warning."""

        class MockAlgo:
            policy = _NoForwardNet()

        algo = MockAlgo()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compile_policy(algo)
            assert any("no forward()" in str(warning.message) for warning in w)

    def test_compile_returns_algo(self) -> None:
        """compile_policy should return the algo for chaining."""

        class MockAlgo:
            policy = _SimpleNet()

        algo = MockAlgo()
        result = compile_policy(algo)
        assert result is algo
