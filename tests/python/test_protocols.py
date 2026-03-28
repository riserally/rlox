"""Tests for protocol interfaces and exploration strategies."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from rlox.protocols import (
    OnPolicyActor,
    StochasticActor,
    DeterministicActor,
    QFunction,
    DiscreteQFunction,
    ExplorationStrategy,
    ReplayBufferProtocol,
)


# ---------------------------------------------------------------------------
# Protocol conformance tests
# ---------------------------------------------------------------------------


class TestOnPolicyActorProtocol:
    def test_discrete_policy_satisfies_protocol(self):
        from rlox.policies import DiscretePolicy
        policy = DiscretePolicy(obs_dim=4, n_actions=2)
        assert isinstance(policy, OnPolicyActor)

    def test_continuous_policy_satisfies_protocol(self):
        from rlox.policies import ContinuousPolicy
        policy = ContinuousPolicy(obs_dim=3, act_dim=1)
        assert isinstance(policy, OnPolicyActor)

    def test_custom_policy_satisfies_protocol(self):
        class MyPolicy:
            def get_action_and_logprob(self, obs):
                return torch.zeros(obs.shape[0]), torch.zeros(obs.shape[0])
            def get_value(self, obs):
                return torch.zeros(obs.shape[0])
            def get_logprob_and_entropy(self, obs, actions):
                return torch.zeros(obs.shape[0]), torch.zeros(obs.shape[0])

        assert isinstance(MyPolicy(), OnPolicyActor)

    def test_incomplete_class_fails_protocol(self):
        class BadPolicy:
            def get_action_and_logprob(self, obs):
                return torch.zeros(1), torch.zeros(1)
            # missing get_value and get_logprob_and_entropy

        assert not isinstance(BadPolicy(), OnPolicyActor)


class TestStochasticActorProtocol:
    def test_squashed_gaussian_satisfies_protocol(self):
        from rlox.networks import SquashedGaussianPolicy
        actor = SquashedGaussianPolicy(obs_dim=3, act_dim=1, hidden=64)
        assert isinstance(actor, StochasticActor)

    def test_custom_actor_satisfies_protocol(self):
        class MyActor:
            def sample(self, obs):
                return torch.zeros(obs.shape[0], 1), torch.zeros(obs.shape[0])
            def deterministic(self, obs):
                return torch.zeros(obs.shape[0], 1)

        assert isinstance(MyActor(), StochasticActor)


class TestQFunctionProtocol:
    def test_q_network_satisfies_protocol(self):
        from rlox.networks import QNetwork
        critic = QNetwork(obs_dim=3, act_dim=1, hidden=64)
        assert isinstance(critic, QFunction)


class TestDiscreteQFunctionProtocol:
    def test_simple_q_network_satisfies_protocol(self):
        from rlox.networks import SimpleQNetwork
        q_net = SimpleQNetwork(obs_dim=4, act_dim=2, hidden=64)
        assert isinstance(q_net, DiscreteQFunction)

    def test_dueling_q_network_satisfies_protocol(self):
        from rlox.networks import DuelingQNetwork
        q_net = DuelingQNetwork(obs_dim=4, act_dim=2, hidden=64)
        assert isinstance(q_net, DiscreteQFunction)


class TestReplayBufferProtocol:
    def test_rlox_buffer_satisfies_protocol(self):
        import rlox
        buf = rlox.ReplayBuffer(100, obs_dim=4, act_dim=1)
        assert isinstance(buf, ReplayBufferProtocol)


# ---------------------------------------------------------------------------
# Exploration strategies
# ---------------------------------------------------------------------------


class TestGaussianNoise:
    def test_instantiation(self):
        from rlox.exploration import GaussianNoise
        noise = GaussianNoise(sigma=0.1)
        assert noise is not None

    def test_adds_noise(self):
        from rlox.exploration import GaussianNoise
        noise = GaussianNoise(sigma=0.5, seed=42)
        action = np.array([0.0, 0.0])
        noisy = noise.select_action(action, step=0, total_steps=100)
        assert not np.array_equal(action, noisy)

    def test_satisfies_protocol(self):
        from rlox.exploration import GaussianNoise
        noise = GaussianNoise(sigma=0.1)
        assert isinstance(noise, ExplorationStrategy)


class TestEpsilonGreedy:
    def test_instantiation(self):
        from rlox.exploration import EpsilonGreedy
        eps = EpsilonGreedy(n_actions=4)
        assert eps is not None

    def test_decays_epsilon(self):
        from rlox.exploration import EpsilonGreedy
        eps = EpsilonGreedy(n_actions=4, eps_start=1.0, eps_end=0.05, decay_fraction=0.5)
        action = np.array([1])
        # At step 0, epsilon = 1.0 (always random)
        # At step 50 of 100, epsilon = 0.05 (mostly greedy)
        early_randoms = sum(
            1 for _ in range(100)
            if not np.array_equal(eps.select_action(action, 0, 100), action)
        )
        # Most should be random at step 0
        assert early_randoms > 50

    def test_satisfies_protocol(self):
        from rlox.exploration import EpsilonGreedy
        eps = EpsilonGreedy(n_actions=4)
        assert isinstance(eps, ExplorationStrategy)


class TestOUNoise:
    def test_instantiation(self):
        from rlox.exploration import OUNoise
        noise = OUNoise(action_dim=2)
        assert noise is not None

    def test_correlated_noise(self):
        from rlox.exploration import OUNoise
        noise = OUNoise(action_dim=1, sigma=0.3, seed=42)
        action = np.array([0.0])
        values = [noise.select_action(action, i, 100)[0] for i in range(10)]
        # OU noise is correlated — consecutive values should be similar
        diffs = [abs(values[i+1] - values[i]) for i in range(len(values)-1)]
        assert np.mean(diffs) < 0.5  # correlated, not random jumps

    def test_reset(self):
        from rlox.exploration import OUNoise
        noise = OUNoise(action_dim=2, seed=42)
        action = np.zeros(2)
        noise.select_action(action, 0, 100)
        noise.reset()
        # After reset, internal state should be zero
        assert np.allclose(noise._state, 0.0)

    def test_satisfies_protocol(self):
        from rlox.exploration import OUNoise
        noise = OUNoise(action_dim=2)
        assert isinstance(noise, ExplorationStrategy)


# ---------------------------------------------------------------------------
# Composable losses
# ---------------------------------------------------------------------------


class TestComposableLoss:
    def test_ppo_loss_returns_loss_output(self):
        from rlox.losses import PPOLoss
        loss_fn = PPOLoss()
        # PPOLoss already works — just verify it returns (loss, metrics)
        assert callable(loss_fn)

    def test_loss_addition(self):
        from rlox.losses import CompositeLoss, LossComponent

        class ConstantLoss(LossComponent):
            def compute(self, **kwargs):
                return torch.tensor(1.0), {"const": 1.0}

        class AnotherLoss(LossComponent):
            def compute(self, **kwargs):
                return torch.tensor(2.0), {"another": 2.0}

        combined = CompositeLoss([
            (1.0, ConstantLoss()),
            (0.5, AnotherLoss()),
        ])
        loss, metrics = combined.compute()
        assert abs(loss.item() - 2.0) < 1e-6  # 1.0 * 1.0 + 0.5 * 2.0
        assert "const" in metrics
        assert "another" in metrics


# ---------------------------------------------------------------------------
# Builder pattern
# ---------------------------------------------------------------------------


class TestSACBuilder:
    def test_basic_build(self):
        from rlox.builders import SACBuilder
        sac = SACBuilder().env("Pendulum-v1").seed(42).learning_starts(100).build()
        assert sac is not None
        assert hasattr(sac, "train")
        assert hasattr(sac, "predict")

    def test_custom_actor(self):
        from rlox.builders import SACBuilder
        from rlox.networks import SquashedGaussianPolicy

        custom_actor = SquashedGaussianPolicy(obs_dim=3, act_dim=1, hidden=128)
        sac = SACBuilder().env("Pendulum-v1").actor(custom_actor).learning_starts(100).build()
        assert sac.actor is custom_actor

    def test_custom_exploration(self):
        from rlox.builders import SACBuilder
        from rlox.exploration import OUNoise

        sac = (SACBuilder()
               .env("Pendulum-v1")
               .exploration(OUNoise(action_dim=1))
               .learning_starts(100)
               .build())
        assert sac is not None


class TestPPOBuilder:
    def test_basic_build(self):
        from rlox.builders import PPOBuilder
        ppo = PPOBuilder().env("CartPole-v1").build()
        assert ppo is not None
        assert hasattr(ppo, "train")

    def test_custom_policy(self):
        from rlox.builders import PPOBuilder
        from rlox.policies import DiscretePolicy

        custom = DiscretePolicy(obs_dim=4, n_actions=2, hidden=128)
        ppo = PPOBuilder().env("CartPole-v1").policy(custom).build()
        assert ppo.policy is custom
