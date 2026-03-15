"""Tests for ContinuousPolicy."""

from __future__ import annotations

import torch
import pytest

from rlox.policies import ContinuousPolicy


@pytest.fixture
def policy() -> ContinuousPolicy:
    return ContinuousPolicy(obs_dim=3, act_dim=2, hidden=32)


class TestContinuousPolicy:
    def test_action_shape(self, policy: ContinuousPolicy) -> None:
        obs = torch.randn(8, 3)
        actions, log_probs = policy.get_action_and_logprob(obs)
        assert actions.shape == (8, 2)
        assert log_probs.shape == (8,)

    def test_logprob_shape(self, policy: ContinuousPolicy) -> None:
        obs = torch.randn(4, 3)
        actions = torch.randn(4, 2)
        log_probs, entropy = policy.get_logprob_and_entropy(obs, actions)
        assert log_probs.shape == (4,)
        assert entropy.shape == (4,)

    def test_value_shape(self, policy: ContinuousPolicy) -> None:
        obs = torch.randn(5, 3)
        values = policy.get_value(obs)
        assert values.shape == (5,)

    def test_entropy_positive(self, policy: ContinuousPolicy) -> None:
        obs = torch.randn(10, 3)
        actions, _ = policy.get_action_and_logprob(obs)
        _, entropy = policy.get_logprob_and_entropy(obs, actions)
        assert (entropy > 0).all()

    def test_logprob_gradient_flows(self, policy: ContinuousPolicy) -> None:
        obs = torch.randn(4, 3)
        actions, log_probs = policy.get_action_and_logprob(obs)
        loss = -log_probs.mean()
        loss.backward()

        has_grad = False
        for param in policy.actor.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "Gradient did not flow through actor"

    def test_get_logprob_and_entropy_matches(self, policy: ContinuousPolicy) -> None:
        """get_logprob_and_entropy should give same log_probs as manual computation."""
        torch.manual_seed(0)
        obs = torch.randn(8, 3)
        actions = torch.randn(8, 2)

        log_probs, entropy = policy.get_logprob_and_entropy(obs, actions)

        # Manual check: forward through actor, build distribution, compute
        mean = policy.actor(obs)
        std = policy.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        expected_lp = dist.log_prob(actions).sum(dim=-1)
        expected_ent = dist.entropy().sum(dim=-1)

        torch.testing.assert_close(log_probs, expected_lp)
        torch.testing.assert_close(entropy, expected_ent)

    def test_log_std_init(self, policy: ContinuousPolicy) -> None:
        """log_std should be initialized to -0.5."""
        expected = torch.full((2,), -0.5)
        torch.testing.assert_close(policy.log_std.data, expected)

    def test_single_obs(self) -> None:
        """Should work with a single observation (batch dim = 1)."""
        policy = ContinuousPolicy(obs_dim=4, act_dim=1, hidden=16)
        obs = torch.randn(1, 4)
        actions, log_probs = policy.get_action_and_logprob(obs)
        assert actions.shape == (1, 1)
        assert log_probs.shape == (1,)
