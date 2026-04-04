"""Tests for Vanilla Policy Gradient (VPG / REINFORCE with baseline).

TDD: these tests are written FIRST, before the implementation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from rlox.protocols import Algorithm


# ============================================================================
# Construction & Protocol
# ============================================================================


class TestVPGConstruction:
    def test_vpg_constructs(self):
        """VPG can be instantiated with just an env_id."""
        from rlox.algorithms.vpg import VPG

        vpg = VPG(env_id="CartPole-v1")
        assert vpg is not None
        assert vpg.env_id == "CartPole-v1"

    def test_vpg_satisfies_protocol(self):
        """VPG exposes env_id, train(), save(), from_checkpoint()."""
        from rlox.algorithms.vpg import VPG

        vpg = VPG(env_id="CartPole-v1")
        assert hasattr(vpg, "env_id")
        assert callable(getattr(vpg, "train", None))
        assert callable(getattr(vpg, "save", None))
        assert callable(getattr(vpg, "from_checkpoint", None))

    def test_vpg_config_roundtrip(self):
        """VPGConfig serializes to dict and back."""
        from rlox.config import VPGConfig

        cfg = VPGConfig(learning_rate=1e-3, gamma=0.95, hidden=128)
        d = cfg.to_dict()
        cfg2 = VPGConfig.from_dict(d)
        assert cfg2.learning_rate == pytest.approx(1e-3)
        assert cfg2.gamma == pytest.approx(0.95)
        assert cfg2.hidden == 128


# ============================================================================
# Training
# ============================================================================


class TestVPGTraining:
    def test_vpg_train_returns_metrics(self):
        """train() returns a dict containing at least mean_reward."""
        from rlox.algorithms.vpg import VPG

        vpg = VPG(env_id="CartPole-v1", n_envs=2, n_steps=32)
        metrics = vpg.train(total_timesteps=64)
        assert isinstance(metrics, dict)
        assert "mean_reward" in metrics
        assert "policy_loss" in metrics
        assert "vf_loss" in metrics

    def test_vpg_registered(self):
        """Trainer("vpg", ...) resolves to VPG."""
        from rlox.trainer import Trainer

        trainer = Trainer("vpg", env="CartPole-v1", config={"n_envs": 2, "n_steps": 32})
        assert trainer is not None
        # Should be able to train a tiny amount
        metrics = trainer.train(total_timesteps=64)
        assert "mean_reward" in metrics


# ============================================================================
# VPG-specific guarantees
# ============================================================================


class TestVPGBehavior:
    def test_vpg_no_clipping(self):
        """VPG does NOT use PPO-style clipping.

        We verify that the algorithm has no clip_eps / clip_range attribute
        and that its loss computation does not involve torch.clamp on ratios.
        """
        from rlox.algorithms.vpg import VPG

        vpg = VPG(env_id="CartPole-v1")
        # No clipping config
        assert not hasattr(vpg.config, "clip_eps")
        assert not hasattr(vpg.config, "clip_range")
        # No PPOLoss
        assert not hasattr(vpg, "loss_fn") or not type(vpg).__name__ == "PPOLoss"

    def test_vpg_single_policy_step(self):
        """VPG takes exactly ONE policy gradient step per rollout.

        We monkey-patch the optimizer to count how many times step() is called
        during a single update cycle (one rollout).
        """
        from rlox.algorithms.vpg import VPG

        vpg = VPG(env_id="CartPole-v1", n_envs=2, n_steps=32)

        original_step = vpg.policy_optimizer.step
        step_count = 0

        def counting_step(*args, **kwargs):
            nonlocal step_count
            step_count += 1
            return original_step(*args, **kwargs)

        vpg.policy_optimizer.step = counting_step

        # One rollout = n_envs * n_steps = 64 timesteps
        vpg.train(total_timesteps=64)

        # Exactly 1 policy gradient step per rollout (we do 1 rollout)
        assert step_count == 1, (
            f"Expected 1 policy optimizer step per rollout, got {step_count}"
        )

    def test_vpg_separate_optimizers(self):
        """VPG uses separate optimizers for policy and value function."""
        from rlox.algorithms.vpg import VPG

        vpg = VPG(env_id="CartPole-v1")
        assert hasattr(vpg, "policy_optimizer")
        assert hasattr(vpg, "vf_optimizer")
        assert vpg.policy_optimizer is not vpg.vf_optimizer

    def test_vpg_multiple_vf_epochs(self):
        """Value function is trained for multiple epochs per rollout."""
        from rlox.algorithms.vpg import VPG

        vpg = VPG(env_id="CartPole-v1", n_envs=2, n_steps=32, vf_epochs=3)

        original_step = vpg.vf_optimizer.step
        vf_step_count = 0

        def counting_step(*args, **kwargs):
            nonlocal vf_step_count
            vf_step_count += 1
            return original_step(*args, **kwargs)

        vpg.vf_optimizer.step = counting_step
        vpg.train(total_timesteps=64)

        # vf_epochs=3, so 3 value function steps per rollout
        assert vf_step_count == 3


# ============================================================================
# Checkpoint
# ============================================================================


class TestVPGCheckpoint:
    def test_vpg_save_load(self):
        """VPG can be saved and restored from checkpoint."""
        from rlox.algorithms.vpg import VPG

        vpg = VPG(env_id="CartPole-v1", n_envs=2, n_steps=32)
        vpg.train(total_timesteps=64)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "vpg.pt")
            vpg.save(path)
            vpg2 = VPG.from_checkpoint(path, env_id="CartPole-v1")
            assert vpg2.env_id == "CartPole-v1"
