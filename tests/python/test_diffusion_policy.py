"""Tests for Diffusion Policy algorithm.

TDD: These tests are written FIRST, before the implementation.
"""

from __future__ import annotations

import pytest
import torch
import numpy as np


class TestDiffusionPolicyConstruction:
    """Tests for DiffusionPolicy construction and registration."""

    def test_diffusion_constructs(self):
        """DiffusionPolicy can be instantiated with an environment ID."""
        from rlox.algorithms.diffusion_policy import DiffusionPolicy

        dp = DiffusionPolicy(env_id="Pendulum-v1")
        assert dp is not None
        assert dp.env_id == "Pendulum-v1"

    def test_diffusion_satisfies_protocol(self):
        """DiffusionPolicy satisfies the Algorithm protocol."""
        from rlox.algorithms.diffusion_policy import DiffusionPolicy
        from rlox.protocols import Algorithm

        assert isinstance(DiffusionPolicy, type)
        dp = DiffusionPolicy(env_id="Pendulum-v1")
        # Protocol requires: env_id, train(), save(), from_checkpoint()
        assert hasattr(dp, "env_id")
        assert callable(getattr(dp, "train", None))
        assert callable(getattr(dp, "save", None))
        assert callable(getattr(dp, "from_checkpoint", None))

    def test_diffusion_registered(self):
        """DiffusionPolicy is registered in the algorithm registry."""
        from rlox.trainer import ALGORITHM_REGISTRY

        # Force registration
        import rlox.algorithms.diffusion_policy  # noqa: F401

        assert "diffusion" in ALGORITHM_REGISTRY

    def test_diffusion_train_returns_metrics(self):
        """train() returns a dict with a 'loss' key."""
        from rlox.algorithms.diffusion_policy import DiffusionPolicy

        dp = DiffusionPolicy(
            env_id="Pendulum-v1",
            n_diffusion_steps=10,
            action_horizon=4,
            obs_horizon=2,
            hidden_dim=32,
            batch_size=16,
            warmup_steps=100,
        )
        metrics = dp.train(total_timesteps=200)
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)


class TestDiffusionNoiseSchedule:
    """Tests for noise schedule construction and shapes."""

    def test_diffusion_noise_schedule_shapes(self):
        """Noise schedule produces correct tensor shapes."""
        from rlox.algorithms.diffusion_policy import NoiseSchedule

        n_steps = 50
        sched = NoiseSchedule(n_steps=n_steps, schedule_type="cosine")

        assert sched.betas.shape == (n_steps,)
        assert sched.alphas.shape == (n_steps,)
        assert sched.alpha_cumprod.shape == (n_steps,)

        # Betas should be in (0, 1)
        assert (sched.betas > 0).all()
        assert (sched.betas < 1).all()

        # Alpha cumulative product should be monotonically decreasing
        diffs = sched.alpha_cumprod[1:] - sched.alpha_cumprod[:-1]
        assert (diffs <= 0).all()

    def test_linear_noise_schedule(self):
        """Linear schedule produces betas linearly spaced."""
        from rlox.algorithms.diffusion_policy import NoiseSchedule

        sched = NoiseSchedule(
            n_steps=100,
            schedule_type="linear",
            beta_start=0.0001,
            beta_end=0.02,
        )
        assert sched.betas.shape == (100,)
        assert abs(sched.betas[0].item() - 0.0001) < 1e-6
        assert abs(sched.betas[-1].item() - 0.02) < 1e-6

    def test_cosine_noise_schedule(self):
        """Cosine schedule alpha_cumprod follows cosine shape."""
        from rlox.algorithms.diffusion_policy import NoiseSchedule

        sched = NoiseSchedule(n_steps=50, schedule_type="cosine")
        # First alpha_cumprod should be close to 1, last should be close to 0
        assert sched.alpha_cumprod[0].item() > 0.9
        assert sched.alpha_cumprod[-1].item() < 0.1


class TestDiffusionDenoising:
    """Tests for the denoising process."""

    def test_diffusion_denoising_reduces_noise(self):
        """After training on constant actions, denoising should recover them.

        We train the network on a dataset of identical actions using
        a linear schedule (more numerically stable for few steps),
        then verify that reverse diffusion from noise produces output
        closer to the training target than the initial noise.
        """
        from rlox.algorithms.diffusion_policy import DiffusionPolicy

        torch.manual_seed(42)

        act_dim = 1
        obs_dim = 3
        action_horizon = 2
        obs_horizon = 2
        n_steps = 20

        dp = DiffusionPolicy(
            env_id="Pendulum-v1",
            n_diffusion_steps=n_steps,
            n_inference_steps=n_steps,
            action_horizon=action_horizon,
            obs_horizon=obs_horizon,
            hidden_dim=128,
            noise_schedule="linear",
            beta_start=0.0001,
            beta_end=0.02,
            learning_rate=3e-4,
        )

        # Train on constant actions = 0.5
        target_action = torch.full((64, action_horizon, act_dim), 0.5)
        obs_batch = torch.zeros(64, obs_horizon, obs_dim)

        dp.model.train()
        for _ in range(500):
            t = torch.randint(0, n_steps, (64,))
            noise = torch.randn_like(target_action)
            x_t, _ = dp._forward_diffusion(target_action, t, noise)
            eps_pred = dp.model(x_t, t, obs_batch)
            loss = torch.nn.functional.mse_loss(eps_pred, noise)
            dp.optimizer.zero_grad()
            loss.backward()
            dp.optimizer.step()

        # Denoise from standard normal noise (not scaled up)
        torch.manual_seed(123)
        x_noise = torch.randn(1, action_horizon, act_dim)
        obs = torch.zeros(1, obs_horizon, obs_dim)
        noise_error = (x_noise - 0.5).abs().mean().item()

        with torch.no_grad():
            denoised = dp._reverse_diffusion(obs, x_noise.clone())

        denoised_error = (denoised - 0.5).abs().mean().item()

        # Denoised output should be closer to target than random noise was
        assert denoised_error < noise_error, (
            f"Denoised error ({denoised_error:.4f}) should be less than "
            f"noise error ({noise_error:.4f})"
        )


class TestDiffusionConfig:
    """Tests for DiffusionPolicyConfig."""

    def test_diffusion_config_roundtrip(self):
        """Config can be created and round-tripped via to_dict/from_dict."""
        from rlox.config import DiffusionPolicyConfig

        cfg = DiffusionPolicyConfig(
            n_diffusion_steps=100,
            action_horizon=16,
            hidden_dim=512,
        )
        d = cfg.to_dict()
        cfg2 = DiffusionPolicyConfig.from_dict(d)

        assert cfg2.n_diffusion_steps == 100
        assert cfg2.action_horizon == 16
        assert cfg2.hidden_dim == 512
        assert cfg2.obs_horizon == 2  # default

    def test_diffusion_config_validation(self):
        """Config raises ValueError for invalid parameters."""
        from rlox.config import DiffusionPolicyConfig

        with pytest.raises(ValueError):
            DiffusionPolicyConfig(learning_rate=-1.0)

        with pytest.raises(ValueError):
            DiffusionPolicyConfig(n_diffusion_steps=0)

        with pytest.raises(ValueError):
            DiffusionPolicyConfig(batch_size=0)
