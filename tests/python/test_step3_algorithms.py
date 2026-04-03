"""Step 3 tests: ICM, Population-Based Training, TRPO.

TDD: These tests are written FIRST, before the implementations.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from rlox.protocols import IntrinsicMotivation


# ============================================================================
# TestICM
# ============================================================================


class TestICM:
    """Tests for Intrinsic Curiosity Module."""

    def test_icm_constructs(self):
        """ICM can be instantiated with obs_dim and action_dim."""
        from rlox.intrinsic.icm import ICM

        icm = ICM(obs_dim=4, action_dim=2)
        assert icm is not None

    def test_icm_reward_shape(self):
        """Intrinsic reward shape matches batch size."""
        from rlox.intrinsic.icm import ICM

        icm = ICM(obs_dim=4, action_dim=2)
        obs = torch.randn(16, 4)
        next_obs = torch.randn(16, 4)
        actions = torch.randint(0, 2, (16,))
        reward = icm.compute_intrinsic_reward(obs, next_obs, actions)
        assert reward.shape == (16,)

    def test_icm_novelty_higher_for_unseen(self):
        """Novel (out-of-distribution) states should get higher intrinsic reward
        than states the module has been trained on."""
        from rlox.intrinsic.icm import ICM

        icm = ICM(obs_dim=4, action_dim=2, learning_rate=1e-3)

        # Train on a narrow distribution around zero
        for _ in range(200):
            obs = torch.randn(32, 4) * 0.1
            next_obs = obs + torch.randn(32, 4) * 0.01
            actions = torch.randint(0, 2, (32,))
            icm.update(obs, next_obs, actions)

        # Familiar states
        familiar_obs = torch.randn(32, 4) * 0.1
        familiar_next = familiar_obs + torch.randn(32, 4) * 0.01
        familiar_actions = torch.randint(0, 2, (32,))
        familiar_reward = icm.compute_intrinsic_reward(
            familiar_obs, familiar_next, familiar_actions
        )

        # Novel states -- far from training distribution
        novel_obs = torch.randn(32, 4) * 10.0
        novel_next = novel_obs + torch.randn(32, 4) * 5.0
        novel_actions = torch.randint(0, 2, (32,))
        novel_reward = icm.compute_intrinsic_reward(
            novel_obs, novel_next, novel_actions
        )

        assert novel_reward.mean() > familiar_reward.mean()

    def test_icm_inverse_model_predicts_actions(self):
        """Inverse model output shape matches action space for discrete actions."""
        from rlox.intrinsic.icm import ICM

        action_dim = 5
        icm = ICM(obs_dim=8, action_dim=action_dim)
        obs = torch.randn(16, 8)
        next_obs = torch.randn(16, 8)

        # Access inverse model output directly
        with torch.no_grad():
            feat_s = icm.encoder(obs)
            feat_s_next = icm.encoder(next_obs)
            action_pred = icm.inverse_model(torch.cat([feat_s, feat_s_next], dim=-1))

        assert action_pred.shape == (16, action_dim)

    def test_icm_protocol_compliance(self):
        """ICM satisfies the IntrinsicMotivation protocol."""
        from rlox.intrinsic.icm import ICM

        icm = ICM(obs_dim=4, action_dim=2)
        assert isinstance(icm, IntrinsicMotivation)

    def test_icm_update_returns_loss_dict(self):
        """update() returns a dict containing forward_loss and inverse_loss."""
        from rlox.intrinsic.icm import ICM

        icm = ICM(obs_dim=4, action_dim=2)
        obs = torch.randn(16, 4)
        next_obs = torch.randn(16, 4)
        actions = torch.randint(0, 2, (16,))
        info = icm.update(obs, next_obs, actions)
        assert isinstance(info, dict)
        assert "forward_loss" in info
        assert "inverse_loss" in info
        assert isinstance(info["forward_loss"], float)
        assert isinstance(info["inverse_loss"], float)


# ============================================================================
# TestPBT
# ============================================================================


class TestPBT:
    """Tests for Population-Based Training."""

    def test_pbt_constructs(self):
        """PBT can be instantiated with an algorithm name and environment."""
        from rlox.pbt import PBT

        pbt = PBT(algo="ppo", env="CartPole-v1", population_size=2)
        assert pbt is not None

    def test_pbt_trains_population(self):
        """A short PBT run completes without error."""
        from rlox.pbt import PBT

        pbt = PBT(
            algo="ppo",
            env="CartPole-v1",
            population_size=2,
            interval=256,
            n_iterations=1,
        )
        results = pbt.run()
        assert isinstance(results, dict)
        assert "best_fitness" in results

    def test_pbt_exploit_copies_best_to_worst(self):
        """Exploit step copies weights from best performer to worst."""
        from rlox.pbt import PBT

        pbt = PBT(
            algo="ppo",
            env="CartPole-v1",
            population_size=3,
            exploit_fraction=0.34,  # bottom 1 of 3 gets replaced
        )
        # Manually set fitness so we know the ranking
        pbt._fitnesses = [100.0, 50.0, 10.0]

        # Grab state dicts before exploit
        best_params = {
            k: v.clone()
            for k, v in pbt._agents[0].policy.state_dict().items()
        }

        pbt._exploit()

        # Worst agent (index 2) should now have best agent's weights
        worst_params = pbt._agents[2].policy.state_dict()
        for k in best_params:
            torch.testing.assert_close(best_params[k], worst_params[k])

    def test_pbt_explore_perturbs_hps(self):
        """Explore step perturbs hyperparameters of exploited agents."""
        from rlox.pbt import PBT

        pbt = PBT(
            algo="ppo",
            env="CartPole-v1",
            population_size=3,
            perturb_factor=0.5,
        )
        # Mark agent 2 as just-exploited so explore targets it
        pbt._exploited_indices = {2}

        lr_before = pbt._agents[2].optimizer.param_groups[0]["lr"]
        pbt._explore()
        lr_after = pbt._agents[2].optimizer.param_groups[0]["lr"]

        # LR should have changed (extremely unlikely to stay the same with 0.5 perturb)
        assert lr_before != lr_after

    def test_pbt_config_roundtrip(self):
        """PBTConfig serializes and deserializes correctly."""
        from rlox.config import PBTConfig

        cfg = PBTConfig(population_size=4, interval=5000)
        d = cfg.to_dict()
        cfg2 = PBTConfig.from_dict(d)
        assert cfg2.population_size == 4
        assert cfg2.interval == 5000


# ============================================================================
# TestTRPO
# ============================================================================


class TestTRPO:
    """Tests for Trust Region Policy Optimization."""

    def test_trpo_constructs(self):
        """TRPO can be instantiated."""
        from rlox.algorithms.trpo import TRPO

        trpo = TRPO(env_id="CartPole-v1")
        assert trpo is not None

    def test_trpo_satisfies_protocol(self):
        """TRPO satisfies the Algorithm protocol."""
        from rlox.algorithms.trpo import TRPO
        from rlox.protocols import Algorithm

        trpo = TRPO(env_id="CartPole-v1")
        assert isinstance(trpo, Algorithm)

    def test_trpo_train_returns_metrics(self):
        """Short TRPO training returns a metrics dict with mean_reward."""
        from rlox.algorithms.trpo import TRPO

        trpo = TRPO(env_id="CartPole-v1", n_envs=2, n_steps=64)
        metrics = trpo.train(total_timesteps=128)
        assert isinstance(metrics, dict)
        assert "mean_reward" in metrics

    def test_trpo_registered(self):
        """TRPO is accessible through the unified Trainer."""
        from rlox.trainer import Trainer

        trainer = Trainer("trpo", env="CartPole-v1", config={"n_envs": 2, "n_steps": 64})
        metrics = trainer.train(total_timesteps=128)
        assert "mean_reward" in metrics

    def test_trpo_kl_constraint(self):
        """After an update the KL divergence stays below max_kl."""
        from rlox.algorithms.trpo import TRPO

        max_kl = 0.01
        trpo = TRPO(env_id="CartPole-v1", n_envs=2, n_steps=64, max_kl=max_kl)
        metrics = trpo.train(total_timesteps=128)
        # The algorithm should report KL and it should be within constraint
        if "kl" in metrics:
            assert metrics["kl"] <= max_kl * 1.5  # small tolerance for numerical issues

    def test_trpo_conjugate_gradient(self):
        """CG solver finds a direction that approximately solves Ax = b."""
        from rlox.algorithms.trpo import _conjugate_gradient

        n = 10
        # Create a positive-definite matrix
        A_raw = torch.randn(n, n)
        A = A_raw @ A_raw.T + torch.eye(n) * 0.1
        b = torch.randn(n)

        def Ax_fn(x: torch.Tensor) -> torch.Tensor:
            return A @ x

        x = _conjugate_gradient(Ax_fn, b, cg_iters=20)
        # Check Ax approximately equals b
        residual = (A @ x - b).norm() / b.norm()
        assert residual < 0.05

    def test_trpo_config_roundtrip(self):
        """TRPOConfig serializes and deserializes correctly."""
        from rlox.config import TRPOConfig

        cfg = TRPOConfig(max_kl=0.02, cg_iters=15)
        d = cfg.to_dict()
        cfg2 = TRPOConfig.from_dict(d)
        assert cfg2.max_kl == 0.02
        assert cfg2.cg_iters == 15
