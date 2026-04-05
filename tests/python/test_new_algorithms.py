"""Tests for Decision Transformer, QMIX, and Cal-QL algorithms.

Written FIRST (TDD red phase) -- implementations follow.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from rlox.protocols import Algorithm


# ---------------------------------------------------------------------------
# Decision Transformer
# ---------------------------------------------------------------------------


class TestDecisionTransformer:
    """Decision Transformer: RL via sequence modeling."""

    def test_dt_constructs(self):
        from rlox.algorithms.decision_transformer import DecisionTransformer

        dt = DecisionTransformer(env_id="CartPole-v1", seed=0)
        assert dt is not None
        assert dt.env_id == "CartPole-v1"

    def test_dt_satisfies_protocol(self):
        from rlox.algorithms.decision_transformer import DecisionTransformer

        dt = DecisionTransformer(env_id="CartPole-v1", seed=0)
        assert isinstance(dt, Algorithm)

    def test_dt_train_returns_metrics(self):
        from rlox.algorithms.decision_transformer import DecisionTransformer

        dt = DecisionTransformer(
            env_id="CartPole-v1",
            seed=0,
            batch_size=4,
            context_length=5,
        )
        metrics = dt.train(total_timesteps=200)
        assert isinstance(metrics, dict)
        assert "loss" in metrics

    def test_dt_registered(self):
        from rlox.trainer import Trainer

        trainer = Trainer("dt", env="CartPole-v1", config={"batch_size": 4, "context_length": 5})
        assert trainer.algo is not None

    def test_dt_save_load(self):
        from rlox.algorithms.decision_transformer import DecisionTransformer

        dt = DecisionTransformer(env_id="CartPole-v1", seed=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "dt_ckpt.pt")
            dt.save(path)
            dt2 = DecisionTransformer.from_checkpoint(path)
            assert dt2.env_id == "CartPole-v1"

    def test_dt_context_length(self):
        """Model handles sequences shorter than context_length."""
        from rlox.algorithms.decision_transformer import DecisionTransformer

        dt = DecisionTransformer(
            env_id="CartPole-v1",
            seed=0,
            context_length=20,
        )
        # Predict with a short sequence (length 3, less than context_length=20)
        obs_dim = dt._obs_dim
        short_states = torch.randn(1, 3, obs_dim)
        short_actions = torch.zeros(1, 3, dtype=torch.long)
        short_rtg = torch.ones(1, 3, 1)
        short_timesteps = torch.arange(3).unsqueeze(0)

        action = dt.predict(
            states=short_states,
            actions=short_actions,
            returns_to_go=short_rtg,
            timesteps=short_timesteps,
        )
        assert action is not None

    def test_dt_config_roundtrip(self):
        from rlox.config import DecisionTransformerConfig

        cfg = DecisionTransformerConfig(context_length=10, n_heads=2)
        d = cfg.to_dict()
        cfg2 = DecisionTransformerConfig.from_dict(d)
        assert cfg2.context_length == 10
        assert cfg2.n_heads == 2


# ---------------------------------------------------------------------------
# QMIX
# ---------------------------------------------------------------------------


class TestQMIX:
    """QMIX: value decomposition for cooperative multi-agent RL."""

    def test_qmix_constructs(self):
        from rlox.algorithms.qmix import QMIX

        qmix = QMIX(env_id="CartPole-v1", n_agents=3, seed=0)
        assert qmix is not None
        assert qmix.env_id == "CartPole-v1"

    def test_qmix_satisfies_protocol(self):
        from rlox.algorithms.qmix import QMIX

        qmix = QMIX(env_id="CartPole-v1", n_agents=3, seed=0)
        assert isinstance(qmix, Algorithm)

    def test_qmix_monotonic_mixing(self):
        """Q_tot must increase when any individual Q_i increases (monotonicity)."""
        from rlox.algorithms.qmix import QMIX

        qmix = QMIX(env_id="CartPole-v1", n_agents=3, seed=0)

        obs_dim = qmix._obs_dim
        n_agents = 3
        batch = 8

        # Create fake per-agent Q-values and global state
        agent_qs = torch.randn(batch, n_agents, requires_grad=True)
        global_state = torch.randn(batch, obs_dim * n_agents)

        q_tot = qmix.mixing_network(agent_qs, global_state)
        assert q_tot.shape == (batch, 1)

        # Check monotonicity: dQ_tot / dQ_i >= 0 for all i
        q_tot.sum().backward()
        assert agent_qs.grad is not None
        assert (agent_qs.grad >= 0).all(), (
            "Mixing network violates monotonicity constraint"
        )

    def test_qmix_registered(self):
        from rlox.trainer import Trainer

        trainer = Trainer(
            "qmix",
            env="CartPole-v1",
            config={"n_agents": 2, "batch_size": 4, "buffer_size": 500},
        )
        assert trainer.algo is not None

    def test_qmix_config_roundtrip(self):
        from rlox.config import QMIXConfig

        cfg = QMIXConfig(n_agents=5, hidden_dim=128)
        d = cfg.to_dict()
        cfg2 = QMIXConfig.from_dict(d)
        assert cfg2.n_agents == 5
        assert cfg2.hidden_dim == 128


# ---------------------------------------------------------------------------
# Cal-QL
# ---------------------------------------------------------------------------


class TestCalQL:
    """Cal-QL: CQL with calibrated conservative penalty."""

    def test_calql_constructs(self):
        from rlox.algorithms.calql import CalQL

        # Cal-QL requires a continuous action space (SAC backbone)
        agent = CalQL(env_id="Pendulum-v1", seed=0)
        assert agent is not None
        assert agent.env_id == "Pendulum-v1"

    def test_calql_rejects_discrete_env(self):
        """Cal-QL must raise ValueError for discrete action spaces."""
        from rlox.algorithms.calql import CalQL

        with pytest.raises(ValueError, match="continuous action space"):
            CalQL(env_id="CartPole-v1", seed=0)

    def test_calql_calibration_scales_penalty(self):
        """Calibration tau should scale the CQL penalty."""
        from rlox.algorithms.calql import CalQL

        agent_low = CalQL(env_id="Pendulum-v1", seed=0, calibration_tau=0.1)
        agent_high = CalQL(env_id="Pendulum-v1", seed=0, calibration_tau=0.9)

        # With higher calibration_tau, the penalty weight should be different
        # The calibration_tau is stored and used in loss computation
        assert agent_low._calibration_tau == 0.1
        assert agent_high._calibration_tau == 0.9

        # Verify that the calibrated penalty computation produces different
        # results for different tau values
        obs_dim = agent_low._obs_dim
        batch_size = 8
        fake_q_current = torch.randn(batch_size)
        fake_q_offline = torch.randn(batch_size) - 1.0  # Lower offline Q

        penalty_low = agent_low._compute_calibrated_penalty(
            fake_q_current, fake_q_offline
        )
        penalty_high = agent_high._compute_calibrated_penalty(
            fake_q_current, fake_q_offline
        )

        # Different tau should yield different penalty scaling
        assert not torch.allclose(penalty_low, penalty_high), (
            "Calibration tau must affect the penalty"
        )

    def test_calql_registered(self):
        from rlox.trainer import Trainer

        # Cal-QL requires a continuous env
        trainer = Trainer(
            "calql",
            env="Pendulum-v1",
            config={"batch_size": 4, "buffer_size": 500},
        )
        assert trainer.algo is not None

    def test_calql_config_roundtrip(self):
        from rlox.config import CalQLConfig

        cfg = CalQLConfig(calibration_tau=0.7, cql_alpha=2.0)
        d = cfg.to_dict()
        cfg2 = CalQLConfig.from_dict(d)
        assert cfg2.calibration_tau == 0.7
        assert cfg2.cql_alpha == 2.0
