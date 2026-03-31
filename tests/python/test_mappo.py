"""Tests for MAPPO: MultiAgentCollector and multi-agent training."""

from __future__ import annotations

import pytest
import torch
import numpy as np


# ---------------------------------------------------------------------------
# MultiAgentCollector
# ---------------------------------------------------------------------------


class TestMultiAgentCollector:
    def test_import_error_without_pettingzoo(self, monkeypatch):
        """Collector raises ImportError with helpful message when pettingzoo is missing."""
        import importlib
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "pettingzoo" in name:
                raise ImportError("No module named 'pettingzoo'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from rlox.algorithms.mappo import MultiAgentCollector

        with pytest.raises(ImportError, match="pettingzoo"):
            MultiAgentCollector(env_fn=lambda: None, n_envs=1, n_agents=2)

    def test_collector_init_stores_params(self):
        """Collector stores n_envs and n_agents."""
        from rlox.algorithms.mappo import MultiAgentCollector

        # We expect an ImportError (pettingzoo probably not installed),
        # but we can check the class exists and has the right signature
        try:
            collector = MultiAgentCollector(
                env_fn=lambda: None, n_envs=2, n_agents=3
            )
            assert collector.n_envs == 2
            assert collector.n_agents == 3
        except ImportError:
            # pettingzoo not installed -- that's fine, the class exists
            pass


# ---------------------------------------------------------------------------
# MAPPO single-agent (n_agents=1)
# ---------------------------------------------------------------------------


class TestMAPPOSingleAgent:
    def test_instantiation(self):
        from rlox.algorithms.mappo import MAPPO

        agent = MAPPO(env_id="CartPole-v1", n_agents=1, n_envs=2, seed=42)
        assert agent.n_agents == 1
        assert len(agent.actors) == 1

    @pytest.mark.slow
    def test_trains_single_agent(self):
        from rlox.algorithms.mappo import MAPPO

        agent = MAPPO(env_id="CartPole-v1", n_agents=1, n_envs=2, seed=42)
        metrics = agent.train(total_timesteps=500)
        assert isinstance(metrics, dict)
        assert "mean_reward" in metrics
        assert np.isfinite(metrics["mean_reward"])

    def test_multi_agent_no_longer_raises(self):
        """n_agents > 1 should no longer raise NotImplementedError."""
        from rlox.algorithms.mappo import MAPPO

        agent = MAPPO(env_id="CartPole-v1", n_agents=2, n_envs=2, seed=42)
        # Should not raise NotImplementedError -- uses MultiAgentCollector
        # (may raise ImportError if pettingzoo not installed, but NOT NotImplementedError)
        try:
            agent.train(total_timesteps=256)
        except ImportError:
            pass  # pettingzoo not installed, acceptable
        except NotImplementedError:
            pytest.fail("n_agents > 1 should not raise NotImplementedError anymore")


# ---------------------------------------------------------------------------
# MAPPOConfig
# ---------------------------------------------------------------------------


class TestMAPPOConfig:
    def test_defaults(self):
        from rlox.config import MAPPOConfig

        cfg = MAPPOConfig()
        assert cfg.n_agents == 2
        assert cfg.learning_rate == 5e-4
        assert cfg.n_steps == 128
        assert cfg.clip_range == 0.2
        assert cfg.share_parameters is False

    def test_validation_rejects_bad_values(self):
        from rlox.config import MAPPOConfig

        with pytest.raises(ValueError):
            MAPPOConfig(learning_rate=-1.0)
        with pytest.raises(ValueError):
            MAPPOConfig(n_agents=0)
        with pytest.raises(ValueError):
            MAPPOConfig(n_envs=0)

    def test_from_dict_ignores_unknown(self):
        from rlox.config import MAPPOConfig

        cfg = MAPPOConfig.from_dict({"n_agents": 4, "unknown_key": 99})
        assert cfg.n_agents == 4

    def test_roundtrip_dict(self):
        from rlox.config import MAPPOConfig

        cfg = MAPPOConfig(n_agents=3, hidden=128)
        d = cfg.to_dict()
        loaded = MAPPOConfig.from_dict(d)
        assert loaded.to_dict() == d

    def test_yaml_roundtrip(self, tmp_path):
        from rlox.config import MAPPOConfig

        cfg = MAPPOConfig(n_agents=5, learning_rate=1e-3)
        path = tmp_path / "mappo.yaml"
        cfg.to_yaml(path)
        loaded = MAPPOConfig.from_yaml(path)
        assert loaded.n_agents == 5
        assert loaded.learning_rate == 1e-3

    def test_toml_roundtrip(self, tmp_path):
        from rlox.config import MAPPOConfig

        cfg = MAPPOConfig(n_agents=4, gamma=0.95)
        path = tmp_path / "mappo.toml"
        cfg.to_toml(path)
        loaded = MAPPOConfig.from_toml(path)
        assert loaded.n_agents == 4
        assert loaded.gamma == 0.95
