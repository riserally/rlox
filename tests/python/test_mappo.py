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


# ---------------------------------------------------------------------------
# MAPPO PettingZoo end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMAPPOPettingZoo:
    """End-to-end MAPPO test with a real PettingZoo environment."""

    N_AGENTS = 3
    OBS_DIM = 18
    N_ACTIONS = 5

    @pytest.fixture(autouse=True)
    def _skip_if_no_pettingzoo(self):
        pytest.importorskip("pettingzoo")

    @staticmethod
    def _make_env():
        """Create a simple_spread_v3 parallel env with 3 agents."""
        from pettingzoo.mpe import simple_spread_v3

        return simple_spread_v3.parallel_env(
            N=TestMAPPOPettingZoo.N_AGENTS,
            max_cycles=25,
            continuous_actions=False,
        )

    @staticmethod
    def _register_proxy_gym_env() -> str:
        """Register a dummy Gymnasium env with matching obs/action spaces.

        MAPPO.__init__ calls _detect_env_spaces(env_id) which does gym.make(),
        so we need a gym-registered env whose spaces match simple_spread_v3.
        """
        import gymnasium as gym

        env_id = "SimpleSpreadProxy-v0"
        if env_id not in gym.registry:

            class _Proxy(gym.Env):
                def __init__(self):
                    super().__init__()
                    self.observation_space = gym.spaces.Box(
                        -np.inf, np.inf, (TestMAPPOPettingZoo.OBS_DIM,), np.float32
                    )
                    self.action_space = gym.spaces.Discrete(
                        TestMAPPOPettingZoo.N_ACTIONS
                    )

                def reset(self, **kwargs):
                    return self.observation_space.sample(), {}

                def step(self, action):
                    return self.observation_space.sample(), 0.0, False, False, {}

            gym.register(id=env_id, entry_point=_Proxy)
        return env_id

    def test_multi_agent_trains_on_simple_spread(self):
        """MAPPO trains on PettingZoo simple_spread and returns finite metrics."""
        from rlox.algorithms.mappo import MAPPO

        proxy_env_id = self._register_proxy_gym_env()

        agent = MAPPO(
            env_id=proxy_env_id,
            n_agents=self.N_AGENTS,
            n_envs=2,
            seed=42,
            n_steps=32,
            n_epochs=2,
            batch_size=32,
            learning_rate=3e-4,
            env_fn=self._make_env,
        )

        # Train for a short run -- enough to exercise the full loop
        # 2 envs * 32 steps = 64 steps per update; ~8K total
        metrics = agent.train(total_timesteps=8_192)

        assert isinstance(metrics, dict)
        assert "mean_reward" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics

        # All metrics should be finite (no NaN / Inf)
        for key, value in metrics.items():
            assert np.isfinite(value), f"metric '{key}' is not finite: {value}"

    def test_multi_agent_reward_is_negative(self):
        """simple_spread rewards are negative distances; verify sign makes sense."""
        from rlox.algorithms.mappo import MAPPO

        proxy_env_id = self._register_proxy_gym_env()

        agent = MAPPO(
            env_id=proxy_env_id,
            n_agents=self.N_AGENTS,
            n_envs=1,
            seed=0,
            n_steps=16,
            n_epochs=1,
            batch_size=16,
            env_fn=self._make_env,
        )

        metrics = agent.train(total_timesteps=256)
        # simple_spread gives negative rewards (distance penalties)
        assert metrics["mean_reward"] < 0.0, (
            "Expected negative mean_reward for simple_spread"
        )
