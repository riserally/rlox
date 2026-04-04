"""Tests for the plugin ecosystem and model zoo (TDD -- written first)."""

from __future__ import annotations

import ast
import difflib

import pytest
from dataclasses import fields
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# TestPluginRegistry
# ---------------------------------------------------------------------------


class TestPluginRegistry:
    """Tests for environment, buffer, and reward registries."""

    def setup_method(self) -> None:
        """Clear registries before each test to ensure isolation."""
        from rlox.plugins import ENV_REGISTRY, BUFFER_REGISTRY, REWARD_REGISTRY

        ENV_REGISTRY.clear()
        BUFFER_REGISTRY.clear()
        REWARD_REGISTRY.clear()

    def test_register_env(self) -> None:
        from rlox.plugins import ENV_REGISTRY, register_env

        @register_env("custom-env-v0")
        class CustomEnv:
            def step(self, action): pass
            def reset(self): pass

        assert "custom-env-v0" in ENV_REGISTRY
        assert ENV_REGISTRY["custom-env-v0"] is CustomEnv

    def test_register_env_preserves_class(self) -> None:
        """Decorator must return the original class unchanged."""
        from rlox.plugins import register_env

        @register_env("my-env")
        class MyEnv:
            sentinel = 42
            def step(self, action): pass
            def reset(self): pass

        assert MyEnv.sentinel == 42

    def test_register_buffer(self) -> None:
        from rlox.plugins import BUFFER_REGISTRY, register_buffer

        @register_buffer("custom-buffer")
        class CustomBuffer:
            def push(self, *args): pass
            def sample(self, n): pass
            def __len__(self): return 0

        assert "custom-buffer" in BUFFER_REGISTRY
        assert BUFFER_REGISTRY["custom-buffer"] is CustomBuffer

    def test_register_reward(self) -> None:
        from rlox.plugins import REWARD_REGISTRY, register_reward

        @register_reward("curiosity")
        class CuriosityReward:
            def shape(self, rewards, obs, next_obs, dones): return rewards

        assert "curiosity" in REWARD_REGISTRY
        assert REWARD_REGISTRY["curiosity"] is CuriosityReward

    def test_list_registered(self) -> None:
        from rlox.plugins import (
            register_env,
            register_buffer,
            register_reward,
            list_registered,
        )

        @register_env("env-a")
        class A:
            def step(self, action): pass
            def reset(self): pass

        @register_buffer("buf-b")
        class B:
            def push(self, *args): pass
            def sample(self, n): pass
            def __len__(self): return 0

        @register_reward("rew-c")
        class C:
            def shape(self, rewards, obs, next_obs, dones): return rewards

        result = list_registered()
        assert "environments" in result
        assert "buffers" in result
        assert "rewards" in result
        assert "algorithms" in result
        assert "env-a" in result["environments"]
        assert "buf-b" in result["buffers"]
        assert "rew-c" in result["rewards"]

    def test_algorithm_registry_included(self) -> None:
        """list_registered must include algorithms from trainer.ALGORITHM_REGISTRY."""
        from rlox.plugins import list_registered
        from rlox.trainer import ALGORITHM_REGISTRY

        result = list_registered()
        # The trainer module registers builtins on import -- verify they appear
        for name in ALGORITHM_REGISTRY:
            assert name in result["algorithms"]

    def test_duplicate_registration_overwrites(self) -> None:
        from rlox.plugins import ENV_REGISTRY, register_env

        @register_env("dup")
        class First:
            def step(self, action): pass
            def reset(self): pass

        @register_env("dup")
        class Second:
            def step(self, action): pass
            def reset(self): pass

        assert ENV_REGISTRY["dup"] is Second

    def test_list_registered_sorted(self) -> None:
        """Registry lists must be sorted alphabetically."""
        from rlox.plugins import register_env, list_registered

        @register_env("z-env")
        class Z:
            def step(self, action): pass
            def reset(self): pass

        @register_env("a-env")
        class A:
            def step(self, action): pass
            def reset(self): pass

        result = list_registered()
        assert result["environments"] == ["a-env", "z-env"]

    def test_discover_plugins_loads_entry_points(self) -> None:
        """discover_plugins should load every entry point in the given group."""
        from rlox.plugins import discover_plugins

        mock_ep = MagicMock()
        mock_ep.load = MagicMock()

        with patch("importlib.metadata.entry_points", return_value=[mock_ep, mock_ep]):
            discover_plugins("rlox_plugins")

        assert mock_ep.load.call_count == 2


# ---------------------------------------------------------------------------
# TestModelZoo
# ---------------------------------------------------------------------------


class TestModelZoo:
    """Tests for the pretrained model registry."""

    def setup_method(self) -> None:
        from rlox.zoo import ModelZoo

        ModelZoo.PRETRAINED.clear()

    def test_register_pretrained(self) -> None:
        from rlox.zoo import ModelZoo

        ModelZoo.register(
            name="ppo-cartpole-v1",
            algo="ppo",
            env="CartPole-v1",
            checkpoint_url="/tmp/model.pt",
            mean_return=475.0,
        )

        assert "ppo-cartpole-v1" in ModelZoo.PRETRAINED
        info = ModelZoo.PRETRAINED["ppo-cartpole-v1"]
        assert info["algo"] == "ppo"
        assert info["env"] == "CartPole-v1"
        assert info["mean_return"] == 475.0

    def test_register_with_extra_metadata(self) -> None:
        from rlox.zoo import ModelZoo

        ModelZoo.register(
            name="sac-halfcheetah",
            algo="sac",
            env="HalfCheetah-v4",
            checkpoint_url="/tmp/sac.pt",
            mean_return=10_000.0,
            training_steps=1_000_000,
            author="test",
        )

        info = ModelZoo.PRETRAINED["sac-halfcheetah"]
        assert info["training_steps"] == 1_000_000
        assert info["author"] == "test"

    def test_list_models(self) -> None:
        from rlox.zoo import ModelZoo

        ModelZoo.register("m1", "ppo", "CartPole-v1", "/tmp/m1.pt", 400.0)
        ModelZoo.register("m2", "sac", "Pendulum-v1", "/tmp/m2.pt", -200.0)

        models = ModelZoo.list()
        assert len(models) == 2
        names = {m["name"] for m in models}
        assert names == {"m1", "m2"}

    def test_list_models_empty(self) -> None:
        from rlox.zoo import ModelZoo

        assert ModelZoo.list() == []

    def test_search_by_env(self) -> None:
        from rlox.zoo import ModelZoo

        ModelZoo.register("a", "ppo", "CartPole-v1", "/a", 400.0)
        ModelZoo.register("b", "sac", "Pendulum-v1", "/b", -200.0)
        ModelZoo.register("c", "dqn", "CartPole-v1", "/c", 450.0)

        results = ModelZoo.search(env="CartPole-v1")
        assert len(results) == 2
        assert all(r["env"] == "CartPole-v1" for r in results)

    def test_search_by_algo(self) -> None:
        from rlox.zoo import ModelZoo

        ModelZoo.register("a", "ppo", "CartPole-v1", "/a", 400.0)
        ModelZoo.register("b", "ppo", "Pendulum-v1", "/b", -200.0)
        ModelZoo.register("c", "sac", "CartPole-v1", "/c", 450.0)

        results = ModelZoo.search(algo="ppo")
        assert len(results) == 2
        assert all(r["algo"] == "ppo" for r in results)

    def test_search_by_env_and_algo(self) -> None:
        from rlox.zoo import ModelZoo

        ModelZoo.register("a", "ppo", "CartPole-v1", "/a", 400.0)
        ModelZoo.register("b", "ppo", "Pendulum-v1", "/b", -200.0)
        ModelZoo.register("c", "sac", "CartPole-v1", "/c", 450.0)

        results = ModelZoo.search(env="CartPole-v1", algo="ppo")
        assert len(results) == 1
        assert results[0]["name"] == "a"

    def test_search_no_match(self) -> None:
        from rlox.zoo import ModelZoo

        ModelZoo.register("a", "ppo", "CartPole-v1", "/a", 400.0)
        assert ModelZoo.search(env="Nonexistent-v0") == []

    def test_load_unknown_model_raises(self) -> None:
        from rlox.zoo import ModelZoo

        with pytest.raises(KeyError):
            ModelZoo.load("nonexistent")


# ---------------------------------------------------------------------------
# TestModelCard
# ---------------------------------------------------------------------------


class TestModelCard:
    """Tests for ModelCard dataclass."""

    def _make_card(self, **overrides) -> "ModelCard":
        from rlox.zoo import ModelCard

        defaults = {
            "algorithm": "PPO",
            "environment": "CartPole-v1",
            "mean_return": 475.0,
            "std_return": 12.3,
            "total_timesteps": 100_000,
            "training_time_s": 42.5,
            "hyperparameters": {"lr": 3e-4, "gamma": 0.99},
            "framework_version": "1.0.0",
        }
        defaults.update(overrides)
        return ModelCard(**defaults)

    def test_model_card_creation(self) -> None:
        card = self._make_card()
        assert card.algorithm == "PPO"
        assert card.environment == "CartPole-v1"
        assert card.mean_return == 475.0

    def test_model_card_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        card = self._make_card()
        with pytest.raises(FrozenInstanceError):
            card.algorithm = "SAC"  # type: ignore[misc]

    def test_model_card_to_dict_roundtrip(self) -> None:
        from rlox.zoo import ModelCard

        card = self._make_card()
        d = card.to_dict()

        assert isinstance(d, dict)
        assert d["algorithm"] == "PPO"
        assert d["mean_return"] == 475.0

        # Roundtrip
        card2 = ModelCard(**d)
        assert card2 == card

    def test_model_card_to_markdown(self) -> None:
        card = self._make_card()
        md = card.to_markdown()

        assert isinstance(md, str)
        assert "PPO" in md
        assert "CartPole-v1" in md
        assert "475.0" in md
        assert "100000" in md or "100,000" in md or "100_000" in md

    def test_model_card_to_markdown_contains_hyperparams(self) -> None:
        card = self._make_card()
        md = card.to_markdown()
        assert "lr" in md
        assert "gamma" in md

    def test_model_card_from_trainer(self) -> None:
        from rlox.zoo import ModelCard

        # Build a mock trainer with enough structure
        mock_algo = MagicMock()
        mock_algo.__class__.__name__ = "PPO"

        mock_trainer = MagicMock()
        mock_trainer.algo = mock_algo
        mock_trainer.env = "CartPole-v1"

        metrics = {
            "mean_return": 475.0,
            "std_return": 12.3,
            "total_timesteps": 100_000,
            "training_time_s": 42.5,
        }

        card = ModelCard.from_trainer(mock_trainer, metrics)
        assert card.algorithm == "PPO"
        assert card.environment == "CartPole-v1"
        assert card.mean_return == 475.0
        assert card.total_timesteps == 100_000

    def test_model_card_from_trainer_with_config(self) -> None:
        """from_trainer should extract hyperparameters from algo config if available."""
        from rlox.zoo import ModelCard

        mock_config = MagicMock()
        mock_config.to_dict.return_value = {"lr": 1e-3, "batch_size": 64}

        mock_algo = MagicMock()
        mock_algo.__class__.__name__ = "SAC"
        mock_algo.config = mock_config

        mock_trainer = MagicMock()
        mock_trainer.algo = mock_algo
        mock_trainer.env = "Pendulum-v1"

        metrics = {
            "mean_return": -150.0,
            "std_return": 25.0,
            "total_timesteps": 50_000,
            "training_time_s": 30.0,
        }

        card = ModelCard.from_trainer(mock_trainer, metrics)
        assert card.hyperparameters == {"lr": 1e-3, "batch_size": 64}


# ---------------------------------------------------------------------------
# TestPluginValidation -- Part 4: registration validates interface
# ---------------------------------------------------------------------------


class TestPluginValidation:
    """Decorators reject classes missing required methods."""

    def setup_method(self) -> None:
        from rlox.plugins import ENV_REGISTRY, BUFFER_REGISTRY, REWARD_REGISTRY

        ENV_REGISTRY.clear()
        BUFFER_REGISTRY.clear()
        REWARD_REGISTRY.clear()

    def test_env_validation_rejects_missing_step(self) -> None:
        from rlox.plugins import register_env

        with pytest.raises(TypeError, match="step"):

            @register_env("bad-env-v0")
            class BadEnv:
                def reset(self):
                    pass

    def test_env_validation_rejects_missing_reset(self) -> None:
        from rlox.plugins import register_env

        with pytest.raises(TypeError, match="reset"):

            @register_env("bad-env-v1")
            class BadEnv:
                def step(self, action):
                    pass

    def test_env_validation_accepts_valid_env(self) -> None:
        from rlox.plugins import ENV_REGISTRY, register_env

        @register_env("good-env-v0")
        class GoodEnv:
            def step(self, action):
                pass

            def reset(self):
                pass

        assert "good-env-v0" in ENV_REGISTRY

    def test_buffer_validation_rejects_missing_push(self) -> None:
        from rlox.plugins import register_buffer

        with pytest.raises(TypeError, match="push"):

            @register_buffer("bad-buf")
            class BadBuf:
                def sample(self, n):
                    pass

                def __len__(self):
                    return 0

    def test_buffer_validation_rejects_missing_sample(self) -> None:
        from rlox.plugins import register_buffer

        with pytest.raises(TypeError, match="sample"):

            @register_buffer("bad-buf")
            class BadBuf:
                def push(self, *args):
                    pass

                def __len__(self):
                    return 0

    def test_buffer_validation_rejects_missing_len(self) -> None:
        from rlox.plugins import register_buffer

        with pytest.raises(TypeError, match="__len__"):

            @register_buffer("bad-buf")
            class BadBuf:
                def push(self, *args):
                    pass

                def sample(self, n):
                    pass

    def test_buffer_validation_accepts_valid_buffer(self) -> None:
        from rlox.plugins import BUFFER_REGISTRY, register_buffer

        @register_buffer("good-buf")
        class GoodBuf:
            def push(self, *args):
                pass

            def sample(self, n):
                pass

            def __len__(self):
                return 0

        assert "good-buf" in BUFFER_REGISTRY

    def test_reward_validation_rejects_bad_class(self) -> None:
        from rlox.plugins import register_reward

        with pytest.raises(TypeError, match="shape.*__call__"):

            @register_reward("bad-rew")
            class BadRew:
                pass

    def test_reward_validation_accepts_shape_method(self) -> None:
        from rlox.plugins import REWARD_REGISTRY, register_reward

        @register_reward("good-rew")
        class GoodRew:
            def shape(self, rewards, obs, next_obs, dones):
                return rewards

        assert "good-rew" in REWARD_REGISTRY

    def test_reward_validation_accepts_callable(self) -> None:
        from rlox.plugins import REWARD_REGISTRY, register_reward

        @register_reward("call-rew")
        class CallRew:
            def __call__(self, rewards, obs, next_obs, dones):
                return rewards

        assert "call-rew" in REWARD_REGISTRY


# ---------------------------------------------------------------------------
# TestRegistryHelpers -- Part 2 & 3: get_buffer, get_reward_shaper
# ---------------------------------------------------------------------------


class TestRegistryHelpers:
    """Test helper functions for buffer and reward creation from registries."""

    def setup_method(self) -> None:
        from rlox.plugins import BUFFER_REGISTRY, REWARD_REGISTRY

        BUFFER_REGISTRY.clear()
        REWARD_REGISTRY.clear()

    def test_get_buffer_creates_from_registry(self) -> None:
        from rlox.plugins import BUFFER_REGISTRY, register_buffer, get_buffer

        @register_buffer("test-buf")
        class TestBuf:
            def __init__(self, capacity, obs_dim, act_dim):
                self.capacity = capacity
                self.obs_dim = obs_dim
                self.act_dim = act_dim

            def push(self, *args):
                pass

            def sample(self, n):
                pass

            def __len__(self):
                return 0

        buf = get_buffer("test-buf", capacity=1000, obs_dim=4, act_dim=2)
        assert isinstance(buf, TestBuf)
        assert buf.capacity == 1000
        assert buf.obs_dim == 4

    def test_get_buffer_creates_builtin_replay(self) -> None:
        from rlox.plugins import get_buffer

        buf = get_buffer("replay", capacity=100, obs_dim=4, act_dim=1)
        assert len(buf) == 0  # empty buffer

    def test_get_buffer_unknown_raises(self) -> None:
        from rlox.plugins import get_buffer

        with pytest.raises(ValueError, match="Unknown buffer"):
            get_buffer("nonexistent", capacity=100, obs_dim=4, act_dim=1)

    def test_get_reward_shaper_creates_from_registry(self) -> None:
        from rlox.plugins import REWARD_REGISTRY, register_reward, get_reward_shaper

        @register_reward("test-rew")
        class TestRew:
            def __init__(self, scale=1.0):
                self.scale = scale

            def shape(self, rewards, obs, next_obs, dones):
                return rewards * self.scale

        shaper = get_reward_shaper("test-rew", scale=2.0)
        assert isinstance(shaper, TestRew)
        assert shaper.scale == 2.0

    def test_get_reward_shaper_unknown_raises(self) -> None:
        from rlox.plugins import get_reward_shaper

        with pytest.raises(ValueError, match="Unknown reward shaper"):
            get_reward_shaper("nonexistent")


# ---------------------------------------------------------------------------
# TestSuggestSimilar -- Part 8: fuzzy matching
# ---------------------------------------------------------------------------


class TestSuggestSimilar:
    """Test 'Did you mean?' suggestions for typos."""

    def test_suggest_similar_on_close_match(self) -> None:
        from rlox.plugins import _suggest_similar

        registry = {"ppo": object, "sac": object, "dqn": object}
        suggestion = _suggest_similar("pppo", registry)
        assert "ppo" in suggestion

    def test_suggest_similar_no_match(self) -> None:
        from rlox.plugins import _suggest_similar

        registry = {"ppo": object, "sac": object}
        suggestion = _suggest_similar("zzzzzzz", registry)
        assert suggestion == ""

    def test_suggest_similar_empty_registry(self) -> None:
        from rlox.plugins import _suggest_similar

        assert _suggest_similar("ppo", {}) == ""

    def test_suggest_similar_used_in_get_buffer_error(self) -> None:
        from rlox.plugins import get_buffer, BUFFER_REGISTRY

        # Register a buffer so we have something to suggest
        BUFFER_REGISTRY["prioritized"] = type("P", (), {})

        with pytest.raises(ValueError, match="prioritized"):
            get_buffer("prioitized", capacity=100, obs_dim=4, act_dim=1)

    def test_suggest_similar_used_in_get_reward_error(self) -> None:
        from rlox.plugins import get_reward_shaper, REWARD_REGISTRY

        REWARD_REGISTRY["potential"] = type("P", (), {})

        with pytest.raises(ValueError, match="potential"):
            get_reward_shaper("potntial")


# ---------------------------------------------------------------------------
# TestResolveEnv -- Part 1: resolve_env_id registers with gymnasium
# ---------------------------------------------------------------------------


class TestResolveEnv:
    """Test resolve_env_id integration."""

    def setup_method(self) -> None:
        from rlox.plugins import ENV_REGISTRY

        ENV_REGISTRY.clear()

    def test_resolve_env_registers_with_gymnasium(self) -> None:
        import gymnasium as gym
        from rlox.plugins import ENV_REGISTRY
        from rlox.trainer import resolve_env_id

        # Create a minimal gymnasium env
        class _TestEnv(gym.Env):
            def __init__(self, **kwargs):
                super().__init__()
                self.observation_space = gym.spaces.Box(-1, 1, shape=(2,))
                self.action_space = gym.spaces.Discrete(2)

            def step(self, action):
                return self.observation_space.sample(), 0.0, False, False, {}

            def reset(self, **kwargs):
                return self.observation_space.sample(), {}

        ENV_REGISTRY["test-env-v99"] = _TestEnv

        result = resolve_env_id("test-env-v99")
        assert result == "test-env-v99"
        # Should now be in gymnasium registry
        assert "test-env-v99" in gym.envs.registry

        # Clean up
        del gym.envs.registry["test-env-v99"]

    def test_resolve_env_passthrough_for_standard_envs(self) -> None:
        from rlox.trainer import resolve_env_id

        result = resolve_env_id("CartPole-v1")
        assert result == "CartPole-v1"

    def test_resolve_env_idempotent(self) -> None:
        """Calling resolve_env_id twice does not fail."""
        import gymnasium as gym
        from rlox.plugins import ENV_REGISTRY
        from rlox.trainer import resolve_env_id

        class _TestEnv2(gym.Env):
            def __init__(self, **kwargs):
                super().__init__()
                self.observation_space = gym.spaces.Box(-1, 1, shape=(2,))
                self.action_space = gym.spaces.Discrete(2)

            def step(self, action):
                return self.observation_space.sample(), 0.0, False, False, {}

            def reset(self, **kwargs):
                return self.observation_space.sample(), {}

        ENV_REGISTRY["test-env-v100"] = _TestEnv2
        resolve_env_id("test-env-v100")
        resolve_env_id("test-env-v100")  # Should not raise

        del gym.envs.registry["test-env-v100"]
