"""Tests for the plugin ecosystem and model zoo (TDD -- written first)."""

from __future__ import annotations

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
            pass

        assert "custom-env-v0" in ENV_REGISTRY
        assert ENV_REGISTRY["custom-env-v0"] is CustomEnv

    def test_register_env_preserves_class(self) -> None:
        """Decorator must return the original class unchanged."""
        from rlox.plugins import register_env

        @register_env("my-env")
        class MyEnv:
            sentinel = 42

        assert MyEnv.sentinel == 42

    def test_register_buffer(self) -> None:
        from rlox.plugins import BUFFER_REGISTRY, register_buffer

        @register_buffer("custom-buffer")
        class CustomBuffer:
            pass

        assert "custom-buffer" in BUFFER_REGISTRY
        assert BUFFER_REGISTRY["custom-buffer"] is CustomBuffer

    def test_register_reward(self) -> None:
        from rlox.plugins import REWARD_REGISTRY, register_reward

        @register_reward("curiosity")
        class CuriosityReward:
            pass

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
            pass

        @register_buffer("buf-b")
        class B:
            pass

        @register_reward("rew-c")
        class C:
            pass

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
            pass

        @register_env("dup")
        class Second:
            pass

        assert ENV_REGISTRY["dup"] is Second

    def test_list_registered_sorted(self) -> None:
        """Registry lists must be sorted alphabetically."""
        from rlox.plugins import register_env, list_registered

        @register_env("z-env")
        class Z:
            pass

        @register_env("a-env")
        class A:
            pass

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
