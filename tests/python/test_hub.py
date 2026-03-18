"""Tests for rlox.hub -- HuggingFace Hub integration."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from rlox.hub import create_model_card, _extract_info, _require_huggingface_hub


# ---------------------------------------------------------------------------
# create_model_card
# ---------------------------------------------------------------------------


class TestCreateModelCard:
    def test_has_expected_sections(self) -> None:
        card = create_model_card("PPO", "CartPole-v1")
        assert "# PPO on CartPole-v1" in card
        assert "## Algorithm" in card
        assert "## Usage" in card
        assert "library_name: rlox" in card

    def test_includes_metrics(self) -> None:
        metrics = {"mean_reward": 475.0, "policy_loss": 0.012}
        card = create_model_card("PPO", "CartPole-v1", metrics=metrics)
        assert "## Metrics" in card
        assert "mean_reward" in card
        assert "475.0000" in card
        assert "policy_loss" in card
        assert "0.0120" in card

    def test_no_metrics_section_when_empty(self) -> None:
        card = create_model_card("SAC", "Pendulum-v1")
        assert "## Metrics" not in card

    def test_tags_include_algo_and_env(self) -> None:
        card = create_model_card("DQN", "CartPole-v1")
        assert "- dqn" in card
        assert "- CartPole-v1" in card
        assert "- reinforcement-learning" in card

    def test_custom_library_name(self) -> None:
        card = create_model_card("PPO", "CartPole-v1", library_name="custom-lib")
        assert "library_name: custom-lib" in card


# ---------------------------------------------------------------------------
# push_to_hub -- requires huggingface_hub
# ---------------------------------------------------------------------------


class TestPushToHubDependency:
    def test_push_to_hub_requires_huggingface_hub(self) -> None:
        """Graceful ImportError when huggingface_hub is not installed."""
        with patch.dict(sys.modules, {"huggingface_hub": None}):
            with pytest.raises(ImportError, match="huggingface_hub"):
                _require_huggingface_hub()


# ---------------------------------------------------------------------------
# _extract_info
# ---------------------------------------------------------------------------


class TestExtractInfo:
    def test_from_trainer_like_object(self) -> None:
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {"lr": 3e-4}

        mock_algo = MagicMock()
        mock_algo.env_id = "CartPole-v1"
        mock_algo.config = mock_config
        type(mock_algo).__name__ = "PPO"

        mock_trainer = MagicMock()
        mock_trainer.algo = mock_algo
        mock_trainer.env = "CartPole-v1"

        info = _extract_info(mock_trainer)
        assert info["algo_name"] == "PPO"
        assert info["env_id"] == "CartPole-v1"
        assert info["config"] == {"lr": 3e-4}

    def test_from_algo_directly(self) -> None:
        mock_algo = MagicMock(spec=[])
        mock_algo.env_id = "Pendulum-v1"
        type(mock_algo).__name__ = "SAC"

        info = _extract_info(mock_algo)
        assert info["algo_name"] == "SAC"
        assert info["env_id"] == "Pendulum-v1"
        assert info["config"] == {}
