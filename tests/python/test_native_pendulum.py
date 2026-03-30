"""Tests for native Pendulum environment support.

The Rust-side Pendulum implementation may not be available yet.
All tests skip gracefully if the native env cannot be created.
"""

from __future__ import annotations

import numpy as np
import pytest

import rlox
from rlox.collectors import _NATIVE_ENV_IDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_create_pendulum(n: int = 4, seed: int = 0) -> rlox.VecEnv:
    """Attempt to create a native Pendulum VecEnv; skip test if unavailable."""
    try:
        env = rlox.VecEnv(n=n, seed=seed, env_id="Pendulum-v1")
    except (ValueError, RuntimeError) as exc:
        pytest.skip(f"Native Pendulum not available: {exc}")
    return env


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestNativeEnvRegistry:
    """Verify that Pendulum variants are listed as native env IDs."""

    def test_pendulum_v1_in_native_ids(self) -> None:
        assert "Pendulum-v1" in _NATIVE_ENV_IDS

    def test_pendulum_bare_in_native_ids(self) -> None:
        assert "Pendulum" in _NATIVE_ENV_IDS


# ---------------------------------------------------------------------------
# Environment creation & basic API
# ---------------------------------------------------------------------------

class TestPendulumCreation:
    """Tests that require the Rust Pendulum build."""

    def test_create_vec_env(self) -> None:
        env = _try_create_pendulum(n=4)
        assert env is not None

    def test_action_space_is_dict(self) -> None:
        env = _try_create_pendulum(n=2)
        action_space = getattr(env, "action_space", None)
        assert isinstance(action_space, dict), (
            f"Expected dict action_space, got {type(action_space)}"
        )

    def test_action_space_type_continuous(self) -> None:
        env = _try_create_pendulum(n=2)
        action_space = env.action_space
        assert action_space.get("type") != "discrete", (
            "Pendulum should have a continuous action space"
        )

    def test_reset_all_obs_shape(self) -> None:
        n_envs = 4
        env = _try_create_pendulum(n=n_envs)
        obs = env.reset_all()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (n_envs, 3), f"Expected (4, 3), got {obs.shape}"

    def test_step_all_with_continuous_actions(self) -> None:
        n_envs = 4
        env = _try_create_pendulum(n=n_envs)
        env.reset_all()

        # Pendulum action space: 1-dim continuous in [-2, 2]
        actions = np.random.default_rng(0).uniform(-2.0, 2.0, size=(n_envs, 1)).astype(np.float32)
        result = env.step_all(actions)

        assert "obs" in result
        assert "rewards" in result
        assert "terminated" in result
        assert "truncated" in result
        assert result["obs"].shape == (n_envs, 3)

    def test_multi_step_no_crash(self) -> None:
        """Run 50 random steps to verify stability."""
        n_envs = 4
        env = _try_create_pendulum(n=n_envs)
        env.reset_all()

        rng = np.random.default_rng(42)
        for _ in range(50):
            actions = rng.uniform(-2.0, 2.0, size=(n_envs, 1)).astype(np.float32)
            result = env.step_all(actions)
            assert result["obs"].shape == (n_envs, 3)


# ---------------------------------------------------------------------------
# Collector integration
# ---------------------------------------------------------------------------

class TestPendulumCollector:
    """Verify RolloutCollector works with Pendulum action-space detection."""

    def test_collector_detects_continuous(self) -> None:
        """Collector should set _is_discrete=False for Pendulum."""
        env = _try_create_pendulum(n=2)
        collector = rlox.RolloutCollector(
            env_id="Pendulum-v1",
            n_envs=2,
            seed=0,
            env=env,
        )
        assert collector._is_discrete is False, (
            "Pendulum should be detected as continuous"
        )
