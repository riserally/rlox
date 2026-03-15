"""Tests for GymVecEnv wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from rlox.gym_vec_env import GymVecEnv


@pytest.fixture
def env() -> GymVecEnv:
    return GymVecEnv("CartPole-v1", n_envs=4, seed=42)


class TestGymVecEnv:
    def test_step_returns_correct_keys(self, env: GymVecEnv) -> None:
        env.reset_all()
        actions = np.array([0, 1, 0, 1], dtype=np.int64)
        result = env.step_all(actions)

        expected_keys = {"obs", "rewards", "terminated", "truncated", "terminal_obs"}
        assert set(result.keys()) == expected_keys

    def test_obs_shape(self, env: GymVecEnv) -> None:
        env.reset_all()
        actions = np.array([0, 1, 0, 1], dtype=np.int64)
        result = env.step_all(actions)

        assert result["obs"].shape == (4, 4)
        assert result["obs"].dtype == np.float64

    def test_auto_reset(self, env: GymVecEnv) -> None:
        """After enough random steps, at least one env should auto-reset."""
        obs = env.reset_all()
        any_done = False
        for _ in range(500):
            actions = np.array(
                [env.action_space.sample() for _ in range(4)], dtype=np.int64
            )
            result = env.step_all(actions)
            terminated = result["terminated"].astype(bool)
            truncated = result["truncated"].astype(bool)
            if np.any(terminated | truncated):
                any_done = True
                # After auto-reset, obs should still be valid (not NaN)
                assert not np.any(np.isnan(result["obs"]))
                break
        assert any_done, "Expected at least one done in 500 steps"

    def test_terminal_obs_present_on_done(self, env: GymVecEnv) -> None:
        """terminal_obs should contain the pre-reset observation for done envs."""
        env.reset_all()
        for _ in range(500):
            actions = np.array(
                [env.action_space.sample() for _ in range(4)], dtype=np.int64
            )
            result = env.step_all(actions)
            terminated = result["terminated"].astype(bool)
            truncated = result["truncated"].astype(bool)
            dones = terminated | truncated
            if np.any(dones):
                terminal_obs = result["terminal_obs"]
                for i in range(4):
                    if dones[i]:
                        assert terminal_obs[i] is not None
                        assert terminal_obs[i].shape == (4,)
                    else:
                        assert terminal_obs[i] is None
                return
        pytest.fail("No done encountered in 500 steps")

    def test_reset_returns_correct_shape(self, env: GymVecEnv) -> None:
        obs = env.reset_all()
        assert obs.shape == (4, 4)
        assert obs.dtype == np.float64

    def test_reset_with_seed(self) -> None:
        env1 = GymVecEnv("CartPole-v1", n_envs=2, seed=0)
        env2 = GymVecEnv("CartPole-v1", n_envs=2, seed=0)
        obs1 = env1.reset_all(seed=123)
        obs2 = env2.reset_all(seed=123)
        np.testing.assert_array_equal(obs1, obs2)

    def test_num_envs(self, env: GymVecEnv) -> None:
        assert env.num_envs() == 4

    def test_reward_dtype(self, env: GymVecEnv) -> None:
        env.reset_all()
        actions = np.array([0, 1, 0, 1], dtype=np.int64)
        result = env.step_all(actions)
        assert result["rewards"].dtype == np.float64

    def test_done_dtype(self, env: GymVecEnv) -> None:
        env.reset_all()
        actions = np.array([0, 1, 0, 1], dtype=np.int64)
        result = env.step_all(actions)
        assert result["terminated"].dtype == np.uint8
        assert result["truncated"].dtype == np.uint8
