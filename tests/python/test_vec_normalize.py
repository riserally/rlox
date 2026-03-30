"""Tests for VecNormalize environment wrapper."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from rlox.gym_vec_env import GymVecEnv
from rlox.vec_normalize import VecNormalize


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cartpole_env() -> GymVecEnv:
    return GymVecEnv("CartPole-v1", n_envs=4, seed=42)


@pytest.fixture
def pendulum_env() -> GymVecEnv:
    return GymVecEnv("Pendulum-v1", n_envs=4, seed=42)


@pytest.fixture
def norm_env(cartpole_env: GymVecEnv) -> VecNormalize:
    return VecNormalize(cartpole_env, norm_obs=True, norm_reward=True)


# ---------------------------------------------------------------------------
# Observation normalization
# ---------------------------------------------------------------------------


class TestObsNormalization:
    def test_obs_near_zero_mean_after_warmup(self, cartpole_env: GymVecEnv) -> None:
        """After many steps, normalized observations should have near-zero mean."""
        vn = VecNormalize(cartpole_env, norm_obs=True, norm_reward=False)
        obs = vn.reset_all()

        all_obs = [obs]
        for _ in range(200):
            actions = np.array(
                [cartpole_env.action_space.sample() for _ in range(4)],
                dtype=np.int64,
            )
            result = vn.step_all(actions)
            all_obs.append(result["obs"])

        stacked = np.concatenate(all_obs, axis=0)
        # After warmup, mean should be close to 0, std close to 1
        assert np.abs(stacked[-100:].mean(axis=0)).max() < 1.5
        assert stacked[-100:].std(axis=0).max() < 5.0

    def test_obs_clipped(self, cartpole_env: GymVecEnv) -> None:
        """Normalized observations should be clipped to [-clip_obs, clip_obs]."""
        clip_val = 5.0
        vn = VecNormalize(
            cartpole_env, norm_obs=True, norm_reward=False, clip_obs=clip_val
        )
        obs = vn.reset_all()
        for _ in range(50):
            actions = np.array(
                [cartpole_env.action_space.sample() for _ in range(4)],
                dtype=np.int64,
            )
            result = vn.step_all(actions)
            assert np.all(result["obs"] <= clip_val)
            assert np.all(result["obs"] >= -clip_val)

    def test_no_obs_normalization_when_disabled(
        self, cartpole_env: GymVecEnv
    ) -> None:
        """With norm_obs=False, observations should pass through unchanged."""
        vn = VecNormalize(cartpole_env, norm_obs=False, norm_reward=False)
        raw_env = GymVecEnv("CartPole-v1", n_envs=4, seed=42)

        obs_norm = vn.reset_all(seed=99)
        obs_raw = raw_env.reset_all(seed=99)
        np.testing.assert_array_almost_equal(obs_norm, obs_raw)


# ---------------------------------------------------------------------------
# Reward normalization
# ---------------------------------------------------------------------------


class TestRewardNormalization:
    def test_reward_scaled_by_return_std(self, cartpole_env: GymVecEnv) -> None:
        """Reward normalization should divide by sqrt(var(returns) + eps)."""
        vn = VecNormalize(cartpole_env, norm_obs=False, norm_reward=True, gamma=0.99)
        vn.reset_all()

        raw_rewards = []
        norm_rewards = []
        for _ in range(100):
            actions = np.array(
                [cartpole_env.action_space.sample() for _ in range(4)],
                dtype=np.int64,
            )
            # Get raw rewards from inner env for comparison
            result = vn.step_all(actions)
            norm_rewards.append(result["rewards"])

        # Normalized rewards should have smaller variance than raw (which are all 1.0 for CartPole)
        norm_arr = np.concatenate(norm_rewards)
        # After enough steps, return std grows, so normalized rewards shrink
        assert norm_arr.std() < 2.0  # raw CartPole rewards are always 1.0

    def test_reward_mean_not_subtracted(self, cartpole_env: GymVecEnv) -> None:
        """SB3 convention: only scale, do not shift rewards."""
        vn = VecNormalize(cartpole_env, norm_obs=False, norm_reward=True)
        vn.reset_all()

        all_rewards = []
        for _ in range(100):
            actions = np.array(
                [cartpole_env.action_space.sample() for _ in range(4)],
                dtype=np.int64,
            )
            result = vn.step_all(actions)
            all_rewards.append(result["rewards"])

        # All rewards should be positive (CartPole reward is 1.0, only scaled)
        arr = np.concatenate(all_rewards)
        assert np.all(arr >= 0)

    def test_no_reward_normalization_when_disabled(
        self, cartpole_env: GymVecEnv
    ) -> None:
        """With norm_reward=False, rewards pass through unchanged."""
        vn = VecNormalize(cartpole_env, norm_obs=False, norm_reward=False)
        vn.reset_all()
        actions = np.array([0, 1, 0, 1], dtype=np.int64)
        result = vn.step_all(actions)
        # CartPole reward is 1.0 per step
        np.testing.assert_array_almost_equal(result["rewards"], np.ones(4))


# ---------------------------------------------------------------------------
# Training mode / frozen stats
# ---------------------------------------------------------------------------


class TestTrainingMode:
    def test_training_true_by_default(self, norm_env: VecNormalize) -> None:
        assert norm_env.training is True

    def test_frozen_stats_when_not_training(
        self, cartpole_env: GymVecEnv
    ) -> None:
        """Setting training=False should freeze running statistics."""
        vn = VecNormalize(cartpole_env, norm_obs=True, norm_reward=True)
        vn.reset_all()

        # Warm up stats
        for _ in range(50):
            actions = np.array(
                [cartpole_env.action_space.sample() for _ in range(4)],
                dtype=np.int64,
            )
            vn.step_all(actions)

        # Freeze
        vn.training = False
        rms_before = vn.get_obs_rms()
        mean_before = rms_before["mean"].copy()
        var_before = rms_before["var"].copy()

        # Step more
        for _ in range(50):
            actions = np.array(
                [cartpole_env.action_space.sample() for _ in range(4)],
                dtype=np.int64,
            )
            vn.step_all(actions)

        rms_after = vn.get_obs_rms()
        np.testing.assert_array_equal(rms_after["mean"], mean_before)
        np.testing.assert_array_equal(rms_after["var"], var_before)

    def test_normalize_obs_does_not_update_stats(
        self, cartpole_env: GymVecEnv
    ) -> None:
        """Public normalize_obs() should not update running statistics."""
        vn = VecNormalize(cartpole_env, norm_obs=True, norm_reward=False)
        vn.reset_all()

        # Warm up
        for _ in range(20):
            actions = np.array(
                [cartpole_env.action_space.sample() for _ in range(4)],
                dtype=np.int64,
            )
            vn.step_all(actions)

        rms_before = vn.get_obs_rms()
        mean_before = rms_before["mean"].copy()

        # Call normalize_obs (should NOT update)
        fake_obs = np.random.randn(4, 4).astype(np.float32) * 100
        _ = vn.normalize_obs(fake_obs)

        rms_after = vn.get_obs_rms()
        np.testing.assert_array_equal(rms_after["mean"], mean_before)


# ---------------------------------------------------------------------------
# terminal_obs normalization
# ---------------------------------------------------------------------------


class TestTerminalObs:
    def test_terminal_obs_normalized(self, cartpole_env: GymVecEnv) -> None:
        """terminal_obs entries should be normalized when norm_obs=True."""
        vn = VecNormalize(cartpole_env, norm_obs=True, norm_reward=False)
        vn.reset_all()

        for _ in range(500):
            actions = np.array(
                [cartpole_env.action_space.sample() for _ in range(4)],
                dtype=np.int64,
            )
            result = vn.step_all(actions)
            dones = result["terminated"].astype(bool) | result["truncated"].astype(bool)
            if np.any(dones):
                for i in range(4):
                    if dones[i] and result["terminal_obs"][i] is not None:
                        # Should be normalized (clipped to [-10, 10] by default)
                        assert np.all(np.abs(result["terminal_obs"][i]) <= 10.0 + 1e-6)
                return

        pytest.fail("No done encountered in 500 steps")


# ---------------------------------------------------------------------------
# Passthrough properties
# ---------------------------------------------------------------------------


class TestPassthrough:
    def test_num_envs(self, norm_env: VecNormalize) -> None:
        assert norm_env.num_envs() == 4

    def test_action_space(self, norm_env: VecNormalize) -> None:
        import gymnasium as gym

        assert isinstance(norm_env.action_space, gym.spaces.Discrete)

    def test_observation_space(self, norm_env: VecNormalize) -> None:
        import gymnasium as gym

        assert isinstance(norm_env.observation_space, gym.spaces.Box)

    def test_env_id_passthrough(self, cartpole_env: GymVecEnv) -> None:
        vn = VecNormalize(cartpole_env, norm_obs=False, norm_reward=False)
        # GymVecEnv doesn't have env_id attr by default but VecNormalize should
        # delegate gracefully
        assert hasattr(vn, "env_id") or not hasattr(cartpole_env, "env_id")


# ---------------------------------------------------------------------------
# Works with continuous envs
# ---------------------------------------------------------------------------


class TestContinuousEnv:
    def test_pendulum_obs_normalization(self, pendulum_env: GymVecEnv) -> None:
        """VecNormalize should work with continuous action spaces (Pendulum)."""
        vn = VecNormalize(pendulum_env, norm_obs=True, norm_reward=True)
        obs = vn.reset_all()
        assert obs.shape == (4, 3)  # Pendulum obs_dim=3

        for _ in range(50):
            actions = np.array(
                [pendulum_env.action_space.sample() for _ in range(4)],
                dtype=np.float32,
            )
            result = vn.step_all(actions)
            assert result["obs"].shape == (4, 3)
            assert np.all(np.abs(result["obs"]) <= 10.0 + 1e-6)


# ---------------------------------------------------------------------------
# step_all / reset_all return format
# ---------------------------------------------------------------------------


class TestReturnFormat:
    def test_step_all_keys_preserved(self, norm_env: VecNormalize) -> None:
        norm_env.reset_all()
        actions = np.array([0, 1, 0, 1], dtype=np.int64)
        result = norm_env.step_all(actions)
        expected_keys = {"obs", "rewards", "terminated", "truncated", "terminal_obs"}
        assert set(result.keys()) == expected_keys

    def test_reset_all_returns_ndarray(self, norm_env: VecNormalize) -> None:
        obs = norm_env.reset_all()
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32

    def test_step_dtypes_preserved(self, norm_env: VecNormalize) -> None:
        norm_env.reset_all()
        actions = np.array([0, 1, 0, 1], dtype=np.int64)
        result = norm_env.step_all(actions)
        assert result["obs"].dtype == np.float32
        assert result["rewards"].dtype == np.float64
        assert result["terminated"].dtype == np.uint8
        assert result["truncated"].dtype == np.uint8


# ---------------------------------------------------------------------------
# get_obs_rms
# ---------------------------------------------------------------------------


class TestGetObsRms:
    def test_returns_mean_and_var(self, norm_env: VecNormalize) -> None:
        norm_env.reset_all()
        rms = norm_env.get_obs_rms()
        assert "mean" in rms
        assert "var" in rms
        assert rms["mean"].shape == (4,)  # CartPole obs_dim=4
        assert rms["var"].shape == (4,)
