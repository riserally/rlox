"""Integration tests for rlox Python bindings."""

import numpy as np
import pytest


def test_cartpole_create_and_reset():
    from rlox import CartPole

    env = CartPole(seed=42)
    obs = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)
    assert obs.dtype == np.float32


def test_cartpole_step():
    from rlox import CartPole

    env = CartPole(seed=42)
    result = env.step(1)
    assert "obs" in result
    assert "reward" in result
    assert "terminated" in result
    assert "truncated" in result
    assert isinstance(result["obs"], np.ndarray)
    assert result["obs"].shape == (4,)
    assert result["reward"] == 1.0


def test_cartpole_deterministic():
    from rlox import CartPole

    def run(seed):
        env = CartPole(seed=seed)
        observations = [env.reset(seed=seed).tolist()]
        for _ in range(50):
            result = env.step(1)
            observations.append(result["obs"].tolist())
            if result["terminated"] or result["truncated"]:
                break
        return observations

    run1 = run(123)
    run2 = run(123)
    assert run1 == run2


def test_vecenv_step_all():
    from rlox import VecEnv

    n = 4
    venv = VecEnv(n=n, seed=42)
    actions = [1, 0, 1, 0]
    result = venv.step_all(actions)

    assert result["obs"].shape == (4, 4)
    assert result["rewards"].shape == (4,)
    assert result["terminated"].shape == (4,)
    assert result["truncated"].shape == (4,)


def test_vecenv_reset_all():
    from rlox import VecEnv

    venv = VecEnv(n=4, seed=42)
    obs = venv.reset_all(seed=99)
    assert obs.shape == (4, 4)
    assert obs.dtype == np.float32

    # Deterministic
    venv2 = VecEnv(n=4, seed=42)
    obs2 = venv2.reset_all(seed=99)
    np.testing.assert_array_equal(obs, obs2)


def test_vecenv_many_steps_auto_reset():
    """VecEnv should auto-reset done envs without errors."""
    from rlox import VecEnv

    venv = VecEnv(n=4, seed=42)
    for _ in range(200):
        actions = [1, 1, 1, 1]
        result = venv.step_all(actions)
        assert result["obs"].shape == (4, 4)


def test_gym_env_wrapper():
    """GymEnv wraps a gymnasium CartPole and steps it."""
    try:
        import gymnasium as gym
    except ImportError:
        pytest.skip("gymnasium not installed")

    from rlox import GymEnv

    gym_env = gym.make("CartPole-v1")
    wrapped = GymEnv(gym_env)

    obs = wrapped.reset(seed=42)
    assert obs is not None

    result = wrapped.step(1)
    assert "obs" in result
    assert "reward" in result
    assert "terminated" in result
    assert "truncated" in result
