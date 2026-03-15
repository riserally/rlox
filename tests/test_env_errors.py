"""Tests for VecEnv error handling on unknown env_ids."""

import pytest
import rlox


def test_unknown_env_id_raises_value_error():
    """VecEnv must raise ValueError for unrecognized env IDs."""
    with pytest.raises(ValueError, match="Unknown env_id 'FakeEnv-v99'"):
        rlox.VecEnv(4, env_id="FakeEnv-v99")


def test_cartpole_v1_works():
    """Explicit CartPole-v1 env_id must still construct successfully."""
    vec_env = rlox.VecEnv(4, env_id="CartPole-v1")
    assert vec_env.num_envs() == 4


def test_default_env_id_works():
    """Omitting env_id must default to CartPole-v1 (backward compat)."""
    vec_env = rlox.VecEnv(4)
    assert vec_env.num_envs() == 4
