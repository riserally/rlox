"""Tests for Decision Tree Policies (RWDTP / RCDTP).

Based on: Koirala & Fleming, "Solving Offline Reinforcement Learning with
Decision Tree Regression," arXiv:2401.11630, 2024.
"""

from __future__ import annotations

import numpy as np
import pytest

import rlox
from rlox.config import DTPConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(n=500, obs_dim=4, act_dim=2, ep_len=50):
    """Create a synthetic offline dataset with known episode boundaries."""
    rng = np.random.RandomState(0)
    obs = rng.randn(n, obs_dim).astype(np.float32)
    next_obs = obs + 0.01 * rng.randn(n, obs_dim).astype(np.float32)
    actions = rng.randn(n, act_dim).astype(np.float32)
    rewards = rng.randn(n).astype(np.float32)
    terminated = np.zeros(n, dtype=np.uint8)
    truncated = np.zeros(n, dtype=np.uint8)
    for i in range(ep_len - 1, n, ep_len):
        terminated[i] = 1
    buf = rlox.OfflineDatasetBuffer(
        obs.ravel(), next_obs.ravel(), actions.ravel(),
        rewards, terminated, truncated,
    )
    return buf, obs_dim, act_dim


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestDTPConfig:
    def test_defaults(self):
        cfg = DTPConfig()
        assert cfg.method == "rwdtp"
        assert cfg.gamma == 1.0
        assert cfg.return_power == 1.0
        assert cfg.n_trees == 500
        assert cfg.max_depth == 6

    def test_from_dict(self):
        cfg = DTPConfig.from_dict({"method": "rcdtp", "n_trees": 100})
        assert cfg.method == "rcdtp"
        assert cfg.n_trees == 100

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            DTPConfig(method="invalid")

    def test_invalid_n_trees(self):
        with pytest.raises(ValueError, match="n_trees"):
            DTPConfig(n_trees=0)

    def test_merge(self):
        cfg = DTPConfig()
        cfg2 = cfg.merge({"n_trees": 200, "method": "rcdtp"})
        assert cfg2.n_trees == 200
        assert cfg2.method == "rcdtp"
        # Original unchanged
        assert cfg.n_trees == 500


# ---------------------------------------------------------------------------
# Return computation tests
# ---------------------------------------------------------------------------


class TestReturnComputation:
    def test_undiscounted_returns(self):
        """Return-to-go with gamma=1 should be cumulative sum from end."""
        from rlox.algorithms.dtp import _compute_episode_returns

        rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        terminated = np.array([0, 0, 1], dtype=np.uint8)
        returns = _compute_episode_returns(rewards, terminated, gamma=1.0)
        np.testing.assert_allclose(returns, [6.0, 5.0, 3.0], atol=1e-5)

    def test_discounted_returns(self):
        from rlox.algorithms.dtp import _compute_episode_returns

        rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        terminated = np.array([0, 0, 1], dtype=np.uint8)
        gamma = 0.9
        # R_0 = 1 + 0.9 + 0.81 = 2.71, R_1 = 1 + 0.9 = 1.9, R_2 = 1.0
        returns = _compute_episode_returns(rewards, terminated, gamma=gamma)
        np.testing.assert_allclose(returns, [2.71, 1.9, 1.0], atol=1e-5)

    def test_multi_episode(self):
        from rlox.algorithms.dtp import _compute_episode_returns

        rewards = np.array([1.0, 2.0, 3.0, 10.0, 20.0], dtype=np.float32)
        terminated = np.array([0, 0, 1, 0, 1], dtype=np.uint8)
        returns = _compute_episode_returns(rewards, terminated, gamma=1.0)
        np.testing.assert_allclose(returns, [6.0, 5.0, 3.0, 30.0, 20.0], atol=1e-5)


# ---------------------------------------------------------------------------
# Data preparation tests
# ---------------------------------------------------------------------------


class TestDataPreparation:
    def test_rwdtp_weights_normalized(self):
        """RWDTP weights should be in [0, 1]."""
        from rlox.algorithms.dtp import _normalize_returns

        returns = np.array([1.0, 5.0, 3.0, 10.0, 2.0])
        normed = _normalize_returns(returns)
        assert normed.min() >= 0.0
        assert normed.max() <= 1.0
        # Max return -> weight 1.0, min -> weight 0.0
        assert np.isclose(normed[np.argmax(returns)], 1.0)
        assert np.isclose(normed[np.argmin(returns)], 0.0)

    def test_rwdtp_weights_with_power(self):
        from rlox.algorithms.dtp import _normalize_returns

        returns = np.array([0.0, 5.0, 10.0])
        normed_p1 = _normalize_returns(returns, power=1.0)
        normed_p2 = _normalize_returns(returns, power=2.0)
        # power=2 should amplify the difference between high and low returns
        np.testing.assert_allclose(normed_p1, [0.0, 0.5, 1.0])
        np.testing.assert_allclose(normed_p2, [0.0, 0.25, 1.0])

    def test_rwdtp_constant_returns(self):
        """If all returns are equal, weights should be uniform (all 1)."""
        from rlox.algorithms.dtp import _normalize_returns

        returns = np.array([5.0, 5.0, 5.0])
        normed = _normalize_returns(returns, power=1.0)
        np.testing.assert_allclose(normed, [1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# Training smoke tests
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset_and_dims():
    return _make_dataset(n=200, obs_dim=4, act_dim=2, ep_len=50)


class TestRWDTP:
    def test_train_smoke(self, dataset_and_dims):
        pytest.importorskip("xgboost")
        from rlox.algorithms.dtp import DecisionTreePolicy

        buf, obs_dim, act_dim = dataset_and_dims
        dtp = DecisionTreePolicy(
            dataset=buf,
            obs_dim=obs_dim,
            act_dim=act_dim,
            method="rwdtp",
            n_trees=10,
            max_depth=3,
        )
        metrics = dtp.train(n_gradient_steps=1)
        assert "mse" in metrics
        assert metrics["mse"] >= 0.0

    def test_predict_shape(self, dataset_and_dims):
        pytest.importorskip("xgboost")
        from rlox.algorithms.dtp import DecisionTreePolicy

        buf, obs_dim, act_dim = dataset_and_dims
        dtp = DecisionTreePolicy(
            dataset=buf, obs_dim=obs_dim, act_dim=act_dim,
            method="rwdtp", n_trees=10, max_depth=3,
        )
        dtp.train(n_gradient_steps=1)
        obs = np.random.randn(obs_dim).astype(np.float32)
        action = dtp.predict(obs)
        assert action.shape == (act_dim,)

    def test_predict_batch(self, dataset_and_dims):
        pytest.importorskip("xgboost")
        from rlox.algorithms.dtp import DecisionTreePolicy

        buf, obs_dim, act_dim = dataset_and_dims
        dtp = DecisionTreePolicy(
            dataset=buf, obs_dim=obs_dim, act_dim=act_dim,
            method="rwdtp", n_trees=10, max_depth=3,
        )
        dtp.train(n_gradient_steps=1)
        obs = np.random.randn(8, obs_dim).astype(np.float32)
        actions = dtp.predict(obs)
        assert actions.shape == (8, act_dim)


class TestRCDTP:
    def test_train_smoke(self, dataset_and_dims):
        pytest.importorskip("xgboost")
        from rlox.algorithms.dtp import DecisionTreePolicy

        buf, obs_dim, act_dim = dataset_and_dims
        dtp = DecisionTreePolicy(
            dataset=buf, obs_dim=obs_dim, act_dim=act_dim,
            method="rcdtp", n_trees=10, max_depth=3,
        )
        metrics = dtp.train(n_gradient_steps=1)
        assert "mse" in metrics

    def test_predict_with_rtg(self, dataset_and_dims):
        pytest.importorskip("xgboost")
        from rlox.algorithms.dtp import DecisionTreePolicy

        buf, obs_dim, act_dim = dataset_and_dims
        dtp = DecisionTreePolicy(
            dataset=buf, obs_dim=obs_dim, act_dim=act_dim,
            method="rcdtp", n_trees=10, max_depth=3,
        )
        dtp.train(n_gradient_steps=1)
        obs = np.random.randn(obs_dim).astype(np.float32)
        action = dtp.predict(obs, rtg=100.0, timestep=0)
        assert action.shape == (act_dim,)


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_roundtrip(self, dataset_and_dims, tmp_path):
        pytest.importorskip("xgboost")
        from rlox.algorithms.dtp import DecisionTreePolicy

        buf, obs_dim, act_dim = dataset_and_dims
        dtp = DecisionTreePolicy(
            dataset=buf, obs_dim=obs_dim, act_dim=act_dim,
            method="rwdtp", n_trees=10, max_depth=3,
        )
        dtp.train(n_gradient_steps=1)

        path = str(tmp_path / "dtp_checkpoint")
        dtp.save(path)

        dtp2 = DecisionTreePolicy.from_checkpoint(path)
        obs = np.random.randn(obs_dim).astype(np.float32)
        np.testing.assert_allclose(
            dtp.predict(obs), dtp2.predict(obs), atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Single action dim
# ---------------------------------------------------------------------------


class TestSingleActionDim:
    def test_act_dim_1(self):
        pytest.importorskip("xgboost")
        from rlox.algorithms.dtp import DecisionTreePolicy

        buf, obs_dim, _ = _make_dataset(n=200, obs_dim=4, act_dim=1, ep_len=50)
        dtp = DecisionTreePolicy(
            dataset=buf, obs_dim=obs_dim, act_dim=1,
            method="rwdtp", n_trees=10, max_depth=3,
        )
        dtp.train(n_gradient_steps=1)
        obs = np.random.randn(obs_dim).astype(np.float32)
        action = dtp.predict(obs)
        assert action.shape == (1,)


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_rwdtp_registered(self):
        from rlox.trainer import ALGORITHM_REGISTRY
        assert "rwdtp" in ALGORITHM_REGISTRY

    def test_rcdtp_registered(self):
        from rlox.trainer import ALGORITHM_REGISTRY
        assert "rcdtp" in ALGORITHM_REGISTRY
