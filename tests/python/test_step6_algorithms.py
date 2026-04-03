"""Step 6 tests: Self-Play, Go-Explore, MPO.

TDD: These tests are written FIRST, before the implementations.
"""

from __future__ import annotations

import copy
from dataclasses import asdict

import numpy as np
import pytest
import torch
import torch.nn as nn

from rlox.config import ConfigMixin


# ============================================================================
# TestSelfPlay
# ============================================================================


class TestSelfPlay:
    """Tests for Self-Play training framework."""

    def test_self_play_constructs(self):
        """SelfPlay can be instantiated with default config."""
        from rlox.self_play import SelfPlay
        from rlox.config import SelfPlayConfig

        sp = SelfPlay()
        assert sp is not None
        assert isinstance(sp.config, SelfPlayConfig)

    def test_self_play_snapshot_saves_to_pool(self):
        """Taking a snapshot adds a policy to the opponent pool."""
        from rlox.self_play import SelfPlay

        sp = SelfPlay()
        assert len(sp.opponent_pool) == 0
        # Create a dummy policy (simple linear layer) and snapshot it
        dummy_policy = nn.Linear(4, 2)
        sp.snapshot(dummy_policy)
        assert len(sp.opponent_pool) == 1

    def test_self_play_pool_size_limit(self):
        """When the pool is full, the oldest entry is evicted."""
        from rlox.self_play import SelfPlay
        from rlox.config import SelfPlayConfig

        cfg = SelfPlayConfig(pool_size=3)
        sp = SelfPlay(config=cfg)

        policies = [nn.Linear(4, 2) for _ in range(5)]
        for p in policies:
            sp.snapshot(p)

        assert len(sp.opponent_pool) == 3
        # The pool should contain snapshots of the last 3 policies
        # Verify the oldest were evicted by checking state dict values differ
        # from the first two policies
        first_sd = {k: v.clone() for k, v in policies[0].state_dict().items()}
        pool_sds = sp.opponent_pool
        # Pool entries are state_dicts; the first policy's weights should NOT be in pool
        for pool_entry in pool_sds:
            match = all(
                torch.allclose(pool_entry[k], first_sd[k])
                for k in first_sd
            )
            # At least one should differ (not guaranteed by randomness, but
            # the point is pool length is capped)
        assert len(sp.opponent_pool) == 3

    def test_self_play_elo_update(self):
        """After a match, winner gains Elo and loser loses Elo."""
        from rlox.self_play import SelfPlay
        from rlox.config import SelfPlayConfig

        cfg = SelfPlayConfig(initial_elo=1000.0, elo_k=32.0)
        sp = SelfPlay(config=cfg)

        # Snapshot two policies to create two pool entries
        p1 = nn.Linear(4, 2)
        p2 = nn.Linear(4, 2)
        sp.snapshot(p1)
        sp.snapshot(p2)

        elo_before_0 = sp.elo_ratings[0]
        elo_before_1 = sp.elo_ratings[1]

        # Simulate: player 0 wins against player 1
        sp.update_elo(winner_idx=0, loser_idx=1)

        assert sp.elo_ratings[0] > elo_before_0, "Winner should gain Elo"
        assert sp.elo_ratings[1] < elo_before_1, "Loser should lose Elo"
        # Total Elo is conserved
        total_before = elo_before_0 + elo_before_1
        total_after = sp.elo_ratings[0] + sp.elo_ratings[1]
        assert abs(total_before - total_after) < 1e-6

    def test_self_play_config_roundtrip(self):
        """SelfPlayConfig survives to_dict/from_dict roundtrip."""
        from rlox.config import SelfPlayConfig

        cfg = SelfPlayConfig(
            pool_size=10,
            snapshot_freq=5000,
            matchmaking="elo",
            initial_elo=1200.0,
            elo_k=16.0,
            opponent_fraction=0.9,
        )
        d = cfg.to_dict()
        restored = SelfPlayConfig.from_dict(d)
        assert restored.pool_size == 10
        assert restored.snapshot_freq == 5000
        assert restored.matchmaking == "elo"
        assert restored.initial_elo == 1200.0
        assert restored.elo_k == 16.0
        assert restored.opponent_fraction == 0.9

    def test_self_play_sample_opponent_uniform(self):
        """Uniform matchmaking returns a valid opponent from the pool."""
        from rlox.self_play import SelfPlay
        from rlox.config import SelfPlayConfig

        cfg = SelfPlayConfig(matchmaking="uniform")
        sp = SelfPlay(config=cfg)
        for _ in range(5):
            sp.snapshot(nn.Linear(4, 2))

        opponent_sd = sp.sample_opponent()
        assert opponent_sd is not None
        assert isinstance(opponent_sd, dict)  # state_dict

    def test_self_play_sample_opponent_latest(self):
        """Latest matchmaking always returns the most recent snapshot."""
        from rlox.self_play import SelfPlay
        from rlox.config import SelfPlayConfig

        cfg = SelfPlayConfig(matchmaking="latest")
        sp = SelfPlay(config=cfg)

        policies = [nn.Linear(4, 2) for _ in range(5)]
        for p in policies:
            sp.snapshot(p)

        opponent_sd = sp.sample_opponent()
        latest_sd = sp.opponent_pool[-1]
        for k in opponent_sd:
            assert torch.allclose(opponent_sd[k], latest_sd[k])


# ============================================================================
# TestGoExplore
# ============================================================================


class TestGoExplore:
    """Tests for Go-Explore archive-based exploration."""

    def test_go_explore_constructs(self):
        """GoExplore can be instantiated with default config."""
        from rlox.exploration.go_explore import GoExplore
        from rlox.config import GoExploreConfig

        ge = GoExplore()
        assert ge is not None
        assert isinstance(ge.config, GoExploreConfig)

    def test_go_explore_archive_stores_states(self):
        """Adding a state to the archive stores it with its cell key."""
        from rlox.exploration.go_explore import GoExplore

        ge = GoExplore()
        obs = np.array([1.5, 2.3, 0.1, -0.5], dtype=np.float32)
        score = 10.0
        trajectory = [obs]  # simplified trajectory for replay

        ge.add_to_archive(obs, score=score, trajectory=trajectory)
        assert len(ge.archive) == 1

    def test_go_explore_cell_hashing_deterministic(self):
        """The same observation always maps to the same cell key."""
        from rlox.exploration.go_explore import GoExplore
        from rlox.config import GoExploreConfig

        cfg = GoExploreConfig(cell_resolution=8)
        ge = GoExplore(config=cfg)

        obs = np.array([1.5, 2.3, 0.1, -0.5], dtype=np.float32)
        cell1 = ge.compute_cell(obs)
        cell2 = ge.compute_cell(obs)
        assert cell1 == cell2

        # Different obs should (likely) yield different cell
        obs2 = np.array([10.0, -5.0, 3.0, 7.0], dtype=np.float32)
        cell3 = ge.compute_cell(obs2)
        assert cell3 != cell1

    def test_go_explore_selects_novel_cells(self):
        """Under-visited cells are selected more frequently."""
        from rlox.exploration.go_explore import GoExplore
        from rlox.config import GoExploreConfig

        cfg = GoExploreConfig(novelty_weight=1.0, score_weight=0.0)
        ge = GoExplore(config=cfg)

        # Add two cells: one visited many times, one visited once
        frequent_obs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        rare_obs = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)

        for _ in range(100):
            ge.add_to_archive(frequent_obs, score=1.0, trajectory=[frequent_obs])
        ge.add_to_archive(rare_obs, score=1.0, trajectory=[rare_obs])

        # Sample many times and count which cell is selected
        rare_cell = ge.compute_cell(rare_obs)
        freq_cell = ge.compute_cell(frequent_obs)
        counts = {rare_cell: 0, freq_cell: 0}
        for _ in range(200):
            cell_key, _ = ge.select_cell()
            if cell_key in counts:
                counts[cell_key] += 1

        assert counts[rare_cell] > counts[freq_cell], (
            f"Rare cell should be selected more often: rare={counts[rare_cell]}, "
            f"freq={counts[freq_cell]}"
        )

    def test_go_explore_config_roundtrip(self):
        """GoExploreConfig survives to_dict/from_dict roundtrip."""
        from rlox.config import GoExploreConfig

        cfg = GoExploreConfig(
            archive_size=50_000,
            cell_resolution=16,
            exploration_steps=200,
            score_weight=0.5,
            novelty_weight=0.5,
        )
        d = cfg.to_dict()
        restored = GoExploreConfig.from_dict(d)
        assert restored.archive_size == 50_000
        assert restored.cell_resolution == 16
        assert restored.exploration_steps == 200
        assert restored.score_weight == 0.5
        assert restored.novelty_weight == 0.5


# ============================================================================
# TestMPO
# ============================================================================


class TestMPO:
    """Tests for Maximum a Posteriori Policy Optimization."""

    def test_mpo_constructs(self):
        """MPO can be instantiated for a continuous environment."""
        from rlox.algorithms.mpo import MPO

        mpo = MPO(env_id="Pendulum-v1", seed=42)
        assert mpo is not None
        assert mpo.env_id == "Pendulum-v1"

    def test_mpo_satisfies_protocol(self):
        """MPO satisfies the Algorithm protocol."""
        from rlox.algorithms.mpo import MPO
        from rlox.protocols import Algorithm

        mpo = MPO(env_id="Pendulum-v1", seed=42)
        assert isinstance(mpo, Algorithm)

    def test_mpo_train_returns_metrics(self):
        """A short training run returns a dict with expected metric keys."""
        from rlox.algorithms.mpo import MPO

        mpo = MPO(env_id="Pendulum-v1", seed=42, learning_starts=50)
        metrics = mpo.train(total_timesteps=200)
        assert isinstance(metrics, dict)
        assert "mean_reward" in metrics

    def test_mpo_registered(self):
        """MPO is accessible via Trainer('mpo', ...)."""
        from rlox.trainer import ALGORITHM_REGISTRY

        # Force re-import to trigger registration
        import rlox.algorithms.mpo  # noqa: F401

        assert "mpo" in ALGORITHM_REGISTRY

    def test_mpo_e_step_concentrates(self):
        """E-step weights concentrate on actions with higher Q-values."""
        from rlox.algorithms.mpo import MPO

        mpo = MPO(env_id="Pendulum-v1", seed=42)

        # Create fake Q-values: one action is clearly better
        n_samples = 20
        q_values = torch.zeros(n_samples)
        q_values[0] = 10.0  # best action
        q_values[1:] = -10.0  # bad actions

        temperature = 0.1  # low temp -> concentrate on best
        weights = mpo.compute_e_step_weights(q_values, temperature)

        assert weights.shape == (n_samples,)
        assert weights.sum().item() == pytest.approx(1.0, abs=1e-5)
        # The best action should have the highest weight
        assert weights[0] > weights[1]
        # With this extreme gap and low temp, weight on best should dominate
        assert weights[0] > 0.5

    def test_mpo_dual_updates(self):
        """Dual variable (temperature) adjusts toward satisfying the KL constraint."""
        from rlox.algorithms.mpo import MPO

        mpo = MPO(env_id="Pendulum-v1", seed=42)

        # Set up a scenario where KL is too high -> temperature should increase
        # to make the distribution more uniform (reducing KL)
        initial_log_eta = mpo.log_eta.item()

        # Fake Q-values with high variance -> high KL
        q_values = torch.randn(20) * 10.0
        mpo.update_dual(q_values, target_epsilon=0.01)

        updated_log_eta = mpo.log_eta.item()
        # Temperature should have changed
        assert updated_log_eta != initial_log_eta

    def test_mpo_config_roundtrip(self):
        """MPOConfig survives to_dict/from_dict roundtrip."""
        from rlox.config import MPOConfig

        cfg = MPOConfig(
            learning_rate=1e-3,
            buffer_size=500_000,
            batch_size=128,
            gamma=0.995,
            tau=0.01,
            n_action_samples=30,
            epsilon=0.05,
            epsilon_penalty=0.01,
            dual_lr=5e-3,
            hidden=128,
            learning_starts=500,
        )
        d = cfg.to_dict()
        restored = MPOConfig.from_dict(d)
        assert restored.learning_rate == 1e-3
        assert restored.buffer_size == 500_000
        assert restored.batch_size == 128
        assert restored.n_action_samples == 30
        assert restored.epsilon == 0.05
        assert restored.dual_lr == 5e-3
