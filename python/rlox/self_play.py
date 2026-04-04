"""Self-play training for competitive environments.

Maintains a pool of historical policy snapshots and trains the current
agent against sampled opponents. Supports uniform, latest, and Elo-based
matchmaking strategies.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from rlox.config import SelfPlayConfig


class SelfPlay:
    """Self-play training framework for competitive environments.

    Parameters
    ----------
    config : SelfPlayConfig, optional
        Configuration. Uses defaults if not provided.
    """

    def __init__(self, config: SelfPlayConfig | None = None) -> None:
        self.config = config or SelfPlayConfig()
        self._opponent_pool: list[dict[str, torch.Tensor]] = []
        self._elo_ratings: list[float] = []

    @property
    def opponent_pool(self) -> list[dict[str, torch.Tensor]]:
        """List of stored policy state dicts."""
        return self._opponent_pool

    @property
    def elo_ratings(self) -> list[float]:
        """Elo ratings for each pool entry."""
        return self._elo_ratings

    def snapshot(self, policy: nn.Module) -> None:
        """Save a snapshot of the current policy to the opponent pool.

        If the pool is at capacity, the oldest entry is evicted.

        Parameters
        ----------
        policy : nn.Module
            The policy network to snapshot.
        """
        sd = {k: v.clone().detach() for k, v in policy.state_dict().items()}

        if len(self._opponent_pool) >= self.config.pool_size:
            self._opponent_pool.pop(0)
            self._elo_ratings.pop(0)

        self._opponent_pool.append(sd)
        self._elo_ratings.append(self.config.initial_elo)

    def sample_opponent(self) -> dict[str, torch.Tensor]:
        """Sample an opponent from the pool using the configured strategy.

        Returns
        -------
        state_dict : dict
            State dict of the sampled opponent policy.

        Raises
        ------
        RuntimeError
            If the pool is empty.
        """
        if not self._opponent_pool:
            raise RuntimeError("Cannot sample from an empty opponent pool")

        matchmaking = self.config.matchmaking

        if matchmaking == "latest":
            return self._opponent_pool[-1]

        if matchmaking == "elo":
            return self._sample_elo_weighted()

        # Default: uniform
        idx = random.randrange(len(self._opponent_pool))
        return self._opponent_pool[idx]

    def _sample_elo_weighted(self) -> dict[str, torch.Tensor]:
        """Sample opponent weighted by Elo rating (higher = more likely)."""
        import numpy as np

        ratings = np.array(self._elo_ratings, dtype=np.float64)
        # Softmax-style weighting
        ratings = ratings - ratings.max()
        weights = np.exp(ratings / max(self.config.elo_k, 1.0))
        probs = weights / weights.sum()
        idx = np.random.choice(len(self._opponent_pool), p=probs)
        return self._opponent_pool[idx]

    def update_elo(self, winner_idx: int, loser_idx: int) -> None:
        """Update Elo ratings after a match.

        Uses the standard Elo formula with K-factor from config.

        Parameters
        ----------
        winner_idx : int
            Index of the winning pool entry.
        loser_idx : int
            Index of the losing pool entry.
        """
        k = self.config.elo_k
        r_w = self._elo_ratings[winner_idx]
        r_l = self._elo_ratings[loser_idx]

        expected_w = 1.0 / (1.0 + 10.0 ** ((r_l - r_w) / 400.0))
        expected_l = 1.0 - expected_w

        self._elo_ratings[winner_idx] = r_w + k * (1.0 - expected_w)
        self._elo_ratings[loser_idx] = r_l + k * (0.0 - expected_l)
