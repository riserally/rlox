"""Go-Explore: archive-based exploration for sparse rewards.

Ecoffet et al., 2021. Maintains an archive of interesting states indexed
by discretized cell keys. Selects under-explored cells weighted by novelty
and score, then explores from the archived state.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rlox.config import GoExploreConfig


@dataclass(slots=True)
class _CellEntry:
    """A single cell in the archive."""

    obs: np.ndarray
    score: float
    visit_count: int
    trajectory: list[np.ndarray]


class GoExplore:
    """Go-Explore: archive-based exploration for sparse rewards.

    Parameters
    ----------
    config : GoExploreConfig, optional
        Configuration. Uses defaults if not provided.
    """

    def __init__(self, config: GoExploreConfig | None = None) -> None:
        self.config = config or GoExploreConfig()
        # cell_key -> _CellEntry
        self._archive: dict[str, _CellEntry] = {}

    @property
    def archive(self) -> dict[str, _CellEntry]:
        """Read-only access to the archive."""
        return self._archive

    def compute_cell(self, obs: np.ndarray) -> str:
        """Discretize an observation into a cell key.

        Downscales the observation by rounding to ``cell_resolution``
        granularity, then hashes the result.

        Parameters
        ----------
        obs : (D,) array
            Raw observation.

        Returns
        -------
        cell_key : str
            Deterministic hash string for this cell.
        """
        resolution = self.config.cell_resolution
        discretized = np.round(obs * resolution).astype(np.int32)
        key = hashlib.md5(discretized.tobytes()).hexdigest()
        return key

    def add_to_archive(
        self,
        obs: np.ndarray,
        score: float,
        trajectory: list[np.ndarray],
    ) -> None:
        """Add or update a state in the archive.

        If the cell already exists, the visit count is incremented and
        the entry is updated if the new score is higher.  If the archive
        is full, the lowest-score cell is evicted.

        Parameters
        ----------
        obs : (D,) array
            Observation to archive.
        score : float
            Cumulative reward or fitness associated with this state.
        trajectory : list of arrays
            Sequence of observations leading to this state (for replay).
        """
        cell_key = self.compute_cell(obs)

        if cell_key in self._archive:
            entry = self._archive[cell_key]
            entry.visit_count += 1
            if score > entry.score:
                entry.obs = obs.copy()
                entry.score = score
                entry.trajectory = list(trajectory)
        else:
            # Evict lowest-score cell if archive is full
            if len(self._archive) >= self.config.archive_size:
                worst_key = min(
                    self._archive,
                    key=lambda k: self._archive[k].score,
                )
                del self._archive[worst_key]

            self._archive[cell_key] = _CellEntry(
                obs=obs.copy(),
                score=score,
                visit_count=1,
                trajectory=list(trajectory),
            )

    def select_cell(self) -> tuple[str, _CellEntry]:
        """Select a cell for exploration, weighted by novelty and score.

        Novelty is defined as ``1 / visit_count``. The selection
        probability for each cell is proportional to
        ``novelty_weight / visit_count + score_weight * score``.

        Returns
        -------
        cell_key : str
        entry : _CellEntry
        """
        if not self._archive:
            raise RuntimeError("Cannot select from an empty archive")

        cfg = self.config
        keys = list(self._archive.keys())
        entries = [self._archive[k] for k in keys]

        weights = np.array(
            [
                cfg.novelty_weight / e.visit_count + cfg.score_weight * max(e.score, 0.0)
                for e in entries
            ],
            dtype=np.float64,
        )
        # Ensure all weights are positive
        weights = np.maximum(weights, 1e-10)
        probs = weights / weights.sum()

        idx = np.random.choice(len(keys), p=probs)
        return keys[idx], entries[idx]
