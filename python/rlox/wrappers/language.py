"""Language-Conditioned and Goal-Conditioned RL wrappers.

Provides wrappers that augment observations with language embeddings
or goal vectors and compute sparse goal-conditioned rewards.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class LanguageWrapper:
    """Wrap any env with language instruction conditioning.

    Appends a language embedding to the observation vector.  Uses a
    frozen sentence transformer for encoding.

    Parameters
    ----------
    env : object
        Inner vectorized environment with ``step_all`` / ``reset_all``.
    instructions : list[str]
        List of natural-language instructions.
    encoder : str
        Sentence-transformers model name (default ``"all-MiniLM-L6-v2"``).
    instruction_mode : str
        How to select instructions:

        - ``"fixed"`` -- use the first instruction for all episodes.
        - ``"random"`` -- sample a random instruction per reset.
        - ``"curriculum"`` -- cycle through instructions sequentially.
    """

    def __init__(
        self,
        env: Any,
        instructions: list[str],
        encoder: str = "all-MiniLM-L6-v2",
        instruction_mode: str = "fixed",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.env = env
        self.instructions = instructions
        self.instruction_mode = instruction_mode

        self._model = SentenceTransformer(encoder)
        # Pre-compute all instruction embeddings
        self._embeddings = self._model.encode(
            instructions, convert_to_numpy=True
        ).astype(np.float32)
        self._embed_dim = self._embeddings.shape[1]
        self._n_envs = env.num_envs()

        # Current instruction embedding per env: (n_envs, embed_dim)
        self._current_embeddings = self._select_embeddings()
        self._curriculum_idx = 0

    def _select_embeddings(self) -> np.ndarray:
        """Select instruction embeddings based on mode."""
        if self.instruction_mode == "fixed":
            return np.tile(self._embeddings[0], (self._n_envs, 1))
        elif self.instruction_mode == "random":
            indices = np.random.randint(0, len(self.instructions), size=self._n_envs)
            return self._embeddings[indices]
        elif self.instruction_mode == "curriculum":
            idx = self._curriculum_idx % len(self.instructions)
            self._curriculum_idx += 1
            return np.tile(self._embeddings[idx], (self._n_envs, 1))
        else:
            raise ValueError(
                f"Unknown instruction_mode: {self.instruction_mode!r}. "
                f"Expected 'fixed', 'random', or 'curriculum'."
            )

    def _append_instruction(self, obs: np.ndarray) -> np.ndarray:
        """Concatenate instruction embeddings to observations."""
        return np.concatenate([obs, self._current_embeddings], axis=1).astype(
            np.float32
        )

    def step_all(self, actions: np.ndarray | list[Any]) -> dict[str, Any]:
        """Step the inner env and append instruction embeddings."""
        result = self.env.step_all(actions)
        result["obs"] = self._append_instruction(result["obs"])
        return result

    def reset_all(self, **kwargs: Any) -> np.ndarray:
        """Reset inner env, re-select instructions, and append embeddings."""
        obs = self.env.reset_all(**kwargs)
        self._current_embeddings = self._select_embeddings()
        return self._append_instruction(obs)

    def num_envs(self) -> int:
        """Return the number of parallel sub-environments."""
        return self._n_envs


class GoalConditionedWrapper:
    """Wrap env to append goal to observation and compute sparse reward.

    The last ``goal_dim`` dimensions of the original observation are
    treated as the *achieved goal*.  A random desired goal is sampled
    at each reset and appended to the observation.

    Sparse reward: ``0`` if distance < ``distance_threshold``, else ``-1``.

    Parameters
    ----------
    env : object
        Inner vectorized environment.
    goal_dim : int
        Dimensionality of the goal vector.
    distance_threshold : float
        L2 distance below which the goal is considered reached (default 0.05).
    """

    def __init__(
        self,
        env: Any,
        goal_dim: int,
        distance_threshold: float = 0.05,
    ) -> None:
        self.env = env
        self.goal_dim = goal_dim
        self.distance_threshold = distance_threshold
        self._n_envs = env.num_envs()
        self._goals: np.ndarray | None = None

    def _sample_goals(self, obs: np.ndarray) -> np.ndarray:
        """Sample random goals from the achieved-goal space."""
        # Use the last goal_dim dims of observation as the achieved goal space
        # Sample goals as random perturbations of achieved goals
        achieved = obs[:, -self.goal_dim :]
        noise = np.random.randn(*achieved.shape).astype(np.float32) * 0.1
        return (achieved + noise).astype(np.float32)

    def _compute_sparse_reward(self, obs: np.ndarray) -> np.ndarray:
        """Compute sparse reward: 0 if within threshold, -1 otherwise."""
        achieved = obs[:, -self.goal_dim :]
        assert self._goals is not None
        distances = np.linalg.norm(achieved - self._goals, axis=1)
        return np.where(distances < self.distance_threshold, 0.0, -1.0)

    def step_all(self, actions: np.ndarray | list[Any]) -> dict[str, Any]:
        """Step inner env, append goal, and compute sparse reward."""
        result = self.env.step_all(actions)
        obs = result["obs"]

        # Compute sparse reward before appending goal
        result["rewards"] = self._compute_sparse_reward(obs)

        # Append goal to observation
        assert self._goals is not None
        result["obs"] = np.concatenate([obs, self._goals], axis=1).astype(
            np.float32
        )
        return result

    def reset_all(self, **kwargs: Any) -> np.ndarray:
        """Reset inner env, sample new goals, and append them."""
        obs = self.env.reset_all(**kwargs)
        self._goals = self._sample_goals(obs)
        return np.concatenate([obs, self._goals], axis=1).astype(np.float32)

    def num_envs(self) -> int:
        """Return the number of parallel sub-environments."""
        return self._n_envs
