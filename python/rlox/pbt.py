"""Population-Based Training (PBT) -- concurrent agents with exploit/explore.

Jaderberg et al., 2017. Maintains a population of agents that train
concurrently. Periodically, the worst performers copy weights from the
best performers (exploit), and then perturb their hyperparameters (explore).
"""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import torch

from rlox.config import PBTConfig
from rlox.trainer import ALGORITHM_REGISTRY


class PBT:
    """Population-Based Training -- concurrent agents with exploit/explore.

    Parameters
    ----------
    algo : str
        Algorithm name (must be registered in ALGORITHM_REGISTRY).
    env : str
        Gymnasium environment ID.
    population_size : int
        Number of agents in the population (default 8).
    interval : int
        Training timesteps between exploit/explore cycles (default 10_000).
    n_iterations : int
        Number of PBT iterations (default 20).
    exploit_fraction : float
        Bottom fraction of population replaced each cycle (default 0.2).
    perturb_factor : float
        Hyperparameter perturbation range (default 0.2).
    seed : int
        Base random seed (default 42).
    algo_config : dict
        Extra config forwarded to each algorithm instance.
    """

    def __init__(
        self,
        algo: str = "ppo",
        env: str = "CartPole-v1",
        population_size: int = 8,
        interval: int = 10_000,
        n_iterations: int = 20,
        exploit_fraction: float = 0.2,
        perturb_factor: float = 0.2,
        seed: int = 42,
        algo_config: dict[str, Any] | None = None,
    ) -> None:
        self.algo_name = algo.lower()
        self.env = env
        self.config = PBTConfig(
            population_size=population_size,
            interval=interval,
            n_iterations=n_iterations,
            exploit_fraction=exploit_fraction,
            perturb_factor=perturb_factor,
        )
        self.seed = seed
        self._algo_config = algo_config or {}

        algo_cls = ALGORITHM_REGISTRY.get(self.algo_name)
        if algo_cls is None:
            raise ValueError(
                f"Unknown algorithm {self.algo_name!r}. "
                f"Registered: {sorted(ALGORITHM_REGISTRY)}"
            )

        # Create population of agents
        self._agents = []
        for i in range(population_size):
            agent = algo_cls(
                env_id=env,
                seed=seed + i,
                **self._algo_config,
            )
            self._agents.append(agent)

        self._fitnesses: list[float] = [0.0] * population_size
        self._exploited_indices: set[int] = set()

    def _evaluate(self, agent_idx: int) -> float:
        """Evaluate an agent by running a short rollout and returning mean reward."""
        agent = self._agents[agent_idx]
        # Use the agent's collector to gather a short rollout
        n_steps = 128
        if hasattr(agent, "collector") and hasattr(agent, "policy"):
            batch = agent.collector.collect(agent.policy, n_steps=n_steps)
            n_envs = agent.config.n_envs if hasattr(agent.config, "n_envs") else 1
            return batch.rewards.sum().item() / max(n_envs, 1)
        return 0.0

    def _exploit(self) -> None:
        """Copy weights from best performers to worst performers."""
        cfg = self.config
        n = cfg.population_size
        n_replace = max(1, int(n * cfg.exploit_fraction))

        # Rank by fitness (descending)
        ranked = sorted(range(n), key=lambda i: self._fitnesses[i], reverse=True)
        best_indices = ranked[:n_replace]
        worst_indices = ranked[-n_replace:]

        self._exploited_indices = set()
        for worst_idx, best_idx in zip(worst_indices, best_indices):
            best_agent = self._agents[best_idx]
            worst_agent = self._agents[worst_idx]

            # Copy policy weights
            if hasattr(best_agent, "policy") and hasattr(worst_agent, "policy"):
                worst_agent.policy.load_state_dict(
                    copy.deepcopy(best_agent.policy.state_dict())
                )
            self._exploited_indices.add(worst_idx)

    def _explore(self) -> None:
        """Perturb hyperparameters of recently exploited agents."""
        cfg = self.config
        for idx in self._exploited_indices:
            agent = self._agents[idx]
            # Perturb learning rate
            if hasattr(agent, "optimizer"):
                for pg in agent.optimizer.param_groups:
                    factor = 1.0 + random.uniform(-cfg.perturb_factor, cfg.perturb_factor)
                    pg["lr"] = pg["lr"] * factor

    def run(self) -> dict[str, Any]:
        """Execute the full PBT loop.

        Returns
        -------
        dict with 'best_fitness', 'all_fitnesses', 'n_iterations'.
        """
        cfg = self.config

        for iteration in range(cfg.n_iterations):
            # Train each agent for `interval` timesteps
            for i, agent in enumerate(self._agents):
                agent.train(total_timesteps=cfg.interval)

            # Evaluate fitness
            for i in range(cfg.population_size):
                self._fitnesses[i] = self._evaluate(i)

            # Exploit: copy best to worst
            self._exploit()

            # Explore: perturb HPs of exploited agents
            self._explore()

        return {
            "best_fitness": max(self._fitnesses),
            "all_fitnesses": list(self._fitnesses),
            "n_iterations": cfg.n_iterations,
        }
