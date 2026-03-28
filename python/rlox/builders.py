"""Builder pattern for algorithm construction.

Builders provide a fluent API for progressively customizing algorithms
while keeping the zero-config path intact.

Example
-------
>>> from rlox.builders import SACBuilder, PPOBuilder
>>> from rlox.exploration import OUNoise
>>>
>>> # Zero-config (same as SAC("Pendulum-v1"))
>>> sac = SACBuilder().env("Pendulum-v1").build()
>>>
>>> # Custom actor + exploration
>>> sac = (SACBuilder()
...     .env("Pendulum-v1")
...     .actor(MyCNNActor(...))
...     .critic(MyCNNCritic(...))
...     .exploration(OUNoise(action_dim=1))
...     .learning_rate(1e-4)
...     .build())
>>>
>>> # Zero-config PPO
>>> ppo = PPOBuilder().env("CartPole-v1").build()
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn


class PPOBuilder:
    """Fluent builder for PPO algorithm.

    All methods return ``self`` for chaining. Call ``.build()`` to create
    the PPO instance.
    """

    def __init__(self):
        self._env_id: str | None = None
        self._n_envs: int = 8
        self._seed: int = 42
        self._policy: nn.Module | None = None
        self._compile: bool = False
        self._callbacks: list | None = None
        self._logger = None
        self._config: dict[str, Any] = {}

    def env(self, env_id: str) -> PPOBuilder:
        self._env_id = env_id
        return self

    def n_envs(self, n: int) -> PPOBuilder:
        self._n_envs = n
        return self

    def seed(self, s: int) -> PPOBuilder:
        self._seed = s
        return self

    def policy(self, p: nn.Module) -> PPOBuilder:
        self._policy = p
        return self

    def compile(self, enabled: bool = True) -> PPOBuilder:
        self._compile = enabled
        return self

    def callbacks(self, cbs: list) -> PPOBuilder:
        self._callbacks = cbs
        return self

    def logger(self, l) -> PPOBuilder:
        self._logger = l
        return self

    def learning_rate(self, lr: float) -> PPOBuilder:
        self._config["learning_rate"] = lr
        return self

    def n_steps(self, n: int) -> PPOBuilder:
        self._config["n_steps"] = n
        return self

    def n_epochs(self, n: int) -> PPOBuilder:
        self._config["n_epochs"] = n
        return self

    def batch_size(self, n: int) -> PPOBuilder:
        self._config["batch_size"] = n
        return self

    def gamma(self, g: float) -> PPOBuilder:
        self._config["gamma"] = g
        return self

    def clip_eps(self, e: float) -> PPOBuilder:
        self._config["clip_eps"] = e
        return self

    def config(self, **kwargs) -> PPOBuilder:
        """Set arbitrary config parameters."""
        self._config.update(kwargs)
        return self

    def build(self):
        """Create and return the PPO instance."""
        from rlox.algorithms.ppo import PPO
        if self._env_id is None:
            raise ValueError("env_id is required. Call .env('CartPole-v1') before .build()")
        return PPO(
            env_id=self._env_id,
            n_envs=self._n_envs,
            seed=self._seed,
            policy=self._policy,
            logger=self._logger,
            callbacks=self._callbacks,
            compile=self._compile,
            **self._config,
        )


class SACBuilder:
    """Fluent builder for SAC algorithm.

    Supports custom actor, critic, exploration strategy, and all hyperparameters.
    """

    def __init__(self):
        self._env_id: str | Any = None
        self._seed: int = 42
        self._actor: nn.Module | None = None
        self._critic1: nn.Module | None = None
        self._critic2: nn.Module | None = None
        self._exploration = None
        self._compile: bool = False
        self._callbacks: list | None = None
        self._logger = None
        self._config: dict[str, Any] = {}

    def env(self, env_id) -> SACBuilder:
        self._env_id = env_id
        return self

    def seed(self, s: int) -> SACBuilder:
        self._seed = s
        return self

    def actor(self, a: nn.Module) -> SACBuilder:
        self._actor = a
        return self

    def critic(self, c: nn.Module) -> SACBuilder:
        """Set both critics to the same architecture (separate instances)."""
        import copy
        self._critic1 = c
        self._critic2 = copy.deepcopy(c)
        return self

    def critics(self, c1: nn.Module, c2: nn.Module) -> SACBuilder:
        """Set critics independently."""
        self._critic1 = c1
        self._critic2 = c2
        return self

    def exploration(self, strategy) -> SACBuilder:
        self._exploration = strategy
        return self

    def compile(self, enabled: bool = True) -> SACBuilder:
        self._compile = enabled
        return self

    def callbacks(self, cbs: list) -> SACBuilder:
        self._callbacks = cbs
        return self

    def logger(self, l) -> SACBuilder:
        self._logger = l
        return self

    def learning_rate(self, lr: float) -> SACBuilder:
        self._config["learning_rate"] = lr
        return self

    def learning_starts(self, n: int) -> SACBuilder:
        self._config["learning_starts"] = n
        return self

    def buffer_size(self, n: int) -> SACBuilder:
        self._config["buffer_size"] = n
        return self

    def batch_size(self, n: int) -> SACBuilder:
        self._config["batch_size"] = n
        return self

    def gamma(self, g: float) -> SACBuilder:
        self._config["gamma"] = g
        return self

    def tau(self, t: float) -> SACBuilder:
        self._config["tau"] = t
        return self

    def hidden(self, h: int) -> SACBuilder:
        self._config["hidden"] = h
        return self

    def config(self, **kwargs) -> SACBuilder:
        self._config.update(kwargs)
        return self

    def build(self):
        """Create and return the SAC instance."""
        from rlox.algorithms.sac import SAC
        if self._env_id is None:
            raise ValueError("env_id is required. Call .env('Pendulum-v1') before .build()")
        sac = SAC(
            env_id=self._env_id,
            seed=self._seed,
            callbacks=self._callbacks,
            logger=self._logger,
            compile=self._compile,
            **self._config,
        )
        # Inject custom components
        if self._actor is not None:
            sac.actor = self._actor
        if self._critic1 is not None:
            sac.critic1 = self._critic1
        if self._critic2 is not None:
            sac.critic2 = self._critic2
        return sac


class DQNBuilder:
    """Fluent builder for DQN algorithm."""

    def __init__(self):
        self._env_id = None
        self._seed: int = 42
        self._q_network: nn.Module | None = None
        self._exploration = None
        self._compile: bool = False
        self._callbacks: list | None = None
        self._logger = None
        self._config: dict[str, Any] = {}

    def env(self, env_id) -> DQNBuilder:
        self._env_id = env_id
        return self

    def seed(self, s: int) -> DQNBuilder:
        self._seed = s
        return self

    def q_network(self, net: nn.Module) -> DQNBuilder:
        self._q_network = net
        return self

    def exploration(self, strategy) -> DQNBuilder:
        self._exploration = strategy
        return self

    def compile(self, enabled: bool = True) -> DQNBuilder:
        self._compile = enabled
        return self

    def callbacks(self, cbs: list) -> DQNBuilder:
        self._callbacks = cbs
        return self

    def double_dqn(self, enabled: bool = True) -> DQNBuilder:
        self._config["double_dqn"] = enabled
        return self

    def dueling(self, enabled: bool = True) -> DQNBuilder:
        self._config["dueling"] = enabled
        return self

    def prioritized(self, enabled: bool = True) -> DQNBuilder:
        self._config["prioritized"] = enabled
        return self

    def config(self, **kwargs) -> DQNBuilder:
        self._config.update(kwargs)
        return self

    def build(self):
        from rlox.algorithms.dqn import DQN
        if self._env_id is None:
            raise ValueError("env_id is required")
        dqn = DQN(
            env_id=self._env_id,
            seed=self._seed,
            callbacks=self._callbacks,
            logger=self._logger,
            compile=self._compile,
            **self._config,
        )
        if self._q_network is not None:
            import copy
            dqn.q_network = self._q_network
            dqn.target_network = copy.deepcopy(self._q_network)
        return dqn
