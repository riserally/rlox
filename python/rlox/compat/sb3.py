"""Drop-in SB3-compatible API wrappers.

Usage:
    # Replace: from stable_baselines3 import PPO
    from rlox.compat.sb3 import PPO

    model = PPO("MlpPolicy", "CartPole-v1")
    model.learn(total_timesteps=50_000)
    model.save("ppo_cartpole")
    model = PPO.load("ppo_cartpole")
"""

from __future__ import annotations

import warnings
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


def _resolve_env_id(env: str | gym.Env) -> str:
    """Extract environment ID from a string or gymnasium.Env instance."""
    if isinstance(env, str):
        return env
    spec = getattr(env, "spec", None)
    if spec is not None and spec.id is not None:
        return spec.id
    raise ValueError(
        "Cannot determine env_id from the provided environment. "
        "Pass a string env_id or an env created via gymnasium.make()."
    )


def _is_mlp_policy(policy: str | nn.Module) -> bool:
    """Return True if the policy argument means 'use default MLP'."""
    if isinstance(policy, str):
        return policy in ("MlpPolicy", "CnnPolicy", "MultiInputPolicy")
    return False


# ---------------------------------------------------------------------------
# SB3 kwarg -> rlox config key mapping helpers
# ---------------------------------------------------------------------------

_PPO_KWARG_MAP: dict[str, str] = {
    "learning_rate": "learning_rate",
    "n_steps": "n_steps",
    "batch_size": "batch_size",
    "n_epochs": "n_epochs",
    "gamma": "gamma",
    "gae_lambda": "gae_lambda",
    "clip_range": "clip_eps",
    "ent_coef": "ent_coef",
    "vf_coef": "vf_coef",
    "max_grad_norm": "max_grad_norm",
    "normalize_advantage": "normalize_advantages",
}

_SAC_KWARG_MAP: dict[str, str] = {
    "learning_rate": "learning_rate",
    "buffer_size": "buffer_size",
    "batch_size": "batch_size",
    "tau": "tau",
    "gamma": "gamma",
    "learning_starts": "learning_starts",
}

_DQN_KWARG_MAP: dict[str, str] = {
    "learning_rate": "learning_rate",
    "buffer_size": "buffer_size",
    "batch_size": "batch_size",
    "gamma": "gamma",
    "target_update_interval": "target_update_freq",
    "exploration_fraction": "exploration_fraction",
    "exploration_initial_eps": "exploration_initial_eps",
    "exploration_final_eps": "exploration_final_eps",
    "learning_starts": "learning_starts",
}

# SB3 kwargs that we silently ignore (no rlox equivalent yet).
_IGNORED_KWARGS = frozenset(
    {
        "verbose",
        "device",
        "policy_kwargs",
        "stats_window_size",
        "train_freq",
        "gradient_steps",
        "optimize_memory_usage",
        "replay_buffer_class",
        "replay_buffer_kwargs",
        "target_policy_noise",
        "target_noise_clip",
        "use_sde",
        "sde_sample_freq",
        "use_sde_at_warmup",
        "clip_range_vf",
        "target_kl",
        "create_eval_env",
        "monitor_wrapper",
        "_init_setup_model",
    }
)


def _translate_kwargs(
    kwargs: dict[str, Any],
    kwarg_map: dict[str, str],
) -> dict[str, Any]:
    """Map SB3-style kwargs to rlox config kwargs.

    Returns a dict of rlox kwargs. Unknown keys that are not in
    ``_IGNORED_KWARGS`` produce a warning.
    """
    rlox_kwargs: dict[str, Any] = {}
    for sb3_key, value in kwargs.items():
        if sb3_key in kwarg_map:
            rlox_kwargs[kwarg_map[sb3_key]] = value
        elif sb3_key in _IGNORED_KWARGS:
            continue
        elif sb3_key in ("seed", "tensorboard_log"):
            # Handled separately by the wrapper classes.
            continue
        else:
            warnings.warn(
                f"SB3 kwarg '{sb3_key}' has no rlox mapping and will be ignored.",
                stacklevel=3,
            )
    return rlox_kwargs


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------


class PPO:
    """SB3-compatible PPO wrapper around ``rlox.algorithms.ppo.PPO``."""

    def __init__(
        self,
        policy: str | nn.Module,
        env: str | gym.Env,
        verbose: int = 0,
        seed: int | None = None,
        tensorboard_log: str | None = None,
        **kwargs: Any,
    ):
        from rlox.algorithms.ppo import PPO as _PPO

        self._env_id = _resolve_env_id(env)
        self._seed = seed if seed is not None else 42
        self._verbose = verbose

        rlox_kwargs = _translate_kwargs(kwargs, _PPO_KWARG_MAP)

        # Logger
        logger = None
        if tensorboard_log is not None:
            from rlox.logging import TensorBoardLogger

            logger = TensorBoardLogger(log_dir=tensorboard_log)

        # Policy: string -> auto-detect; nn.Module -> pass through
        policy_module: nn.Module | None = None
        if not _is_mlp_policy(policy):
            if isinstance(policy, nn.Module):
                policy_module = policy
            else:
                raise ValueError(
                    f"Unsupported policy type: {policy!r}. "
                    "Use 'MlpPolicy' or pass an nn.Module instance."
                )

        self._algo = _PPO(
            env_id=self._env_id,
            seed=self._seed,
            policy=policy_module,
            logger=logger,
            **rlox_kwargs,
        )

    # -- SB3 public API -------------------------------------------------------

    def learn(
        self,
        total_timesteps: int,
        callback: Any = None,
        log_interval: int = 1,
        progress_bar: bool = False,
    ) -> PPO:
        """Train the agent. Returns ``self`` for chaining (SB3 convention)."""
        self._algo.train(total_timesteps=total_timesteps)
        return self

    def predict(
        self,
        observation: np.ndarray | torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, None]:
        """Run policy inference on a single observation.

        Returns
        -------
        tuple[np.ndarray, None]
            ``(action, None)`` matching SB3's ``predict`` signature.
            The second element is ``None`` because rlox does not track
            hidden states.
        """
        obs_t = torch.as_tensor(np.asarray(observation), dtype=torch.float32)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                # Discrete: argmax over logits; Continuous: use mean
                if hasattr(self._algo.policy, "actor"):
                    logits_or_mean = self._algo.policy.actor(obs_t)
                    if hasattr(self._algo.policy, "log_std"):
                        # Continuous -- mean is the deterministic action
                        action = logits_or_mean.squeeze(0).numpy()
                    else:
                        # Discrete -- argmax
                        action = logits_or_mean.argmax(dim=-1).squeeze(0).numpy()
                else:
                    actions, _ = self._algo.policy.get_action_and_logprob(obs_t)
                    action = actions.squeeze(0).numpy()
            else:
                actions, _ = self._algo.policy.get_action_and_logprob(obs_t)
                action = actions.squeeze(0).numpy()

        return action, None

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        self._algo.save(path)

    @classmethod
    def load(cls, path: str, env: str | gym.Env | None = None) -> PPO:
        """Load model from checkpoint.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        env : str or gymnasium.Env, optional
            Environment (used only for env_id resolution).
        """
        from rlox.algorithms.ppo import PPO as _PPO

        env_id = _resolve_env_id(env) if env is not None else None
        algo = _PPO.from_checkpoint(path, env_id=env_id)

        wrapper = object.__new__(cls)
        wrapper._algo = algo
        wrapper._env_id = algo.env_id
        wrapper._seed = algo.seed
        wrapper._verbose = 0
        return wrapper

    @property
    def policy(self) -> nn.Module:
        return self._algo.policy


# ---------------------------------------------------------------------------
# SAC
# ---------------------------------------------------------------------------


class SAC:
    """SB3-compatible SAC wrapper around ``rlox.algorithms.sac.SAC``."""

    def __init__(
        self,
        policy: str | nn.Module,
        env: str | gym.Env,
        verbose: int = 0,
        seed: int | None = None,
        tensorboard_log: str | None = None,
        **kwargs: Any,
    ):
        from rlox.algorithms.sac import SAC as _SAC

        self._env_id = _resolve_env_id(env)
        self._seed = seed if seed is not None else 42
        self._verbose = verbose

        rlox_kwargs = _translate_kwargs(kwargs, _SAC_KWARG_MAP)

        self._algo = _SAC(
            env_id=self._env_id,
            seed=self._seed,
            **rlox_kwargs,
        )

    def learn(
        self,
        total_timesteps: int,
        callback: Any = None,
        log_interval: int = 1,
        progress_bar: bool = False,
    ) -> SAC:
        self._algo.train(total_timesteps=total_timesteps)
        return self

    def predict(
        self,
        observation: np.ndarray | torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, None]:
        obs_t = torch.as_tensor(np.asarray(observation), dtype=torch.float32)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                mean, _ = self._algo.actor(obs_t)
                action = torch.tanh(mean).squeeze(0).numpy()
            else:
                action_t, _ = self._algo.actor.sample(obs_t)
                action = action_t.squeeze(0).numpy()

        return action * self._algo.act_high, None

    def save(self, path: str) -> None:
        self._algo.save(path)

    @classmethod
    def load(cls, path: str, env: str | gym.Env | None = None) -> SAC:
        from rlox.algorithms.sac import SAC as _SAC

        env_id = _resolve_env_id(env) if env is not None else None
        algo = _SAC.from_checkpoint(path, env_id=env_id)

        wrapper = object.__new__(cls)
        wrapper._algo = algo
        wrapper._env_id = algo.env_id
        wrapper._seed = 42
        wrapper._verbose = 0
        return wrapper

    @property
    def policy(self) -> nn.Module:
        return self._algo.actor


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------


class DQN:
    """SB3-compatible DQN wrapper around ``rlox.algorithms.dqn.DQN``."""

    def __init__(
        self,
        policy: str | nn.Module,
        env: str | gym.Env,
        verbose: int = 0,
        seed: int | None = None,
        tensorboard_log: str | None = None,
        **kwargs: Any,
    ):
        from rlox.algorithms.dqn import DQN as _DQN

        self._env_id = _resolve_env_id(env)
        self._seed = seed if seed is not None else 42
        self._verbose = verbose

        rlox_kwargs = _translate_kwargs(kwargs, _DQN_KWARG_MAP)

        self._algo = _DQN(
            env_id=self._env_id,
            seed=self._seed,
            **rlox_kwargs,
        )

    def learn(
        self,
        total_timesteps: int,
        callback: Any = None,
        log_interval: int = 1,
        progress_bar: bool = False,
    ) -> DQN:
        self._algo.train(total_timesteps=total_timesteps)
        return self

    def predict(
        self,
        observation: np.ndarray | torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, None]:
        obs_t = torch.as_tensor(np.asarray(observation), dtype=torch.float32)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)

        with torch.no_grad():
            q_values = self._algo.q_network(obs_t)
            action = q_values.argmax(dim=-1).squeeze(0).numpy()

        return action, None

    def save(self, path: str) -> None:
        self._algo.save(path)

    @classmethod
    def load(cls, path: str, env: str | gym.Env | None = None) -> DQN:
        from rlox.algorithms.dqn import DQN as _DQN

        env_id = _resolve_env_id(env) if env is not None else None
        algo = _DQN.from_checkpoint(path, env_id=env_id)

        wrapper = object.__new__(cls)
        wrapper._algo = algo
        wrapper._env_id = algo.env_id
        wrapper._seed = 42
        wrapper._verbose = 0
        return wrapper

    @property
    def policy(self) -> nn.Module:
        return self._algo.q_network
