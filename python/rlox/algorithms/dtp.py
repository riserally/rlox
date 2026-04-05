"""Decision Tree Policies for Offline RL (RWDTP / RCDTP).

Implements the two frameworks from:
    P. Koirala and C. Fleming, "Solving Offline Reinforcement Learning with
    Decision Tree Regression," arXiv:2401.11630, 2024.

RWDTP: return-weighted regression -- fits XGBoost on (state -> action) with
    per-sample weights proportional to normalized discounted returns.

RCDTP: return-conditioned regression -- fits XGBoost on
    (state, return-to-go, timestep) -> action, conditioning actions on the
    desired return (similar to Decision Transformer but with gradient-boosted
    trees instead of a Transformer).

Both methods are offline-only and train in seconds-to-minutes on CPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing import TypeVar
    Self = TypeVar("Self")

import numpy as np

from rlox.callbacks import Callback, CallbackList
from rlox.logging import LoggerCallback
from rlox.offline.base import OfflineAlgorithm, OfflineDataset
from rlox.trainer import register_algorithm


# ---------------------------------------------------------------------------
# Utility functions (tested independently)
# ---------------------------------------------------------------------------


def _compute_episode_returns(
    rewards: np.ndarray,
    terminated: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    """Compute per-step discounted returns, resetting at episode boundaries.

    Parameters
    ----------
    rewards : (N,) array of per-step rewards.
    terminated : (N,) uint8 array; 1 at the last step of each episode.
    gamma : discount factor.

    Returns
    -------
    returns : (N,) array of R_t = sum_{k=t}^{T} gamma^{k-t} * r_k
    """
    n = len(rewards)
    returns = np.zeros(n, dtype=np.float64)
    running = 0.0
    for i in range(n - 1, -1, -1):
        if terminated[i]:
            running = 0.0
        running = float(rewards[i]) + gamma * running
        returns[i] = running
    return returns.astype(np.float32)


def _compute_timesteps(terminated: np.ndarray) -> np.ndarray:
    """Compute per-step timestep within each episode.

    Parameters
    ----------
    terminated : (N,) uint8; 1 at episode boundaries.

    Returns
    -------
    timesteps : (N,) int32 array, 0-indexed within each episode.
    """
    n = len(terminated)
    timesteps = np.zeros(n, dtype=np.int32)
    t = 0
    for i in range(n):
        timesteps[i] = t
        if terminated[i]:
            t = 0
        else:
            t += 1
    return timesteps


def _normalize_returns(
    returns: np.ndarray,
    power: float = 1.0,
) -> np.ndarray:
    """Min-max normalize returns to [0, 1] and raise to power p.

    If all returns are identical, returns uniform weight of 1.0.

    Parameters
    ----------
    returns : (N,) array of discounted returns.
    power : exponent p for weight shaping.

    Returns
    -------
    weights : (N,) array in [0, 1].
    """
    r_min = returns.min()
    r_max = returns.max()
    if np.isclose(r_max, r_min):
        return np.ones_like(returns, dtype=np.float32)
    normed = (returns - r_min) / (r_max - r_min)
    return np.power(normed, power).astype(np.float32)


# ---------------------------------------------------------------------------
# DecisionTreePolicy
# ---------------------------------------------------------------------------


class DecisionTreePolicy(OfflineAlgorithm):
    """Decision Tree Policy for offline RL.

    Supports both RWDTP (return-weighted) and RCDTP (return-conditioned)
    frameworks.  Uses one XGBRegressor per action dimension.

    Parameters
    ----------
    dataset : OfflineDataset
        Offline dataset (must support ``sample()`` and ``__len__()``).
    obs_dim : int
        Observation dimension.
    act_dim : int
        Action dimension.
    method : str
        ``"rwdtp"`` or ``"rcdtp"`` (default ``"rwdtp"``).
    gamma : float
        Discount factor for return computation (default 1.0).
    return_power : float
        Exponent p for RWDTP weight shaping (default 1.0).
    n_trees : int
        Number of XGBoost boosting rounds (default 500).
    max_depth : int
        Maximum depth per tree (default 6).
    learning_rate_xgb : float
        XGBoost shrinkage (default 0.1).
    target_return : float or None
        RCDTP target return for inference.  None = dataset max.
    subsample : float
        Row subsampling (default 1.0).
    colsample_bytree : float
        Column subsampling per tree (default 1.0).
    reg_alpha : float
        L1 regularization (default 0.0).
    reg_lambda : float
        L2 regularization (default 1.0).
    callbacks : list[Callback], optional
    logger : LoggerCallback, optional
    """

    def __init__(
        self,
        dataset: OfflineDataset,
        obs_dim: int,
        act_dim: int,
        method: str = "rwdtp",
        gamma: float = 1.0,
        return_power: float = 1.0,
        n_trees: int = 500,
        max_depth: int = 6,
        learning_rate_xgb: float = 0.1,
        target_return: float | None = None,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        callbacks: list[Callback] | None = None,
        logger: LoggerCallback | None = None,
    ):
        super().__init__(dataset, batch_size=len(dataset), callbacks=callbacks, logger=logger)
        if method not in ("rwdtp", "rcdtp"):
            raise ValueError(f"method must be 'rwdtp' or 'rcdtp', got {method!r}")

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.method = method
        self.gamma = gamma
        self.return_power = return_power
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.learning_rate_xgb = learning_rate_xgb
        self.target_return = target_return
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

        # One XGBRegressor per action dimension (filled during train)
        self.models: list[Any] = []
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, n_gradient_steps: int = 1) -> dict[str, float]:
        """Train XGBoost models on the full offline dataset.

        The ``n_gradient_steps`` parameter is accepted for API compatibility
        but ignored -- XGBoost trains to convergence in a single call.

        Returns
        -------
        metrics : dict with ``"mse"`` key.
        """
        import xgboost as xgb

        self.callbacks.on_training_start()

        # Pull the entire dataset
        all_data = self.dataset.sample(len(self.dataset), seed=0)
        obs = all_data["obs"]           # (N, obs_dim)
        actions = all_data["actions"]   # (N, act_dim)
        rewards = all_data["rewards"]   # (N,)
        terminated = all_data["terminated"].astype(np.uint8)  # (N,)

        # Compute returns
        returns = _compute_episode_returns(rewards, terminated, gamma=self.gamma)

        # Build features and weights
        if self.method == "rwdtp":
            X = obs
            weights = _normalize_returns(returns, power=self.return_power)
        else:
            # RCDTP: return-to-go (undiscounted) + timestep as extra features
            rtg = _compute_episode_returns(rewards, terminated, gamma=1.0)
            timesteps = _compute_timesteps(terminated).astype(np.float32)
            X = np.column_stack([obs, rtg.reshape(-1, 1), timesteps.reshape(-1, 1)])
            weights = None

        # Store dataset-level stats for inference defaults
        if self.method == "rcdtp":
            self._rtg_max = float(rtg.max()) if rtg.max() > 0 else float(rtg.mean())

        # Fit one XGBRegressor per action dim
        self.models = []
        total_mse = 0.0

        xgb_params = dict(
            n_estimators=self.n_trees,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate_xgb,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective="reg:squarederror",
            verbosity=0,
        )

        for d in range(self.act_dim):
            model = xgb.XGBRegressor(**xgb_params)
            y = actions[:, d] if actions.ndim > 1 else actions
            if weights is not None:
                model.fit(X, y, sample_weight=weights)
            else:
                model.fit(X, y)
            self.models.append(model)
            preds = model.predict(X)
            total_mse += float(np.mean((y - preds) ** 2))

        avg_mse = total_mse / max(self.act_dim, 1)
        self._trained = True

        metrics = {"mse": avg_mse}
        self.callbacks.on_training_end()
        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        obs: np.ndarray,
        rtg: float | None = None,
        timestep: int | None = None,
    ) -> np.ndarray:
        """Predict actions from observations.

        Parameters
        ----------
        obs : (obs_dim,) or (B, obs_dim) array.
        rtg : float, optional
            Return-to-go for RCDTP.  Ignored for RWDTP.
        timestep : int, optional
            Current timestep for RCDTP.  Ignored for RWDTP.

        Returns
        -------
        actions : (act_dim,) or (B, act_dim) array.
        """
        if not self._trained:
            raise RuntimeError("Model not trained yet; call .train() first.")

        single = obs.ndim == 1
        if single:
            obs = obs.reshape(1, -1)

        if self.method == "rcdtp":
            if rtg is None:
                rtg = getattr(self, "_rtg_max", 0.0)
            if timestep is None:
                timestep = 0
            B = obs.shape[0]
            rtg_col = np.full((B, 1), rtg, dtype=np.float32)
            ts_col = np.full((B, 1), timestep, dtype=np.float32)
            X = np.column_stack([obs, rtg_col, ts_col])
        else:
            X = obs

        preds = np.column_stack([m.predict(X) for m in self.models])
        if preds.shape[1] == 1 and self.act_dim == 1:
            pass  # keep (B, 1)

        return preds.squeeze(0) if single else preds

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save models and metadata to disk.

        Creates ``{path}/`` directory with one ``.json`` per XGB model
        and a ``meta.json`` with config.
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        meta = {
            "method": self.method,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "gamma": self.gamma,
            "return_power": self.return_power,
            "n_trees": self.n_trees,
            "max_depth": self.max_depth,
            "learning_rate_xgb": self.learning_rate_xgb,
            "target_return": self.target_return,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
        }
        if hasattr(self, "_rtg_max"):
            meta["_rtg_max"] = self._rtg_max

        with open(p / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        for i, model in enumerate(self.models):
            model.save_model(str(p / f"model_{i}.json"))

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs: Any) -> Self:
        """Load a saved DecisionTreePolicy.

        Parameters
        ----------
        path : str
            Directory containing ``meta.json`` and model files.

        Returns
        -------
        DecisionTreePolicy with loaded models.
        """
        import xgboost as xgb

        p = Path(path)
        with open(p / "meta.json") as f:
            meta = json.load(f)

        # Create a dummy instance (no dataset needed for inference)
        instance = object.__new__(cls)
        instance.obs_dim = meta["obs_dim"]
        instance.act_dim = meta["act_dim"]
        instance.method = meta["method"]
        instance.gamma = meta["gamma"]
        instance.return_power = meta["return_power"]
        instance.n_trees = meta["n_trees"]
        instance.max_depth = meta["max_depth"]
        instance.learning_rate_xgb = meta["learning_rate_xgb"]
        instance.target_return = meta.get("target_return")
        instance.subsample = meta.get("subsample", 1.0)
        instance.colsample_bytree = meta.get("colsample_bytree", 1.0)
        instance.reg_alpha = meta.get("reg_alpha", 0.0)
        instance.reg_lambda = meta.get("reg_lambda", 1.0)
        instance._trained = True
        instance.callbacks = CallbackList(None)
        instance.logger = None
        instance.dataset = None
        instance.batch_size = 0
        instance._global_step = 0

        if "_rtg_max" in meta:
            instance._rtg_max = meta["_rtg_max"]

        instance.models = []
        for i in range(meta["act_dim"]):
            model = xgb.XGBRegressor()
            model.load_model(str(p / f"model_{i}.json"))
            instance.models.append(model)

        return instance


# ---------------------------------------------------------------------------
# Register both variants with the rlox Trainer
# ---------------------------------------------------------------------------

# RWDTP and RCDTP are the same class with different default method.
# We register thin wrappers so that Trainer("rwdtp", ...) and
# Trainer("rcdtp", ...) both work.


@register_algorithm("rwdtp")
class RWDTP(DecisionTreePolicy):
    """Return-Weighted Decision Tree Policy (RWDTP).

    Convenience subclass that defaults ``method="rwdtp"``.
    """

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("method", "rwdtp")
        super().__init__(**kwargs)


@register_algorithm("rcdtp")
class RCDTP(DecisionTreePolicy):
    """Return-Conditioned Decision Tree Policy (RCDTP).

    Convenience subclass that defaults ``method="rcdtp"``.
    """

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("method", "rcdtp")
        super().__init__(**kwargs)
