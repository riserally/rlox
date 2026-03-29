"""Behavioral Cloning — supervised learning on demonstrations.

The simplest imitation learning algorithm: directly regress
actions from observations using the offline dataset.

Continuous: MSE loss
Discrete: cross-entropy loss

Reference:
    M. Bain and C. Sammut,
    "A Framework for Behavioural Cloning,"
    Machine Intelligence 15, pp. 103-129, 1995.

See also:
    D. A. Pomerleau,
    "ALVINN: An Autonomous Land Vehicle in a Neural Network,"
    NeurIPS, 1989. https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlox.callbacks import Callback
from rlox.logging import LoggerCallback
from rlox.offline.base import OfflineAlgorithm, OfflineDataset


class BC(OfflineAlgorithm):
    """Behavioral Cloning.

    Parameters
    ----------
    dataset : OfflineDataset
        Offline dataset with expert demonstrations.
    obs_dim : int
        Observation dimension.
    act_dim : int
        Action dimension (or number of discrete actions).
    continuous : bool
        Whether the action space is continuous (default True).
    hidden : int
        Hidden layer width (default 256).
    learning_rate : float
        Learning rate (default 3e-4).
    batch_size : int
        Minibatch size (default 256).
    callbacks : list[Callback], optional
    logger : LoggerCallback, optional
    """

    def __init__(
        self,
        dataset: OfflineDataset,
        obs_dim: int,
        act_dim: int,
        continuous: bool = True,
        hidden: int = 256,
        learning_rate: float = 3e-4,
        batch_size: int = 256,
        callbacks: list[Callback] | None = None,
        logger: LoggerCallback | None = None,
    ):
        super().__init__(dataset, batch_size, callbacks, logger)
        self.continuous = continuous

        if continuous:
            self.policy = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, act_dim), nn.Tanh(),
            )
        else:
            self.policy = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, act_dim),
            )

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

    def _update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"])

        pred = self.policy(obs)

        if self.continuous:
            actions = actions.float()
            if pred.shape != actions.shape:
                actions = actions.reshape(pred.shape)
            loss = F.mse_loss(pred, actions)
        else:
            actions = actions.long()
            if actions.dim() > 1:
                actions = actions.squeeze(-1)
            loss = F.cross_entropy(pred, actions)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Get action from the learned policy."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            pred = self.policy(obs_t).squeeze(0)
            if self.continuous:
                return pred.numpy()
            return pred.argmax().item()
