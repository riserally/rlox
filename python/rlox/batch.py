"""RolloutBatch: flat-tensor container for on-policy rollout data.

Used by :class:`~rlox.collectors.RolloutCollector` and on-policy algorithms
(PPO, A2C) to pass collected experience through the training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RolloutBatch:
    """Flat batch of rollout data. All tensors have leading dim = ``n_envs * n_steps``.

    This is the standard data format passed between rollout collection and
    PPO/A2C training. Use :meth:`sample_minibatches` to iterate over
    shuffled minibatches during the optimisation phase.

    Attributes
    ----------
    obs : Tensor of shape (N, obs_dim)
        Observations at each timestep.
    actions : Tensor of shape (N,) or (N, act_dim)
        Actions taken.
    rewards : Tensor of shape (N,)
        Rewards received.
    dones : Tensor of shape (N,)
        Episode termination flags (1.0 = done).
    log_probs : Tensor of shape (N,)
        Log-probability of the taken action under the collection policy.
    values : Tensor of shape (N,)
        Value estimates V(s) from the critic at collection time.
    advantages : Tensor of shape (N,)
        GAE-computed advantages.
    returns : Tensor of shape (N,)
        Discounted returns (advantages + values).
    """

    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def sample_minibatches(self, batch_size: int, shuffle: bool = True):
        """Yield ``RolloutBatch`` minibatches for SGD updates.

        Parameters
        ----------
        batch_size : int
            Number of transitions per minibatch.
        shuffle : bool
            Whether to randomly permute indices (default True).

        Yields
        ------
        RolloutBatch
            A minibatch with ``batch_size`` transitions. Drops any remainder
            that doesn't fill a complete minibatch.
        """
        n = self.obs.shape[0]
        if shuffle:
            indices = torch.randperm(n)
        else:
            indices = torch.arange(n)

        n_complete = (n // batch_size) * batch_size
        for start in range(0, n_complete, batch_size):
            idx = indices[start : start + batch_size]
            yield RolloutBatch(
                obs=self.obs[idx],
                actions=self.actions[idx],
                rewards=self.rewards[idx],
                dones=self.dones[idx],
                log_probs=self.log_probs[idx],
                values=self.values[idx],
                advantages=self.advantages[idx],
                returns=self.returns[idx],
            )
