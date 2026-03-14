"""RolloutBatch: flat-tensor container for on-policy rollout data."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RolloutBatch:
    """Flat batch of rollout data.  All tensors have leading dim = n_envs * n_steps."""

    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def sample_minibatches(self, batch_size: int, shuffle: bool = True):
        """Yield ``RolloutBatch`` minibatches.

        Drops any remainder that doesn't fill a complete minibatch.
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
