"""DrQ-v2 random shift augmentation backed by Rust.

Wraps ``rlox.random_shift_batch`` for use with PyTorch tensors,
handling torch <-> numpy conversion transparently.
"""

from __future__ import annotations

import numpy as np
import torch

import rlox._rlox_core as _core


class RandomShift:
    """DrQ-v2 random shift augmentation.

    Pads the image with zeros, then randomly crops back to the original
    size -- effectively translating the image by up to ``pad`` pixels.

    Parameters
    ----------
    pad : int
        Number of zero-padding pixels on each side (default 4).
    """

    def __init__(self, pad: int = 4) -> None:
        self.pad = pad

    def __call__(self, obs: torch.Tensor, seed: int) -> torch.Tensor:
        """Augment a batch of image observations.

        Parameters
        ----------
        obs : (B, C, H, W) float32 tensor
        seed : RNG seed for reproducibility

        Returns
        -------
        augmented : (B, C, H, W) float32 tensor
        """
        B, C, H, W = obs.shape
        device = obs.device

        # Move to CPU for numpy conversion if needed
        arr = obs.detach().cpu().numpy().astype(np.float32)
        flat = arr.ravel()

        result = _core.random_shift_batch(flat, B, C, H, W, self.pad, seed)

        out = torch.from_numpy(result.reshape(B, C, H, W)).to(device)
        return out
