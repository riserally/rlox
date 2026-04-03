"""Potential-based reward shaping (PBRS) with PPO on CartPole.

Demonstrates using rlox.PotentialShaping to add a shaping bonus that
encourages the pole to stay upright. PBRS preserves optimal policy
invariance (Ng et al., 1999) while accelerating early learning.
"""

import numpy as np

from rlox import Trainer
from rlox.reward_shaping import PotentialShaping


# Define a potential function: higher potential when pole angle is small.
# CartPole obs: [x, x_dot, theta, theta_dot]
# Pole angle (theta) is obs[2]; upright = 0 radians.
def pole_potential(obs: np.ndarray) -> np.ndarray:
    """Potential: negative absolute pole angle (upright = 0 = max potential)."""
    theta = obs[:, 2]  # (N,)
    return -np.abs(theta)


# Create the shaping wrapper
shaper = PotentialShaping(potential_fn=pole_potential, gamma=0.99)

# Train PPO with shaped rewards via a callback
trainer = Trainer(
    "ppo",
    env="CartPole-v1",
    seed=42,
    config={
        "n_envs": 8,
        "reward_shaping_fn": shaper.shape,  # plug into the training loop
    },
)
metrics = trainer.train(total_timesteps=30_000)
print(f"Mean reward with PBRS: {metrics['mean_reward']:.1f}")
