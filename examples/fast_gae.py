"""Drop-in replacement for NumPy GAE — 142x faster.

Use rlox.compute_gae() anywhere you compute advantages.
Works with any framework (SB3, CleanRL, custom loops).
"""

import numpy as np
import rlox

# Simulate a rollout (2048 steps)
n_steps = 2048
rewards = np.random.randn(n_steps).astype(np.float64)
values = np.random.randn(n_steps).astype(np.float64)
dones = (np.random.rand(n_steps) < 0.01).astype(np.float64)  # ~1% episode ends
last_value = 0.0

# Compute GAE in Rust — 142x faster than a Python loop
advantages, returns = rlox.compute_gae(
    rewards=rewards,
    values=values,
    dones=dones,
    last_value=last_value,
    gamma=0.99,
    lam=0.95,
)

print(f"Advantages: mean={advantages.mean():.3f}, std={np.std(advantages):.3f}")
print(f"Returns:    mean={returns.mean():.3f}, std={np.std(returns):.3f}")

# Also works for LLM post-training (GRPO advantages — 35x faster)
group_rewards = np.array([1.0, 0.5, 0.8, 0.2, 0.9, 0.3], dtype=np.float64)
group_advantages = rlox.compute_batch_group_advantages(group_rewards, group_size=3)
print(f"\nGRPO advantages: {group_advantages}")
