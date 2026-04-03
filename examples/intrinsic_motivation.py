"""RND intrinsic motivation with PPO on MountainCar.

Uses Random Network Distillation (Burda et al., 2019) to generate
intrinsic reward for exploration. MountainCar's sparse reward makes
it a good test case -- the agent rarely reaches the goal without
exploration bonuses.
"""

import torch

from rlox import Trainer
from rlox.intrinsic.rnd import RND

# MountainCar-v0: obs_dim=2 (position, velocity), discrete actions (3)
obs_dim = 2
rnd = RND(obs_dim=obs_dim, hidden=128, output_dim=32, learning_rate=1e-3)


def intrinsic_reward_fn(obs_batch):
    """Compute RND intrinsic reward and update the predictor."""
    obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32)
    intrinsic = rnd.compute_intrinsic_reward(obs_tensor)
    rnd.update(obs_tensor)
    return intrinsic.numpy()


# Train PPO with RND intrinsic motivation
trainer = Trainer(
    "ppo",
    env="MountainCar-v0",
    seed=42,
    config={
        "n_envs": 8,
        "intrinsic_reward_fn": intrinsic_reward_fn,
        "intrinsic_reward_coef": 0.1,  # scale intrinsic vs extrinsic
    },
)

metrics = trainer.train(total_timesteps=50_000)
print(f"Mean reward: {metrics['mean_reward']:.1f}")
