"""Train PPO on a custom Gymnasium environment using rlox.

Shows how to wrap any gym-compatible environment and use it with the
unified Trainer API. The custom env here is a simple grid where the
agent moves left/right to reach a goal position.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlox import Trainer


class SimpleGridEnv(gym.Env):
    """1-D grid: agent starts at 0, goal is at position 9."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=9, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0=left, 1=right
        self.goal = 9
        self.pos = 0
        self.max_steps = 20
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = 0
        self.steps = 0
        return np.array([self.pos], dtype=np.float32), {}

    def step(self, action):
        self.pos = np.clip(self.pos + (1 if action == 1 else -1), 0, 9)
        self.steps += 1
        reward = 1.0 if self.pos == self.goal else -0.01
        terminated = self.pos == self.goal
        truncated = self.steps >= self.max_steps
        return np.array([self.pos], dtype=np.float32), reward, terminated, truncated, {}


# Register and train
gym.register(id="SimpleGrid-v0", entry_point=lambda: SimpleGridEnv())

trainer = Trainer("ppo", env="SimpleGrid-v0", seed=42, config={"n_envs": 4})
metrics = trainer.train(total_timesteps=20_000)
print(f"Mean reward: {metrics['mean_reward']:.2f}")
