"""Train SAC on Pendulum-v1."""

from rlox import Trainer

trainer = Trainer("sac", env="Pendulum-v1", seed=42, config={
    "learning_starts": 500,
})
metrics = trainer.train(total_timesteps=20_000)
print(f"Mean reward: {metrics['mean_reward']:.1f}")
