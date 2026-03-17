"""Train SAC on Pendulum-v1 (continuous control)."""

from rlox.trainers import SACTrainer

trainer = SACTrainer(env="Pendulum-v1", config={"learning_starts": 500})
metrics = trainer.train(total_timesteps=20_000)
print(f"Mean reward: {metrics['mean_reward']:.1f}")
