"""Reptile meta-learning across CartPole variants.

Trains a meta-policy using Reptile (Nichol et al., 2018) across multiple
CartPole configurations, then adapts quickly to a specific variant.
The outer-loop weight update uses Rust-accelerated reptile_update.
"""

from rlox.meta.reptile import Reptile

# Define task distribution: standard CartPole variants.
# In practice you would register custom envs with different physics params.
task_envs = [
    "CartPole-v1",
    "CartPole-v1",  # same env, different seeds create different trajectories
    "CartPole-v1",
]

# Create Reptile meta-learner
reptile = Reptile(
    algorithm_cls_name="ppo",
    env_ids=task_envs,
    meta_lr=0.1,
    inner_steps=2_000,
    inner_kwargs={"n_envs": 4},
)

# Meta-train for 5 outer iterations
print("Meta-training...")
meta_metrics = reptile.meta_train(n_iterations=5)
print(f"Meta-training mean reward: {meta_metrics['mean_reward']:.1f}")

# Adapt to a specific task with only 1000 steps
print("\nAdapting to CartPole-v1...")
adapted_algo = reptile.adapt("CartPole-v1", n_steps=1_000)
print("Adaptation complete.")
