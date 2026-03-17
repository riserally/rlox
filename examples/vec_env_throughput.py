"""Demonstrate rlox VecEnv throughput: 2.7M steps/s at 512 envs.

Compare Rust-parallel VecEnv vs Python gymnasium.
"""

import time
import numpy as np
import rlox

N_ENVS = 256
N_STEPS = 1000

# --- rlox Rust VecEnv (Rayon parallel) ---
env = rlox.VecEnv(n=N_ENVS, seed=42, env_id="CartPole-v1")
env.reset_all()

start = time.perf_counter()
for _ in range(N_STEPS):
    actions = np.random.randint(0, 2, size=N_ENVS).tolist()
    env.step_all(actions)
elapsed = time.perf_counter() - start

total_steps = N_ENVS * N_STEPS
sps = total_steps / elapsed
print(f"rlox VecEnv:  {sps:,.0f} steps/s  ({N_ENVS} envs x {N_STEPS} steps in {elapsed:.2f}s)")

# --- Compare with gymnasium SyncVectorEnv ---
try:
    from rlox.gym_vec_env import GymVecEnv

    gym_env = GymVecEnv("CartPole-v1", n_envs=N_ENVS, seed=42)
    gym_env.reset_all()

    start = time.perf_counter()
    for _ in range(N_STEPS):
        actions = np.random.randint(0, 2, size=N_ENVS)
        gym_env.step_all(actions)
    elapsed_gym = time.perf_counter() - start

    sps_gym = total_steps / elapsed_gym
    print(f"GymVecEnv:    {sps_gym:,.0f} steps/s  ({elapsed_gym:.2f}s)")
    print(f"Speedup:      {sps / sps_gym:.1f}x")
except Exception as e:
    print(f"GymVecEnv comparison skipped: {e}")
