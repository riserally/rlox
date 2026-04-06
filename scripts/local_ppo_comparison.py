#!/usr/bin/env python3
"""Run both PPO paths on HalfCheetah locally and compare.

Path A: benchmark runner (_run_ppo from rlox_runner.py)
Path B: Trainer API (Trainer("ppo", ...))

Short run (200K steps) to identify which path converges faster.
"""

import time
import numpy as np
import torch
import gymnasium as gym

TOTAL_STEPS = 200_000
ENV_ID = "HalfCheetah-v4"
SEED = 42

CONFIG = {
    "n_envs": 8,
    "n_steps": 2048,
    "n_epochs": 10,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}


def evaluate(policy_fn, env_id, n_episodes=20):
    """Evaluate with a function that maps obs → action."""
    env = gym.make(env_id)
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_r = 0
        done = False
        while not done:
            action = policy_fn(obs)
            obs, r, term, trunc, _ = env.step(action)
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


# ============================================================
# Path A: Trainer API WITH normalization
# ============================================================
print("=" * 60)
print("Path A: Trainer('ppo') WITH VecNormalize")
print("=" * 60)

from rlox import Trainer

t0 = time.time()
trainer_a = Trainer("ppo", env=ENV_ID, seed=SEED, config={
    **CONFIG,
    "normalize_obs": True,
    "normalize_rewards": True,
})
metrics_a = trainer_a.train(total_timesteps=TOTAL_STEPS)
elapsed_a = time.time() - t0

vn = getattr(trainer_a.algo, "vec_normalize", None)
def policy_fn_a(obs):
    if vn:
        obs = vn.normalize_obs(obs.reshape(1, -1)).flatten()
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return trainer_a.algo.policy.actor(obs_t).squeeze(0).numpy()

mean_a, std_a = evaluate(policy_fn_a, ENV_ID)
sps_a = TOTAL_STEPS / elapsed_a
print(f"  Return: {mean_a:.1f} +/- {std_a:.1f}")
print(f"  SPS: {sps_a:.0f}, Wall: {elapsed_a:.0f}s")


# ============================================================
# Path B: Trainer API WITHOUT normalization
# ============================================================
print()
print("=" * 60)
print("Path B: Trainer('ppo') WITHOUT VecNormalize")
print("=" * 60)

t0 = time.time()
trainer_b = Trainer("ppo", env=ENV_ID, seed=SEED, config={
    **CONFIG,
    "normalize_obs": False,
    "normalize_rewards": False,
})
metrics_b = trainer_b.train(total_timesteps=TOTAL_STEPS)
elapsed_b = time.time() - t0

def policy_fn_b(obs):
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return trainer_b.algo.policy.actor(obs_t).squeeze(0).numpy()

mean_b, std_b = evaluate(policy_fn_b, ENV_ID)
sps_b = TOTAL_STEPS / elapsed_b
print(f"  Return: {mean_b:.1f} +/- {std_b:.1f}")
print(f"  SPS: {sps_b:.0f}, Wall: {elapsed_b:.0f}s")


# ============================================================
# Path C: Trainer API with clip_obs=5.0
# ============================================================
print()
print("=" * 60)
print("Path C: Trainer('ppo') VecNormalize clip_obs=5.0")
print("=" * 60)

# Need to pass clip_obs through — check if PPOConfig supports it
from rlox.config import PPOConfig
from dataclasses import fields
has_clip_obs = "clip_obs" in {f.name for f in fields(PPOConfig)}
print(f"  PPOConfig has clip_obs: {has_clip_obs}")

if not has_clip_obs:
    # Manually create with custom clip_obs
    from rlox.algorithms.ppo import PPO
    from rlox.gym_vec_env import GymVecEnv
    from rlox.vec_normalize import VecNormalize as VN
    from rlox.collectors import RolloutCollector
    from rlox.utils import detect_env_spaces

    obs_dim, action_space, is_discrete = detect_env_spaces(ENV_ID)
    raw_env = GymVecEnv(ENV_ID, n_envs=8, seed=SEED)
    vec_norm = VN(raw_env, norm_obs=True, norm_reward=True, gamma=0.99,
                  clip_obs=5.0, obs_dim=obs_dim)

    ppo_c = PPO(env_id=ENV_ID, seed=SEED, **CONFIG)
    # Override the env
    ppo_c.vec_normalize = vec_norm
    ppo_c.collector = RolloutCollector(
        env_id=ENV_ID, n_envs=8, seed=SEED, gamma=0.99, gae_lambda=0.95,
        env=vec_norm,
    )

    t0 = time.time()
    metrics_c = ppo_c.train(total_timesteps=TOTAL_STEPS)
    elapsed_c = time.time() - t0

    def policy_fn_c(obs):
        obs_n = vec_norm.normalize_obs(obs.reshape(1, -1)).flatten()
        obs_t = torch.as_tensor(obs_n, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return ppo_c.policy.actor(obs_t).squeeze(0).numpy()

    mean_c, std_c = evaluate(policy_fn_c, ENV_ID)
    sps_c = TOTAL_STEPS / elapsed_c
    print(f"  Return: {mean_c:.1f} +/- {std_c:.1f}")
    print(f"  SPS: {sps_c:.0f}, Wall: {elapsed_c:.0f}s")
else:
    mean_c, std_c, sps_c = 0, 0, 0
    print("  Skipped — clip_obs in config, use config directly")


# ============================================================
# Summary
# ============================================================
print()
print("=" * 60)
print("SUMMARY (200K steps, seed=42)")
print("=" * 60)
print(f"  {'Config':30s} {'Return':>10s} {'SPS':>8s}")
print(f"  {'-'*30} {'-'*10} {'-'*8}")
print(f"  {'WITH VecNormalize':30s} {mean_a:>10.1f} {sps_a:>8.0f}")
print(f"  {'WITHOUT VecNormalize':30s} {mean_b:>10.1f} {sps_b:>8.0f}")
if mean_c != 0:
    print(f"  {'VecNorm clip_obs=5.0':30s} {mean_c:>10.1f} {sps_c:>8.0f}")
print()
print("If 'WITHOUT' >> 'WITH', VecNormalize is the problem.")
print("If both are similar, the gap is seed variance.")
