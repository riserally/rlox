#!/usr/bin/env python3
"""Investigate PPO Hopper performance gap between benchmark runner and Trainer.

v6 benchmark runner: 3,342 (using _run_ppo + _collect_rollout_gym)
v8 Trainer API: 2,374 (using Trainer("ppo") → PPO → RolloutCollector)

Both use VecNormalize, ent_coef=0.0, same hyperparameters.
The gap must be in subtle differences in execution.

This script runs both paths on Hopper for 200K steps and compares.
"""

import time
import numpy as np
import torch
import gymnasium as gym

ENV_ID = "Hopper-v4"
SEED = 42
STEPS = 200_000

HP = {
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
    env = gym.make(env_id)
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_r = 0; done = False
        while not done:
            action = policy_fn(obs)
            obs, r, term, trunc, _ = env.step(action)
            ep_r += r; done = term or trunc
        rewards.append(ep_r)
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


# ============================================================
# Path 1: Trainer API (same as v8)
# ============================================================
print("=" * 60)
print(f"Path 1: Trainer API — {ENV_ID}, {STEPS} steps")
print("=" * 60)

from rlox import Trainer

t0 = time.time()
trainer = Trainer("ppo", env=ENV_ID, seed=SEED, config={
    **HP, "normalize_obs": True, "normalize_rewards": True,
})
trainer.train(total_timesteps=STEPS)
elapsed1 = time.time() - t0

vn1 = getattr(trainer.algo, "vec_normalize", None)
def policy_fn_1(obs):
    if vn1: obs = vn1.normalize_obs(obs.reshape(1, -1)).flatten()
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return trainer.algo.policy.actor(obs_t).squeeze(0).numpy()

mean1, std1 = evaluate(policy_fn_1, ENV_ID)
print(f"  Return: {mean1:.1f} +/- {std1:.1f}, SPS={STEPS/elapsed1:.0f}")


# ============================================================
# Path 2: Trainer API WITHOUT normalization
# ============================================================
print()
print("=" * 60)
print(f"Path 2: Trainer API NO normalization — {ENV_ID}, {STEPS} steps")
print("=" * 60)

t0 = time.time()
trainer2 = Trainer("ppo", env=ENV_ID, seed=SEED, config={
    **HP, "normalize_obs": False, "normalize_rewards": False,
})
trainer2.train(total_timesteps=STEPS)
elapsed2 = time.time() - t0

def policy_fn_2(obs):
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return trainer2.algo.policy.actor(obs_t).squeeze(0).numpy()

mean2, std2 = evaluate(policy_fn_2, ENV_ID)
print(f"  Return: {mean2:.1f} +/- {std2:.1f}, SPS={STEPS/elapsed2:.0f}")


# ============================================================
# Path 3: Trainer API with norm_obs ONLY (no reward norm)
# ============================================================
print()
print("=" * 60)
print(f"Path 3: Trainer API obs-norm only — {ENV_ID}, {STEPS} steps")
print("=" * 60)

t0 = time.time()
trainer3 = Trainer("ppo", env=ENV_ID, seed=SEED, config={
    **HP, "normalize_obs": True, "normalize_rewards": False,
})
trainer3.train(total_timesteps=STEPS)
elapsed3 = time.time() - t0

vn3 = getattr(trainer3.algo, "vec_normalize", None)
def policy_fn_3(obs):
    if vn3: obs = vn3.normalize_obs(obs.reshape(1, -1)).flatten()
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return trainer3.algo.policy.actor(obs_t).squeeze(0).numpy()

mean3, std3 = evaluate(policy_fn_3, ENV_ID)
print(f"  Return: {mean3:.1f} +/- {std3:.1f}, SPS={STEPS/elapsed3:.0f}")


# ============================================================
# Summary
# ============================================================
print()
print("=" * 60)
print("SUMMARY (Hopper, 200K steps, seed=42)")
print("=" * 60)
print(f"  {'Config':30s} {'Return':>10s} {'SPS':>8s}")
print(f"  {'-'*30} {'-'*10} {'-'*8}")
print(f"  {'norm_obs + norm_reward':30s} {mean1:>10.1f} {STEPS/elapsed1:>8.0f}")
print(f"  {'no normalization':30s} {mean2:>10.1f} {STEPS/elapsed2:>8.0f}")
print(f"  {'norm_obs only':30s} {mean3:>10.1f} {STEPS/elapsed3:>8.0f}")
print()
print("  SB3 reference (1M steps): 3,578")
print("  v6 benchmark runner (1M steps): 3,342")
print()
if mean2 > mean1:
    print("  → Normalization HURTS Hopper at 200K steps")
else:
    print("  → Normalization HELPS Hopper at 200K steps")
