#!/usr/bin/env python3
"""Compare the two PPO code paths side-by-side.

Path A: benchmark runner (rlox_runner.py) — gets 4,226 on HalfCheetah
Path B: Trainer API (PPO class) — gets 1,526 on HalfCheetah

This script traces both paths and prints every parameter that differs.
"""

import numpy as np
import torch

print("=" * 70)
print("PATH A: Benchmark Runner (rlox_runner.py)")
print("=" * 70)

# Simulate what the benchmark runner does
yaml_config = {
    "n_envs": 8,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "normalize_obs": True,
    "normalize_reward": True,  # Note: no 's'
}

print("Config from YAML:")
for k, v in sorted(yaml_config.items()):
    print(f"  {k:25s} = {v!r}")

print()
print("Network: ContinuousPolicy(obs_dim=17, act_dim=6, hidden=64)")
print("  activation: Tanh")
print("  init: orthogonal(sqrt(2)) hidden, orthogonal(0.01) actor, orthogonal(1.0) critic")
print("  log_std: init=0.0 (learnable)")
print()
print("Optimizer: Adam(lr=3e-4, eps=1e-5)")
print("LR annealing: linear from 3e-4 to 0")
print("Advantage normalization: per-minibatch")
print("Value loss: clipped (clip_range=0.2)")

print()
print("=" * 70)
print("PATH B: Trainer API (PPO class)")
print("=" * 70)

from rlox.config import PPOConfig
from dataclasses import fields

# Simulate what Trainer passes to PPO
trainer_config = {
    "n_envs": 8,
    "n_steps": 2048,
    "n_epochs": 10,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "normalize_obs": True,
    "normalize_rewards": True,
    "ent_coef": 0.0,
}

cfg_fields = {f.name for f in fields(PPOConfig)}
accepted = {k: v for k, v in trainer_config.items() if k in cfg_fields}
ppo_config = PPOConfig(**accepted)

print("Config received by PPO:")
for f in sorted(fields(PPOConfig), key=lambda x: x.name):
    val = getattr(ppo_config, f.name)
    yaml_val = yaml_config.get(f.name, "N/A")
    match = "✓" if str(val) == str(yaml_val) or f.name not in yaml_config else "✗"
    print(f"  {match} {f.name:25s} = {val!r:15s}  (yaml: {yaml_val!r})")

print()

# Check network
from rlox.policies import ContinuousPolicy
policy = ContinuousPolicy(obs_dim=17, act_dim=6)
print("Network: ContinuousPolicy(obs_dim=17, act_dim=6)")
print(f"  hidden: {policy.actor[0].out_features if hasattr(policy, 'actor') else '?'}")
print(f"  log_std init: {policy.log_std.data[0].item():.3f}")

print()
print("=" * 70)
print("DIFFERENCES:")
print("=" * 70)

diffs = []
# Check each YAML param vs PPOConfig
for k, yaml_v in yaml_config.items():
    if k == "normalize_reward":
        # Check if normalize_rewards (with s) matches
        ppo_v = ppo_config.normalize_rewards
        if yaml_v != ppo_v:
            diffs.append(f"  YAML 'normalize_reward'={yaml_v} vs PPOConfig 'normalize_rewards'={ppo_v}")
    elif hasattr(ppo_config, k):
        ppo_v = getattr(ppo_config, k)
        if yaml_v != ppo_v:
            diffs.append(f"  '{k}': yaml={yaml_v} vs ppo={ppo_v}")

# Check defaults for params NOT in the v8 config
for f in fields(PPOConfig):
    if f.name not in trainer_config and f.name != "n_envs":
        yaml_val = yaml_config.get(f.name, None)
        ppo_val = getattr(ppo_config, f.name)
        if yaml_val is not None and yaml_val != ppo_val:
            diffs.append(f"  '{f.name}': yaml={yaml_val} vs ppo_default={ppo_val}")

if diffs:
    for d in diffs:
        print(d)
else:
    print("  No differences found — configs are identical!")

print()
print("=" * 70)
print("RECOMMENDATION:")
print("=" * 70)
print("If configs are identical, the gap is in:")
print("  1. Seed / weight initialization differences")
print("  2. GAE computation (batched Rust vs per-env)")
print("  3. Minibatch shuffling order")
print("  4. Numerical precision in normalization stats")
print("  5. Single-seed variance (run 5+ seeds)")
