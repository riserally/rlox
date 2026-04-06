#!/usr/bin/env python3
"""Debug PPO config flow: trace exactly what parameters reach PPO.__init__."""

from rlox.config import PPOConfig
from dataclasses import fields

print("=" * 60)
print("PPOConfig field names:")
print("=" * 60)
for f in sorted(fields(PPOConfig), key=lambda x: x.name):
    print(f"  {f.name:25s} default={f.default!r}")

print()

# Simulate what v8 script passes
v8_config = {
    "n_envs": 8,
    "n_steps": 2048,
    "n_epochs": 10,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "normalize_obs": True,
    "normalize_rewards": True,
    "ent_coef": 0.0,
}

print("=" * 60)
print("v8 config dict:")
print("=" * 60)
for k, v in v8_config.items():
    print(f"  {k:25s} = {v!r}")

print()

# Check which keys are accepted vs dropped
cfg_fields = {f.name for f in fields(PPOConfig)}
accepted = {k: v for k, v in v8_config.items() if k in cfg_fields}
dropped = {k: v for k, v in v8_config.items() if k not in cfg_fields}

print("=" * 60)
print("ACCEPTED (in PPOConfig):")
print("=" * 60)
for k, v in accepted.items():
    print(f"  {k:25s} = {v!r}")

print()
print("=" * 60)
print("DROPPED (not in PPOConfig, SILENTLY LOST):")
print("=" * 60)
for k, v in dropped.items():
    print(f"  {k:25s} = {v!r}  ← THIS IS LOST!")

print()

# Create the actual config
cfg = PPOConfig(**accepted)
print("=" * 60)
print("Resulting PPOConfig:")
print("=" * 60)
print(f"  normalize_obs     = {cfg.normalize_obs}")
print(f"  normalize_rewards = {cfg.normalize_rewards}")
print(f"  ent_coef          = {cfg.ent_coef}")
print(f"  n_steps           = {cfg.n_steps}")
print(f"  n_epochs          = {cfg.n_epochs}")
print(f"  batch_size        = {cfg.batch_size}")
print(f"  learning_rate     = {cfg.learning_rate}")

print()
print("=" * 60)
print("YAML config key comparison:")
print("=" * 60)
yaml_keys = {"normalize_obs": True, "normalize_reward": True}
for k, v in yaml_keys.items():
    in_config = k in cfg_fields
    print(f"  YAML key '{k}': {'ACCEPTED' if in_config else 'DROPPED (wrong name!)'}")
    if not in_config:
        # Find similar
        import difflib
        matches = difflib.get_close_matches(k, cfg_fields, n=1, cutoff=0.7)
        if matches:
            print(f"    → Did you mean: '{matches[0]}'?")
