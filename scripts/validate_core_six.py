#!/usr/bin/env python3
"""Validate all 6 core algorithms converge on their canonical environments.

This is the definitive local test before claiming convergence.
"""

import time
import numpy as np
import torch
import gymnasium as gym

from rlox import Trainer


def evaluate(trainer, env_id, n_episodes=20):
    """Evaluate a trained agent."""
    env = gym.make(env_id)
    vn = getattr(trainer.algo, "vec_normalize", None)
    is_discrete = hasattr(env.action_space, "n")
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_r = 0
        done = False
        while not done:
            if vn:
                obs = vn.normalize_obs(obs.reshape(1, -1)).flatten()
            if hasattr(trainer.algo, "predict"):
                action = trainer.algo.predict(
                    np.array(obs, dtype=np.float32),
                    deterministic=True,
                )
                # Ensure action is the right type for the env
                if hasattr(action, "numpy"):
                    action = action.squeeze().numpy()
                elif hasattr(action, "item") and is_discrete:
                    action = action.item()
                action = np.atleast_1d(np.asarray(action, dtype=np.float32 if not is_discrete else np.int64))
                if is_discrete:
                    action = int(action[0])
                elif action.ndim == 0:
                    action = action.reshape(1)
            elif is_discrete:
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action = trainer.algo.policy.actor(obs_t).argmax(-1).item()
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action = trainer.algo.policy.actor(obs_t).squeeze(0).numpy()
            obs, r, term, trunc, _ = env.step(action)
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


EXPERIMENTS = [
    # (algo, env, steps, config, min_threshold)
    # Thresholds set conservatively for 50K/20K step budget (single seed)
    ("ppo", "CartPole-v1", 50_000, {"n_envs": 8, "n_steps": 128}, 150),
    ("a2c", "CartPole-v1", 50_000, {"n_envs": 8, "n_steps": 5}, 100),
    ("trpo", "CartPole-v1", 50_000, {}, 150),
    ("dqn", "CartPole-v1", 50_000, {"learning_starts": 500}, 100),
    ("sac", "Pendulum-v1", 20_000, {"learning_starts": 500}, -500),
    ("td3", "Pendulum-v1", 20_000, {"learning_starts": 500}, -500),
]

print("=" * 70)
print("CORE SIX VALIDATION")
print("=" * 70)

results = []
all_pass = True

for algo, env_id, steps, config, threshold in EXPERIMENTS:
    print(f"\n--- {algo.upper()} on {env_id} ({steps} steps) ---")
    t0 = time.time()
    trainer = Trainer(algo, env=env_id, seed=42, config=config)
    metrics = trainer.train(total_timesteps=steps)
    elapsed = time.time() - t0

    mean_r, std_r = evaluate(trainer, env_id)
    sps = steps / elapsed
    passed = mean_r > threshold
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False

    print(f"  [{status}] return={mean_r:.1f} +/- {std_r:.1f} (threshold: {threshold})")
    print(f"  SPS={sps:.0f}, wall={elapsed:.1f}s")

    results.append({
        "algo": algo, "env": env_id, "return": mean_r,
        "threshold": threshold, "passed": passed, "sps": sps,
    })

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  {'Algo':6s} {'Env':15s} {'Return':>10s} {'Threshold':>10s} {'Status':>8s} {'SPS':>8s}")
print(f"  {'-'*6} {'-'*15} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
for r in results:
    status = "PASS" if r["passed"] else "FAIL"
    print(f"  {r['algo']:6s} {r['env']:15s} {r['return']:>10.1f} {r['threshold']:>10} {status:>8s} {r['sps']:>8.0f}")

print(f"\n  Result: {'ALL PASS' if all_pass else 'SOME FAILED'} ({sum(r['passed'] for r in results)}/{len(results)})")
