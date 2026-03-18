"""Convergence verification for off-policy algorithm fixes.

Trains PPO (sanity check), DQN, and SAC on standard environments and verifies
they reach minimum reward thresholds. This confirms the off-policy bug fixes
(next_obs handling, action scaling, n-step flushing) actually produce learning.

Usage:
    .venv/bin/python benchmarks/convergence/verify_convergence.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class ConvergenceSpec:
    """Specification for a single convergence test."""

    algorithm: str
    env_id: str
    total_timesteps: int
    reward_threshold: float
    n_eval_episodes: int = 20
    seed: int = 1


def evaluate_dqn(agent, env_id: str, n_episodes: int, seed: int) -> list[float]:
    """Run greedy evaluation episodes for a DQN agent."""
    env = gym.make(env_id)
    returns: list[float] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                q_values = agent.q_network(obs_t)
                action = int(q_values.argmax(dim=-1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            done = terminated or truncated
        returns.append(ep_return)
    env.close()
    return returns


def evaluate_sac(agent, env_id: str, n_episodes: int, seed: int) -> list[float]:
    """Run deterministic evaluation episodes for a SAC agent."""
    env = gym.make(env_id)
    returns: list[float] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = agent.actor.deterministic(obs_t).squeeze(0).numpy()
                action = action * agent.act_high
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            done = terminated or truncated
        returns.append(ep_return)
    env.close()
    return returns


def evaluate_ppo(agent, env_id: str, n_episodes: int, seed: int) -> list[float]:
    """Run greedy evaluation episodes for a PPO agent."""
    env = gym.make(env_id)
    returns: list[float] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                dist, _ = agent.policy(obs_t)
                action = dist.probs.argmax(dim=-1).item()
            obs, reward, terminated, truncated, _ = env.step(int(action))
            ep_return += float(reward)
            done = terminated or truncated
        returns.append(ep_return)
    env.close()
    return returns


def train_and_evaluate_ppo(spec: ConvergenceSpec) -> tuple[float, float, list[float]]:
    """Train PPO on CartPole-v1 and return (mean_reward, std_reward, episode_rewards)."""
    from rlox.algorithms.ppo import PPO

    print(f"\n{'='*60}")
    print(f"PPO on {spec.env_id} ({spec.total_timesteps} steps, seed={spec.seed})")
    print(f"{'='*60}")

    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)

    agent = PPO(
        env_id=spec.env_id,
        n_envs=4,
        seed=spec.seed,
        n_steps=128,
        n_epochs=4,
        batch_size=64,
        learning_rate=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
    )

    t0 = time.perf_counter()
    metrics = agent.train(total_timesteps=spec.total_timesteps)
    elapsed = time.perf_counter() - t0
    print(f"  Training took {elapsed:.1f}s")
    print(f"  Training mean_reward: {metrics.get('mean_reward', 'N/A')}")

    # Evaluation
    eval_returns = evaluate_ppo(agent, spec.env_id, spec.n_eval_episodes, seed=spec.seed + 1000)
    mean_r = float(np.mean(eval_returns))
    std_r = float(np.std(eval_returns))
    print(f"  Eval ({spec.n_eval_episodes} episodes): {mean_r:.1f} +/- {std_r:.1f}")

    return mean_r, std_r, eval_returns


def train_and_evaluate_dqn(spec: ConvergenceSpec) -> tuple[float, float, list[float]]:
    """Train DQN on CartPole-v1 and return (mean_reward, std_reward, episode_rewards)."""
    from rlox.algorithms.dqn import DQN

    print(f"\n{'='*60}")
    print(f"DQN on {spec.env_id} ({spec.total_timesteps} steps, seed={spec.seed})")
    print(f"{'='*60}")

    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)

    agent = DQN(
        env_id=spec.env_id,
        buffer_size=50_000,
        learning_rate=1e-3,
        batch_size=64,
        gamma=0.99,
        target_update_freq=500,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        learning_starts=1000,
        double_dqn=True,
        dueling=False,
        n_step=1,
        hidden=128,
        seed=spec.seed,
    )

    t0 = time.perf_counter()

    # Print training episode rewards periodically
    obs, _ = agent.env.reset(seed=spec.seed)
    episode_rewards: list[float] = []
    ep_reward = 0.0

    for step in range(spec.total_timesteps):
        eps = agent._get_epsilon(step, spec.total_timesteps)

        if np.random.random() < eps or step < agent.learning_starts:
            action = int(agent.env.action_space.sample())
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                q_values = agent.q_network(obs_t)
                action = int(q_values.argmax(dim=-1).item())

        next_obs, reward, terminated, truncated, info = agent.env.step(action)
        ep_reward += float(reward)

        agent._store_transition(obs, action, reward, next_obs, terminated, truncated)

        obs = next_obs
        if terminated or truncated:
            # Flush n-step buffer
            while agent._n_step_buffer:
                R = 0.0
                for i in reversed(range(len(agent._n_step_buffer))):
                    _, _, r, _, done, trunc = agent._n_step_buffer[i]
                    R = r + agent.gamma * R * (1.0 - float(done or trunc))
                first_obs_b, first_action_b, _, _, _, _ = agent._n_step_buffer[0]
                _, _, _, last_next_obs_b, last_done_b, last_trunc_b = agent._n_step_buffer[-1]
                agent.buffer.push(
                    np.asarray(first_obs_b, dtype=np.float32),
                    np.array([float(first_action_b)], dtype=np.float32),
                    float(R),
                    bool(last_done_b),
                    bool(last_trunc_b),
                    np.asarray(last_next_obs_b, dtype=np.float32),
                )
                agent._n_step_buffer.pop(0)

            episode_rewards.append(ep_reward)
            if len(episode_rewards) % 10 == 0:
                recent = episode_rewards[-10:]
                print(f"  Episode {len(episode_rewards):4d} | "
                      f"reward={ep_reward:6.1f} | "
                      f"last-10 mean={np.mean(recent):6.1f} | "
                      f"eps={eps:.3f}")
            ep_reward = 0.0
            obs, _ = agent.env.reset()

        # Update
        if step >= agent.learning_starts and len(agent.buffer) >= agent.batch_size:
            agent._update(step, spec.total_timesteps)

        # Target network update
        if step % agent.target_update_freq == 0:
            agent.target_network.load_state_dict(agent.q_network.state_dict())

    elapsed = time.perf_counter() - t0
    print(f"  Training took {elapsed:.1f}s ({len(episode_rewards)} episodes)")

    # Evaluation
    eval_returns = evaluate_dqn(agent, spec.env_id, spec.n_eval_episodes, seed=spec.seed + 1000)
    mean_r = float(np.mean(eval_returns))
    std_r = float(np.std(eval_returns))
    print(f"  Eval ({spec.n_eval_episodes} episodes): {mean_r:.1f} +/- {std_r:.1f}")

    return mean_r, std_r, eval_returns


def train_and_evaluate_sac(spec: ConvergenceSpec) -> tuple[float, float, list[float]]:
    """Train SAC on Pendulum-v1 and return (mean_reward, std_reward, episode_rewards)."""
    from rlox.algorithms.sac import SAC

    print(f"\n{'='*60}")
    print(f"SAC on {spec.env_id} ({spec.total_timesteps} steps, seed={spec.seed})")
    print(f"{'='*60}")

    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)

    agent = SAC(
        env_id=spec.env_id,
        buffer_size=50_000,
        learning_rate=3e-4,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        learning_starts=1000,
        hidden=256,
        seed=spec.seed,
        auto_entropy=True,
    )

    t0 = time.perf_counter()

    # Manual training loop to print episode rewards
    obs, _ = agent.env.reset(seed=spec.seed)
    episode_rewards: list[float] = []
    ep_reward = 0.0

    for step in range(spec.total_timesteps):
        if step < agent.learning_starts:
            action = agent.env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action_t, _ = agent.actor.sample(obs_t)
                action = action_t.squeeze(0).numpy()
            action = action * agent.act_high

        next_obs, reward, terminated, truncated, info = agent.env.step(action)
        ep_reward += float(reward)

        agent.buffer.push(
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            np.asarray(next_obs, dtype=np.float32),
        )

        obs = next_obs
        if terminated or truncated:
            episode_rewards.append(ep_reward)
            if len(episode_rewards) % 10 == 0:
                recent = episode_rewards[-10:]
                print(f"  Episode {len(episode_rewards):4d} | "
                      f"reward={ep_reward:7.1f} | "
                      f"last-10 mean={np.mean(recent):7.1f}")
            ep_reward = 0.0
            obs, _ = agent.env.reset()

        # Update
        if step >= agent.learning_starts and len(agent.buffer) >= agent.batch_size:
            agent._update(step)

    elapsed = time.perf_counter() - t0
    print(f"  Training took {elapsed:.1f}s ({len(episode_rewards)} episodes)")

    # Evaluation
    eval_returns = evaluate_sac(agent, spec.env_id, spec.n_eval_episodes, seed=spec.seed + 1000)
    mean_r = float(np.mean(eval_returns))
    std_r = float(np.std(eval_returns))
    print(f"  Eval ({spec.n_eval_episodes} episodes): {mean_r:.1f} +/- {std_r:.1f}")

    return mean_r, std_r, eval_returns


def run_all() -> bool:
    """Run all convergence checks. Returns True if all pass."""
    specs = [
        ConvergenceSpec(
            algorithm="PPO",
            env_id="CartPole-v1",
            total_timesteps=50_000,
            reward_threshold=400.0,
            seed=1,
        ),
        ConvergenceSpec(
            algorithm="DQN",
            env_id="CartPole-v1",
            total_timesteps=50_000,
            reward_threshold=200.0,
            seed=1,
        ),
        ConvergenceSpec(
            algorithm="SAC",
            env_id="Pendulum-v1",
            total_timesteps=20_000,
            reward_threshold=-300.0,
            seed=1,
        ),
    ]

    train_fns = {
        "PPO": train_and_evaluate_ppo,
        "DQN": train_and_evaluate_dqn,
        "SAC": train_and_evaluate_sac,
    }

    results: list[tuple[str, float, float, bool]] = []

    for spec in specs:
        fn = train_fns[spec.algorithm]
        mean_r, std_r, _ = fn(spec)

        # For Pendulum (negative rewards), ">" means "less negative"
        passed = mean_r >= spec.reward_threshold
        results.append((f"{spec.algorithm}/{spec.env_id}", mean_r, spec.reward_threshold, passed))

    # Summary
    print(f"\n{'='*60}")
    print("CONVERGENCE VERIFICATION SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, mean_r, threshold, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name:<25s}  mean={mean_r:8.1f}  threshold={threshold:8.1f}")
        if not passed:
            all_pass = False

    print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return all_pass


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
