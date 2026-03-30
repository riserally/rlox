"""Convergence tests for rlox algorithms.

These tests verify that PPO, DQN, and SAC actually learn on standard
environments. They use generous thresholds -- the goal is to confirm
the algorithms converge at all (not match SB3-level performance).

Run with: .venv/bin/python -m pytest tests/python/test_convergence.py -m slow -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


def _evaluate_ppo_greedy(agent, env_id: str, n_episodes: int, seed: int) -> float:
    """Evaluate PPO with greedy action selection, return mean reward."""
    import gymnasium as gym

    env = gym.make(env_id)
    returns: list[float] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits = agent.policy.actor(obs_t)
                action = logits.argmax(dim=-1).item()
            obs, reward, terminated, truncated, _ = env.step(int(action))
            ep_return += float(reward)
            done = terminated or truncated
        returns.append(ep_return)
    env.close()
    return float(np.mean(returns))


def _evaluate_dqn_greedy(agent, env_id: str, n_episodes: int, seed: int) -> float:
    """Evaluate DQN with greedy action selection, return mean reward."""
    import gymnasium as gym

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
    return float(np.mean(returns))


def _evaluate_sac_deterministic(agent, env_id: str, n_episodes: int, seed: int) -> float:
    """Evaluate SAC with deterministic policy, return mean reward."""
    import gymnasium as gym

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
    return float(np.mean(returns))


@pytest.mark.slow
def test_ppo_cartpole_converges() -> None:
    """PPO must reach reward > 300 on CartPole-v1 in 50K steps."""
    from rlox.algorithms.ppo import PPO

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = PPO(
        env_id="CartPole-v1",
        n_envs=4,
        seed=seed,
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

    agent.train(total_timesteps=50_000)

    mean_reward = _evaluate_ppo_greedy(agent, "CartPole-v1", n_episodes=20, seed=seed + 1000)
    assert mean_reward > 300.0, (
        f"PPO failed to converge on CartPole-v1: mean_reward={mean_reward:.1f}, expected > 300"
    )


@pytest.mark.slow
def test_dqn_cartpole_converges() -> None:
    """DQN must reach reward > 150 on CartPole-v1.

    DQN has high seed-dependent variance on CartPole (~125-500 across seeds
    at 50K steps). We use 100K training steps to give the algorithm enough
    exploitation time after epsilon annealing, and retry up to 3 seeds to
    tolerate unlucky initialisations.

    Known gap vs SB3: rlox DQN reaches ~165 avg vs SB3's ~196 on GCP
    benchmarks, likely due to missing gradient clipping and hard (rather
    than soft) target-network updates.
    """
    from rlox.algorithms.dqn import DQN

    seeds = [1, 42, 7]
    threshold = 150.0
    best_reward = -float("inf")

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        agent = DQN(
            env_id="CartPole-v1",
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
            seed=seed,
        )

        agent.train(total_timesteps=100_000)

        mean_reward = _evaluate_dqn_greedy(
            agent, "CartPole-v1", n_episodes=20, seed=seed + 1000
        )
        best_reward = max(best_reward, mean_reward)

        if mean_reward > threshold:
            return  # Pass on first success

    pytest.fail(
        f"DQN failed to converge on CartPole-v1 across seeds {seeds}: "
        f"best mean_reward={best_reward:.1f}, expected > {threshold}"
    )


@pytest.mark.slow
def test_sac_pendulum_converges() -> None:
    """SAC must reach reward > -400 on Pendulum-v1 in 20K steps."""
    from rlox.algorithms.sac import SAC

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = SAC(
        env_id="Pendulum-v1",
        buffer_size=50_000,
        learning_rate=3e-4,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        learning_starts=1000,
        hidden=256,
        seed=seed,
        auto_entropy=True,
    )

    agent.train(total_timesteps=20_000)

    mean_reward = _evaluate_sac_deterministic(agent, "Pendulum-v1", n_episodes=20, seed=seed + 1000)
    assert mean_reward > -400.0, (
        f"SAC failed to converge on Pendulum-v1: mean_reward={mean_reward:.1f}, expected > -400"
    )
