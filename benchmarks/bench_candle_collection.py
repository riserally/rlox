"""Benchmark: Candle hybrid collection vs PyTorch collection.

Compares wall-clock time for PPO training on CartPole using:
1. Standard PPO (PyTorch inference + Python env stepping)
2. HybridPPO (Candle inference + Rust VecEnv, zero Python in collection)

Usage:
    python benchmarks/bench_candle_collection.py [--timesteps N]
"""

import argparse
import time

import numpy as np


def bench_standard_ppo(total_timesteps: int, n_envs: int, hidden: int, seed: int):
    """Standard PPO with PyTorch collection."""
    from rlox.algorithms.ppo import PPO

    start = time.perf_counter()
    ppo = PPO(
        env_id="CartPole-v1",
        n_envs=n_envs,
        seed=seed,
        n_steps=128,
        n_epochs=4,
        learning_rate=2.5e-4,
        batch_size=256,
    )
    metrics = ppo.train(total_timesteps=total_timesteps)
    elapsed = time.perf_counter() - start

    sps = total_timesteps / elapsed
    return {
        "method": "Standard PPO (PyTorch)",
        "elapsed_s": elapsed,
        "sps": sps,
        "mean_reward": metrics.get("mean_reward", 0),
    }


def bench_hybrid_ppo(total_timesteps: int, n_envs: int, hidden: int, seed: int):
    """Hybrid PPO with Candle collection."""
    from rlox.algorithms.hybrid_ppo import HybridPPO

    start = time.perf_counter()
    ppo = HybridPPO(
        env_id="CartPole-v1",
        n_envs=n_envs,
        seed=seed,
        hidden=hidden,
        n_steps=128,
        n_epochs=4,
        learning_rate=2.5e-4,
        batch_size=256,
    )
    metrics = ppo.train(total_timesteps=total_timesteps)
    elapsed = time.perf_counter() - start

    timing = ppo.timing_summary()
    sps = total_timesteps / elapsed
    return {
        "method": "Hybrid PPO (Candle)",
        "elapsed_s": elapsed,
        "sps": sps,
        "mean_reward": metrics.get("mean_reward", 0),
        "collection_pct": timing["collection_pct"],
        "training_pct": timing["training_pct"],
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Candle vs PyTorch collection")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Benchmarking PPO on CartPole-v1")
    print(f"  timesteps={args.timesteps}, n_envs={args.n_envs}, hidden={args.hidden}")
    print()

    # Warmup
    print("Warming up...")
    bench_hybrid_ppo(2048, args.n_envs, args.hidden, args.seed)

    print()
    print("=" * 60)

    # Hybrid PPO
    print("Running Hybrid PPO (Candle)...")
    hybrid = bench_hybrid_ppo(args.timesteps, args.n_envs, args.hidden, args.seed)
    print(f"  Time: {hybrid['elapsed_s']:.2f}s")
    print(f"  SPS:  {hybrid['sps']:.0f}")
    print(f"  Reward: {hybrid['mean_reward']:.1f}")
    print(f"  Collection: {hybrid.get('collection_pct', 0):.1f}%")
    print(f"  Training:   {hybrid.get('training_pct', 0):.1f}%")

    print()

    # Standard PPO
    print("Running Standard PPO (PyTorch)...")
    standard = bench_standard_ppo(args.timesteps, args.n_envs, args.hidden, args.seed)
    print(f"  Time: {standard['elapsed_s']:.2f}s")
    print(f"  SPS:  {standard['sps']:.0f}")
    print(f"  Reward: {standard['mean_reward']:.1f}")

    print()
    print("=" * 60)
    speedup = standard["elapsed_s"] / hybrid["elapsed_s"]
    print(f"SPEEDUP: {speedup:.2f}x")
    print(f"  Hybrid:   {hybrid['sps']:.0f} SPS")
    print(f"  Standard: {standard['sps']:.0f} SPS")


if __name__ == "__main__":
    main()
