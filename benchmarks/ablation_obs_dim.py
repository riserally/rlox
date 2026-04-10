#!/usr/bin/env python3
"""Ablation: buffer speedup vs observation dimension.

Measures replay buffer push + sample latency for Rust vs Python at
varying obs_dim to find the crossover where memcpy dominates.

Usage:
    python benchmarks/ablation_obs_dim.py --seeds 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def bench_rust_buffer(obs_dim, n_push=10_000, batch_size=256, n_reps=30):
    """Measure Rust ReplayBuffer push + sample latency."""
    import rlox
    buf = rlox.ReplayBuffer(n_push, obs_dim, 1)
    obs = np.random.randn(obs_dim).astype(np.float32)
    act = np.array([0.0], dtype=np.float32)
    nobs = np.random.randn(obs_dim).astype(np.float32)

    # Push
    t0 = time.perf_counter_ns()
    for _ in range(n_push):
        buf.push(obs, act, 1.0, False, False, nobs)
    push_ns = (time.perf_counter_ns() - t0)

    # Sample
    times = []
    for rep in range(n_reps):
        t0 = time.perf_counter_ns()
        buf.sample(batch_size, rep)
        times.append(time.perf_counter_ns() - t0)

    return push_ns, float(np.median(times))


def bench_python_buffer(obs_dim, n_push=10_000, batch_size=256, n_reps=30):
    """Measure a naive Python replay buffer push + sample latency."""
    obs_buf = np.zeros((n_push, obs_dim), dtype=np.float32)
    act_buf = np.zeros((n_push, 1), dtype=np.float32)
    rew_buf = np.zeros(n_push, dtype=np.float32)
    done_buf = np.zeros(n_push, dtype=bool)
    nobs_buf = np.zeros((n_push, obs_dim), dtype=np.float32)
    obs = np.random.randn(obs_dim).astype(np.float32)
    nobs = np.random.randn(obs_dim).astype(np.float32)

    # Push
    t0 = time.perf_counter_ns()
    for i in range(n_push):
        obs_buf[i] = obs
        act_buf[i] = 0.0
        rew_buf[i] = 1.0
        done_buf[i] = False
        nobs_buf[i] = nobs
    push_ns = (time.perf_counter_ns() - t0)

    # Sample
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter_ns()
        idx = np.random.randint(0, n_push, size=batch_size)
        _ = obs_buf[idx], act_buf[idx], rew_buf[idx], done_buf[idx], nobs_buf[idx]
        times.append(time.perf_counter_ns() - t0)

    return push_ns, float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description="Obs-dim scaling ablation")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", default="results/ablation")
    args = parser.parse_args()

    obs_dims = [4, 17, 64, 256, 1024, 4096]

    print(f"Obs-dim scaling: buffer push + sample, {args.seeds} seeds")
    print("=" * 70)

    results = []
    for od in obs_dims:
        rust_push, rust_sample = [], []
        py_push, py_sample = [], []
        for i in range(args.seeds):
            rp, rs = bench_rust_buffer(od)
            pp, ps = bench_python_buffer(od)
            rust_push.append(rp); rust_sample.append(rs)
            py_push.append(pp); py_sample.append(ps)

        push_ratio = float(np.mean(py_push) / np.mean(rust_push))
        sample_ratio = float(np.mean(py_sample) / np.mean(rust_sample))

        results.append({
            "obs_dim": od,
            "push_speedup": push_ratio,
            "sample_speedup": sample_ratio,
            "rust_push_ms": float(np.mean(rust_push)) / 1e6,
            "python_push_ms": float(np.mean(py_push)) / 1e6,
            "rust_sample_us": float(np.mean(rust_sample)) / 1e3,
            "python_sample_us": float(np.mean(py_sample)) / 1e3,
        })

        print(f"  obs_dim={od:>6d}  push={push_ratio:>5.1f}x  "
              f"sample={sample_ratio:>5.1f}x")

    print()
    print("=" * 70)
    print(f"{'obs_dim':>8s} {'push':>8s} {'sample':>8s}")
    for r in results:
        print(f"{r['obs_dim']:>8d} {r['push_speedup']:>7.1f}x "
              f"{r['sample_speedup']:>7.1f}x")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "ablation_obs_dim.json").open("w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nSaved to {out / 'ablation_obs_dim.json'}")


if __name__ == "__main__":
    main()
