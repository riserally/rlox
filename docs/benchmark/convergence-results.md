# Convergence Benchmark Results

!!! note "In Progress"
    Benchmark v5 is currently running on GCP (26/32 experiments complete).
    This page will be populated with full results upon completion.

## Methodology

- **Frameworks:** rlox v0.2.3 vs Stable-Baselines3
- **Hardware:** e2-standard-8 (8 vCPU, 32GB RAM), CPU-only
- **Environments:** CartPole-v1, Pendulum-v1, HalfCheetah-v4, Hopper-v4, Walker2d-v4
- **Algorithms:** PPO, SAC, TD3, DQN
- **Evaluation:** 30 episodes every 10K steps, deterministic policy
- **Seeds:** Single seed (multi-seed planned for next run)

## Preliminary Results (In Progress)

| Algorithm | Environment | rlox Return | SB3 Return | Steps |
|-----------|-------------|-------------|------------|-------|
| PPO | CartPole-v1 | ~500 | ~500 | 100K |
| PPO | Hopper-v4 | ~1390 | ~1390 | 1M |
| SAC | Hopper-v4 | ~3250 | TBD | 1M |
| SAC | Walker2d-v4 | ~5005 | TBD | 2M |
| TD3 | HalfCheetah-v4 | TBD | ~9000 | 1M |
| DQN | CartPole-v1 | ~500 | ~500 | 100K |

## Speed Comparison

| Algorithm | Environment | rlox SPS | SB3 SPS | Speedup |
|-----------|-------------|----------|---------|---------|
| PPO | CartPole-v1 | TBD | TBD | TBD |
| SAC | Hopper-v4 | ~82 | ~100 | TBD |
| TD3 | HalfCheetah-v4 | TBD | ~101 | TBD |

## Candle Hybrid Collection Benchmark

Measured on Apple M-series, CartPole-v1, 16 envs, 200K timesteps:

| Method | SPS | Collection % | Training % | Speedup |
|--------|-----|-------------|------------|---------|
| Standard PPO (PyTorch) | 31,849 | ~50% | ~50% | 1.0x |
| **Hybrid PPO (Candle)** | **48,243** | **28.5%** | **71.5%** | **1.51x** |

The Candle hybrid approach eliminates Python dispatch overhead during collection,
shifting the bottleneck entirely to the PyTorch training backward pass.

!!! info
    Full convergence results with learning curves, IQM statistics, and multi-seed
    confidence intervals will be published after benchmark completion.
    Results will be uploaded to `gs://rkox-bench-results/convergence-*/`.
