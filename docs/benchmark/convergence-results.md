# Convergence Benchmark Results

!!! note "27/32 Experiments Complete — v6 Re-benchmark Pending"
    Benchmark v5 ran on GCP with 27 of 32 planned experiments completed.
    Missing: TD3 Hopper-v4, TD3 Walker2d-v4, SAC Walker2d-v4 (SB3), A2C Acrobot-v1, DQN Acrobot-v1.
    Six convergence bugs were identified and fixed in v0.3.0/v1.0.0 (see Known Issues below).
    A v6 re-benchmark will validate these fixes with multi-seed runs and IQM statistics.

## Methodology

- **Frameworks:** rlox v0.2.3 vs Stable-Baselines3 (v6 will use rlox v1.0.0)
- **Hardware:** e2-standard-8 (8 vCPU, 32GB RAM), CPU-only
- **Environments:** CartPole-v1, Pendulum-v1, HalfCheetah-v4, Hopper-v4, Walker2d-v4, Acrobot-v1, MountainCar-v0
- **Algorithms:** PPO, SAC, TD3, DQN, A2C
- **Evaluation:** 30 episodes every 10K steps, deterministic policy
- **Seeds:** Single seed (multi-seed planned for next run)

## Convergence Results

| Algorithm | Environment | Steps | rlox Return | SB3 Return | Winner |
|-----------|-------------|-------|-------------|------------|--------|
| PPO | CartPole-v1 | 100K | 453.9 | 420.9 | rlox |
| PPO | Acrobot-v1 | 500K | **-88.5** | -118.1 | rlox |
| PPO | HalfCheetah-v4 | 1M | **4225.6** | 3142.5 | rlox |
| PPO | Hopper-v4 | 1M | 628.1 | **3577.5** | SB3 |
| PPO | Walker2d-v4 | 2M | **5007.4** | 4384.3 | rlox |
| SAC | Pendulum-v1 | 50K | -168.5 | **-167.1** | Tie |
| SAC | HalfCheetah-v4 | 1M | **11468.1** | 10562.7 | rlox |
| SAC | Hopper-v4 | 1M | **3290.6** | 3170.2 | rlox |
| SAC | Walker2d-v4 | 2M | **4978.0** | -- | rlox* |
| TD3 | Pendulum-v1 | 50K | -162.7 | **-169.4** | Tie |
| TD3 | HalfCheetah-v4 | 1M | **10400.4** | 9899.3 | rlox |
| DQN | CartPole-v1 | 100K | 164.8 | **195.8** | SB3 |
| DQN | MountainCar-v0 | 500K | -178.7 | **-109.5** | SB3 |
| A2C | CartPole-v1 | 100K | 53.8 | **500.0** | SB3 |

\* SB3 experiment not yet completed for this pair.

## Speed Comparison

| Algorithm | Environment | rlox SPS | SB3 SPS | Speedup |
|-----------|-------------|----------|---------|---------|
| PPO | CartPole-v1 | 1,691 | 687 | **2.46x** |
| PPO | Acrobot-v1 | 2,520 | 1,306 | **1.93x** |
| PPO | HalfCheetah-v4 | 800 | 437 | **1.83x** |
| PPO | Hopper-v4 | 1,237 | 770 | **1.61x** |
| PPO | Walker2d-v4 | 931 | 762 | **1.22x** |
| SAC | Pendulum-v1 | 46 | 42 | **1.11x** |
| SAC | HalfCheetah-v4 | 42 | 63 | 0.68x |
| SAC | Hopper-v4 | 77 | 66 | **1.18x** |
| SAC | Walker2d-v4 | 75 | -- | -- |
| TD3 | Pendulum-v1 | 76 | 65 | **1.17x** |
| TD3 | HalfCheetah-v4 | 117 | 101 | **1.16x** |
| DQN | CartPole-v1 | 462 | 642 | 0.72x |
| DQN | MountainCar-v0 | 479 | 634 | 0.76x |
| A2C | CartPole-v1 | 2,028 | 489 | **4.15x** |

## Known Issues (Fixed in v0.3.0 / v1.0.0)

All six convergence bugs identified during v5 benchmarking have been fixed. The v6 re-benchmark will validate these fixes.

| Bug | Fix (v0.3.0) | Affected Results |
|-----|-------------|------------------|
| Truncation bootstrap missing | V(terminal_obs) bootstrap for truncated episodes | PPO Hopper (628 vs 3577) |
| Scalar obs normalization | Per-dimension Welford stats via `RunningStatsVec` | All MuJoCo envs |
| Raw reward normalization | Return-based std (SB3 convention) | All normalized envs |
| Train/collect obs mismatch | Consistent normalization via `VecNormalize` wrapper | All normalized envs |
| A2C advantage normalization | Default changed to False for small batches | A2C CartPole (54 vs 500) |
| log_std init = -0.5 | Changed to 0.0 (std=1.0, matching SB3) | All continuous envs |

### Pre-fix notes (v5 results above)

- **PPO Hopper gap (628 vs 3577):** Truncation bootstrap + normalization bugs. Fixed.
- **A2C CartPole instability (54 vs 500):** Advantage normalization default. Fixed.
- **DQN underperformance:** DQN results lag behind SB3 on both CartPole and MountainCar; under investigation.

## Candle Hybrid Collection Benchmark

Measured on Apple M-series, CartPole-v1, PPO (n_steps=128, n_epochs=4, hidden=64):

| n_envs | Hybrid SPS | Standard SPS | Speedup | Collection % |
|--------|-----------|-------------|---------|-------------|
| 4 | 32,460 | 18,779 | **1.73x** | 45.6% |
| 8 | 40,020 | 23,037 | **1.74x** | 41.2% |
| 16 | 47,863 | 32,204 | **1.49x** | 30.7% |
| 32 | 53,721 | 42,748 | **1.26x** | 23.4% |

The speedup is strongest at lower env counts (4-8 envs: 1.7x) where per-step
Python dispatch overhead (~113us) dominates. With more envs, PyTorch's BLAS
amortizes the overhead, narrowing the gap.

The Candle hybrid approach eliminates Python dispatch overhead during collection,
shifting the bottleneck entirely to the PyTorch training backward pass.

!!! info
    A v6 re-benchmark is planned using rlox v1.0.0 with all six convergence fixes applied.
    This will include multi-seed runs (5 seeds) with IQM statistics and learning curve plots.
    Results will be uploaded to `gs://rkox-bench-results/convergence-*/`.
