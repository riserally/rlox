# rlox

Rust-accelerated reinforcement learning — the Polars architecture pattern applied to RL.

**Rust data plane + Python control plane**, connected via PyO3. Environments step in Rust with Rayon work-stealing parallelism; Python stays in charge of training logic.

## Architecture

```
┌─────────────────────────────────────────┐
│  Python (control plane)                 │
│  Training loops, Gymnasium bridge,      │
│  config, logging                        │
├────────── PyO3 boundary ────────────────┤
│  Rust (data plane)                      │
│  rlox-core: envs, parallel stepping,    │
│             buffers, GAE, GRPO          │
│  rlox-python: thin PyO3 bindings        │
└─────────────────────────────────────────┘
```

Two-crate workspace:
- **rlox-core** — pure Rust, no PyO3 dependency, testable independently
- **rlox-python** — thin PyO3 wrappers exposing `rlox-core` to Python

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Skeleton (workspace, PyO3, maturin) | Done |
| 1 | Environment Engine (CartPole, VecEnv, GymEnv bridge) | Done |
| 2 | Experience Storage (columnar buffer, ring buffer, VarLenStore) | Done |
| 3 | Training Core (GAE, KL controller) | Done |
| 4 | LLM Post-Training (GRPO, DPO, token KL) | Done |
| 5 | Polish & API (type stubs, proptest) | Done |
| 6 | Three-Framework Benchmark (rlox vs TorchRL vs SB3) | Done |

## Three-Framework Benchmark Results

All benchmarks run on Apple M4 with bootstrap 95% confidence intervals (10,000 resamples).
Speedup > 1.0x means rlox is faster. All results marked *** are statistically significant (CI lower bound > 1.0).

> **Full details**: [docs/benchmark/](docs/benchmark/) — includes [setup & methodology](docs/benchmark/setup.md), per-benchmark analysis, raw timing data, and reproducibility instructions.

### [GAE Computation](docs/benchmark/gae.md)

| Trajectory | rlox | NumPy Loop | TorchRL | vs NumPy | vs TorchRL |
|-----------|------|-----------|---------|----------|------------|
| 128 steps | 0.7 us | 34 us | 453 us | **51x** *** | **679x** *** |
| 2048 steps | 4.0 us | 558 us | 6798 us | **139x** *** | **1700x** *** |
| 32768 steps | 60 us | 8906 us | 108441 us | **147x** *** | **1791x** *** |

### [Buffer Operations](docs/benchmark/buffer-ops.md)

| Benchmark | rlox | TorchRL | SB3 | vs TorchRL | vs SB3 |
|-----------|------|---------|-----|------------|--------|
| Push 10K (obs=4) | 1.5 ms | 229 ms | 15 ms | **148x** *** | **9.7x** *** |
| Sample batch=32 | 1.5 us | 20 us | 18 us | **13x** *** | **11x** *** |
| Sample batch=1024 | 9.2 us | 96 us | 75 us | **10x** *** | **8.1x** *** |

### [End-to-End Rollout](docs/benchmark/e2e-rollout.md) (step + store + GAE)

| Config | rlox | SB3 | TorchRL | vs SB3 | vs TorchRL |
|--------|------|-----|---------|--------|------------|
| 16 envs × 128 steps | 6.1 ms | 10.2 ms | 129 ms | **1.7x** *** | **21x** *** |
| 64 envs × 512 steps | 44 ms | 135 ms | 1768 ms | **3.1x** *** | **41x** *** |
| 256 envs × 2048 steps | 539 ms | 2080 ms | 28432 ms | **3.9x** *** | **53x** *** |

### [LLM Operations](docs/benchmark/llm-ops.md) (vs NumPy / PyTorch)

| Benchmark | rlox | NumPy | PyTorch | vs NumPy | vs PyTorch |
|-----------|------|-------|---------|----------|------------|
| GRPO 256×16 | 36 us | 1252 us | 1241 us | **35x** *** | **34x** *** |
| Token KL 128 | 0.4 us | 1.7 us | 2.5 us | **4.0x** *** | **5.9x** *** |
| Token KL 8192 | 17 us | 28 us | 51 us | **1.6x** *** | **3.0x** *** |

### Env Stepping (from Phase 1)

<!-- BENCH:START -->
| Benchmark | rlox | Baseline | Speedup | Significant |
|-----------|------|----------|---------|-------------|
| single_step | 958 ns | 2041 ns (Gymnasium) | **2.1x** | Yes |
| vecenv_1 | 250.5 us | 530.2 us (Gymnasium Sync) | **2.1x** | Yes |
| vecenv_1 | 250.5 us | 1560.8 us (Gymnasium Async) | **6.2x** | Yes |
| vecenv_4 | 2769.2 us | 1336.9 us (Gymnasium Sync) | **0.5x** | No |
| vecenv_4 | 2769.2 us | 2831.6 us (Gymnasium Async) | **1.0x** | No |
| vecenv_16 | 3984.4 us | 4365.3 us (Gymnasium Sync) | **1.1x** | Yes |
| vecenv_16 | 3984.4 us | 7511.5 us (Gymnasium Async) | **1.9x** | Yes |
| vecenv_64 | 6595.6 us | 16963.6 us (Gymnasium Sync) | **2.6x** | Yes |
| vecenv_64 | 6595.6 us | 23450.5 us (Gymnasium Async) | **3.6x** | Yes |
| vecenv_128 | 8967.2 us | 34115.0 us (Gymnasium Sync) | **3.8x** | Yes |
| vecenv_128 | 8967.2 us | 50058.5 us (Gymnasium Async) | **5.6x** | Yes |
| vecenv_256 | 13407.6 us | 68815.0 us (Gymnasium Sync) | **5.1x** | Yes |
| vecenv_512 | 24365.9 us | 138668.1 us (Gymnasium Sync) | **5.7x** | Yes |
<!-- BENCH:END -->

**Key takeaways:**
- **GAE**: 140x faster than Python loops, 1700x faster than TorchRL. The sequential backward scan eliminates Python interpreter overhead entirely.
- **Buffer push**: 148x faster than TorchRL (per-item TensorDict overhead), 10x faster than SB3. For large observations (Atari-sized), memcpy dominates and the gap narrows.
- **Buffer sample**: 8-13x faster than both TorchRL and SB3. Pre-allocated ring buffer + ChaCha8 RNG with predictable latency (p99 < 15us even for batch=1024).
- **End-to-end rollout**: 3.9x faster than SB3, 53x faster than TorchRL at 256 envs × 2048 steps. Advantages compound across the pipeline.
- **GRPO advantages**: 34x faster than both NumPy and PyTorch — dominated by per-call overhead for small arrays.
- At small env counts (4), Rayon scheduling overhead exceeds CartPole compute (~37ns/step) — Gymnasium wins. This is expected and honest.

## Convergence Benchmarks (rlox vs SB3)

End-to-end training comparison: same hyperparameters (rl-zoo3 defaults), 5 seeds per experiment, bootstrap 95% CI. Measures both **wall-clock training speed** and **convergence to reward threshold**.

> **Full details**: [benchmarks/convergence/](benchmarks/convergence/) — configs, runners, raw JSON logs, and reproducibility instructions.

### Training Throughput (Steps Per Second)

![SPS Comparison](docs/benchmark/convergence/sps_comparison.png)

On-policy algorithms (PPO, A2C) using rlox's Rust GAE show **1.6-2.5x SPS improvements**. Off-policy algorithms (SAC, TD3) are bottlenecked by single-env gymnasium stepping and NN updates, showing ~1.1x — as expected, since PyTorch compute dominates.

### Learning Curves

**PPO on CartPole-v1** — rlox converges to same reward, 3.3x faster wall-clock:

![PPO CartPole](docs/benchmark/convergence/learning_curve_PPO_CartPole-v1.png)

**PPO on Acrobot-v1** — both converge to ~-83, rlox reaches threshold 1.4x faster:

![PPO Acrobot](docs/benchmark/convergence/learning_curve_PPO_Acrobot-v1.png)

**A2C on CartPole-v1** — matched convergence, rlox 2.5x faster throughput:

![A2C CartPole](docs/benchmark/convergence/learning_curve_A2C_CartPole-v1.png)

### Aggregate Results (Phase E1 — Classic Control)

| Algorithm | Environment | Framework | IQM Return | Steps to T | Wall-clock to T | SPS |
|-----------|-------------|-----------|------------|------------|----------------|-----|
| PPO | CartPole-v1 | **rlox** | **436.5** | 21,504 | **1.6s** | **9,121** |
| PPO | CartPole-v1 | SB3 | 465.8 | 33,800 | 5.2s | 4,026 |
| A2C | CartPole-v1 | **rlox** | 385.4 | 24,800 | **1.8s** | **10,445** |
| A2C | CartPole-v1 | SB3 | 401.9 | 12,800 | 2.1s | 4,206 |
| PPO | Acrobot-v1 | **rlox** | -83.6 | 58,163 | **6.4s** | **12,030** |
| PPO | Acrobot-v1 | SB3 | -81.4 | 26,600 | 9.1s | 7,727 |
| DQN | MountainCar-v0 | rlox | -200.0 | N/A | N/A | 2,698 |
| DQN | MountainCar-v0 | SB3 | -200.0 | 386,000 | 94.0s | 3,936 |

**Where rlox wins**: On-policy algorithms (PPO, A2C) where the Rust GAE computation and vectorized stepping deliver compounding speedups. Wall-clock improvement is **1.4-3.3x** — training reaches reward thresholds faster.

**Known limitations**: Off-policy algorithms (SAC, TD3, DQN) in the rlox Python layer have convergence issues under investigation. The Rust data-plane primitives (buffers, GAE) are correct — the bugs are in the Python training loop wiring. SB3's off-policy implementations converge correctly on Pendulum (SAC: -192, TD3: -188).

### Performance Profile (Agarwal et al., 2021)

![Performance Profile](docs/benchmark/convergence/performance_profile.png)

The performance profile aggregates across all environments. SB3 currently leads due to off-policy convergence gaps. On the on-policy subset (PPO, A2C), rlox matches SB3's convergence while training faster.

### Running the Benchmarks

```bash
# Phase E1: Classic Control (7 configs x 5 seeds x 2 frameworks = 70 runs)
cd benchmarks/convergence
python run_experiment.py --phase e1 --seeds 0-4

# Single experiment
python run_experiment.py configs/ppo_cartpole.yaml --seed 0 --framework rlox

# Analysis and plots
python analyze.py results/ --csv
python plot_learning_curves.py results/
python plot_profiles.py results/
```

## Quick Start

```bash
# Create venv and install
python3 -m venv .venv
source .venv/bin/activate
pip install maturin numpy gymnasium

# Build and install
maturin develop --release

# Verify
python -c "from rlox import CartPole; print('rlox ready')"
```

## Running Tests

```bash
# All tests (Rust + Python)
./scripts/test.sh

# Tests + benchmarks, updates README table
./scripts/test.sh --bench
```

Or manually:

```bash
# Rust only
cargo test --package rlox-core

# Python only (after maturin develop)
.venv/bin/python -m pytest tests/python/ -v

# Full benchmark suite (rlox vs TorchRL vs SB3)
.venv/bin/python benchmarks/run_all.py

# Individual benchmarks
.venv/bin/python benchmarks/bench_buffer_ops.py
.venv/bin/python benchmarks/bench_gae.py
.venv/bin/python benchmarks/bench_llm_ops.py
.venv/bin/python benchmarks/bench_e2e_rollout.py
```

## Project Layout

```
crates/
  rlox-core/       Pure Rust: envs, spaces, buffers, GAE, GRPO
  rlox-python/     PyO3 bindings
  rlox-bench/      Criterion benchmarks
python/
  rlox/            Python package (imports Rust via _rlox_core)
benchmarks/        Three-framework benchmark suite
tests/python/      Python integration & benchmark TDD tests
scripts/           Test runner, README updater
docs/benchmark/    Detailed benchmark results & methodology
docs/plans/        Phase-by-phase implementation plans
docs/              User guides, math reference, paper references
```

## Documentation

- [Getting Started](docs/getting-started.md) -- installation, first agent, tutorial
- [Python User Guide](docs/python-guide.md) -- Trainer, Algorithm, and Primitive APIs
- [Rust User Guide](docs/rust-guide.md) -- using `rlox-core` directly from Rust
- [Mathematical Reference](docs/math-reference.md) -- full derivations for all algorithms
- [References](docs/references.md) -- academic papers behind each algorithm

## Citing rlox

If you use rlox in your research, please cite it:

```bibtex
@software{rlox2026,
  title        = {rlox: Rust-Accelerated Reinforcement Learning},
  year         = {2026},
  url          = {https://github.com/wojciechkpl/rlox},
  version      = {1.0.0},
  note         = {Rust data plane + Python control plane via PyO3}
}
```
