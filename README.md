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
| 2 | Experience Storage (Arrow columnar, ring buffer, VarLenStore) | Planned |
| 3 | Training Orchestrator (GAE, batch assembly, PPO) | Planned |
| 4 | LLM Post-Training (GRPO, DPO, token KL) | Planned |
| 5 | Polish & API (callbacks, logging, proptest) | Planned |

## Benchmark Results

All benchmarks run on the same machine with bootstrap 95% confidence intervals (10,000 resamples).
Speedup > 1.0x means rlox is faster. "Significant" = CI lower bound > 1.0.

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
- At small env counts (4), Rayon scheduling overhead exceeds CartPole compute (~37ns/step) — Gymnasium wins. This is expected and honest.
- Speedup scales with env count: **5.7x at 512 envs** vs Gymnasium Sync.
- Peak throughput: **2.1M env-steps/second** (512 envs).
- Bridge overhead (Python ↔ Rust): ~2.7 us per step.

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

# Benchmarks
.venv/bin/python benchmarks/run_all.py
```

## Project Layout

```
crates/
  rlox-core/       Pure Rust: envs, spaces, parallel stepping
  rlox-python/     PyO3 bindings
  rlox-bench/      Criterion benchmarks
python/
  rlox/            Python package (imports Rust via _rlox_core)
benchmarks/        Python benchmark suite with statistical framework
tests/python/      Python integration & benchmark TDD tests
scripts/           Test runner, README updater
docs/              PRD, feature spec, phase plans, benchmarking plan
```
