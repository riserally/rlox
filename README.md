<p align="center">
  <img src="assets/rlox_logo.png" alt="rlox logo" width="500">
</p>

<h1 align="center">rlox</h1>

<p align="center">Rust-accelerated reinforcement learning — the Polars architecture pattern applied to RL.</p>

<p align="center">
  <a href="https://riserally.github.io/rlox/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-7c4dff.svg" alt="Documentation"></a>
  <a href="https://crates.io/crates/rlox-core"><img src="https://img.shields.io/crates/v/rlox-core.svg" alt="crates.io"></a>
  <a href="https://pypi.org/project/rlox/"><img src="https://img.shields.io/pypi/v/rlox.svg" alt="PyPI"></a>
  <a href="https://github.com/riserally/rlox/actions/workflows/ci.yml"><img src="https://github.com/riserally/rlox/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://deepwiki.com/riserally/rlox"><img src="https://img.shields.io/badge/DeepWiki-rlox-blue.svg" alt="DeepWiki"></a>
  <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg" alt="License"></a>
  <a href="https://pepy.tech/project/rlox"><img src="https://static.pepy.tech/badge/rlox" alt="Downloads"></a>
</p>

## Why rlox?

RL frameworks like Stable-Baselines3 and TorchRL do everything in Python — environment stepping, buffer storage, advantage computation. This works, but Python interpreter overhead becomes the bottleneck long before your GPU does.

rlox applies the **Polars architecture pattern** to RL: a **Rust data plane** handles the compute-heavy, latency-sensitive work (env stepping, buffers, GAE) while a **Python control plane** stays in charge of training logic, configs, and neural networks via PyTorch. The two connect through PyO3 with zero-copy where possible.

The result: **3-50x faster** than SB3/TorchRL on data-plane operations, with the same Python training API you're used to.

## Quick Start

### Prerequisites

- **Rust 1.75+** -- install via [rustup](https://rustup.rs/):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
- **Python 3.10-3.13**
- **Optional**: `pip install gymnasium[mujoco]` for MuJoCo environments
- **Optional**: `pip install pettingzoo` for multi-agent environments

### Installation

```bash
pip install rlox
```

Or build from source:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy gymnasium torch
maturin develop --release
```

**Train PPO on CartPole in 3 lines:**

```python
from rlox import Trainer

trainer = Trainer("ppo", env="CartPole-v1", seed=42)
metrics = trainer.train(total_timesteps=50_000)
print(f"Mean reward: {metrics['mean_reward']:.1f}")
```

**Train SAC on Pendulum:**

```python
from rlox import Trainer

trainer = Trainer("sac", env="Pendulum-v1", config={"learning_starts": 500})
metrics = trainer.train(total_timesteps=20_000)
```

> **Note:** Per-algorithm trainers (`PPOTrainer`, `SACTrainer`, etc.) are deprecated. Use the unified `Trainer("algo", ...)` API instead.

**Config-driven training (YAML):**

```bash
python -m rlox train --config config.yaml
```

```python
from rlox import TrainingConfig, train_from_config

config = TrainingConfig.from_yaml("config.yaml")
metrics = train_from_config(config)
```

**Use Rust primitives directly:**

```python
import rlox

# 140x faster GAE than Python loops
advantages, returns = rlox.compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95)

# 35x faster GRPO advantages
advantages = rlox.compute_batch_group_advantages(rewards, group_size=4)

# Parallel env stepping (2.7M steps/s at 512 envs)
env = rlox.VecEnv(n=256, seed=42, env_id="CartPole-v1")
result = env.step_all(actions)
```

> More examples in [`examples/`](examples/) — PPO, SAC, GRPO custom rewards, fast GAE, VecEnv throughput.

## Documentation

| Resource | Link |
|----------|------|
| **Full Documentation** | [riserally.github.io/rlox](https://riserally.github.io/rlox/) |
| Getting Started | [Tutorial](https://riserally.github.io/rlox/python/getting-started/) |
| Python API Guide | [User Guide](https://riserally.github.io/rlox/python/) |
| Examples | [Code Examples](https://riserally.github.io/rlox/python/examples/) |
| Rust API | [cargo doc](https://riserally.github.io/rlox/rust/rlox_core/) |
| Migrating from SB3 | [Migration Guide](https://riserally.github.io/rlox/python/tutorials/migration-sb3/) |
| API Reference | [Autodoc](https://riserally.github.io/rlox/python/api/) |

## Architecture

```
┌──────────────────────────────────────────────────┐
│  Python (control plane)                          │
│  PPO, SAC, DQN, TD3, A2C, MAPPO, DreamerV3,     │
│  IMPALA, GRPO, DPO                               │
│  GymVecEnv, VecNormalize, callbacks,             │
│  YAML/TOML configs, trainers, checkpointing,     │
│  diagnostics dashboard                           │
│  vLLM/TGI/SGLang backends, multi-GPU (DDP)       │
├────────────── PyO3 boundary ─────────────────────┤
│  Rust (data plane)                               │
│  rlox-core:   envs (CartPole, Pendulum),         │
│               Rayon parallel stepping,           │
│               buffers (ring, mmap, priority),    │
│               GAE, V-trace, GRPO, pipeline       │
│  rlox-nn:     RL algorithm traits                │
│  rlox-burn:   Burn backend (NdArray)             │
│  rlox-candle: Candle backend (CPU)               │
│  rlox-python: PyO3 bindings                      │
└──────────────────────────────────────────────────┘
```

Multi-crate workspace ([crates.io](https://crates.io/crates/rlox-core)):
- **rlox-core** — pure Rust: environments, buffers (ring, mmap, priority), GAE, V-trace, GRPO, pipeline
- **rlox-nn** — RL algorithm traits (`ActorCritic`, `QFunction`, `StochasticPolicy`, etc.)
- **rlox-burn** — Burn `Autodiff<NdArray>` implementations
- **rlox-candle** — Candle CPU implementations
- **rlox-python** — PyO3 bindings exposing `rlox-core` to Python

> For a deep-dive into the architecture, module relationships, and API reference, see the [DeepWiki](https://deepwiki.com/riserally/rlox).

## Benchmark Highlights

All benchmarks on Apple M4 with bootstrap 95% CI (10,000 resamples). All results statistically significant (CI lower bound > 1.0).

| Component | vs SB3 | vs TorchRL | Details |
|-----------|--------|------------|---------|
| GAE (32K steps) | 147x vs NumPy | **1,700x** | [docs/benchmark/gae.md](docs/benchmark/gae.md) |
| Buffer push (10K) | **9.7x** | **148x** | [docs/benchmark/buffer-ops.md](docs/benchmark/buffer-ops.md) |
| Buffer sample (1024) | **8.1x** | **10x** | [docs/benchmark/buffer-ops.md](docs/benchmark/buffer-ops.md) |
| E2E rollout (256x2048) | **3.9x** | **53x** | [docs/benchmark/e2e-rollout.md](docs/benchmark/e2e-rollout.md) |
| GRPO advantages | 35x vs NumPy | 34x vs PyTorch | [docs/benchmark/llm-ops.md](docs/benchmark/llm-ops.md) |
| Env stepping (512 envs) | -- | -- | [2.7M steps/s](docs/benchmark/env-stepping.md) |

> **Full methodology, raw data, and reproducibility instructions**: [docs/benchmark/](docs/benchmark/)

### Performance

Key numbers at a glance (Apple M4, single-threaded unless noted):

| Operation | rlox | SB3/NumPy | Speedup |
|-----------|------|-----------|---------|
| GAE (32K steps) | ~0.02 ms | ~3 ms | **147x** |
| Buffer push (10K) | ~0.1 ms | ~1 ms | **9.7x** |
| Buffer sample (1024) | ~0.03 ms | ~0.25 ms | **8.1x** |
| PPO SPS (8 envs, CartPole) | 9,121 | 4,026 | **2.3x** |
| A2C SPS (8 envs, CartPole) | 10,445 | 4,206 | **2.5x** |
| VecEnv throughput (512 envs) | 2.7M steps/s | -- | -- |
| GRPO advantages (batch) | ~0.01 ms | ~0.35 ms | **35x** |

### Convergence (rlox vs SB3)

Same hyperparameters (rl-zoo3 defaults), 5 seeds per experiment. On-policy algorithms (PPO, A2C) show **1.4-3.3x faster wall-clock** convergence with matching reward thresholds.

| Algorithm | Environment | rlox Wall-clock | SB3 Wall-clock | rlox SPS | SB3 SPS |
|-----------|-------------|-----------------|----------------|----------|---------|
| PPO | CartPole-v1 | **1.6s** | 5.2s | **9,121** | 4,026 |
| A2C | CartPole-v1 | **1.8s** | 2.1s | **10,445** | 4,206 |
| PPO | Acrobot-v1 | **6.4s** | 9.1s | **12,030** | 7,727 |

![SPS Comparison](docs/benchmark/convergence/sps_comparison.png)

> Full convergence results, learning curves, and performance profiles: [benchmarks/convergence/](benchmarks/convergence/)

## Features

- **22 Algorithms**: PPO, SAC, DQN, TD3, A2C, VPG, TRPO, MAPPO, DreamerV3, IMPALA, and more (+ GRPO, DPO for LLM)
- **Trainers**: Each algorithm has a high-level `Trainer` with `train()`, `save()`, `from_checkpoint()`, `predict()`
- **Environments**: Gymnasium-compatible, Rayon-parallel VecEnv, CartPole and Pendulum-v1 built-in
- **Visual RL wrappers**: `FrameStack`, `ImagePreprocess`, `AtariWrapper`, `DMControlWrapper` for pixel-based RL
- **Language RL wrappers**: `LanguageWrapper`, `GoalConditionedWrapper` for language-grounded tasks
- **Plugin ecosystem**: `ENV_REGISTRY`, `BUFFER_REGISTRY`, `REWARD_REGISTRY`, `discover_plugins` for extensibility
- **Model zoo**: `ModelZoo.register`, `ModelZoo.load` for sharing and reusing pretrained agents
- **VecNormalize**: Obs/reward normalization at the environment boundary (SB3-compatible)
- **Buffers**: ring, mmap, priority replay — all in Rust with zero-copy Python access
- **Config-driven training**: YAML/TOML configs via `TrainingConfig` and `python -m rlox train --config config.yaml`
- **Diagnostics dashboard**: `TerminalDashboard`, `HTMLReport`, entropy/KL/gradient monitoring
- **LLM post-training**: GRPO, DPO, token KL, sequence packing, vLLM/TGI/SGLang backends
- **Cloud deploy**: Dockerfile generator, Kubernetes manifest generator, SageMaker integration
- **Distributed**: pipeline parallelism (crossbeam), gRPC workers, multi-GPU (DDP)
- **Production**: callbacks, checkpointing, eval toolkit (IQM, bootstrap CI, performance profiles)
- **NN backends**: Burn (NdArray) and Candle (CPU) for pure-Rust inference, PyTorch for training
- **444 Rust tests, ~1094 Python tests** — comprehensive coverage

## Tutorials & Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | Installation, first training run, basic API |
| [Custom Rewards & Training Loops](docs/tutorials/custom-rewards-and-training-loops.md) | Reward shaping, GRPO reward functions, custom algorithms |
| [Python Guide](docs/python-guide.md) | Python API reference and patterns |
| [Rust Guide](docs/rust-guide.md) | Rust crate architecture and extending in Rust |
| [Math Reference](docs/math-reference.md) | GAE, V-trace, GRPO, DPO derivations |
| [Benchmark Details](docs/benchmark/) | Full methodology, per-benchmark analysis, reproducibility |
| [DeepWiki](https://deepwiki.com/riserally/rlox) | Auto-generated architecture docs and API reference |

## Running Tests

```bash
# Rust tests (444 tests across all crates)
cargo test --workspace

# Python tests (~1094 tests, after maturin develop)
pip install -e ".[all]"
pytest tests/python/ -q

# Quick smoke test (skip slow tests)
pytest tests/python/ -m "not slow" -q

# Single crate
cargo test --package rlox-core

# All tests (Rust + Python)
./scripts/test.sh

# Full benchmark suite (rlox vs TorchRL vs SB3)
.venv/bin/python benchmarks/run_all.py
```

## Project Layout

```
crates/
  rlox-core/       Pure Rust: envs, buffers (ring, mmap, priority), GAE,
                   V-trace, GRPO, pipeline (crossbeam), sequence packing
  rlox-nn/         RL algorithm traits (ActorCritic, QFunction, etc.)
  rlox-burn/       Burn backend (Autodiff<NdArray>)
  rlox-candle/     Candle backend (CPU)
  rlox-python/     PyO3 bindings
  rlox-bench/      Criterion benchmarks (env stepping, NN backends)
python/rlox/
  algorithms/      PPO, SAC, DQN, TD3, A2C, GRPO, DPO, MAPPO, DreamerV3, IMPALA
  distributed/     Pipeline, vLLM/TGI/SGLang backends, multi-GPU (DDP)
  llm/             LLM environment, reward model serving
  *.py             Collectors, configs, callbacks, policies, trainers,
                   evaluation toolkit, diagnostics, checkpointing
benchmarks/        Three-framework benchmark suite + convergence tests
tests/python/      Python integration & benchmark TDD tests
docs/              Guides, tutorials, benchmark methodology
```

## Citation

If you use rlox in your research, please cite:

```bibtex
@software{kowalinski2026rlox,
  author       = {Kowalinski, Wojciech},
  title        = {rlox: Rust-Accelerated Reinforcement Learning},
  year         = {2026},
  url          = {https://github.com/riserally/rlox},
  version      = {1.0.0},
  license      = {MIT OR Apache-2.0}
}
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE), at your option.
