# Introduction

Welcome to the **rlox Guide** -- the documentation for rlox, a high-performance reinforcement learning framework with a Rust core and Python interface.

rlox follows the Polars architecture pattern: Rust handles the performance-critical data plane (buffers, GAE computation, environment stepping), while Python serves as the researcher-facing control plane with full PyTorch and JAX interoperability. It supports both classic RL (MuJoCo, Atari, custom environments) and LLM post-training workflows (RLHF, DPO, GRPO).

## What's in this guide

- **[Getting Started](getting-started.md)** -- Install rlox, train your first agent, and understand the core architecture.
- **[Python Guide](python-guide.md)** -- Three levels of Python API, from one-liner training to full custom loops.
- **[Rust Guide](rust-guide.md)** -- Using `rlox-core` directly in Rust applications without Python.
- **[Tutorials](tutorials/custom-rewards-and-training-loops.md)** -- Hands-on guides for custom reward functions and training loops.
- **[Architecture](architecture/polars-pattern.md)** -- The Polars-inspired design and feature roadmap.
- **[Math Reference](math-reference.md)** -- Full mathematical formulations for every algorithm in rlox.
- **[Benchmarks](benchmark/README.md)** -- Performance comparisons against TorchRL, Stable-Baselines3, and TRL.
- **[Research Notes](research/README.md)** -- Survey of the top 10 RL algorithms (2026) that rlox implements.
- **[References](references.md)** -- Academic papers that rlox implements or builds upon.
