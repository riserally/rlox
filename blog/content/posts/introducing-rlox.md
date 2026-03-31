---
title: "Introducing rlox: Rust-Accelerated Reinforcement Learning"
date: 2026-03-18
description: "Today we're open-sourcing rlox, a reinforcement learning framework that applies the Polars architecture pattern to RL."
author: "Wojciech Kowalinski"
tags: ["release", "rust", "reinforcement-learning", "benchmarks"]
ShowToc: true
TocOpen: true
---

Today we're open-sourcing **rlox**, a reinforcement learning framework that applies the [Polars architecture pattern](https://pola.rs/) to RL: a Rust data plane for the heavy lifting, a Python control plane for everything else.

## The Problem

If you've trained RL agents with Stable-Baselines3 or TorchRL, you've probably noticed something frustrating: your GPU sits idle while Python loops through environment steps, shuffles replay buffers, and computes advantages. The GIL turns embarrassingly parallel work into a serial bottleneck.

This isn't a Python problem per se — it's an architecture problem. Polars solved the same issue for DataFrames by pushing compute-intensive operations into Rust while keeping the user-facing API in Python. We asked: can the same pattern work for RL?

## The Polars Pattern

Before diving into rlox's architecture, here's the pattern it borrows from. Polars doesn't try to make Python faster — it moves the work out of Python entirely:

<div class="mermaid">
graph LR
    subgraph Traditional["Traditional RL (SB3 / TorchRL)"]
        direction TB
        P1[Python: env.step]
        P2[Python: buffer.push]
        P3[Python: compute GAE]
        P4[Python: sample batch]
        P5[Python: optimizer.step]
        P1 --> P2 --> P3 --> P4 --> P5
    end

    subgraph Polars["rlox (Polars Pattern)"]
        direction TB
        R1[Rust: env.step ∥ Rayon]
        R2[Rust: buffer.push zero-copy]
        R3[Rust: compute GAE]
        R4[Rust: sample batch]
        PY[Python: optimizer.step]
        R1 --> R2 --> R3 --> R4 --> PY
    end
</div>

Python only runs where it adds value: neural network training via PyTorch. Everything else is Rust.

## The Architecture

The full system has three layers connected by PyO3:

<div class="mermaid">
graph TB
    subgraph Python["Python Control Plane"]
        API[Researcher API<br/>train / evaluate / sweep]
        Torch[PyTorch<br/>Autograd & Models]
        HF[HuggingFace<br/>Transformers & Datasets]
        WB[W&B / MLflow<br/>Logging]
    end

    subgraph Rust["Rust Data Plane (rlox-core)"]
        ENV[Environment Engine<br/>parallel stepping via Rayon]
        BUF[Experience Store<br/>ring, mmap, priority buffers]
        LOOP[Training Orchestrator<br/>GAE, V-trace, GRPO, batching]
        SER[Serialization<br/>zero-copy Arrow/numpy]
        DIST[Distribution Layer<br/>gRPC workers, pipeline]
    end

    subgraph Envs["Environment Backends"]
        GYM[Gymnasium<br/>via PyO3 bridge]
        LLM_ENV[LLM Generation<br/>vLLM / TGI / SGLang]
        CUSTOM[Custom Rust Envs<br/>CartPole built-in]
    end

    API -->|PyO3 FFI| ENV
    API -->|PyO3 FFI| BUF
    API -->|PyO3 FFI| LOOP
    Torch <-->|zero-copy tensors| SER
    HF <-->|tokenized batches| SER
    ENV --> GYM
    ENV --> LLM_ENV
    ENV --> CUSTOM
    ENV -->|transitions| BUF
    BUF -->|batches| LOOP
    LOOP -->|grads request| Torch
    LOOP <-->|distributed sync| DIST
</div>

The boundary is deliberate. Everything above the line is where researchers spend their time — algorithm logic, hyperparameter tuning, experiment configs. Everything below is plumbing that should be fast and invisible. PyO3 connects the two with zero-copy where possible.

### Data Flow: One Training Step

Here's what happens during a single PPO training iteration:

<div class="mermaid">
sequenceDiagram
    participant P as Python (PPOTrainer)
    participant R as Rust (rlox-core)
    participant E as Environments (Rayon)
    participant T as PyTorch

    P->>R: collect_rollout(policy)
    R->>E: step_all(actions) [parallel]
    E-->>R: obs, rewards, dones
    R->>R: buffer.push(transitions)
    R->>R: compute_gae(rewards, values)
    R-->>P: RolloutBatch (zero-copy)
    P->>T: forward + backward pass
    T-->>P: gradients
    P->>P: optimizer.step()
    P->>P: log metrics, callbacks
</div>

The critical insight: Rust handles steps 2-6 (the data plane) as a single fused operation. There's no Python interpreter overhead between env stepping, buffer storage, and advantage computation — it's one Rust call that returns a ready-to-train batch.

### Crate Architecture

The Rust side is organized as a multi-crate workspace, each with a single responsibility:

<div class="mermaid">
graph TB
    subgraph Workspace["rlox workspace"]
        CORE[rlox-core<br/>envs, buffers, GAE,<br/>V-trace, GRPO, pipeline]
        NN[rlox-nn<br/>ActorCritic, QFunction,<br/>StochasticPolicy traits]
        BURN[rlox-burn<br/>Burn Autodiff NdArray]
        CANDLE[rlox-candle<br/>Candle CPU inference]
        GRPC[rlox-grpc<br/>tonic gRPC workers]
        PY[rlox-python<br/>PyO3 bindings]
    end

    NN --> BURN
    NN --> CANDLE
    CORE --> NN
    CORE --> GRPC
    PY --> CORE
    PY --> NN
</div>

## What's Fast and Why

We benchmarked rlox against SB3 and TorchRL on Apple M4 with bootstrap 95% confidence intervals (10,000 resamples). Every result marked below is statistically significant.

### GAE: 140-1,700x faster

Generalized Advantage Estimation is a sequential backward scan — the kind of workload where Python's interpreter overhead dominates. rlox runs it as a tight Rust loop:

| Trajectory | rlox | NumPy Loop | TorchRL | vs NumPy | vs TorchRL |
|-----------|------|-----------|---------|----------|------------|
| 128 steps | 0.7 us | 34 us | 453 us | 51x | 679x |
| 2,048 steps | 4.0 us | 558 us | 6,798 us | 139x | 1,700x |
| 32,768 steps | 60 us | 8,906 us | 108,441 us | 147x | 1,791x |

### Buffers: 10-148x faster

Replay buffers are the RL equivalent of DataFrame append + sample. rlox uses pre-allocated ring buffers with ChaCha8 RNG:

| Operation | rlox | TorchRL | SB3 | vs TorchRL | vs SB3 |
|-----------|------|---------|-----|------------|--------|
| Push 10K transitions | 1.5 ms | 229 ms | 15 ms | 148x | 9.7x |
| Sample batch=1024 | 9.2 us | 96 us | 75 us | 10x | 8.1x |

### End-to-End: 3.9-53x faster

The advantages compound across the pipeline — step, store, compute GAE:

| Config | rlox | SB3 | TorchRL |  vs SB3 | vs TorchRL |
|--------|------|-----|---------|---------|------------|
| 256 envs x 2048 steps | 539 ms | 2,080 ms | 28,432 ms | 3.9x | 53x |

### Convergence: Same Rewards, Faster Wall-Clock

Raw throughput doesn't matter if the agent doesn't learn. We ran PPO and A2C with identical hyperparameters (rl-zoo3 defaults), 5 seeds each:

| Algorithm | Environment | rlox Wall-clock | SB3 Wall-clock | Speedup |
|-----------|-------------|-----------------|----------------|---------|
| PPO | CartPole-v1 | 1.6s | 5.2s | 3.3x |
| A2C | CartPole-v1 | 1.8s | 2.1s | 1.2x |
| PPO | Acrobot-v1 | 6.4s | 9.1s | 1.4x |

Both frameworks converge to the same reward thresholds — rlox just gets there faster because the data plane isn't waiting on Python.

#### Training Throughput (Steps Per Second)

On-policy algorithms (PPO, A2C) show 1.6-2.5x SPS improvements thanks to Rust GAE. Off-policy algorithms (SAC, TD3) are bottlenecked by single-env stepping and NN updates, as expected.

![SPS Comparison](/images/benchmarks/sps_comparison.png)

#### Learning Curves

**PPO on CartPole-v1** — rlox converges to the same reward, 3.3x faster wall-clock:

![PPO CartPole](/images/benchmarks/learning_curve_PPO_CartPole-v1.png)

**PPO on Acrobot-v1** — both converge to ~-83, rlox reaches threshold 1.4x faster:

![PPO Acrobot](/images/benchmarks/learning_curve_PPO_Acrobot-v1.png)

**A2C on CartPole-v1** — matched convergence, rlox 2.5x faster throughput:

![A2C CartPole](/images/benchmarks/learning_curve_A2C_CartPole-v1.png)

#### Performance Profile (Agarwal et al., 2021)

Aggregated across all environments. On the on-policy subset (PPO, A2C), rlox matches SB3's convergence while training 1.4-3.3x faster.

![Performance Profile](/images/benchmarks/performance_profile.png)

## Beyond Classic RL: LLM Post-Training

rlox isn't just for CartPole. We built first-class support for LLM post-training:

- **GRPO and DPO** with Rust-accelerated advantage computation (35x faster than NumPy/PyTorch)
- **Token-level KL divergence** computed in Rust
- **Sequence packing** for efficient batching
- **vLLM, TGI, and SGLang** inference backends with a unified factory interface
- **Multi-GPU training** via PyTorch DDP composition

```python
from rlox.algorithms import GRPO

def math_reward(completions, prompts):
    return [1.0 if verify_answer(c) else 0.0 for c in completions]

grpo = GRPO(model=my_llm, ref_model=ref_llm, reward_fn=math_reward)
grpo.train(prompts, n_epochs=3)
```

## The Rust Crate Ecosystem

rlox is a multi-crate Rust workspace, published on [crates.io](https://crates.io/crates/rlox-core):

- **rlox-core** — environments, buffers, GAE, V-trace, GRPO, pipeline orchestration
- **rlox-nn** — RL algorithm traits (`ActorCritic`, `QFunction`, `StochasticPolicy`)
- **rlox-burn** — [Burn](https://burn.dev/) backend for pure-Rust training
- **rlox-candle** — [Candle](https://github.com/huggingface/candle) backend for low-latency CPU inference

You can use these crates independently in Rust projects without Python at all.

## Getting Started

```bash
pip install rlox
```

Train PPO on CartPole:

```python
from rlox.trainers import PPOTrainer

trainer = PPOTrainer(env="CartPole-v1", seed=42)
metrics = trainer.train(total_timesteps=50_000)
print(f"Mean reward: {metrics['mean_reward']:.1f}")
```

Or use the Rust primitives directly for maximum control:

```python
import rlox

advantages, returns = rlox.compute_gae(
    rewards, values, dones, last_value,
    gamma=0.99, lam=0.95
)

env = rlox.VecEnv(n=256, seed=42, env_id="CartPole-v1")
result = env.step_all(actions)
```

## What's Next

- More convergence benchmarks across MuJoCo and Atari environments
- GPU-accelerated environment stepping
- Broader LLM post-training coverage (online DPO, RLAIF pipelines)
- Community-contributed Rust environments

## Links

- **GitHub**: [github.com/riserally/rlox](https://github.com/riserally/rlox)
- **PyPI**: [pypi.org/project/rlox](https://pypi.org/project/rlox/)
- **crates.io**: [crates.io/crates/rlox-core](https://crates.io/crates/rlox-core)
- **Docs**: [riserally.github.io/rlox](https://riserally.github.io/rlox/)
- **License**: MIT or Apache 2.0

We'd love to hear from you — open an issue, start a discussion, or try `pip install rlox` and let us know what you think.

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
