# rlox: Product Requirements Document

**Version**: 1.0
**Date**: March 2026
**Status**: Living document — update with each major release

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Python RL Ecosystem Analysis](#3-python-rl-ecosystem-analysis)
4. [Gaps and Opportunities](#4-gaps-and-opportunities)
5. [Vision and Positioning](#5-vision-and-positioning)
6. [Target Users and Use Cases](#6-target-users-and-use-cases)
7. [Core Feature Requirements](#7-core-feature-requirements)
8. [API Design Principles](#8-api-design-principles)
9. [Architecture Requirements](#9-architecture-requirements)
10. [Integration Requirements](#10-integration-requirements)
11. [Performance Requirements](#11-performance-requirements)
12. [Testing and Reliability Requirements](#12-testing-and-reliability-requirements)
13. [Deployment Considerations](#13-deployment-considerations)
14. [Competitive Comparison Matrix](#14-competitive-comparison-matrix)
15. [Phased Roadmap](#15-phased-roadmap)
16. [Success Metrics](#16-success-metrics)
17. [Risks and Mitigations](#17-risks-and-mitigations)

---

## 1. Executive Summary

rlox is a Rust-accelerated reinforcement learning library with Python bindings. It follows the "Polars architecture pattern": a high-performance Rust data plane connected to a Python control plane via PyO3. Researchers write Python as they always have; the bottleneck operations — environment stepping, experience storage, advantage computation, and LLM post-training primitives — execute in Rust with no GIL, no interpreter overhead, and compile-time memory safety.

The core thesis is narrow and proven: RL frameworks are bottlenecked by Python's architecture in three specific places (parallel environment stepping, replay buffer I/O, and training loop math), and Rust eliminates all three bottlenecks simultaneously. Benchmarks on Apple M4 confirm this at 140x faster GAE vs NumPy, 148x faster buffer push vs TorchRL, and 3.9x faster end-to-end rollout vs SB3 at 256 environments.

The second distinguishing claim is dual-domain support: rlox explicitly targets both classic simulation RL (MuJoCo, Atari, robotics) and LLM post-training (GRPO, DPO, RLHF-PPO, RLOO). No existing library does both well. TRL dominates LLM post-training but ignores simulation. SB3 dominates simulation but has no LLM primitives. rlox intends to be the high-performance substrate for both.

**Current status (March 2026):** Phases 0–6 complete. Core infrastructure is proven: VecEnv, ExperienceTable, ReplayBuffer, VarLenStore, GAE, GRPO advantages, token KL, DPO pairs. 73 Rust tests, 85 Python tests. The next challenge is algorithm completeness and ecosystem integration to achieve adoption.

---

## 2. Problem Statement

### 2.1 Python's Architectural Ceiling

Every major Python RL library shares the same fundamental bottlenecks. These are not implementation bugs; they are structural consequences of Python's design:

**The GIL serializes parallel work.** DeepMind stated publicly (PEP 703 discussions) that the GIL becomes the bottleneck for their RL workloads with fewer than 10 threads, where they need 50–100. This forces every Python RL framework into multiprocessing with expensive inter-process communication. SB3's SubprocVecEnv is 5x *slower* than sequential DummyVecEnv for simple environments because IPC overhead exceeds computation time. Even for complex environments, speedup plateaus far below linear scaling.

**Interpreter overhead dominates shallow networks.** RL uses shallower networks than supervised learning, making per-operation Python overhead proportionally large. LeanRL demonstrated that torch.compile alone — eliminating Python/C++ boundary crossings without changing algorithm logic — delivers 2.7–6.8x speedups on PPO, SAC, and TD3. The overhead being eliminated is purely Python interpreter dispatch.

**Memory safety is borrowed, not owned.** MuJoCo rendering segfaults in multithreaded contexts (mujoco-py#283). Memory leaks accumulate during long training runs (documented in Google DeepMind's MuJoCo issue tracker). RLlib's ReplayBuffer used Python lists until recently, meaning training fails if data exceeds RAM. These are structural consequences of mixing Python with C extensions across thread and process boundaries.

**Reproducibility is structurally unsolved.** Henderson et al. (AAAI 2018) showed different seeds produce completely different learning curves. Agarwal et al. (NeurIPS 2021 Outstanding Paper) found 50–70% of claimed RL improvements may be spurious under proper statistical analysis. PyTorch warns that "completely reproducible results are not guaranteed across releases, platforms, or individual commits" due to non-associative floating-point in CUDA atomics.

### 2.2 The Two-Domain Fragmentation

Modern RL spans two domains that share theoretical foundations but differ operationally:

- **Classic simulation RL**: MuJoCo, Atari, custom environments. Bottleneck is environment throughput (frames/second). The training loop is a tight collect-process-train cycle. SB3, TorchRL, and CleanRL serve this domain.

- **LLM post-training**: GRPO, DPO, RLHF-PPO, RLOO. Bottleneck is generation throughput (tokens/second) and training memory efficiency. The "environment" is an inference server; the "trajectory" is a token sequence of variable length. TRL serves this domain.

No library serves both. A team training a robotics policy and a language model policy must use two different frameworks with incompatible abstractions for replay buffers, advantage computation, and training loops. rlox's architecture generalizes cleanly across both: the same buffer handles fixed-dimension observations and variable-length token sequences; the same GAE implementation applies to both; the same KL controller manages both PPO and RLHF-PPO.

---

## 3. Python RL Ecosystem Analysis

### 3.1 Stable-Baselines3 (SB3)

**Stars:** 12.9k | **Version:** 2.7.x | **Latest requirements:** PyTorch >= 2.3, Python 3.10+

**What it is:** The workhorse of academic RL research. scikit-learn-inspired API (`PPO("MlpPolicy", "CartPole-v1")`), 7 core algorithms (PPO, SAC, TD3, A2C, DQN, DDPG, HER), 7 more in SB3-Contrib (TRPO, RecurrentPPO, MaskablePPO, QR-DQN, TQC, CrossQ, ARS).

**Strengths:**
- The most imitable API in the RL ecosystem. One-liners work for standard environments.
- Extensive documentation, tutorials, and Hugging Face model hub integration.
- High test coverage with type hints — unusual in RL.
- SB3-Contrib provides experimental algorithms without polluting the core.
- SBX (JAX variant) runs up to 20x faster on identical tasks, proving SB3's design is sound even if Python overhead limits performance.

**Weaknesses:**
- Single-machine, single-agent only — no distributed training, no multi-agent support.
- No model-based RL, no offline RL (beyond HER).
- No LLM post-training support.
- Performance ceiling from Python/NumPy: DummyVecEnv is often faster than SubprocVecEnv for environments with computation time under ~1ms.
- Active development has shifted to maintenance mode for the core; major innovation now happens in SBX or contrib repos.

**Key insight:** SB3 proved that API simplicity drives adoption more than algorithm completeness. The benchmark community uses SB3 as the reference point precisely because it is simple and trustworthy, not because it is fast.

### 3.2 TorchRL

**Stars:** 3.3k | **Version:** 0.10.x | **Owner:** Meta/PyTorch

**What it is:** The most architecturally ambitious Python RL framework. Built around TensorDict — a batched, device-aware tensor container — that keeps data on GPU and avoids numpy-to-tensor conversions. Covers online RL (PPO, SAC, DQN, IMPALA), offline RL (IQL, CQL, Decision Transformer), model-based (DreamerV3, TD-MPC), multi-agent (MAPPO, QMIX via BenchMARL), and is adding LLM support (GRPO, SFT) in v0.10.

**Strengths:**
- ICLR 2024 benchmarks: async collector achieves 19,401 fps on Breakout vs RLlib's 97 fps. GAE 10.6x faster than SB3.
- TensorDict abstraction enables algorithm-agnostic code that works for any RL domain.
- Most comprehensive algorithm coverage of any maintained Python library.
- PyTorch-native — first-class GPU support, compatible with torch.compile and FSDP.
- BenchMARL provides state-of-the-art multi-agent benchmarking.

**Weaknesses:**
- Despite GPU-resident TensorDict, is still 1,700x slower than rlox on GAE (TensorDict metadata overhead per element).
- API complexity is high: TensorDict abstraction has a steep learning curve. The "right" way to do anything is non-obvious.
- Beta status means API churn. Teams building on TorchRL face migration costs.
- LLM features (GRPO, SFT) are marked experimental as of v0.10.
- Per-item overhead: buffer push 148x slower than rlox because each TensorDict add() has significant Python dispatch cost.

**Key insight:** TorchRL proves that keeping data GPU-resident and avoiding numpy conversions matters enormously. But TensorDict's per-operation Python overhead is still the bottleneck for operations that are fundamentally simple (GAE is just a backward scan; buffer push is just a memcpy).

### 3.3 RLlib (Ray)

**Stars:** 41.5k (Ray monorepo) | **Version:** 2.54.0 | **Owner:** Anyscale

**What it is:** The only Python RL framework built for distributed training from the ground up. Three-axis scaling: `num_env_runners` for parallel data collection, `num_envs_per_env_runner` for vectorized sub-environments, `num_learners` for data-parallel GPU training.

**Strengths:**
- True distributed training across machines — the only Python library that does this.
- Native MARL support with independent, collaborative, and adversarial configurations.
- Fault-tolerant EnvRunners — training continues when individual workers fail. Critical for cloud spot instances.
- Full offline RL support via Ray Data integration (BC, CQL, MARWIL).
- Recent "new API stack" (RLModule, Learner, ConnectorV2) significantly improved modularity.
- Algorithms: PPO, DQN/Rainbow, SAC, APPO, IMPALA, DreamerV3, plus offline methods.

**Weaknesses:**
- Per-step throughput is poor for single-machine workloads. Users consistently report difficulty matching SB3 or CleanRL performance on simple environments.
- Ray overhead from serialization, scheduling, and actor management is substantial for non-distributed use.
- Complex configuration — RLlib's `AlgorithmConfig` system historically required navigating nested dicts.
- The "new API stack" transition is ongoing; many old-stack algorithms (DDPG, TD3, MADDPG, AlphaZero) are not yet ported.
- No LLM post-training support.

**Key insight:** RLlib occupies a unique and defensible niche (distributed training, fault tolerance, MARL at scale) but is genuinely poor for single-machine research. Teams that need distributed RL often write custom solutions rather than fight RLlib's abstraction layers.

### 3.4 CleanRL

**Stars:** 9.3k | **Type:** Educational reference, not importable library

**What it is:** Single-file algorithm implementations. PPO on Atari is 340 lines. Every implementation is self-contained and benchmarked across 34+ Atari games. A NeurIPS 2022 paper validated its implementations are statistically equivalent to SB3 on PPO.

**Strengths:**
- Unmatched for understanding implementation details. The single-file constraint forces explicitness.
- 12 PPO variants, 4 DQN variants, SAC, DDPG, TD3 — covering major use cases.
- Strong reproducibility culture: fixed seeds, tracked hyperparameters, published learning curves.
- LeanRL fork demonstrates that torch.compile + CUDA graphs delivers 6.8x PPO speedup — no algorithm changes needed.

**Weaknesses:**
- Not importable as a library. Cannot `import cleanrl`.
- Massive code duplication: changes to a shared component require editing every single file.
- No configuration system, no callbacks, no distributed training.
- Unsuitable for production use or as a dependency.

**Key insight:** CleanRL's success reveals that researchers value correctness and transparency over modularity. The RL community has been burned by buggy "modular" implementations and welcomes flat, readable code. rlox's algorithm layer should be CleanRL-transparent even when the infrastructure layer is not.

### 3.5 Tianshou

**Stars:** 10.3k | **Version:** 2.x | **Owner:** Tsinghua University

**What it is:** Comprehensive Python RL framework with 30+ algorithms. Dual API: high-level for ease of use, procedural API for research customization. Strong MuJoCo benchmark results.

**Strengths:**
- Algorithm breadth rivals or exceeds TorchRL: DQN/Rainbow family, PPO/TRPO, SAC/DDPG/TD3/REDQ, offline (BCQ/CQL/TD3+BC/IQL), imitation (GAIL), model-based (MBPO).
- Clean separation between OnPolicyAlgorithm, OffPolicyAlgorithm, and OfflineAlgorithm.
- Excellent type hints and documentation.
- EnvPool integration for high-throughput environment stepping.

**Weaknesses:**
- Python 3.11+ requirement (recent addition) limits deployment environments.
- Version 2 introduced breaking changes that drove users away.
- No LLM post-training support.
- Less community mindshare than SB3 despite comparable algorithm coverage.

**Key insight:** Tianshou demonstrates that algorithm breadth alone does not win adoption. SB3 has fewer algorithms but 20% more GitHub stars because of API simplicity and documentation quality.

### 3.6 Sample Factory

**Stars:** 975 | **Target:** High-throughput on-policy training

**What it is:** Specialized framework optimized for PPO throughput. Separate worker processes for environment collection and centralized learning, enabling high-throughput asynchronous training. State-of-the-art results on ViZDoom, IsaacGym, DMLab-30, Atari.

**Strengths:**
- Genuine throughput leadership for CPU-based environments when paired with complex observation spaces.
- Population-Based Training support.
- Multi-agent with self-play.
- HuggingFace model hub integration.

**Weaknesses:**
- Single algorithm focus (PPO). No off-policy algorithms.
- Linux/macOS only.
- 975 stars indicates limited adoption despite strong performance.

**Key insight:** Specialization yields performance but limits adoption. Sample Factory's architecture is instructive (decoupled collection/training) but the framework itself remains niche.

### 3.7 EnvPool

**Stars:** 1.3k | **Type:** Environment execution engine only

**What it is:** C++-backed parallel environment execution engine with Python bindings (pybind11). Not an RL framework — just fast environment stepping. 1 million Atari frames/second on 256 CPU cores. 14.9–19.6x faster than Python subprocess-based vectorized environments.

**Strengths:**
- Proven that C++/pybind11 breaks through the GIL for environment stepping.
- Drop-in compatible with SB3, Tianshou, CleanRL via Gymnasium API.
- NUMA-aware optimization for multi-socket systems.
- Thread-pool model (no IPC overhead).

**Weaknesses:**
- C++ compilation required for custom environments.
- No algorithm implementations — must pair with another framework.
- Stars (1.3k) suggest that most users prefer working with one framework even if it is slower.
- Stopped receiving major updates after NeurIPS 2022 paper.

**Key insight:** EnvPool validates rlox's VecEnv approach at a larger scale. Its limited adoption despite impressive benchmarks confirms that users want a complete framework, not just a fast environment runner.

### 3.8 TRL (Hugging Face)

**Stars:** 17.6k | **Version:** active | **Owner:** Hugging Face

**What it is:** The dominant LLM post-training library. Trainer classes wrapping HuggingFace Transformers for SFT, DPO, GRPO, reward model training, PPO. Built on Accelerate + DeepSpeed for multi-GPU/multi-node scaling.

**Strengths:**
- The ecosystem leader for LLM post-training. 17.6k stars dwarfs all simulation RL libraries except OpenAI Baselines.
- First-class PEFT/LoRA/QLoRA support.
- Handles the practical complexity of LLM training: gradient checkpointing, mixed precision, sequence packing, distributed training.
- All major post-training algorithms: SFT, DPO, GRPO, PPO (RLHF), reward training.
- Recently added OpenEnv framework integration for agentic RL workflows.

**Weaknesses:**
- No simulation RL support. Cannot use it with MuJoCo, Atari, or any custom environment.
- Python-only hot paths: reward normalization, advantage computation, KL divergence are all Python/PyTorch — with corresponding overhead.
- Tightly coupled to HuggingFace Transformers — non-HF models require significant adaptation.
- No first-class support for verifiable reward environments (code execution, math verification) beyond basic wrappers.
- The hot path for GRPO advantage computation, token KL, and batch construction has no equivalent to rlox's Rust primitives.

**Key insight:** TRL's 17.6k stars reveal the market size for LLM post-training tooling. rlox does not aim to replace TRL's training orchestration layer (Accelerate + DeepSpeed are excellent), but can provide 34x faster GRPO advantage computation and 4x faster token KL as drop-in NumPy replacements that TRL-based workflows can adopt.

### 3.9 Gymnasium / PettingZoo

**Stars:** 11.5k (Gymnasium) | **Owner:** Farama Foundation

**What it is:** The universal environment API standard. Gymnasium is the maintained successor to OpenAI Gym. PettingZoo is its multi-agent equivalent. 20.6k dependent projects in the ecosystem.

**Strengths:**
- De facto standard: every RL framework supports `gymnasium.Env`.
- 300+ built-in environments (Atari, MuJoCo, Classic Control, Box2D, Toy Text).
- Strict versioning for reproducibility (CartPole-v1 vs CartPole-v0 are different environments).
- Farama Foundation governance ensures long-term maintenance.

**Weaknesses:**
- The standard API is a Python interface, inheriting all Python performance limitations.
- Not a training framework — purely environment interface specification.
- Documentation acknowledges that "environments can be slow to run compared to custom C, C++, or Cython-based environments."

**Key insight:** rlox must be 100% Gymnasium-compatible via the GymEnv bridge. Compatibility with this standard is non-negotiable for adoption.

### 3.10 Other Notable Libraries

**OpenAI Baselines** (16.7k stars, maintenance only): The original reference implementations. No longer receiving new features but still widely cited. SB3 is its practical successor.

**Dopamine** (10.9k stars, Google DeepMind): Research-focused JAX/TF library emphasizing reproducibility for DQN family algorithms. Small algorithm set (DQN, C51, Rainbow, IQN, SAC, PPO) but well-benchmarked.

**ACME** (3.9k stars, Google DeepMind): Research framework with JAX/TF backends. "Written by researchers, for researchers." Explicitly not production-targeted. Strong distributed RL architecture.

**DI-engine** (3.6k stars, OpenDILab): Comprehensive framework covering model-based, MARL, offline, and LLM post-training (PPO-max, DPO, PromptPG). Underexplored outside China.

**Pearl** (3k stars, Meta): Production-focused with unique features: dynamic action spaces (recommender systems), constrained optimization for safety, contextual bandits. Niche but important for industry deployment.

**PARL** (3.5k stars, Baidu/PaddlePaddle): Large-scale distributed RL framework. Won NeurIPS 2018–2020 competitions. Limited adoption outside PaddlePaddle ecosystem.

---

## 4. Gaps and Opportunities

### 4.1 The Performance-Usability Gap

The RL ecosystem presents a clear trade-off: usable frameworks (SB3, CleanRL) are slow; fast frameworks (TorchRL, Sample Factory, EnvPool) are complex. No library combines SB3-level API simplicity with TorchRL-level (or better) performance.

rlox's opportunity: use Rust to deliver TorchRL-exceeding performance through a SB3-quality Python API. The performance advantage compounds: not just faster environment stepping but faster everything (buffer I/O, GAE, batch assembly, KL computation), because all hot paths share the same architectural advantage.

### 4.2 The Dual-Domain Gap

No library serves both simulation RL and LLM post-training from shared infrastructure. The underlying mathematics are identical (policy gradient, advantage estimation, KL divergence, experience replay) but every framework treats them as separate concerns. Teams doing research across both domains maintain two separate codebases with two sets of abstractions.

rlox's opportunity: the same Rust primitives (GAE, KL, variable-length storage, parallel stepping) that accelerate simulation RL directly accelerate LLM post-training. The unification is architectural, not cosmetic.

### 4.3 The Reproducibility Gap

Despite Henderson et al. (2018) and Agarwal et al. (2021) establishing the severity of RL's reproducibility crisis, no framework has made reproducibility a first-class design concern. Seeds are settable but their scope is unclear. Experiment metadata is not automatically captured. Statistical evaluation (IQM, performance profiles, stratified bootstrap) requires external tools (rliable).

rlox's opportunity: Rust's deterministic RNG (ChaCha8, already implemented), combined with automatic experiment metadata capture and built-in statistical evaluation, can make rlox the first RL framework where "bit-exact reproduction" is a documented, tested guarantee.

### 4.4 The LLM Infrastructure Gap

TRL handles algorithm orchestration for LLM post-training but leaves critical hot paths in Python:
- GRPO advantage normalization: 35x slower than rlox's Rust implementation.
- Token KL divergence: 4x slower than rlox.
- Sequence packing: no efficient Rust bin-packing implementation exists.
- Verifiable reward execution: no sandboxed code execution environment exists in any RL framework.

rlox's opportunity: target TRL's hot paths as drop-in replacements callable from Python with `pip install rlox`. Researchers using TRL can adopt rlox incrementally without rewriting their training scripts.

### 4.5 The Memory Safety Gap

Long-running RL training (days to weeks) is where Python's memory unsafety manifests most harmfully: accumulating leaks, segfaults during environment resets, corrupted replay buffers under concurrent access. Rust's ownership model makes these impossible at compile time. No Python RL library can make this guarantee.

rlox's opportunity: document and test the memory safety properties explicitly. "rlox's buffer cannot overflow, leak, or have a data race — by construction" is a meaningful claim that production engineers will value.

### 4.6 The Tail Latency Gap

Most RL benchmarks report median throughput. Tail latency (p99, p999) matters for real-time robotics and simulation pipelines where a single slow step stalls the entire collection batch. Python's GC creates unpredictable pauses. TorchRL/SB3 buffer sample p99 is 135–138µs; rlox's is 14.7µs (batch=1024). No other library documents or optimizes tail latency.

rlox's opportunity: measure and guarantee tail latency bounds. This is a differentiated claim for production robotics teams.

---

## 5. Vision and Positioning

### 5.1 Vision Statement

**rlox is the high-performance substrate for reinforcement learning: the fastest data plane for both simulation RL and LLM post-training, exposed through a Python API that requires no Rust knowledge.**

This is not a rewrite of existing RL frameworks. It is a Rust engine that makes existing Python RL workflows — including those built on SB3, TorchRL, TRL, or CleanRL — measurably faster by replacing their bottlenecks with zero-copy, GIL-free, memory-safe implementations.

### 5.2 Positioning

rlox is positioned at the intersection of two dimensions: performance and domain coverage.

```
                    High Performance
                          |
            rlox (target) |
           /              |   EnvPool (env only)
          /               |   Sample Factory
         /                |
Narrow   ------------------------------------------  Broad
Domain   |                |
         | CleanRL        |  TorchRL
         | (classic only) |  RLlib
         |                |  Tianshou
         |       SB3      |
         |                |
                    Lower Performance
```

The target position: broad domain coverage (simulation + LLM post-training) with the highest sustained throughput of any Python-accessible RL library.

### 5.3 Differentiation Claims

1. **"Faster than TorchRL on every benchmark, easier to use than SB3."** The performance is proven (Phase 6 benchmarks). The usability is the primary engineering challenge for v1.0.

2. **"The only RL library that handles both MuJoCo rollouts and GRPO training from shared infrastructure."** Verified by existing implementation: same GAE function, same buffer abstractions, same KL utilities work for both.

3. **"Drop-in acceleration for TRL-based GRPO pipelines."** `rlox.compute_group_advantages()` replaces the equivalent NumPy/PyTorch call with 35x lower latency. TRL users adopt rlox without rewriting training code.

4. **"The only RL framework where bit-exact reproducibility is a tested, documented guarantee."** ChaCha8 RNG already implemented. Needs experiment metadata capture to complete the story.

5. **"Tail latency that robotics production systems can rely on."** Buffer sample p99 < 15µs regardless of batch size. GC pauses cannot affect rlox's Rust core.

---

## 6. Target Users and Use Cases

### 6.1 Primary Users

**Academic RL Researchers (simulation domain)**
- Current tool: SB3 or CleanRL, sometimes TorchRL
- Pain: waiting for long training runs, cannot scale to 256+ parallel environments without multiprocessing overhead
- rlox value: 3.9–53x faster rollout collection means more experiments per day; same Python API means minimal migration cost
- Use case: PPO/SAC experiments on MuJoCo, Atari, custom Gymnasium environments

**LLM Post-Training Engineers**
- Current tool: TRL, sometimes custom GRPO/DPO implementations
- Pain: GRPO advantage normalization and KL computation are Python/PyTorch hot paths that limit throughput for large groups (G=16+)
- rlox value: 35x faster GRPO advantages, 4x faster token KL as drop-in replacements; variable-length sequence storage for efficient trajectory management
- Use case: GRPO training for reasoning models, DPO for preference alignment, online RLHF

**RL Infrastructure Engineers (production systems)**
- Current tool: Custom C++, sometimes RLlib
- Pain: Python memory safety issues in long-running training (leaks, segfaults), unpredictable tail latency, no compile-time correctness guarantees
- rlox value: Rust ownership model prevents buffer overflow, use-after-free, and data races by construction; p99 latency bounds are predictable and documented
- Use case: Real-time robot control, large-scale simulation farms, production recommendation RL

**RL Researchers Scaling Beyond Single Machine**
- Current tool: RLlib (despite its overhead), custom solutions
- Pain: RLlib's per-step overhead is poor for single-machine workloads; custom solutions are unmaintained
- rlox value: Decoupled collection/training architecture, gRPC-based distributed env workers (v1.0 roadmap), Rust-native async that eliminates IPC serialization cost
- Use case: Distributed PPO/IMPALA across multiple nodes, async actor-learner setups

### 6.2 Secondary Users

**Robotics Researchers**
- Use case: Sim-to-real transfer with Isaac Gym / MuJoCo; need high throughput for domain randomization (hundreds of parallel environments)
- Specific value: VecEnv with true Rayon parallelism, no GIL, predictable latency

**Multi-Agent RL Researchers**
- Use case: MAPPO, QMIX on cooperative/competitive tasks
- Specific value: Shared experience buffers without Python lock contention; observation routing in Rust

**RL Curriculum Developers and Educators**
- Use case: Teaching RL implementation; building reference implementations
- Specific value: Clean Rust code for algorithms is pedagogically valuable for understanding performance

### 6.3 Non-Targets (explicitly out of scope)

- **Supervised ML training**: rlox is not a general ML framework. Use PyTorch, JAX, or their ecosystems.
- **Tabular RL**: Q-tables and simple TD-learning do not benefit from rlox's architecture. Use basic Python.
- **Game development**: rlox provides environment interfaces but is not a game engine. Use Bevy for complex game environments.
- **Non-PyTorch model training**: rlox is designed to interop with PyTorch. JAX interop is a future consideration.

---

## 7. Core Feature Requirements

Priority levels:
- **P0 (Must-have)**: v0.1 cannot launch without this.
- **P1 (High priority)**: Required for v0.5 adoption by research teams.
- **P2 (Medium priority)**: Required for v1.0 production readiness.
- **P3 (Nice-to-have)**: Future roadmap.

### 7.1 Environment Engine

| ID | Feature | Priority | Status | Description |
|----|---------|----------|--------|-------------|
| E1 | Parallel environment stepping (Rayon) | P0 | Done | True multi-threaded step() across N environments with zero GIL contention. Achieved: 5.7x faster than Gymnasium Sync at 512 envs. |
| E2 | Gymnasium-compatible bridge | P0 | Done | Any gymnasium.Env subclass works via GymEnv wrapper. |
| E3 | Rust-native environment trait | P0 | Done | `RLEnv` trait with compile-time action/observation type checking. CartPole implemented. |
| E4 | Async environment collection | P1 | Planned | Environments that finish early don't block others. Critical for LLM generation where episode lengths vary 10–100x. |
| E5 | LLM generation environment wrapper | P0 | Partial | Wrap vLLM/TGI/SGLang as an RL environment. VarLenStore and DPOPair exist; full inference server integration pending. |
| E6 | Auto-vectorization of gymnasium envs | P1 | Planned | Automatic batching of step() calls into contiguous tensor operations for Gymnasium-compatible envs. |
| E7 | Environment checkpointing / forking | P2 | Planned | Save/restore env state for MCTS-style planning and best-of-N sampling. |
| E8 | WASM compilation of Rust envs | P3 | Future | Rust environments compile to WASM for browser demos and edge deployment. |
| E9 | MuJoCo native bindings | P1 | Planned | Direct mujoco-rs bindings without Python subprocess overhead. Eliminate the primary segfault source. |
| E10 | Isaac Gym / Isaac Lab integration | P2 | Planned | GPU-accelerated simulation at scale. Critical for sim-to-real robotics. |

### 7.2 Experience Storage

| ID | Feature | Priority | Status | Description |
|----|---------|----------|--------|-------------|
| B1 | Columnar experience store (ExperienceTable) | P0 | Done | Append-only columnar storage. Zero-copy sharing with PyTorch/NumPy. |
| B2 | Ring buffer replay (ReplayBuffer) | P0 | Done | Fixed-capacity ring buffer, ChaCha8 random sampling. p99 < 15µs at batch=1024. |
| B3 | Variable-length sequence storage (VarLenStore) | P0 | Done | Offset-array backed storage for token sequences. No padding waste. |
| B4 | Prioritized experience replay | P1 | Planned | Sum-tree in Rust: O(log N) sampling and priority update. Thread-safe. |
| B5 | Memory-mapped buffer overflow | P1 | Planned | When buffer exceeds RAM, spill to NVMe transparently. Critical for LLM trajectories (100K+ tokens). |
| B6 | On-disk Parquet/Arrow dataset | P1 | Planned | Read/write experience in Parquet for offline RL and DPO datasets. Lazy loading with predicate pushdown. |
| B7 | Multi-table relational storage | P2 | Planned | Separate tables for prompts, completions, rewards, KL terms. Join on (episode_id, step_id). |
| B8 | Concurrent lock-free sampling | P1 | Partial | Writers (env workers) and readers (learner) without blocking. Existing buffer uses mutex; lock-free upgrade planned. |
| B9 | Trajectory segmentation | P1 | Planned | Automatic episode boundary detection in continuous streams. Per-episode return/advantage computation. |

### 7.3 Training Core

| ID | Feature | Priority | Status | Description |
|----|---------|----------|--------|-------------|
| T1 | Rust-native GAE | P0 | Done | 140x faster than NumPy, 1700x faster than TorchRL. Supports lambda-returns. |
| T2 | Token KL divergence | P0 | Done | Token-level KL: sum(exp(p) * (p - q)). 4x faster than NumPy, 6x faster than PyTorch. |
| T3 | KL controller (adaptive penalty) | P0 | Done | Adaptive KL penalty coefficient management (Ziegler et al. 2019 style). |
| T4 | GRPO group advantage normalization | P0 | Done | Group-relative (reward - mean) / std. 35x faster than NumPy/PyTorch. |
| T5 | Batch assembly pipeline | P1 | Planned | Collate transitions into training batches. Handle padding, masking, and sequence packing. Zero-copy PyTorch tensor output. |
| T6 | V-trace / UPGO advantage variants | P2 | Planned | Alternatives to GAE for IMPALA-style async training. |
| T7 | Reward normalization | P1 | Planned | Running mean/variance (Welford's algorithm) in Rust. Per-token and per-sequence normalization. |
| T8 | Gradient accumulation scheduler | P1 | Planned | Track micro-batch accumulation state; trigger optimizer at step boundaries. |
| T9 | Reference model logprob management | P1 | Planned | Efficient computation of reference model log probabilities for RLHF-PPO. LoRA base vs. separate copy. |
| T10 | Sequence packing | P1 | Planned | Bin-packing of variable-length sequences into fixed GPU batch sizes. Generates position IDs and attention masks. 30–60% GPU utilization improvement. |

### 7.4 Algorithm Implementations

| ID | Algorithm | Priority | Status | Notes |
|----|-----------|----------|--------|-------|
| A1 | PPO (clip + KL variants) | P0 | Planned | Core algorithm for simulation RL and RLHF. Must include all 37 implementation details from Huang et al. 2022. |
| A2 | DPO / IPO / KTO | P0 | Partial | DPOPair storage done. Loss computation and training loop pending. |
| A3 | GRPO | P0 | Partial | Advantage normalization done. Group generation, per-group clipping, loss pending. |
| A4 | SAC | P1 | Planned | Soft Actor-Critic for continuous control. Automatic entropy tuning. |
| A5 | TD3 | P1 | Planned | Twin Delayed DDPG. Cheaper than SAC, deterministic. |
| A6 | DQN family (DQN, C51, Rainbow) | P1 | Planned | Discrete control. Prioritized replay integration (B4). |
| A7 | RLOO (REINFORCE Leave-One-Out) | P1 | Planned | Variance reduction for LLM policy gradient. Rust computes leave-one-out baselines. |
| A8 | Online DPO / OAIF | P1 | Planned | On-policy generation + preference pair construction + DPO update in one loop. |
| A9 | Best-of-N / rejection sampling | P1 | Planned | Generate N completions, score, keep best. Simple but effective LLM baseline. |
| A10 | Reward model training | P1 | Planned | Bradley-Terry preference model. Shares experience storage layer. |
| A11 | MAPPO / QMIX | P2 | Planned | Multi-agent cooperative RL. Centralized critic, decentralized execution. |
| A12 | DreamerV3 | P2 | Planned | World-model RL. Rust-native imagination rollouts in learned latent space. |
| A13 | IMPALA / V-trace | P2 | Planned | Asynchronous distributed RL. Rust async collection, V-trace correction. |
| A14 | A2C | P1 | Planned | Simpler PPO without clipping. Faster per-update, useful baseline. |

### 7.5 LLM Post-Training Specifics

| ID | Feature | Priority | Status | Description |
|----|---------|----------|--------|-------------|
| L1 | Token-level MDP abstraction | P0 | Partial | VarLenStore + DPOPair provide storage. Full token MDP trait pending. |
| L2 | vLLM / TGI / SGLang integration | P0 | Planned | Async Rust client for inference servers. Connection pooling, backpressure, KV-cache lifecycle management. |
| L3 | Reward model serving | P0 | Planned | Batch scoring via REST/gRPC. Ensemble RM support. Multi-objective mixing (helpfulness + safety). |
| L4 | Prompt dataset management | P1 | Planned | Streaming iteration with shuffling, curriculum, deduplication. Parquet/Arrow native. |
| L5 | Verifiable reward sandbox | P1 | Planned | Sandboxed code execution (Rust, WASM) for code/math verification. Timeout and memory limits. |
| L6 | KV-cache aware scheduling | P2 | Planned | Co-locate prompts sharing prefixes. Maximize KV-cache reuse across generation/scoring batches. |
| L7 | Constitutional AI / RLAIF pipeline | P2 | Planned | Generate → critique → revise → preference pairs → train. Full pipeline orchestration. |

### 7.6 Distribution and Scaling

| ID | Feature | Priority | Status | Description |
|----|---------|----------|--------|-------------|
| D1 | Single-machine multi-GPU | P1 | Planned | PyTorch DDP/FSDP for gradient sync; rlox Rust core for data collection. Covers 90% of research. |
| D2 | Decoupled collection / training | P1 | Planned | Env workers and learner as separate Rust async tasks. Shared memory or gRPC for experience transfer. |
| D3 | Distributed env workers (gRPC) | P2 | Planned | Env stepping across multiple machines. Load-balanced Rust gRPC service. |
| D4 | Multi-node training | P2 | Planned | PyTorch DDP/FSDP for gradients; Rust handles data routing and collection. |
| D5 | Elastic scaling | P3 | Future | Add/remove env workers without restarting training. |

### 7.7 Python API and Developer Experience

| ID | Feature | Priority | Status | Description |
|----|---------|----------|--------|-------------|
| API1 | One-liner training API | P0 | Planned | `PPOTrainer(env="HalfCheetah-v4", model=policy).train(1_000_000)`. Must be this simple. |
| API2 | Composable config system | P0 | Planned | Dataclass-based config with full type hints. Merge from YAML/CLI/code. No nested dict hell. |
| API3 | Custom model support | P0 | Planned | Any nn.Module (simulation) or HuggingFace PreTrainedModel (LLM) works as policy. |
| API4 | Zero-copy tensor bridge | P0 | Partial | rust-numpy bridge done. DLPack for direct PyTorch tensor creation (no numpy intermediary) planned. |
| API5 | Callback / hook system | P1 | Planned | on_step, on_episode_end, on_train_batch, on_eval. Rust invokes Python callbacks with batched payloads. |
| API6 | W&B / MLflow / TensorBoard integration | P1 | Planned | Rust collects metrics and flushes to Python loggers periodically. |
| API7 | Jupyter / notebook experience | P1 | Planned | Progress bars, inline plots. Training does not block notebook kernel. |
| API8 | Type-safe space definitions | P1 | Planned | Box[float32, (84,84,4)], Discrete[18]. Early error on space mismatch. |
| API9 | Comprehensive error messages | P0 | Partial | Typed errors implemented. Human-readable context messages (action shape mismatches, etc.) need improvement. |
| API10 | pip install with prebuilt wheels | P0 | Planned | maturin-built wheels for Linux x86_64, ARM64, macOS. No Rust toolchain required for users. |

### 7.8 Reproducibility and Correctness

| ID | Feature | Priority | Status | Description |
|----|---------|----------|--------|-------------|
| R1 | Deterministic seeding | P0 | Done | ChaCha8 RNG. Single seed controls all Rust RNG, env resets, buffer sampling. Bit-exact on same hardware + thread count. |
| R2 | Experiment snapshot | P1 | Planned | Record: git hash, full config, dependency versions, hardware info, seed. JSON alongside checkpoints. |
| R3 | Checkpoint / resume | P0 | Planned | Save: model weights, optimizer state, buffer contents, RNG state, training step. Resume produces identical trajectory to uninterrupted run. |
| R4 | Statistical evaluation toolkit | P1 | Planned | Built-in IQM, optimality gap, performance profiles (Agarwal et al. 2021). Stratified bootstrap CI. No external rliable needed. |
| R5 | Deterministic parallel reduction | P2 | Planned | Kahan summation for cross-environment aggregation. Eliminates floating-point non-associativity across thread counts. |
| R6 | Transition provenance | P2 | Planned | Every transition tagged: env_id, episode_id, step, policy_version, RM_version. Critical for LLM post-training debugging. |

### 7.9 Observability and Debugging

| ID | Feature | Priority | Status | Description |
|----|---------|----------|--------|-------------|
| O1 | Real-time throughput dashboard | P1 | Planned | Live FPS, GPU utilization, buffer occupancy, collect/train ratio. Terminal or web UI. |
| O2 | Training diagnostics | P1 | Planned | Automatic detection: entropy collapse, KL spikes, gradient explosions, reward hacking. Logged as warnings. |
| O3 | Trajectory inspector | P2 | Planned | Browse trajectories: replay env states (simulation) or view prompt → completion → reward (LLM). |
| O4 | Profiler integration | P2 | Planned | Breakdown: env stepping vs buffer ops vs model forward/backward vs optimizer vs data transfer. |

---

## 8. API Design Principles

### 8.1 The Layered API Contract

rlox must offer three distinct API layers without forcing users into a higher-complexity layer than they need:

**Layer 0 — Primitives (current state)**
Raw Rust-backed operations callable from Python. No training loop, no configuration system.
```python
from rlox import VecEnv, ReplayBuffer, compute_gae, compute_group_advantages

env = VecEnv(n=64)
buf = ReplayBuffer(capacity=100_000, obs_dim=4, act_dim=1)
advs, returns = compute_gae(rewards, values, dones, last_val, gamma=0.99, lam=0.95)
```
This layer is already done and benchmarked. Target users: advanced researchers who want rlox's speed but their own training loop.

**Layer 1 — Components (v0.5 target)**
Higher-level components with lifecycle management. Compatible with any PyTorch training loop.
```python
from rlox import RolloutCollector, ExperienceStore, PPOLoss

collector = RolloutCollector(env="HalfCheetah-v4", n_envs=64)
store = ExperienceStore(capacity=2048)
loss_fn = PPOLoss(clip_eps=0.2, entropy_coef=0.01)

for batch in collector.collect(n_steps=2048):
    store.push(batch)
    for minibatch in store.sample_epochs(n_epochs=4, batch_size=64):
        loss = loss_fn(minibatch, policy)
        loss.backward()
```
Target users: researchers who want rlox's performance and components but control their own training loop structure.

**Layer 2 — Trainers (v1.0 target)**
One-liner API inspired by SB3 and HuggingFace Trainers.
```python
from rlox.trainers import PPOTrainer, GRPOTrainer
from rlox.config import PPOConfig

# Simulation RL
trainer = PPOTrainer(
    env="HalfCheetah-v4",
    model=my_policy_network,
    config=PPOConfig(n_envs=64, n_steps=2048, learning_rate=3e-4),
)
trainer.train(total_timesteps=1_000_000)

# LLM post-training
trainer = GRPOTrainer(
    model="deepseek-ai/deepseek-r1-zero",
    reward_fn=math_reward_fn,
    config=GRPOConfig(group_size=16, kl_coef=0.1),
)
trainer.train(dataset=math_problems)
```
Target users: practitioners who want SB3-level simplicity. This is the "getting started" experience.

### 8.2 Core API Principles

**Principle 1: Zero knowledge of Rust required.**
The Python API must feel like a Python library. Type stubs (.pyi), docstrings, error messages, and examples must be written from a Python perspective. Rust implementation details should never leak into the Python surface.

**Principle 2: Fail loudly at the boundary, not in the middle of training.**
Observation shape mismatches, incompatible action spaces, and configuration errors must be caught at construction time and raise clear exceptions with context. "Expected obs_dim=4, got shape (3,) — check your environment's observation_space" is acceptable. A raw Rust panic traceback is not.

**Principle 3: Composition over inheritance.**
Components should be combinable in any order, not locked into a class hierarchy. A researcher should be able to use rlox's ReplayBuffer with their own training loop, or rlox's GRPOTrainer with a custom reward function, without subclassing anything.

**Principle 4: Defaults that work, overrides that are discoverable.**
Every component has sensible defaults based on published best practices. Every default can be overridden via keyword argument. No configuration key buried three levels deep in a nested dict.

**Principle 5: The hot path is in Rust; the decision path is in Python.**
Python should control: what algorithm to run, what hyperparameters to use, what model architecture to train, when to checkpoint, when to log. Rust should control: when to step environments, how to store transitions, how to compute advantages, how to sample batches.

**Principle 6: PyTorch tensors are the lingua franca.**
All data crossing the PyO3 boundary should be expressible as PyTorch tensors or NumPy arrays. No custom rlox tensor types. No mandatory conversion steps. `buffer.sample()` returns a dict of tensors that can be fed directly to `loss.backward()`.

### 8.3 API Anti-Patterns to Avoid

- **TorchRL's TensorDict API**: Powerful but teaches a new abstraction before the user can train anything. rlox's API should be learnable from the PyTorch documentation alone.
- **RLlib's AlgorithmConfig nested dicts**: Every nested dict requires documentation lookup. rlox uses typed dataclasses with IDE completion.
- **SB3's rigid policy string ("MlpPolicy")**: Forces users into a fixed architecture. rlox accepts any `nn.Module`.
- **CleanRL's single-file approach**: Not importable as a library. Every rlox component must be importable.

---

## 9. Architecture Requirements

### 9.1 Module Boundaries

The architecture must maintain strict separation across four boundaries:

```
┌─────────────────────────────────────────────────────────┐
│  Python Control Plane (python/rlox/)                    │
│  Trainers, configs, callbacks, logging integration      │
│  No performance-critical loops; pure orchestration      │
├─────────────── PyO3 boundary ───────────────────────────┤
│  Thin Binding Layer (crates/rlox-python/)               │
│  PyO3 class/function wrappers                           │
│  Type conversion: Python ↔ Rust                         │
│  Error translation: Rust errors → Python exceptions     │
├─────────────── Rust core ───────────────────────────────┤
│  Core Library (crates/rlox-core/)                       │
│  env/ — environment traits, CartPole, VecEnv, GymEnv   │
│  buffer/ — columnar, ring, varlen storage               │
│  training/ — GAE, KL, reward normalization              │
│  llm/ — GRPO ops, token KL, DPO ops                     │
│  (No PyO3 dependency — testable independently)          │
├─────────────── Async runtime ───────────────────────────┤
│  Distribution Layer (future: crates/rlox-dist/)         │
│  gRPC env workers, experience routing                   │
│  Tokio async runtime                                    │
└─────────────────────────────────────────────────────────┘
```

**Critical constraint:** `rlox-core` must never depend on PyO3. All PyO3 code lives in `rlox-python`. This separation enables testing `rlox-core` with standard Rust test tooling, including Criterion benchmarks and proptest property-based tests.

### 9.2 Data Flow

```
Environments (Rust or Gymnasium Python)
    |
    | Transition: (obs, action, reward, terminated, truncated)
    v
Experience Storage (Rust)
    |
    | Batch: zero-copy numpy/torch arrays
    v
Advantage Computation (Rust: GAE, GRPO, RLOO)
    |
    | Augmented batch: (obs, action, advantage, return, log_prob)
    v
PyO3 boundary (zero-copy)
    |
    v
PyTorch Model (Python)
    |
    | Forward + backward pass, optimizer step
    v
Updated policy weights (in Python GPU memory)
    |
    | Weight sync (periodic)
    v
Environments (updated policy for next rollout)
```

Environment stepping, storage, and advantage computation never touch the Python GIL. Model computation stays in Python/PyTorch. The boundary crossing is one batch per gradient step: a dict of arrays, passed without copy.

### 9.3 Concurrency Model

**Intra-process parallelism (Rayon):** Environment stepping uses Rayon's work-stealing thread pool. This is zero-IPC, true thread parallelism, GIL-free. Appropriate for: all Rust-native environments, all Gymnasium environments where the GIL is acquired in batch (one acquisition per step-all, not per-step).

**Async I/O (Tokio):** Inference server communication (vLLM/TGI), gRPC env workers, and checkpoint I/O use Tokio async runtime. Async I/O enables high-concurrency without thread overhead for I/O-bound operations.

**Actor model (future):** Decoupled collection/training uses separate Rust tasks communicating via channels. No shared mutable state across the collection/training boundary. Experience transfer via lock-free queue.

**Python GIL management:** rlox acquires the GIL only when executing Python-defined reward functions, custom environment step functions, or callbacks. Acquisition is batched: one acquire/release per entire step-batch, not one per step. This is the same strategy that made EnvPool 19x faster than Python subprocess envs.

### 9.4 Memory Architecture

**Fixed-capacity pre-allocation:** All buffers are pre-allocated at construction. No heap allocation during training hot paths. No garbage collection pauses in Rust-managed memory.

**Zero-copy cross-boundary data sharing:** `rust-numpy` enables Python to access Rust-owned arrays without copy. Buffer `sample()` returns numpy arrays backed by Rust memory. `torch.from_numpy()` makes this a PyTorch tensor with one additional (zero-copy) step. DLPack support will eliminate the numpy intermediary.

**Observation memory layout:** Observations stored in columnar layout (all obs[0] contiguous, all obs[1] contiguous) for cache-efficient batch sampling. This differs from TorchRL's row-major TensorDict layout and enables SIMD-friendly operations.

---

## 10. Integration Requirements

### 10.1 PyTorch Integration

- All data crossing the PyO3 boundary must be directly consumable as PyTorch tensors via `torch.from_numpy()` or DLPack.
- rlox must not require specific PyTorch versions. The dependency is on numpy for array representation, not PyTorch directly.
- Compatibility with `torch.compile`: rlox's Rust boundary is transparent to torch.compile. The Python training loop using rlox primitives should compile cleanly.
- FSDP/DDP compatibility: rlox handles data collection; PyTorch handles gradient synchronization. No interference.

### 10.2 HuggingFace Ecosystem Integration

- LLM training: rlox must accept HuggingFace `PreTrainedModel` as a policy without modification.
- `DPOPair` and `VarLenStore` must serialize to/from HuggingFace `Dataset` format (Apache Arrow).
- PEFT/LoRA compatibility: reference model management must support LoRA adapters (base model + adapter = reference; active adapters = policy).
- Accelerate compatibility: rlox's experience collection is independent of Accelerate's device management. No conflicts.

### 10.3 Gymnasium Compatibility

- Full Gymnasium API compliance via `GymEnv` wrapper: `env.step()`, `env.reset()`, `observation_space`, `action_space`.
- Support all standard observation space types: Box, Discrete, MultiDiscrete, MultiBinary, Dict, Tuple.
- Vectorized environment protocol: `gymnasium.vector.VectorEnv` compatible output format (obs, rewards, terminated, truncated, infos).
- PettingZoo compatibility for multi-agent environments (v1.0 target).

### 10.4 NumPy Compatibility

- All rlox Python functions that accept arrays must accept any numpy-compatible array (including torch tensors via `__array__`).
- All rlox Python functions that return arrays return numpy arrays by default, convertible to torch tensors via zero-copy.
- NumPy dtype consistency: float32 for observations and actions, float64 for rewards and advantages (following SB3 convention for numerical stability in advantage computation).

### 10.5 Build and Distribution

- Prebuilt wheels via maturin for: Linux x86_64, Linux aarch64, macOS x86_64, macOS arm64.
- No Rust toolchain required for users. `pip install rlox` downloads and installs the prebuilt wheel.
- ABI stability: use `abi3` stable ABI targeting Python 3.10+ to avoid per-Python-version wheel builds.
- `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` already in use for development builds.

### 10.6 Observability Integration

- W&B: `rlox.logging.WandbLogger` wraps the W&B Python SDK. Metrics flushed from Rust to Python logger at configurable intervals (default: every gradient step).
- MLflow: `rlox.logging.MLflowLogger` — identical interface.
- TensorBoard: `rlox.logging.TensorBoardLogger` — identical interface.
- Prometheus: metrics endpoint for production deployments (P2).

---

## 11. Performance Requirements

### 11.1 Benchmark Targets (Already Validated, March 2026)

These are measured results on Apple M4. Maintenance targets: do not regress below CI lower bound.

| Benchmark | rlox Result | vs Best Alternative | CI Lower Bound |
|-----------|------------|---------------------|----------------|
| GAE (2048 steps) | 4.0 µs | 139x vs NumPy, 1700x vs TorchRL | 120x vs NumPy |
| Buffer push (10K, obs=4) | 1.5 ms | 148x vs TorchRL, 9.7x vs SB3 | 100x vs TorchRL |
| Buffer sample (batch=1024) | 9.2 µs | 10x vs TorchRL, 8.1x vs SB3 | 7x vs both |
| E2E rollout (256×2048) | 539 ms | 3.9x vs SB3, 53x vs TorchRL | 3x vs SB3 |
| GRPO advantages (256×16) | 36 µs | 35x vs NumPy/PyTorch | 25x vs NumPy |
| Token KL (128 tokens) | 0.4 µs | 4x vs NumPy, 6x vs PyTorch | 3x vs NumPy |

### 11.2 Future Performance Targets (v0.5 / v1.0)

| Benchmark | Target | Notes |
|-----------|--------|-------|
| VecEnv throughput (256 envs, CartPole) | >500K steps/sec | Current: ~19K steps/sec (539ms / (256×2048)) — primarily env compute |
| VecEnv throughput (256 envs, MuJoCo) | >50K steps/sec | With mujoco-rs native bindings |
| Buffer push (large obs, Atari 28224) | competitive with SB3 | Already 0.9x SB3 (memcpy-dominated). Target: match or beat. |
| Buffer sample p99 (batch=1024) | < 20 µs | Current: 14.7 µs. Maintain under increased concurrency. |
| Sequence packing utilization | > 90% GPU batch fill rate | vs ~60% for naive padding |
| GRPO (256 prompts × 16 completions) | < 50 µs | Current: 36 µs. Maintain as group size scales. |
| E2E PPO training throughput (HalfCheetah) | > 500K SPS | SB3 achieves ~130K SPS. LeanRL achieves ~800K SPS. Target competitive. |

### 11.3 Tail Latency Requirements

| Operation | p50 Target | p99 Target | Rationale |
|-----------|-----------|-----------|-----------|
| Buffer sample (batch=256) | < 5 µs | < 20 µs | Real-time control: 50Hz policy = 20ms budget |
| VecEnv step-all (64 envs) | < 500 µs | < 2 ms | Training loop: 1000 steps/sec target |
| GAE (2048 steps) | < 10 µs | < 50 µs | Training overhead: < 1% of step time |
| PyO3 boundary crossing | < 200 ns | < 1 µs | Per-batch overhead must be negligible |

### 11.4 Scalability Requirements

- VecEnv: linear throughput scaling from 1 to 64 environments on a single machine. (Current: sub-linear below 16 envs due to Rayon scheduling overhead vs CartPole's 37ns step time. This is expected and documented.)
- Replay buffer: O(1) push, O(log N) prioritized sample, O(1) uniform sample. No O(N) operations on the hot path.
- Sequence storage: O(1) append regardless of sequence length. O(1) access by index.
- GAE: O(N) where N is trajectory length. Memory: 3 * N * 8 bytes (rewards, values, dones). 2048-step trajectory: ~50KB working set — cache-resident.

---

## 12. Testing and Reliability Requirements

### 12.1 Test Categories and Coverage

**Rust unit tests (target: 95%+ line coverage of rlox-core)**
- Every data structure: boundary conditions, capacity limits, overflow behavior.
- Every algorithm primitive: numerical correctness against reference Python implementation.
- Thread safety: concurrent push/sample operations on all buffer types.
- Error handling: malformed inputs produce typed errors, never panics.

**Property-based tests (proptest, already in use)**
- Buffer invariants: FIFO eviction, no data loss, correct capacity management.
- Numerical invariants: GAE advantages sum to returns, KL divergence non-negative, GRPO advantages have zero mean and unit variance.
- Round-trip invariants: serialize → deserialize → bit-exact match.

**Python integration tests (target: 85+ tests, currently 85)**
- API surface coverage: every public Python API exercised.
- Type compatibility: numpy dtypes, array shapes, edge cases (empty batch, single env, max batch size).
- Gymnasium bridge: compatibility with 5+ Gymnasium environments.
- PyTorch interop: all buffer outputs consumable as PyTorch tensors without copy.

**Algorithm correctness tests (v0.5 target)**
- PPO must solve CartPole-v1 in < 50K timesteps across 5 seeds.
- SAC must achieve MuJoCo HalfCheetah > 5000 return in 1M steps (matching SB3 baseline).
- DPO must achieve > 90% preference accuracy on a synthetic preference dataset.
- GRPO must improve math accuracy from 0% to > 20% on a small symbolic math task.

**Benchmark regression tests (already in use)**
- CI runs benchmark suite on every PR.
- Fail if any benchmark regresses below 80% of baseline (accounting for CI hardware variability).
- Benchmark results published as PR artifact.

### 12.2 Safety Requirements

- **No panics in production code paths.** Every function that can fail must return `Result<T, RloxError>`. Panics are bugs, not error handling.
- **No unsafe Rust in rlox-core without explicit justification.** Every `unsafe` block requires a comment explaining why it is sound.
- **No memory leaks.** Valgrind/address sanitizer clean on the Python test suite. PyO3 reference counting must be verified.
- **No data races.** Concurrent access patterns must be tested with thread sanitizer and validated with Rust's type system.

### 12.3 CI/CD Requirements

- Every PR: Rust tests (`cargo test`), Python tests (`pytest`), benchmark regression check, `cargo clippy` clean, `cargo fmt` check.
- Nightly: Full benchmark suite on standardized hardware. Publish results to benchmark history.
- Release: Wheel builds for all target platforms, integration test against latest SB3/TorchRL/TRL versions.
- Fuzzing: Continuous fuzzing of the PyO3 boundary with arbitrary inputs (malformed arrays, NaN rewards, infinite values).

---

## 13. Deployment Considerations

### 13.1 Installation Simplicity

The installation story must be: `pip install rlox`. No compilation step for users. Prebuilt wheels handle all supported platforms.

The development story is: clone repo, `maturin develop --release`, run tests. This requires the Rust toolchain but is documented with a one-command setup.

### 13.2 Container Packaging

Provide official Docker images:
- `rlox/base`: Python + rlox, no ML dependencies
- `rlox/torch`: rlox + PyTorch (CUDA and CPU variants)
- `rlox/llm`: rlox + PyTorch + vLLM (for LLM post-training)

Images pinned to specific rlox versions and tested on release.

### 13.3 Edge and Embedded Deployment

rlox's Rust core compiles to static libraries. This enables:
- **Embedded deployment**: rlox environment stepping without Python runtime. Relevant for robotics edge compute.
- **WASM compilation**: Rust environments compile to WASM for browser visualization (P3).
- **iOS/Android**: Rust libraries link into mobile apps for on-device policy inference. (P3, dependent on TorchScript or ONNX model export.)

### 13.4 Version Compatibility Matrix

| rlox version | Python | PyTorch | NumPy | Gymnasium |
|-------------|--------|---------|-------|-----------|
| v0.1 (current) | 3.10+ | 2.3+ | 1.24+ | 1.0+ |
| v0.5 (target) | 3.10+ | 2.5+ | 2.0+ | 1.0+ |
| v1.0 (target) | 3.11+ | 2.6+ | 2.0+ | 1.0+ |

Support policy: latest two minor Python versions; latest minor PyTorch version.

### 13.5 Semantic Versioning

- **0.x**: API may change between minor versions. Document breaking changes in CHANGELOG.
- **1.0+**: Stable API guarantee. Breaking changes only in major versions. Pre-deprecation notice minimum 2 minor versions ahead.

---

## 14. Competitive Comparison Matrix

### 14.1 Feature Comparison

| Feature | rlox | SB3 | TorchRL | RLlib | TRL | CleanRL |
|---------|------|-----|---------|-------|-----|---------|
| **Classic simulation RL** | Yes | Yes | Yes | Yes | No | Yes |
| **LLM post-training** | Yes | No | Partial | No | Yes | No |
| **PPO** | P0 | Yes | Yes | Yes | Yes | Yes |
| **SAC / TD3** | P1 | Yes | Yes | Yes | No | Yes |
| **DQN / Rainbow** | P1 | Yes | Yes | Yes | No | Yes |
| **GRPO** | Yes | No | Beta | No | Yes | No |
| **DPO** | Partial | No | No | No | Yes | No |
| **Multi-agent** | P2 | No | Yes | Yes | No | No |
| **Model-based RL** | P2 | No | Yes | Yes | No | No |
| **Offline RL** | P2 | No | Yes | Yes | No | No |
| **Distributed training** | P2 | No | No | Yes | Yes | No |
| **Rust data plane** | Yes | No | No | No | No | No |
| **GIL-free parallelism** | Yes | No | No | No | No | No |
| **Memory-safe buffers** | Yes | No | No | No | No | No |
| **Pip install (no compile)** | P0 | Yes | Yes | Yes | Yes | N/A |
| **One-liner API** | P0 | Yes | Partial | No | Yes | No |
| **Custom nn.Module support** | P0 | Partial | Yes | Yes | Yes | Yes |

### 14.2 Performance Comparison (Apple M4, measured)

| Metric | rlox | SB3 | TorchRL | CleanRL | TRL |
|--------|------|-----|---------|---------|-----|
| GAE (2048 steps) | 4.0 µs | 558 µs | 6798 µs | ~500 µs | ~500 µs |
| Buffer push (10K) | 1.5 ms | 15 ms | 229 ms | ~15 ms | N/A |
| Buffer sample (b=1024) | 9.2 µs | 75 µs | 96 µs | ~70 µs | N/A |
| E2E rollout (256×2048) | 539 ms | 2080 ms | 28432 ms | ~1800 ms | N/A |
| GRPO advantages | 36 µs | N/A | N/A | N/A | ~1250 µs |
| Token KL (128 tokens) | 0.4 µs | N/A | N/A | N/A | ~1.7 µs (NumPy) |

### 14.3 Positioning Summary

**vs SB3**: rlox targets the same usability standard (one-liner API, simple config, clear docs) while delivering 4–148x better performance across the RL pipeline. SB3 has more algorithms today; rlox will match on P0/P1 algorithms by v0.5.

**vs TorchRL**: rlox is 5–1700x faster on all measured operations. TorchRL has more algorithms today (especially offline RL and world models). TorchRL's TensorDict API is more powerful but harder to learn. rlox's API targets SB3-level simplicity.

**vs RLlib**: rlox is faster per step on single-machine workloads. RLlib's unique advantage — fault-tolerant distributed training at cluster scale — is out of rlox's current scope. Teams that need RLlib for distributed training can use rlox for its faster data collection layer (P2 gRPC integration).

**vs TRL**: Complementary, not competitive. TRL handles LLM training orchestration (Accelerate, DeepSpeed, model loading). rlox replaces TRL's Python hot paths (GRPO advantages, token KL, sequence storage) as drop-in primitives. The full LLM post-training stack can combine TRL's training loop with rlox's data plane.

**vs CleanRL**: rlox is the "production-ready CleanRL" — transparent algorithm implementations with industrial-strength infrastructure. rlox's Layer 0 (raw primitives) targets the same researchers who use CleanRL for educational clarity.

---

## 15. Phased Roadmap

### Phase 0: Skeleton (Complete)
Workspace structure, PyO3 build, maturin packaging, CI.

### Phase 1: Environment Engine (Complete)
CartPole, VecEnv with Rayon, GymEnv bridge. Benchmarks vs Gymnasium.

### Phase 2: Experience Storage (Complete)
ExperienceTable (columnar), ReplayBuffer (ring), VarLenStore (variable-length). Zero-copy numpy bridge.

### Phase 3: Training Core (Complete)
GAE, token KL divergence, KL controller. Benchmarks vs NumPy and TorchRL.

### Phase 4: LLM Post-Training Primitives (Complete)
GRPO advantage normalization, DPOPair storage, token KL. Benchmarks vs NumPy/PyTorch.

### Phase 5: API Polish (Complete)
Type stubs (.pyi), proptest property-based tests, API ergonomics.

### Phase 6: Benchmark Suite (Complete)
Three-framework comparison (rlox vs TorchRL vs SB3) with statistical rigor. Published results.

---

### Phase 7: Algorithm Completeness — v0.5 (Target: Q3 2026)

**Goal:** Complete PPO end-to-end for simulation RL. Complete GRPO end-to-end for LLM post-training. Make rlox usable as a complete training framework, not just a collection of primitives.

**Deliverables:**
- PPO complete (policy loss, value loss, entropy regularization, advantage normalization, gradient clipping). Must solve CartPole in < 50K steps across 5 seeds. Must match SB3 learning curves on MuJoCo within 1 standard deviation.
- GRPO complete (generation loop integration, per-group clipping, sequence storage management). Drop-in replacement for TRL's GRPO training loop with same or better throughput.
- DPO complete (loss computation, training loop, reference model management).
- A2C complete (simpler PPO variant, useful baseline).
- Batch assembly pipeline (T5): collate transitions into training-ready zero-copy torch tensors.
- Reward normalization (T7): running statistics, whitening, per-token normalization.
- Sequence packing (T10): bin-packing for LLM trajectory batching.
- Component API (Layer 1): `RolloutCollector`, `ExperienceStore`, `PPOLoss`.
- Checkpoint/resume (R3).
- W&B / TensorBoard logging (API6).
- Prebuilt wheels for Linux and macOS.

**Success criteria:**
- End-to-end PPO training script for MuJoCo in < 50 lines of Python using rlox Layer 1 API.
- End-to-end GRPO training script for a math reasoning task in < 80 lines using rlox Layer 1 API.
- All Phase 6 benchmarks maintained (no regression).
- 100+ Python tests, 80+ Rust tests.

---

### Phase 8: Production Hardening — v0.7 (Target: Q4 2026)

**Goal:** Harden the framework for production research use. Complete the algorithm set. Add SAC, TD3, DQN. Build the one-liner trainer API.

**Deliverables:**
- SAC, TD3, DQN/Rainbow complete with prioritized replay (B4).
- One-liner trainer API (Layer 2): `PPOTrainer`, `SACTrainer`, `GRPOTrainer`, `DPOTrainer`.
- Composable config system (API2): dataclass-based, YAML/CLI merge.
- Callback/hook system (API5).
- Statistical evaluation toolkit (R4): IQM, performance profiles, stratified bootstrap CI.
- Experiment snapshot (R2): automatic metadata capture alongside checkpoints.
- MuJoCo native bindings (E9) via mujoco-rs.
- Memory-mapped buffer overflow (B5).
- Online DPO / OAIF (A8).
- Best-of-N sampling (A9).
- Verifiable reward sandbox (L5): sandboxed code/math verification.
- Training diagnostics (O2): automatic detection of entropy collapse, KL spikes.

**Success criteria:**
- `PPOTrainer(env="HalfCheetah-v4", model=policy).train(1_000_000)` works in < 10 seconds to start.
- Published benchmark comparing rlox PPO, SAC, DQN against SB3 and CleanRL on standard MuJoCo/Atari tasks.
- Zero regressions on Phase 6/7 benchmarks.
- pip install produces working wheels on all target platforms.

---

### Phase 9: Distributed and Scale — v1.0 (Target: Q2 2027)

**Goal:** Scale rlox to multi-GPU and multi-machine. Add multi-agent support. First stable API release.

**Deliverables:**
- Decoupled collection/training (D2): async Rust tasks for parallel collect and learn.
- Single-machine multi-GPU (D1): PyTorch DDP/FSDP for gradient sync.
- Distributed env workers (D3): gRPC-based env worker mesh.
- MAPPO/QMIX (A11): multi-agent cooperative RL.
- DreamerV3 (A12): world-model RL with Rust imagination rollouts.
- IMPALA/V-trace (A13): async distributed RL with V-trace correction.
- vLLM / TGI integration (L2): full LLM generation environment.
- Reward model serving (L3): batch RM scoring.
- Transition provenance (R6): full lineage tracking for LLM post-training.
- Stable API 1.0: semver guarantees, migration guide from 0.x.

**Success criteria:**
- E2E distributed PPO training across 4 machines with near-linear throughput scaling.
- MAPPO solving StarCraft II micromanagement scenarios.
- DreamerV3 achieving competitive results on visual control benchmarks.
- rlox cited in at least 3 published RL research papers.
- 1,000+ GitHub stars.

---

### Phase 10+: Ecosystem and Advanced Features (2027+)

**Target:** JAX backend support, WASM compilation, Isaac Lab integration, RLHF full pipeline with critic + reward model + policy, Constitutional AI orchestration, model-based LLM planning.

---

## 16. Success Metrics

### 16.1 Performance Metrics (Maintained Continuously)

- All Phase 6 benchmark speedups maintained (no regression below CI lower bound).
- PPO training throughput: > 500K samples/second on HalfCheetah on A100 GPU.
- Tail latency: buffer sample p99 < 20µs at batch=1024.
- Memory overhead: < 10% above theoretical minimum for buffer storage.

### 16.2 Adoption Metrics (Per Release)

| Milestone | Target | Timeframe |
|-----------|--------|-----------|
| GitHub stars | 500 | v0.5 (Q3 2026) |
| GitHub stars | 2,000 | v1.0 (Q2 2027) |
| PyPI downloads/month | 1,000 | v0.5 |
| PyPI downloads/month | 10,000 | v1.0 |
| Research papers citing rlox | 1 | v0.7 |
| Research papers citing rlox | 5 | v1.0 |
| Companies using rlox in production | 1 | v1.0 |

### 16.3 Quality Metrics (Per Release)

- Rust test coverage: > 90% line coverage of rlox-core.
- Python test count: > 150 tests by v1.0.
- Zero open P0/P1 bugs in stable releases.
- Documentation coverage: every public API has docstring + example.
- Benchmark CI: no regression on any measured speedup.

### 16.4 Ecosystem Metrics

- All P0 algorithms reproduce published baseline results within 1 standard deviation.
- rlox works as drop-in acceleration for TRL-based GRPO training (validated by integration test).
- rlox works with SB3's Gymnasium environments without modification.
- Prebuilt wheels available on PyPI for all target platforms within 24 hours of release.

---

## 17. Risks and Mitigations

### 17.1 Performance Risks

**Risk:** Rust's advantage narrows as Python frameworks adopt torch.compile and CUDA graphs. LeanRL demonstrates 6.8x PPO speedup from torch.compile alone.

**Mitigation:** rlox's advantages are in pre-GPU operations (env stepping, buffer I/O, advantage computation) that torch.compile cannot accelerate — these are Python/CPU bottlenecks, not GPU dispatch bottlenecks. rlox and torch.compile are complementary: rlox accelerates the CPU/Rust data plane; torch.compile accelerates the GPU/model compute plane. Target users should use both.

**Risk:** Apple M4 benchmarks may not represent GPU-heavy workloads where Python overhead is proportionally smaller.

**Mitigation:** Publish benchmarks on Linux + A100 GPU hardware for v0.5. The GAE and buffer operations are CPU-only and hardware-independent; the E2E rollout advantage should hold or widen on GPU workloads where env stepping is the true bottleneck.

### 17.2 Adoption Risks

**Risk:** Rust compilation step creates friction for researchers who `pip install` everything.

**Mitigation:** Prebuilt wheels eliminate the compilation step for users. `pip install rlox` is the only required command. The Rust toolchain is only needed for contributors.

**Risk:** Learning Rust is a barrier for potential contributors.

**Mitigation:** The contribution barrier is only for Rust core development. Python API improvements, algorithm implementations (Python layer), documentation, and tests are all Python contributions. Keep the contributing guide explicit about which work requires Rust knowledge.

**Risk:** Existing frameworks (TorchRL, TRL) absorb rlox's differentiators.

**Mitigation:** TorchRL has been working on performance improvements for 4 years and is still 1,700x slower on GAE. The architectural advantage (no TensorDict per-item overhead, no Python interpreter overhead in hot loops) is fundamental, not implementable in Python. TRL could add Rust primitives, but the PyO3 pattern is complex and would require significant investment. First-mover advantage in this specific combination matters.

### 17.3 Technical Risks

**Risk:** PyO3 API churn breaks rlox's bindings across Python or PyO3 versions.

**Mitigation:** ABI3 stable ABI targeting Python 3.10+ reduces per-version wheel maintenance. Pin PyO3 to specific versions in Cargo.lock. Track PyO3 release notes.

**Risk:** Numerical correctness diverges between rlox's Rust implementations and reference Python implementations under edge cases (NaN rewards, very long trajectories, pathological advantage values).

**Mitigation:** Proptest property-based tests (already in use) cover random inputs. Add specific tests for NaN handling, infinity, very large/small values. Algorithm correctness tests compare against reference implementations on standard benchmarks.

**Risk:** The Rayon thread pool causes issues when rlox is embedded in a larger application that also uses threading (e.g., PyTorch DataLoader workers).

**Mitigation:** Rayon's global thread pool is configurable. Provide a `rlox.set_num_threads(n)` function. Document thread count interaction with PyTorch DataLoader's `num_workers`.

### 17.4 Ecosystem Risks

**Risk:** vLLM / TGI / SGLang APIs change frequently, breaking rlox's inference server integration.

**Mitigation:** Abstract the inference server behind a `GenerationBackend` trait. Concrete implementations are versioned separately. Users can implement the trait for any inference server.

**Risk:** Gymnasium deprecates or significantly changes its API.

**Mitigation:** rlox's GymEnv bridge is a thin wrapper. Gymnasium API changes require updating < 100 lines of binding code. The Farama Foundation has committed to API stability for Gymnasium 1.x.

---

*This document reflects the state of rlox as of March 2026 (Phases 0–6 complete) and the planned trajectory to v1.0. It should be updated with each major release to reflect completed phases, revised performance measurements, and updated market context.*

*For the detailed feature specification with dependency graphs and implementation phasing, see [feature_spec.md](feature_spec.md).*

*For algorithm research context, see [../research/README.md](../research/README.md) and the individual algorithm documents.*

*For benchmark methodology and results, see [../benchmark/README.md](../benchmark/README.md).*
