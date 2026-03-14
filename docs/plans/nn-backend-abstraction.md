# Neural Network Backend Abstraction for rlox

## 1. Rust NN Backend Landscape (March 2026)

### Tier 1: Viable for Training

| Backend | Autograd | GPU | Training | Maintenance | Strengths | Weaknesses |
|---------|----------|-----|----------|-------------|-----------|------------|
| **Burn** (tracel-ai/burn) | Yes, `AutodiffBackend` trait | CUDA (via LibTorch, WGPU, or Candle), Metal via WGPU | Full | Very active, funded startup | Backend-agnostic by design; `Backend` trait is the gold standard for this pattern; built-in `nn` module with `Module` derive macro; serialization; LR schedulers | Larger API surface; compile times; still pre-1.0 |
| **Candle** (huggingface/candle) | Yes, `Var` type tracks gradients | CUDA, Metal | Full | Active (HuggingFace) | Lightweight, PyTorch-like API; fast compile; good for inference and fine-tuning; strong LLM ecosystem | No `Backend` trait -- hardcoded device/dtype; distributions not built-in; less structured than Burn |
| **tch-rs** (LaurentMazare/tch-rs) | Yes, `requires_grad` on Tensor | CUDA (LibTorch C++) | Full | Maintained, stable | 1:1 PyTorch parity; every op available; battle-tested; distributions via `torch::distributions` | Requires LibTorch C++ install (~2GB); FFI overhead; not "Rust-native"; platform pain on macOS |
| **dfdx** (coreylowman/dfdx) | Yes, compile-time shape-checked | CUDA | Full | **Stalled** since mid-2024 | Compile-time shape verification; truly zero-cost; beautiful type-level API | Development halted; author moved on; shapes make generic code verbose; missing ops |

### Tier 2: Inference Only / Niche

| Backend | Autograd | GPU | Training | Maintenance | Notes |
|---------|----------|-----|----------|-------------|-------|
| **ort** (pyke-ml/ort) | No | CUDA, TensorRT, CoreML, DirectML | No (inference) | Active | ONNX Runtime bindings; excellent for deploying trained models; 20+ execution providers |
| **tract** (sonos/tract) | No | No (CPU only) | No (inference) | Active | Pure Rust ONNX/NNEF inference; no native deps; great for edge/embedded |
| **luminal** (jafioti/luminal) | Partial (graph-level) | CUDA, Metal | Experimental | Early stage | Graph-based, compiles to optimized kernels; interesting but not production-ready |
| **cust** (Rust-GPU/cust) | No | CUDA only | Manual | Maintained | Raw CUDA driver API; you write kernels yourself; no autograd |
| **wgpu** compute | No | Vulkan/Metal/DX12 | Manual | Active (part of wgpu) | Compute shaders; you write everything; no autograd; portable GPU |
| **metal-rs** | No | Metal only | Manual | Maintained | Apple Metal bindings; raw access; no autograd |
| **cubecl** (tracel-ai/cubecl) | No | CUDA, WGPU | Kernel-level | Active (Burn team) | Burn's underlying compute abstraction; write once, run on CUDA/WGPU; not a NN framework |

### Assessment

**dfdx** is out -- stalled development makes it a liability. **ort** and **tract** are complementary (inference/deployment) not competitors. **luminal** is too immature. **cust/wgpu/metal-rs** are too low-level; you would be writing a framework, not using one.

The real contenders for training are: **Burn**, **Candle**, and **tch-rs**.

---

## 2. What RL Algorithms Actually Need from a NN Backend

Distilling from the existing Python code (PPO, SAC, TD3, DQN, GRPO, DPO), the operations fall into these categories:

### Tensor Operations
- Creation: `zeros`, `ones`, `randn`, `from_slice/data`
- Arithmetic: `+`, `-`, `*`, `/`, `pow`, `exp`, `log`, `abs`
- Reduction: `mean`, `sum`, `max`, `min` (with dim and keepdim variants)
- Shape: `cat`, `squeeze`, `unsqueeze`, `reshape`, `gather`
- Comparison: `clamp`, element-wise `max`/`min` of two tensors
- Boolean/masking: `(ratio - 1.0).abs() > clip_eps` -> float mask

### Neural Network Layers
- `Linear` (with bias)
- Activations: `ReLU`, `Tanh`
- `Sequential` composition
- Parameter iteration (for optimizer, grad clipping, polyak update)

### Autograd
- Forward pass with gradient tracking
- `backward()` on scalar loss
- `zero_grad()` on optimizer
- `no_grad` context for inference / target computation
- Gradient clipping (`clip_grad_norm_`)

### Optimizer
- Adam with configurable `lr`, `eps`
- Per-parameter-group learning rate (for LR annealing)

### Distributions
- `Categorical(logits=...)`: `sample()`, `log_prob(action)`, `entropy()`
- `Normal(mean, std)`: `rsample()` (reparameterized), `log_prob(x)`
- Tanh squashing correction for SAC

### Model Operations
- Deep copy (for target networks)
- Polyak / soft update: `target = tau * source + (1 - tau) * target`
- State dict save/load (for checkpointing)
- Weight initialization (orthogonal)

---

## 3. Burn's Backend Abstraction: Lessons

Burn's design is the most relevant prior art. Key insights:

```
// Burn's approach (simplified):
trait Backend {
    type FloatTensorPrimitive;    // opaque handle
    type Device;
    fn float_add(lhs: ..., rhs: ...) -> ...;
    fn float_matmul(lhs: ..., rhs: ...) -> ...;
    // ~200 methods for every tensor op
}

trait AutodiffBackend: Backend {
    type InnerBackend: Backend;
    fn backward(tensor: ...) -> Gradients;
}
```

**Burn's strength**: complete backend abstraction. Code written against `Backend` runs on LibTorch, WGPU, Candle, NdArray -- zero changes.

**Burn's cost**: massive trait surface (~200 methods), and the `Tensor<B, D>` type carries a const generic `D: usize` for rank, making generic code over arbitrary ranks painful.

**Key question**: should rlox adopt Burn directly, wrap it, or build something lighter?

### Recommendation: Do NOT re-derive Burn's `Backend` trait

Building a tensor-level abstraction (Approach A) means re-implementing 200+ ops across each backend. This is a multi-person-year effort and will always lag behind the backends themselves. Burn already did this work. Competing with it is irrational.

---

## 4. Architecture Evaluation

### Approach A: Thin Trait Over Tensor Ops
```rust
trait Tensor { fn matmul(&self, other: &Self) -> Self; /* ... */ }
trait Backend { type Tensor: Tensor; /* ... */ }
```
- **Effort**: Extreme (200+ ops per backend)
- **Performance**: Zero overhead
- **Ergonomics**: Poor (re-inventing the wheel)
- **Verdict**: REJECT. This is building a framework, not an RL library.

### Approach B: Module-Level Trait
```rust
trait NNModule<B: Backend> {
    fn forward(&self, input: B::Tensor) -> B::Tensor;
    fn parameters(&self) -> Vec<&B::Tensor>;
}
```
- **Effort**: Moderate
- **Performance**: Zero overhead
- **Ergonomics**: Decent, but `B::Tensor` leaks into all algorithm code
- **Verdict**: Plausible but still requires a `Backend` trait underneath

### Approach C: Feature-Flag Compilation (Type Aliases)
```rust
#[cfg(feature = "burn-backend")]
type RloxTensor = burn::Tensor<BurnBackend, 2>;
#[cfg(feature = "candle-backend")]
type RloxTensor = candle::Tensor;
```
- **Effort**: Low per backend
- **Performance**: Zero overhead
- **Ergonomics**: Terrible -- different backends have different APIs; can't unify with aliases alone
- **Verdict**: REJECT as primary approach. Useful as an auxiliary mechanism.

### Approach D: Enum Dispatch (Runtime Backend)
```rust
enum AnyTensor { Burn(burn::Tensor), Candle(candle::Tensor) }
```
- **Effort**: Moderate
- **Performance**: Branch on every op; prevents inlining; terrible for hot loops
- **Ergonomics**: Easy to use
- **Verdict**: REJECT for training. Acceptable only for one-shot inference dispatch.

### Approach E: Hybrid -- Algorithm-Level Trait + Feature Flags (RECOMMENDED)

Abstract at the **policy/module level**, not the tensor level. The trait boundary is where RL algorithms interact with neural networks, not where tensors multiply. Feature flags select the concrete backend at compile time.

**This is the right level of abstraction because**:
- RL algorithms do not need to know about tensors. They need: "give me an action from this observation" and "compute this loss and update weights."
- Each backend has its own tensor type, autograd mechanism, and optimizer API. Trying to abstract these generically produces a leaky abstraction.
- The actual NN architectures (MLP, CNN) are simple and few. Implementing them separately per backend is cheap.

---

## 5. Recommended Trait Design

### Core Traits

```rust
use std::fmt::Debug;

/// Represents a 1D slice of f32 data that can be passed across the
/// trait boundary without depending on any backend's tensor type.
/// This is the "lingua franca" between rlox-core (which operates on
/// raw buffers/slices) and the NN backend.
///
/// For batch operations, data is laid out as [batch_size * dim] in
/// row-major order.
pub struct TensorData {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl TensorData {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        debug_assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "data length must match shape product"
        );
        Self { data, shape }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let len = shape.iter().product();
        Self { data: vec![0.0; len], shape }
    }
}

/// Errors from neural network operations.
#[derive(Debug, thiserror::Error)]
pub enum NNError {
    #[error("Backend error: {0}")]
    Backend(String),
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },
    #[error("Device error: {0}")]
    Device(String),
}

/// Action output from a policy, with associated log-probabilities.
#[derive(Debug, Clone)]
pub struct ActionOutput {
    /// Actions, shape [batch_size, act_dim] or [batch_size] for discrete
    pub actions: TensorData,
    /// Log-probabilities of the selected actions, shape [batch_size]
    pub log_probs: TensorData,
}

/// Output from evaluating a policy on (obs, actions) pairs.
#[derive(Debug, Clone)]
pub struct EvalOutput {
    /// Log-probabilities, shape [batch_size]
    pub log_probs: TensorData,
    /// Entropy of the distribution, shape [batch_size]
    pub entropy: TensorData,
    /// State values, shape [batch_size]
    pub values: TensorData,
}

/// The core trait for actor-critic policies used in on-policy algorithms
/// (PPO, A2C, IMPALA).
pub trait ActorCritic: Send + Sync {
    /// Sample actions from the policy (with gradient tracking disabled).
    /// Used during rollout collection.
    fn act(&self, obs: &TensorData) -> Result<ActionOutput, NNError>;

    /// Compute values for the given observations (no grad).
    /// Used for GAE bootstrap.
    fn value(&self, obs: &TensorData) -> Result<TensorData, NNError>;

    /// Evaluate the policy: compute log_prob, entropy, and value for
    /// given (obs, actions) pairs. This IS differentiable -- used in
    /// the PPO loss computation.
    fn evaluate(&self, obs: &TensorData, actions: &TensorData)
        -> Result<EvalOutput, NNError>;

    /// Perform one gradient step given the PPO loss inputs.
    /// Returns training metrics (policy_loss, value_loss, entropy, etc.)
    ///
    /// This bundles: forward -> loss computation -> backward -> clip grad -> step.
    /// The loss computation lives INSIDE the trait impl, not in generic code,
    /// because it requires autograd-aware tensor operations (clamp, ratio, etc.)
    /// that differ across backends.
    fn ppo_step(
        &mut self,
        obs: &TensorData,
        actions: &TensorData,
        old_log_probs: &TensorData,
        advantages: &TensorData,
        returns: &TensorData,
        old_values: &TensorData,
        config: &PPOStepConfig,
    ) -> Result<TrainMetrics, NNError>;

    /// Get/set learning rate (for annealing).
    fn learning_rate(&self) -> f32;
    fn set_learning_rate(&mut self, lr: f32);

    /// Serialize model state for checkpointing.
    fn save(&self, path: &std::path::Path) -> Result<(), NNError>;
    fn load(&mut self, path: &std::path::Path) -> Result<(), NNError>;
}

/// Configuration passed into `ppo_step` so the trait impl can compute
/// the clipped loss without the caller needing to manipulate tensors.
#[derive(Debug, Clone)]
pub struct PPOStepConfig {
    pub clip_eps: f32,
    pub vf_coef: f32,
    pub ent_coef: f32,
    pub max_grad_norm: f32,
    pub clip_vloss: bool,
}

/// Training metrics dictionary.
#[derive(Debug, Clone, Default)]
pub struct TrainMetrics {
    pub entries: Vec<(String, f64)>,
}

impl TrainMetrics {
    pub fn insert(&mut self, key: impl Into<String>, value: f64) {
        self.entries.push((key.into(), value));
    }

    pub fn get(&self, key: &str) -> Option<f64> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
    }
}

/// The core trait for Q-value networks used in off-policy algorithms
/// (DQN, SAC critics, TD3 critics).
pub trait QFunction: Send + Sync {
    /// Compute Q-values. For DQN: obs -> [batch, n_actions].
    /// For SAC/TD3: (obs, action) -> [batch, 1].
    fn q_values(&self, obs: &TensorData, actions: Option<&TensorData>)
        -> Result<TensorData, NNError>;

    /// Perform one gradient step given TD targets.
    fn td_step(
        &mut self,
        obs: &TensorData,
        actions: &TensorData,
        targets: &TensorData,
        weights: Option<&TensorData>, // for PER
    ) -> Result<(f64, TensorData), NNError>; // (loss, td_errors)
}

/// Stochastic policy for SAC.
pub trait StochasticPolicy: Send + Sync {
    /// Sample actions with reparameterization trick, returning
    /// (squashed_actions, log_probs). Differentiable.
    fn sample(&self, obs: &TensorData)
        -> Result<(TensorData, TensorData), NNError>;

    /// Deterministic action (mean through squashing).
    fn deterministic(&self, obs: &TensorData) -> Result<TensorData, NNError>;

    /// Backprop through actor loss and step optimizer.
    fn actor_step(
        &mut self,
        obs: &TensorData,
        critic: &dyn QFunction,
    ) -> Result<f64, NNError>; // returns actor_loss
}

/// Deterministic policy for TD3.
pub trait DeterministicPolicy: Send + Sync {
    fn act(&self, obs: &TensorData) -> Result<TensorData, NNError>;

    fn actor_step(
        &mut self,
        obs: &TensorData,
        critic: &dyn QFunction,
    ) -> Result<f64, NNError>;
}

/// Manages target network synchronization.
pub trait HasTargetNetwork {
    /// Polyak update: target = tau * self + (1-tau) * target
    fn soft_update(&mut self, tau: f64);

    /// Hard copy: target = self
    fn hard_update(&mut self);
}
```

### Why `TensorData` as the Boundary Type

The crucial insight: **the trait boundary passes `&[f32]` data, not backend tensors**. This means:
1. rlox-core's existing buffer/GAE code (which operates on `&[f64]` / `&[f32]` slices) integrates naturally.
2. No backend tensor types leak into algorithm code.
3. The cost of copying data across the boundary is negligible compared to the NN forward/backward pass (a 256x2048 rollout is ~2MB, memcpy takes <1ms, forward pass takes >10ms).
4. Each backend implementation converts `TensorData -> BackendTensor` internally, runs computation, converts back. This is what happens at the Python<->PyTorch boundary too.

### Alternative Considered: Generic Tensor Type

```rust
// This approach was rejected:
trait ActorCritic<T: TensorOps> {
    fn act(&self, obs: &T) -> Result<ActionOutput<T>, NNError>;
}
```

This forces `T` to propagate through the entire algorithm codebase. Every function that touches a tensor becomes generic over `T`. This is Burn's approach, and it works when you ARE the framework. For a library that wraps frameworks, it creates painful type gymnastics with no benefit -- you just end up monomorphizing to a single backend anyway.

---

## 6. Crate Structure

```
rlox/
  crates/
    rlox-core/         # Existing: buffers, GAE, envs (no NN deps)
    rlox-nn/           # NEW: trait definitions (TensorData, ActorCritic, etc.)
                       #   Zero dependencies beyond thiserror.
                       #   This is the "contract" crate.
    rlox-burn/         # NEW: Burn backend implementation
                       #   Depends on: rlox-nn, burn
    rlox-candle/       # FUTURE: Candle backend implementation
    rlox-tch/          # FUTURE: tch-rs backend implementation
    rlox-python/       # Existing: PyO3 bindings
    rlox-bench/        # Existing: benchmarks
```

**Why a separate `rlox-nn` crate?**
- `rlox-core` stays dependency-free (just rayon, rand, serde, thiserror). This is important for the publication story: the core computational primitives have zero heavy deps.
- Backend crates depend on `rlox-nn` for the trait, not on `rlox-core`. This keeps the dependency graph clean.
- Algorithm implementations in a future `rlox-algo` crate (or in `rlox-core`) depend on `rlox-nn` traits but not on any specific backend.

### Dependency Graph

```
rlox-core (buffers, GAE, envs)
    |
    v
rlox-nn (traits, TensorData) <--- rlox-burn (impl with burn)
    |                          <--- rlox-candle (impl with candle)
    v                          <--- rlox-tch (impl with tch-rs)
rlox-algo (PPO, SAC, TD3, DQN -- generic over traits)
    |
    v
rlox-python (PyO3 bindings, wires concrete backend)
```

---

## 7. What a Concrete Implementation Looks Like (Burn)

```rust
// In rlox-burn/src/lib.rs

use burn::prelude::*;
use burn::module::Module;
use burn::nn;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use rlox_nn::{ActorCritic, ActionOutput, EvalOutput, NNError, PPOStepConfig,
              TensorData, TrainMetrics};

#[derive(Module, Debug)]
pub struct BurnDiscretePolicy<B: burn::tensor::backend::AutodiffBackend> {
    actor: Vec<nn::Linear<B>>,   // or a custom Sequential
    critic: Vec<nn::Linear<B>>,
}

pub struct BurnActorCritic<B: burn::tensor::backend::AutodiffBackend> {
    model: BurnDiscretePolicy<B>,
    optimizer: burn::optim::adaptor::OptimizerAdaptor<
        burn::optim::Adam<B::InnerBackend>,
        BurnDiscretePolicy<B>,
        B,
    >,
    device: B::Device,
    lr: f32,
}

impl<B: burn::tensor::backend::AutodiffBackend> ActorCritic for BurnActorCritic<B>
where
    B::FloatElem: From<f32>,
{
    fn act(&self, obs: &TensorData) -> Result<ActionOutput, NNError> {
        // 1. Convert TensorData -> Burn Tensor
        // 2. Forward through actor (no_grad context)
        // 3. Sample from Categorical
        // 4. Convert back to TensorData
        todo!()
    }

    fn value(&self, obs: &TensorData) -> Result<TensorData, NNError> {
        todo!()
    }

    fn evaluate(&self, obs: &TensorData, actions: &TensorData)
        -> Result<EvalOutput, NNError>
    {
        todo!()
    }

    fn ppo_step(
        &mut self,
        obs: &TensorData,
        actions: &TensorData,
        old_log_probs: &TensorData,
        advantages: &TensorData,
        returns: &TensorData,
        old_values: &TensorData,
        config: &PPOStepConfig,
    ) -> Result<TrainMetrics, NNError> {
        // 1. Convert all TensorData -> Burn tensors
        // 2. Forward pass (with autograd)
        // 3. Compute clipped PPO loss (all in Burn tensor ops)
        // 4. backward()
        // 5. Clip gradients
        // 6. optimizer.step()
        // 7. Extract metrics as f64
        // 8. Return TrainMetrics
        todo!()
    }

    fn learning_rate(&self) -> f32 { self.lr }
    fn set_learning_rate(&mut self, lr: f32) { self.lr = lr; }

    fn save(&self, path: &std::path::Path) -> Result<(), NNError> {
        // Burn has built-in model serialization
        todo!()
    }

    fn load(&mut self, path: &std::path::Path) -> Result<(), NNError> {
        todo!()
    }
}
```

### Burn Backend Selection at Compile Time

```rust
// In user code or rlox-python:

#[cfg(feature = "wgpu")]
type MyBackend = burn::backend::Autodiff<burn::backend::Wgpu>;

#[cfg(feature = "libtorch")]
type MyBackend = burn::backend::Autodiff<burn::backend::LibTorch>;

#[cfg(feature = "candle")]
type MyBackend = burn::backend::Autodiff<burn::backend::Candle>;

fn create_policy(obs_dim: usize, act_dim: usize) -> Box<dyn ActorCritic> {
    Box::new(BurnActorCritic::<MyBackend>::new(obs_dim, act_dim))
}
```

Note the irony: by using Burn as the first backend, you automatically get Candle, LibTorch, WGPU, and NdArray as sub-backends through Burn's own abstraction. This is a major advantage.

---

## 8. Backend Prioritization

### Phase 1: Burn (via `rlox-burn`)
- Burn gives you 4 GPU backends for free (WGPU, LibTorch, Candle, NdArray).
- WGPU works on macOS/Metal, Linux/Vulkan, Windows/DX12 without any system deps.
- LibTorch backend gives PyTorch-equivalent numerics for validation.
- Burn has the best Rust-native `Module` system with derive macros.
- **This single implementation covers 90% of use cases.**

### Phase 2 (defer): Direct Candle (`rlox-candle`)
- Only worth doing if someone needs Candle's lighter weight without Burn's overhead.
- Candle's Metal support is independent of Burn's.
- Relevant for the LLM post-training path (Candle has strong HuggingFace model support).

### Phase 3 (defer): Direct tch-rs (`rlox-tch`)
- Only if someone needs exact LibTorch parity or has an existing LibTorch installation.
- The Burn LibTorch backend already covers this, but direct tch-rs avoids Burn's indirection.

### Do NOT implement: ort, tract (deploy trained models via ONNX export from Burn instead), dfdx (dead), luminal (immature).

---

## 9. Impact on Existing Code

### No changes to `rlox-core`
- GAE, V-trace, KL controller, buffers -- all operate on `&[f32]`/`&[f64]` slices.
- They feed into `TensorData` naturally: `TensorData::new(advantages.iter().map(|x| *x as f32).collect(), vec![n])`.

### Algorithm code moves to `rlox-algo` (or stays in Python initially)
- The PPO training loop from `python/rlox/algorithms/ppo.py` translates to:
  ```rust
  fn ppo_update(
      policy: &mut dyn ActorCritic,
      batch: &RolloutBatch,   // from rlox-core buffers
      config: &PPOConfig,
  ) -> Result<TrainMetrics, NNError> {
      for _epoch in 0..config.n_epochs {
          for mb in batch.minibatches(config.batch_size) {
              let advantages = normalize(&mb.advantages);
              let metrics = policy.ppo_step(
                  &mb.obs, &mb.actions, &mb.old_log_probs,
                  &advantages, &mb.returns, &mb.old_values,
                  &config.step_config(),
              )?;
          }
      }
  }
  ```
- The algorithm code is backend-agnostic. It calls trait methods.

### Python bindings
- `rlox-python` can expose a `NativePolicy` class that wraps `Box<dyn ActorCritic>`.
- Users who want to stay in PyTorch can keep using the existing Python policies.
- Users who want pure-Rust training can use the native policy.

---

## 10. Implementation Effort Estimate

| Work Item | Effort | Notes |
|-----------|--------|-------|
| `rlox-nn` crate (traits, TensorData, errors) | 2-3 days | Small, well-defined |
| `rlox-burn` MLP + Categorical distribution | 3-5 days | Burn Module derive, forward/backward |
| `rlox-burn` PPO `ppo_step` implementation | 2-3 days | Direct translation from Python |
| `rlox-burn` SAC traits + twin Q + squashed Gaussian | 3-5 days | More complex distributions |
| `rlox-burn` DQN traits implementation | 2-3 days | Simpler than SAC |
| Integration tests (PPO solving CartPole natively) | 2-3 days | End-to-end validation |
| Benchmark: native vs Python+PyTorch PPO | 1-2 days | The publishable result |
| **Total for Phase 1 (PPO + DQN working natively)** | **~3 weeks** | |
| Phase 2 additions (SAC, TD3, checkpointing) | 2-3 weeks | |

---

## 11. Publication Impact

### Positive
- "Backend-agnostic RL in Rust" is a strong narrative. No other Rust RL library has this.
- Benchmarks showing rlox end-to-end training (not just buffer/GAE) outperforming SB3/TorchRL would be significant.
- The trait design itself is a contribution -- there is no established pattern for this in the Rust ML ecosystem.

### Risk
- If Burn is the only backend with a real implementation, the "backend-agnostic" claim may feel hollow.
- Mitigation: Burn's own multi-backend support (WGPU, LibTorch, Candle, NdArray) means you genuinely run on multiple backends through one implementation.

### Recommended framing
Position rlox as: "Rust RL training with pluggable backends, shipping with Burn (which itself supports WGPU/CUDA/Metal)." The trait design enables future backends without the paper needing to demonstrate all of them.

---

## 12. Open Design Questions

1. **Should `ppo_step` be on the trait or should loss computation be separate?**
   Putting it on the trait is simpler but means each backend re-implements the loss. Pulling it out requires a generic tensor abstraction. Recommendation: keep it on the trait. The loss is <50 lines and each backend needs to express it in its own tensor API anyway.

2. **f32 vs f64?**
   RL training universally uses f32. The existing `rlox-core` GAE/V-trace use f64 for numerical stability in accumulation, which is fine -- the cast to f32 happens at the `TensorData` boundary. This matches PyTorch's default behavior.

3. **Async training?**
   Not in Phase 1. The trait methods are synchronous. GPU operations are already async under the hood (Burn/Candle batch and pipeline internally). If needed later, add `async` variants as a separate trait.

4. **Distribution abstraction?**
   Keep distributions INSIDE the backend implementations, not as a separate trait. Reason: `Categorical`, `Normal`, and their `log_prob`/`entropy`/`rsample` methods are tightly coupled to the backend's tensor type and autograd. Abstracting them separately gains nothing -- the only consumer is the policy implementation within the same backend crate.

5. **Continuous vs Discrete action spaces?**
   The `ActorCritic` trait handles both via `TensorData`. For discrete: actions are `[batch_size]` with integer values stored as f32. For continuous: actions are `[batch_size, act_dim]`. The backend implementation knows which distribution to use based on its construction (obs_dim, act_dim vs obs_dim, n_actions).
