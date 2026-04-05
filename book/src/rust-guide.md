# rlox Rust User Guide

This guide documents `rlox-core`, the pure-Rust crate that powers rlox's data plane. Use this if you want to embed rlox's RL primitives in a Rust application without any Python dependency.

## Crate Architecture

```
rlox-core/
  src/
    env/         # RLEnv trait + CartPole + Pendulum + MuJoCo + VecEnv
    buffer/      # Ring, PER, Sequence, HER, Concurrent, Mmap, Mixed, Flat/GPU
    training/    # GAE, V-trace, augmentation, reward shaping, weight ops, SIMD
    llm/         # GRPO, DPO, token KL
    error.rs     # Unified error type
    seed.rs      # Deterministic seeding utilities
```

`rlox-core` has **no PyO3 dependency** and can be tested independently with `cargo test --package rlox-core`. The sibling `rlox-python` crate provides thin PyO3 wrappers that expose these APIs to Python.

---

## Environment Engine

### The `RLEnv` Trait

All environments implement `RLEnv`:

```rust
pub trait RLEnv: Send {
    fn step(&mut self, action: &Action) -> Result<Transition, RloxError>;
    fn reset(&mut self, seed: Option<u64>) -> Result<Observation, RloxError>;
    fn action_space(&self) -> &ActionSpace;
    fn obs_space(&self) -> &ObsSpace;
    fn render(&self) -> Option<String>;
}
```

Key types:

| Type | Description |
|------|-------------|
| `Observation(Vec<f32>)` | Flat observation vector |
| `Action::Discrete(u32)` | Discrete action |
| `Action::Continuous(Vec<f32>)` | Continuous action vector |
| `Transition` | Step result: `obs`, `reward`, `terminated`, `truncated`, `info: Option<HashMap>` |
| `ActionSpace::Discrete(n)` | Discrete space with `n` actions |
| `ObsSpace::Box { low, high, shape }` | Bounded continuous observation space |

### CartPole

Faithful port of Gymnasium's `CartPole-v1` with identical physics constants and Euler integration:

```rust
use rlox_core::env::builtins::CartPole;
use rlox_core::env::spaces::Action;
use rlox_core::env::RLEnv;

let mut env = CartPole::new(Some(42));
let obs = env.reset(Some(42)).unwrap();
assert_eq!(obs.as_slice().len(), 4);

let transition = env.step(&Action::Discrete(1)).unwrap();
assert!((transition.reward - 1.0).abs() < f64::EPSILON);
```

Physics constants match Gymnasium exactly:
- Gravity: 9.8 m/s^2
- Cart mass: 1.0 kg, pole mass: 0.1 kg
- Pole half-length: 0.5 m
- Force magnitude: 10.0 N
- Timestep: 0.02 s
- Termination: |x| > 2.4 or |theta| > 12 deg
- Truncation: 500 steps

### VecEnv (Parallel Stepping)

`VecEnv` wraps multiple environments and steps them in parallel using Rayon work-stealing:

```rust
use rlox_core::env::builtins::CartPole;
use rlox_core::env::parallel::VecEnv;
use rlox_core::env::spaces::Action;
use rlox_core::env::RLEnv;
use rlox_core::seed::derive_seed;

// Create 64 parallel CartPole environments
let envs: Vec<Box<dyn RLEnv>> = (0..64)
    .map(|i| {
        Box::new(CartPole::new(Some(derive_seed(42, i)))) as Box<dyn RLEnv>
    })
    .collect();
// VecEnv::new returns Result — validates that all envs have matching spaces
let mut vec_env = VecEnv::new(envs).unwrap();

// Reset all environments
let observations = vec_env.reset_all(Some(42)).unwrap();
assert_eq!(observations.len(), 64);

// Step all in parallel
let actions: Vec<Action> = (0..64)
    .map(|i| Action::Discrete((i % 2) as u32))
    .collect();
let batch = vec_env.step_all(&actions).unwrap();

// Access columnar results
assert_eq!(batch.obs.len(), 64);       // Vec<Vec<f32>>
assert_eq!(batch.rewards.len(), 64);   // Vec<f64>
assert_eq!(batch.terminated.len(), 64); // Vec<bool>
assert_eq!(batch.truncated.len(), 64);  // Vec<bool>
```

> **Note:** `VecEnv::new` returns `Result<VecEnv, RloxError>`. It validates that all environments share compatible action and observation spaces. Use `?` in fallible contexts or `.unwrap()` in examples.

> **Note:** `Transition.info` is `Option<HashMap<String, f64>>`. Most environments return `None` for info; only environments that produce auxiliary metadata (e.g., episode statistics) populate this field. Check with `if let Some(info) = transition.info { ... }`.

**Auto-reset**: When an environment terminates or truncates, `VecEnv` automatically resets it. The returned `batch.obs` contains the post-reset observation, while the pre-reset observation is available in `batch.terminal_obs[i]` (needed for value bootstrapping).

**Deterministic seeding**: `derive_seed(master, index)` derives per-environment seeds deterministically. Parallel execution order does not affect results.

**Scaling behaviour**: At >= 64 environments, Rayon parallelism delivers 3-6x speedup over sequential stepping. At < 4 environments with lightweight workloads like CartPole (~37ns/step), scheduling overhead exceeds compute.

---

## Experience Buffers

### ExperienceRecord

All buffers accept `ExperienceRecord`:

```rust
use rlox_core::buffer::ExperienceRecord;

let record = ExperienceRecord {
    obs: vec![0.1, 0.2, 0.3, 0.4],  // f32
    action: vec![0.5],                // f32, supports multi-dim
    reward: 1.0,                      // f32
    terminated: false,
    truncated: false,
};
```

Actions are `Vec<f32>` to support both discrete (stored as single f32) and continuous multi-dimensional actions.

### ExperienceTable (On-Policy)

Append-only columnar store for on-policy algorithms. Optimised for sequential writes and bulk reads:

```rust
use rlox_core::buffer::columnar::ExperienceTable;
use rlox_core::buffer::ExperienceRecord;

let mut table = ExperienceTable::new(4, 1); // obs_dim=4, act_dim=1

table.push(ExperienceRecord {
    obs: vec![0.1, 0.2, 0.3, 0.4],
    action: vec![0.0],
    reward: 1.0,
    terminated: false,
    truncated: false,
}).unwrap();

assert_eq!(table.len(), 1);

// Bulk access for advantage computation
let obs: &[f32] = table.observations_raw();
let rewards: &[f32] = table.rewards_raw();

// Clear for next rollout (retains allocated capacity)
table.clear();
```

### ReplayBuffer (Off-Policy)

Fixed-capacity ring buffer with uniform random sampling. Pre-allocates all arrays at construction for zero-allocation push:

```rust
use rlox_core::buffer::ringbuf::ReplayBuffer;
use rlox_core::buffer::ExperienceRecord;

let mut buf = ReplayBuffer::new(100_000, 4, 1);

// O(1) push — overwrites oldest when full
for i in 0..200_000 {
    buf.push(ExperienceRecord {
        obs: vec![i as f32; 4],
        action: vec![0.0],
        reward: 1.0,
        terminated: false,
        truncated: false,
    }).unwrap();
}
assert_eq!(buf.len(), 100_000); // capped at capacity

// Deterministic sampling with ChaCha8 RNG
let batch = buf.sample(32, /*seed=*/42).unwrap();
assert_eq!(batch.batch_size, 32);
assert_eq!(batch.observations.len(), 32 * 4);
assert_eq!(batch.rewards.len(), 32);
```

Sampling uses `ChaCha8Rng` seeded per call, giving deterministic cross-platform reproducibility.

### PrioritizedReplayBuffer

Sum-tree backed buffer for Prioritized Experience Replay (Schaul et al., 2016). Provides O(log N) sampling proportional to priority and importance-sampling weight correction:

```rust
use rlox_core::buffer::priority::PrioritizedReplayBuffer;
use rlox_core::buffer::ExperienceRecord;

let mut buf = PrioritizedReplayBuffer::new(
    100_000, // capacity
    4,       // obs_dim
    1,       // act_dim
    0.6,     // alpha (priority exponent)
    0.4,     // beta (IS correction exponent)
);

// Push with initial priority (typically |TD error| + epsilon)
buf.push(ExperienceRecord {
    obs: vec![0.1, 0.2, 0.3, 0.4],
    action: vec![0.0],
    reward: 1.0,
    terminated: false,
    truncated: false,
}, 1.0).unwrap();

// Stratified sampling
let batch = buf.sample(32, 42).unwrap();
let weights = &batch.weights;   // IS weights in [0, 1]
let indices = &batch.indices;   // for priority updates

// Update priorities after computing new TD errors
buf.update_priorities(&indices, &new_td_errors).unwrap();

// Anneal beta toward 1.0 over training
buf.set_beta(0.7);
```

The `SumTree` data structure stores `2 * capacity` nodes where leaves hold `priority^alpha` and internal nodes hold partial sums. Stratified sampling divides the total priority into `batch_size` equal segments and samples one index per segment.

### VarLenStore

Packed variable-length sequence storage for LLM token sequences:

```rust
use rlox_core::buffer::varlen::VarLenStore;

let mut store = VarLenStore::new();
store.push(&[1, 2, 3]);
store.push(&[4, 5]);

assert_eq!(store.num_sequences(), 2);
assert_eq!(store.total_elements(), 5);
assert_eq!(store.get(0), &[1, 2, 3]);
assert_eq!(store.get(1), &[4, 5]);
```

Internally uses a flat `Vec<u32>` with an offsets array. Memory-efficient for heterogeneous sequence lengths common in LLM training.

---

## Training Core

### GAE (Generalized Advantage Estimation)

The core advantage computation used by PPO and A2C. Implements the backward scan from Schulman et al. (2016):

```rust
use rlox_core::training::gae::compute_gae;

let rewards = &[1.0, 1.0, 1.0, 0.0, 1.0];
let values  = &[0.5, 0.6, 0.7, 0.3, 0.8]; // V(s_t)
let dones   = &[0.0, 0.0, 0.0, 1.0, 0.0]; // 1.0 = episode boundary

let (advantages, returns) = compute_gae(
    rewards,
    values,
    dones,
    0.9,   // last_value: V(s_{T+1}) bootstrap
    0.99,  // gamma: discount factor
    0.95,  // gae_lambda: bias-variance tradeoff
);

assert_eq!(advantages.len(), 5);
assert_eq!(returns.len(), 5);
// Invariant: returns[t] == advantages[t] + values[t]
for t in 0..5 {
    assert!((returns[t] - (advantages[t] + values[t])).abs() < 1e-10);
}
```

**Special cases**:
- `gae_lambda = 0`: Reduces to one-step TD error
- `gae_lambda = 1`: Reduces to Monte Carlo returns
- Empty input: Returns empty vectors

**Performance**: 140x faster than a Python loop, 1700x faster than TorchRL. The backward scan is purely sequential with no allocation beyond the output vectors.

### V-trace

Off-policy correction for distributed RL (Espeholt et al., 2018). Used by IMPALA:

```rust
use rlox_core::training::vtrace::compute_vtrace;

let log_rhos = &[0.2, -0.3, 0.8]; // log(pi/mu)
let rewards  = &[1.0, 2.0, 3.0];
let values   = &[0.5, 1.0, 1.5];

let (vs, pg_advantages) = compute_vtrace(
    log_rhos,
    rewards,
    values,
    2.0,   // bootstrap_value
    0.99,  // gamma
    1.0,   // rho_bar: importance weight clip for values
    1.0,   // c_bar: trace-cutting coefficient clip
).unwrap();

assert_eq!(vs.len(), 3);
assert_eq!(pg_advantages.len(), 3);
```

The implementation clips importance weights at both `rho_bar` and `c_bar` thresholds:
- `rho_bar` limits the variance of value updates
- `c_bar` controls how far back the trace propagates

When on-policy (`log_rhos = 0`), V-trace reduces to GAE with `lambda = 1`.

### RunningStats (Welford's Algorithm)

Online mean and variance computation for reward/observation normalisation:

```rust
use rlox_core::training::normalization::RunningStats;

let mut stats = RunningStats::new();
stats.update(1.0);
stats.update(2.0);
stats.update(3.0);

assert!((stats.mean() - 2.0).abs() < 1e-10);
assert!((stats.std() - 0.8165).abs() < 1e-3);
assert_eq!(stats.count(), 3);

// Efficient batch update
stats.batch_update(&[4.0, 5.0, 6.0]);
assert_eq!(stats.count(), 6);
```

Uses Welford's numerically stable single-pass algorithm. O(1) per update.

### Sequence Packing

Packs variable-length token sequences into fixed-length batches for efficient transformer training:

```rust
use rlox_core::training::packing::pack_sequences;

let sequences = vec![
    vec![1u32, 2, 3],
    vec![4, 5],
    vec![6, 7, 8, 9],
];

let packed = pack_sequences(&sequences, 8); // max_length = 8
// Each packed batch contains:
// - input_ids: padded token IDs
// - attention_mask: 1 for real tokens, 0 for padding
// - position_ids: per-sequence position indices
// - sequence_starts: start index of each packed sequence
```

---

## LLM Operations

### GRPO Group Advantages

Group Relative Policy Optimization (Shao et al., 2024) computes advantages relative to the group rather than a learned baseline:

```rust
use rlox_core::llm::ops::compute_group_advantages;

let rewards = &[0.5, 1.2, 0.8, 0.3];
let advantages = compute_group_advantages(rewards);

// advantages = (rewards - mean) / std
// Sum of advantages is approximately zero
let sum: f64 = advantages.iter().sum();
assert!(sum.abs() < 1e-10);

// Constant rewards produce zero advantages
let constant = compute_group_advantages(&[5.0, 5.0, 5.0]);
assert!(constant.iter().all(|&v| v == 0.0));
```

### Token KL Divergence

Forward KL divergence at the token level, used as a regularisation penalty in RLHF:

```rust
use rlox_core::llm::ops::compute_token_kl;

let log_p = &[-1.0, -2.0, -0.5]; // policy log-probs
let log_q = &[-1.0, -2.0, -0.5]; // reference log-probs

// Identical distributions: KL = 0
let kl = compute_token_kl(log_p, log_q).unwrap();
assert!(kl.abs() < 1e-15);

// Mismatched lengths: returns Err
let result = compute_token_kl(&[-1.0, -2.0], &[-1.0]);
assert!(result.is_err());
```

Computes $\sum_t \exp(\log p_t) \cdot (\log p_t - \log q_t)$.

### DPOPair

Container for Direct Preference Optimization preference data:

```rust
use rlox_core::llm::ops::DPOPair;

let pair = DPOPair::new(
    vec![1, 2, 3],       // prompt tokens
    vec![4, 5],           // chosen completion
    vec![6, 7, 8],        // rejected completion
);

assert_eq!(pair.chosen_len(), 2);
assert_eq!(pair.rejected_len(), 3);
```

---

## Advanced Buffers

### SequenceReplayBuffer (Episode-Aware)

Wraps `ReplayBuffer` + `EpisodeTracker` for sampling contiguous sequences within episodes. Essential for DreamerV3, Decision Transformer, and R2D2:

```rust
use rlox_core::buffer::sequence::SequenceReplayBuffer;

let mut buf = SequenceReplayBuffer::new(10_000, 4, 1); // capacity, obs_dim, act_dim

// Push transitions — mark episode boundaries with done=true
for step in 0..100 {
    let done = step % 20 == 19; // episodes of length 20
    buf.push_slices(
        &[step as f32; 4],  // obs
        &[(step + 1) as f32; 4], // next_obs
        &[0.5],             // action
        1.0, false, done,   // reward, terminated, truncated (done)
    ).unwrap();
}

assert_eq!(buf.num_complete_episodes(), 5);

// Sample contiguous sequences of length 10 (never crosses episode boundaries)
let batch = buf.sample_sequences(8, 10, /*seed=*/42).unwrap();
assert_eq!(batch.batch_size, 8);
assert_eq!(batch.seq_len, 10);
assert_eq!(batch.observations.len(), 8 * 10 * 4); // flat [B * T * obs_dim]
```

### HERBuffer (Hindsight Experience Replay)

Goal-conditioned buffer with automatic goal relabeling for sparse-reward robotics:

```rust
use rlox_core::buffer::her::{HERBuffer, HERStrategy};

let mut buf = HERBuffer::new(
    10_000, 4, 1, 3, // capacity, obs_dim, act_dim, goal_dim
    HERStrategy::Future { k: 4 }, // relabel with 4 future achieved goals
    0.5, // her_ratio: 50% relabeled, 50% original
    0.05, // goal tolerance for sparse reward
);

// Push goal-conditioned transitions
buf.push_slices(
    &[0.1, 0.2, 0.3, 0.4], // obs
    &[0.2, 0.3, 0.4, 0.5], // next_obs
    &[0.5],                  // action
    -1.0, false, false,      // reward (sparse: -1 until goal reached)
    &[0.2, 0.3, 0.4],       // achieved_goal
    &[1.0, 1.0, 1.0],       // desired_goal
).unwrap();
```

**Relabeling strategies:**
- `HERStrategy::Final` — relabel with the final achieved goal in the episode
- `HERStrategy::Future { k }` — relabel with k random future achieved goals
- `HERStrategy::Episode` — relabel with any achieved goal from the same episode

### ConcurrentReplayBuffer (Lock-Free)

Multi-producer, single-consumer buffer for IMPALA-style actor-learner architectures:

```rust
use rlox_core::buffer::concurrent::ConcurrentReplayBuffer;
use std::sync::Arc;
use std::thread;

let buf = Arc::new(ConcurrentReplayBuffer::new(100_000, 4, 1));

// Multiple actor threads push concurrently — no locks
let handles: Vec<_> = (0..4).map(|actor_id| {
    let buf = Arc::clone(&buf);
    thread::spawn(move || {
        for i in 0..1000 {
            buf.push(
                &[actor_id as f32; 4], // obs
                &[0.0; 4],             // next_obs
                &[0.5],                // action
                1.0, false, false,     // reward, terminated, truncated
            ).unwrap();
        }
    })
}).collect();

for h in handles { h.join().unwrap(); }
assert_eq!(buf.len(), 4000);

// Single learner thread samples
let batch = buf.sample(32, 42).unwrap();
assert_eq!(batch.batch_size, 32);
```

Uses `AtomicUsize` for lock-free slot claiming with per-slot commit flags.

---

## Training Extensions

### Image Augmentation (DrQ-v2)

Random spatial shift for pixel-based RL. Operates on flat `f32` image arrays for zero-copy integration:

```rust
use rlox_core::training::augmentation::{RandomShift, ImageAugmentation};

let augmenter = RandomShift { pad: 4 };

// Batch of 16 RGB images, 84x84
let images = vec![0.5f32; 16 * 3 * 84 * 84];
let augmented = augmenter.augment_batch(&images, 16, 3, 84, 84, /*seed=*/42).unwrap();
assert_eq!(augmented.len(), images.len()); // same shape
```

The `ImageAugmentation` trait allows adding custom augmentations (CutMix, color jitter, etc.) without modifying existing code.

### Reward Shaping (PBRS)

Potential-Based Reward Shaping — guaranteed to preserve optimal policy (Ng et al., 1999):

```rust
use rlox_core::training::reward_shaping::shape_rewards_pbrs;

let rewards = &[1.0, 1.0, 1.0, 1.0];
let phi_current = &[0.5, 0.6, 0.7, 0.8]; // potential of current state
let phi_next = &[0.6, 0.7, 0.8, 0.9];    // potential of next state
let gamma = 0.99;
let dones = &[0.0, 0.0, 0.0, 1.0];

// r' = r + gamma * phi(s') - phi(s), zeroed at episode boundaries
let shaped = shape_rewards_pbrs(rewards, phi_current, phi_next, gamma, dones).unwrap();
assert_eq!(shaped.len(), 4);
assert!((shaped[3] - 1.0).abs() < 1e-10); // done=1 → raw reward only
```

### Weight Operations (Meta-Learning)

Vectorized weight updates for Reptile, Polyak, and averaging:

```rust
use rlox_core::training::weight_ops::{reptile_update, polyak_update, average_weight_vectors};

// Reptile outer step: params += lr * (task_params - params)
let mut meta_params = vec![1.0f32, 2.0, 3.0];
let task_params = vec![4.0f32, 5.0, 6.0];
reptile_update(&mut meta_params, &task_params, 0.1);
// meta_params is now [1.3, 2.3, 3.3]

// Polyak target update: target = tau * source + (1-tau) * target
let mut target = vec![0.0f32; 3];
let source = vec![1.0f32; 3];
polyak_update(&mut target, &source, 0.005);
// target is now [0.005, 0.005, 0.005]

// Average multiple weight vectors (meta-learning batch)
let v1 = vec![1.0f32, 2.0, 3.0];
let v2 = vec![3.0f32, 2.0, 1.0];
let avg = average_weight_vectors(&[&v1, &v2]);
assert!((avg[0] - 2.0).abs() < 1e-6);
```

### SIMD-Accelerated Operations

Vectorized versions of hot-path operations using `chunks_exact` patterns for LLVM auto-vectorization:

```rust
use rlox_core::training::simd_ops::{
    reptile_update_simd,
    polyak_update_simd,
    pbrs_simd,
    compute_priorities_simd,
    average_weights_simd,
};

// Process 8 f32s at a time (AVX2 = 256-bit = 8 x f32)
let mut target = vec![0.0f32; 1000];
let source = vec![1.0f32; 1000];
reptile_update_simd(&mut target, &source, 0.01);

// Process 4 f64s at a time for reward shaping
let rewards = vec![1.0f64; 500];
let phi = vec![0.5f64; 500];
let phi_next = vec![0.6f64; 500];
let dones = vec![0.0f64; 500];
let shaped = pbrs_simd(&rewards, &phi, &phi_next, 0.99, &dones);
```

### EpisodeTracker

Reusable episode boundary tracker shared by `SequenceReplayBuffer` and `HERBuffer`:

```rust
use rlox_core::buffer::episode::EpisodeTracker;

let mut tracker = EpisodeTracker::new(10_000); // ring capacity

// Notify on each push
tracker.notify_push(0, false); // step 0, not done
tracker.notify_push(1, false); // step 1, not done
tracker.notify_push(2, true);  // step 2, episode ends

assert_eq!(tracker.num_complete_episodes(), 1);

// Sample windows of length 2 from complete episodes
let windows = tracker.sample_windows(4, 2, /*seed=*/42).unwrap();
```

---

## Error Handling

All fallible operations return `Result<T, RloxError>`:

```rust
pub enum RloxError {
    ShapeMismatch { expected: String, got: String },
    BufferError(String),
    EnvError(String),
    InvalidAction(String),
}
```

Shape mismatches (wrong obs_dim, mismatched slice lengths) are caught at the Rust boundary with descriptive error messages that propagate as Python exceptions through PyO3.

---

## Seeding

```rust
use rlox_core::seed::{rng_from_seed, derive_seed};

// Create a ChaCha8 RNG from a seed
let rng = rng_from_seed(42);

// Deterministic child seeds for parallel environments
let seed_0 = derive_seed(42, 0);
let seed_1 = derive_seed(42, 1);
assert_ne!(seed_0, seed_1);
```

All RNG uses `ChaCha8Rng` for:
1. Cross-platform determinism (same results on macOS, Linux, Windows)
2. Low latency (matters for sampling-heavy workloads)
3. Cryptographic quality randomness (no bias in sampling)

---

## Thread Safety

All core types are `Send + Sync`:

```rust
fn assert_send_sync<T: Send + Sync>() {}
assert_send_sync::<ReplayBuffer>();
assert_send_sync::<CartPole>();
```

`VecEnv` holds `Vec<Box<dyn RLEnv>>` where `RLEnv: Send`, enabling Rayon's parallel iteration.

---

## Adding to Your Project

```toml
# Cargo.toml
[dependencies]
rlox-core = { path = "crates/rlox-core" }
```

Or, if published:

```toml
[dependencies]
rlox-core = "1.0"
```

---

## Cross-References

- [Mathematical Reference](math-reference.md) -- full derivations of GAE, V-trace, and all algorithms
- [Python User Guide](python-guide.md) -- using these primitives from Python via PyO3
- [References](references.md) -- academic papers behind each algorithm
