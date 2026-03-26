# rlox Rust Optimization Plan

## Overview

| # | Item | File | Impact | Effort |
|---|------|------|--------|--------|
| P1 | AsyncCollector serial strided GAE | `pipeline/collector.rs:167-193` | 2-5x GAE | Medium |
| P2 | VecEnv per-env obs allocation | `env/parallel.rs`, `builtins.rs:69` | ~10M allocs/s removed | High |
| P3 | PyO3 push() double-copy | `rlox-python/buffer.rs:222-239` | 40-50% push latency | Low |
| P4 | SampledBatch fresh alloc per sample() | `buffer/ringbuf.rs:199-244` | ~56MB/step (Atari) | Low |
| P5 | PER tree_min_prob O(N) | `buffer/priority.rs:320-337` | O(N) -> O(1) | Medium |
| P6 | Release profile optimizations | `Cargo.toml` | 5-15% all code | Trivial |
| P7 | GAE inner loop branch | `training/gae.rs:28` | 1-3% | Trivial |
| P8 | Rayon PAR_THRESHOLD tuning | `llm/ops.rs:124` | Prevent overhead | Trivial |
| P9 | V-trace double exp() | `training/vtrace.rs:49-50` | ~200us/call | Trivial |
| P10 | HashMap::new per CartPole step | `env/builtins.rs:137` | Minor alloc | Trivial |
| P11 | PyO3 sample() missing allow_threads | `rlox-python/buffer.rs` | GIL concurrency | Trivial |
| P12 | ExperienceRecord forces ownership | `buffer/mod.rs:9-19` | Root cause of P3 | Medium |
| P13 | current_obs re-alloc per step | `pipeline/collector.rs:143-147` | 1 alloc/step | Low |

---

## Detailed Items

### P1: AsyncCollector serial strided GAE (CRITICAL)

**File:** `crates/rlox-core/src/pipeline/collector.rs:167-193`

**Problem:** The AsyncCollector computes GAE per-environment by extracting strided per-env slices into temporary Vecs, then calling single-env `compute_gae` in a serial loop. `compute_gae_batched` with Rayon parallelism exists but is not used.

**Current code:**
```rust
for env_idx in 0..n_envs {
    // Strided gather: touches cache lines n_envs * 8 bytes apart
    let env_rewards: Vec<f64> = (0..n_steps)
        .map(|t| all_rewards[t * n_envs + env_idx])
        .collect();
    let env_values: Vec<f64> = (0..n_steps)
        .map(|t| all_values[t * n_envs + env_idx])
        .collect();
    let env_dones: Vec<f64> = (0..n_steps)
        .map(|t| all_dones[t * n_envs + env_idx])
        .collect();

    let (env_adv, env_ret) = gae::compute_gae(
        &env_rewards, &env_values, &env_dones,
        last_values[env_idx], gamma, gae_lambda,
    );

    // Strided write-back
    for t in 0..n_steps {
        advantages[t * n_envs + env_idx] = env_adv[t];
        returns[t * n_envs + env_idx] = env_ret[t];
    }
}
```

**Issues:**
1. Cache-hostile strided access: with 64 envs, every gather touches cache lines 512 bytes apart
2. 3 * n_envs temporary Vec allocations per rollout
3. Serial loop when Rayon batched version exists
4. Strided write-back also thrashes cache

**Proposed fix:**

Option A (minimal): Transpose data once, call `compute_gae_batched`, transpose result back:
```rust
// Transpose step-major -> env-major
let mut env_major_rewards = vec![0.0; total];
let mut env_major_values = vec![0.0; total];
let mut env_major_dones = vec![0.0; total];
for t in 0..n_steps {
    for e in 0..n_envs {
        env_major_rewards[e * n_steps + t] = all_rewards[t * n_envs + e];
        env_major_values[e * n_steps + t] = all_values[t * n_envs + e];
        env_major_dones[e * n_steps + t] = all_dones[t * n_envs + e];
    }
}
let (env_major_adv, env_major_ret) = gae::compute_gae_batched(
    &env_major_rewards, &env_major_values, &env_major_dones,
    &last_values, n_steps, gamma, gae_lambda,
);
// Transpose back env-major -> step-major
for e in 0..n_envs {
    for t in 0..n_steps {
        advantages[t * n_envs + e] = env_major_adv[e * n_steps + t];
        returns[t * n_envs + e] = env_major_ret[e * n_steps + t];
    }
}
```

Option B (better, larger refactor): Store rollout data env-major from the start in the collection loop (lines 130-147), then call `compute_gae_batched` directly without transposing.

**Impact:** 2-5x GAE speedup for typical configs (64 envs, 128 steps). Eliminates O(n_envs) temporary allocations.

**Effort:** Medium

**Dependencies:** None (compute_gae_batched already exists)

**Testing:** Existing proptest `gae_batched_matches_unbatched` validates correctness. Add integration test comparing AsyncCollector output before/after.

---

### P2: VecEnv per-env observation allocation

**File:** `crates/rlox-core/src/env/builtins.rs:68-69`, `crates/rlox-core/src/env/parallel.rs:60-92`

**Problem:** Every `CartPole::obs()` allocates a new `Vec<f32>`:
```rust
fn obs(&self) -> Observation {
    Observation::Flat(self.state.iter().map(|&v| v as f32).collect())
}
```

`BatchTransition` stores observations as `Vec<Vec<f32>>` (Array-of-Structures), requiring n_envs heap allocations per `step_all()`.

**Proposed fix:**

1. Add `write_obs(&self, buf: &mut [f32])` to `RLEnv` trait:
```rust
fn write_obs(&self, buf: &mut [f32]) {
    let obs = self.obs();
    buf.copy_from_slice(obs.as_flat());
}
```

2. Change `BatchTransition` to use flat storage:
```rust
pub struct BatchTransition {
    pub obs: Vec<f32>,       // flat: n_envs * obs_dim
    pub obs_dim: usize,
    pub rewards: Vec<f32>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
}
```

3. Pre-allocate in `VecEnv` and reuse across steps:
```rust
struct VecEnv {
    envs: Vec<Box<dyn RLEnv>>,
    obs_buf: Vec<f32>,  // pre-allocated n_envs * obs_dim
}
```

**Impact:** Eliminates n_envs heap allocations per step. For 1024 envs at 10kHz: ~10M allocations/second removed.

**Effort:** High (requires trait changes, struct changes, PyO3 adapter changes)

**Dependencies:** None

**Testing:** Existing CartPole tests + new test verifying flat obs matches per-env obs.

---

### P3: PyO3 push() double-copy

**File:** `crates/rlox-python/src/buffer.rs:222-239`

**Problem:** Each `push()` call copies data twice:
```rust
fn push(&mut self, obs: PyReadonlyArray1<f32>, ...) -> PyResult<()> {
    let obs_vec = obs.as_slice()?.to_vec();         // copy 1: numpy -> Vec
    let next_obs_vec = n.as_slice()?.to_vec();       // copy 1
    let action_vec = action.as_slice()?.to_vec();     // copy 1
    let record = ExperienceRecord { obs: obs_vec, ... };
    self.inner.push(record)                           // copy 2: Vec -> ring buffer
}
```

**Proposed fix:** Add `push_slices` method to `ReplayBuffer`:
```rust
impl ReplayBuffer {
    pub fn push_slices(
        &mut self,
        obs: &[f32],
        next_obs: &[f32],
        action: &[f32],
        reward: f32,
        terminated: bool,
        truncated: bool,
    ) -> Result<(), RloxError> {
        let idx = self.write_pos % self.capacity;
        let obs_start = idx * self.obs_dim;
        self.observations[obs_start..obs_start + self.obs_dim].copy_from_slice(obs);
        self.next_observations[obs_start..obs_start + self.obs_dim].copy_from_slice(next_obs);
        let act_start = idx * self.act_dim;
        self.actions[act_start..act_start + self.act_dim].copy_from_slice(action);
        self.rewards[idx] = reward;
        self.terminated[idx] = terminated;
        self.truncated[idx] = truncated;
        self.write_pos += 1;
        if self.count < self.capacity { self.count += 1; }
        Ok(())
    }
}
```

Then in PyO3:
```rust
fn push(&mut self, obs: PyReadonlyArray1<f32>, ...) -> PyResult<()> {
    self.inner.push_slices(
        obs.as_slice()?, next_obs.as_slice()?, action.as_slice()?,
        reward, terminated, truncated,
    ).map_err(...)
}
```

**Impact:** 40-50% push latency reduction. For obs_dim=28224 (Atari): saves ~220KB copying per push.

**Effort:** Low

**Dependencies:** P12 (ExperienceRecord ownership) — this effectively replaces it.

**Testing:** Existing buffer push/sample round-trip tests validate correctness.

---

### P4: SampledBatch fresh allocation every sample()

**File:** `crates/rlox-core/src/buffer/ringbuf.rs:199-244`

**Problem:** Every `sample()` creates new Vecs:
```rust
pub fn sample(&self, batch_size: usize, seed: u64) -> Result<SampledBatch, RloxError> {
    let mut batch = SampledBatch::with_capacity(batch_size, self.obs_dim, self.act_dim);
    // ... fills vecs ...
}
```

**Proposed fix:**
```rust
pub fn sample_into(
    &self,
    batch: &mut SampledBatch,
    batch_size: usize,
    seed: u64,
) -> Result<(), RloxError> {
    batch.clear();  // reuse allocations
    // ... same gather logic, extending existing Vecs ...
}
```

Add `SampledBatch::clear()`:
```rust
impl SampledBatch {
    pub fn clear(&mut self) {
        self.observations.clear();
        self.next_observations.clear();
        self.actions.clear();
        self.rewards.clear();
        self.terminated.clear();
        self.truncated.clear();
    }
}
```

The PyO3 layer can cache a `SampledBatch` on the `PyReplayBuffer` struct.

**Impact:** Eliminates allocator pressure. For batch_size=256, obs_dim=28224: saves ~56MB alloc+dealloc per step.

**Effort:** Low

**Dependencies:** None

**Testing:** Existing sample tests. Verify same results with sample vs sample_into.

---

### P5: PrioritizedReplayBuffer tree_min_prob O(N)

**File:** `crates/rlox-core/src/buffer/priority.rs:320-337`

**Problem:**
```rust
fn tree_min_prob(&self) -> f64 {
    let mut min_p = f64::MAX;
    for i in 0..self.count {
        let p = self.tree.get(i);
        // ...
    }
}
```
Linear scan of all leaves per `sample()`. For 1M entries: 1M f64 reads per call.

**Proposed fix:** Add a parallel min-tree alongside the sum-tree:
```rust
pub struct SumTree {
    capacity: usize,
    tree: Vec<f64>,       // sum tree
    min_tree: Vec<f64>,   // min tree (new)
}

impl SumTree {
    pub fn set(&mut self, index: usize, value: f64) {
        // Update sum tree (existing)
        // ...
        // Update min tree (new)
        let mut pos = index + self.capacity;
        self.min_tree[pos] = value;
        while pos > 1 {
            pos /= 2;
            self.min_tree[pos] = self.min_tree[2 * pos].min(self.min_tree[2 * pos + 1]);
        }
    }

    pub fn min(&self) -> f64 {
        self.min_tree[1]  // O(1)
    }
}
```

**Impact:** O(N) -> O(1) for tree_min_prob. O(log N) overhead added to each set().

**Effort:** Medium

**Dependencies:** None

**Testing:** Extend existing SumTree proptests to verify min property.

---

### P6: Release profile optimizations (QUICK WIN)

**File:** Workspace `Cargo.toml`

**Problem:** No `[profile.release]` section. Default codegen-units=16, no LTO, no target-cpu.

**Proposed fix — add to workspace `Cargo.toml`:**
```toml
[profile.release]
lto = "thin"
codegen-units = 1
opt-level = 3
```

**Add `.cargo/config.toml`:**
```toml
[build]
rustflags = ["-C", "target-cpu=native"]
```

**Impact:** 5-15% across all numeric code. `codegen-units=1` enables cross-crate inlining. LTO enables cross-crate optimization. `target-cpu=native` enables AVX2/AVX-512 auto-vectorization.

**Effort:** Trivial (add 4 lines)

**Dependencies:** None

**Testing:** Run existing benchmarks before/after to measure.

---

### P7: GAE inner loop branch (QUICK WIN)

**File:** `crates/rlox-core/src/training/gae.rs:26-36`

**Problem:**
```rust
for t in (0..n).rev() {
    let next_value = if t == n - 1 { last_value } else { values[t + 1] };  // branch every iteration
    // ...
}
```

**Proposed fix:**
```rust
// Handle last step outside loop
let next_non_terminal = 1.0 - dones[n - 1];
let delta = rewards[n - 1] + gamma * last_value * next_non_terminal - values[n - 1];
let mut last_gae = delta;
advantages[n - 1] = last_gae;

// Main loop: no branch
for t in (0..n - 1).rev() {
    let next_non_terminal = 1.0 - dones[t];
    let delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t];
    last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae;
    advantages[t] = last_gae;
}
```

**Impact:** 1-3%. Removes branch prediction miss risk in tight numerical loop.

**Effort:** Trivial

**Dependencies:** None

**Testing:** Existing GAE tests + proptests.

---

### P8: Rayon PAR_THRESHOLD tuning (QUICK WIN)

**File:** `crates/rlox-core/src/llm/ops.rs:124`

**Problem:** `const PAR_THRESHOLD: usize = 16;` — Rayon overhead (~1-10us) may dominate for short sequences.

**Proposed fix:**
```rust
// Gate on total work, not just batch size
let total_elements = log_probs_policy.len();
const PAR_ELEMENT_THRESHOLD: usize = 4096;

let out = if total_elements >= PAR_ELEMENT_THRESHOLD {
    use rayon::prelude::*;
    (0..batch_size).into_par_iter().map(kl_for_seq).collect()
} else {
    (0..batch_size).map(kl_for_seq).collect()
};
```

Also: parallelize `compute_batch_group_advantages` (currently serial).

**Impact:** Prevents Rayon overhead on small inputs. Adds parallelism to GRPO.

**Effort:** Trivial

**Dependencies:** None

---

### P9: V-trace double exp() (QUICK WIN)

**File:** `crates/rlox-core/src/training/vtrace.rs:49-50`

**Problem:**
```rust
let rho_t = rho_bar.min(log_rhos[t].exp());  // exp() call 1
let c_t = c_bar.min(log_rhos[t].exp());       // exp() call 2 (identical)
```

**Proposed fix:**
```rust
let ratio = log_rhos[t].exp();
let rho_t = rho_bar.min(ratio);
let c_t = c_bar.min(ratio);
```

**Impact:** ~20 cycles saved per timestep. For 10K steps: ~200us per vtrace call.

**Effort:** Trivial

**Dependencies:** None

**Testing:** Existing vtrace tests.

---

### P10: HashMap::new() per CartPole step (QUICK WIN)

**File:** `crates/rlox-core/src/env/builtins.rs:137`

**Problem:**
```rust
Ok(Transition {
    info: HashMap::new(),  // heap allocation every step
    // ...
})
```

**Proposed fix:** Change `Transition.info` to `Option<HashMap<String, f64>>`:
```rust
pub struct Transition {
    pub info: Option<HashMap<String, f64>>,  // None = no allocation
    // ...
}
```

Or use `HashMap::with_capacity(0)` (does not allocate on Rust >= 1.36).

**Impact:** Removes 1 allocation per environment per step.

**Effort:** Trivial

**Dependencies:** Requires updating all code that reads `transition.info`.

---

### P11: PyO3 sample() missing allow_threads (QUICK WIN)

**File:** `crates/rlox-python/src/buffer.rs` (ReplayBuffer and PrioritizedReplayBuffer sample methods)

**Problem:** `sample()` holds the GIL during Rust computation.

**Proposed fix:**
```rust
fn sample<'py>(&self, py: Python<'py>, batch_size: usize, seed: u64) -> PyResult<...> {
    let batch = py.allow_threads(|| {
        self.inner.sample(batch_size, seed)
    }).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    // Build Python dict from batch (needs GIL)
    // ...
}
```

**Impact:** Enables Python threads to run during sampling.

**Effort:** Trivial

**Dependencies:** None

---

### P12: ExperienceRecord forces ownership

**File:** `crates/rlox-core/src/buffer/mod.rs:9-19`

**Problem:**
```rust
pub struct ExperienceRecord {
    pub obs: Vec<f32>,       // owned, always cloned
    pub next_obs: Vec<f32>,  // owned, always cloned
    pub action: Vec<f32>,    // owned, always cloned
    // ...
}
```

Root cause of P3 (double-copy in push).

**Proposed fix:** Keep `ExperienceRecord` for the public API but add the `push_slices` path from P3 as the primary internal API. `ExperienceRecord` becomes a convenience wrapper:
```rust
impl ReplayBuffer {
    pub fn push(&mut self, record: ExperienceRecord) -> Result<(), RloxError> {
        self.push_slices(&record.obs, &record.next_obs, &record.action,
                         record.reward, record.terminated, record.truncated)
    }
}
```

**Impact:** Combined with P3, eliminates all unnecessary copies in the push path.

**Effort:** Medium (API change)

**Dependencies:** Implement alongside P3.

---

### P13: current_obs re-allocation per step in AsyncCollector

**File:** `crates/rlox-core/src/pipeline/collector.rs:143-147`

**Problem:**
```rust
current_obs = transition
    .obs
    .into_iter()
    .flat_map(|o| o.into_iter())
    .collect();  // re-allocates every step
```

**Proposed fix:**
```rust
// Before loop:
let mut current_obs = vec![0.0f32; n_envs * obs_dim];

// In loop:
let mut offset = 0;
for obs_vec in &transition.obs {
    current_obs[offset..offset + obs_dim].copy_from_slice(obs_vec);
    offset += obs_dim;
}
```

**Impact:** Eliminates 1 Vec allocation per step.

**Effort:** Low

**Dependencies:** None (but benefits from P2 flat obs layout)

---

## Implementation Order

**Phase 1 — Quick Wins (1-2 hours total):**
P6 (Cargo profiles) -> P9 (vtrace exp) -> P7 (GAE branch) -> P10 (HashMap) -> P11 (sample GIL) -> P8 (threshold tuning)

**Phase 2 — Buffer Performance (1 day):**
P3 + P12 (push_slices) -> P4 (sample_into) -> P5 (min-tree)

**Phase 3 — Pipeline & VecEnv (2-3 days):**
P1 (AsyncCollector GAE) -> P13 (current_obs) -> P2 (flat obs layout)
