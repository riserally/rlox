# Phase 2 — Experience Storage

**Status: NOT STARTED**
**Duration: Weeks 3-5**
**PRD Features: 2.1, 2.2, 2.4, 7.4**
**Depends on: Phase 1 (env trait, Transition struct)**

---

## 1. Overview

Phase 2 builds three storage primitives that hold RL experience data in Rust-owned memory and expose it to Python without copying. This is the second highest-ROI Rust component: Python replay buffers are memory-unsafe, GIL-bound on sampling, and waste memory on per-object overhead.

**Deliverables:**

| Component | Purpose | Key Invariant |
|-----------|---------|---------------|
| `ExperienceTable` | Append-only columnar store for transitions | Zero-copy numpy views via PyO3 |
| `ReplayBuffer` | Ring buffer with uniform random sampling | Deterministic sampling via ChaCha8Rng |
| `VarLenStore` | Packed variable-length sequence storage | <5% overhead vs theoretical minimum |

**Performance targets** (from `tests/python/test_bench_buffer_ops.py`):

| Metric | Target |
|--------|--------|
| Push throughput (CartPole, obs_dim=4) | >1M transitions/sec |
| Push throughput (Atari, obs_dim=28224) | >10K transitions/sec |
| Sample latency (batch_size <= 1024) | <100us |
| Zero-copy observations view | Faster than `np.array(..., copy=True)` |
| VarLenStore memory overhead | <5% vs optimal packed storage |

---

## 2. Architecture

### 2.1 Module Structure

```
crates/rlox-core/src/
    lib.rs              # add `pub mod buffer;`
    buffer/
        mod.rs          # re-exports, ExperienceRecord struct
        columnar.rs     # ExperienceTable
        ringbuf.rs      # ReplayBuffer (wraps ExperienceTable)
        varlen.rs       # VarLenStore
        trajectory.rs   # Episode segmentation + return computation

crates/rlox-python/src/
    lib.rs              # register PyExperienceTable, PyReplayBuffer, PyVarLenStore
    buffer.rs           # PyO3 bindings with zero-copy bridge
```

### 2.2 Crate Organization

All storage logic lives in `rlox-core` with no PyO3 dependency. The `rlox-python` crate provides thin `#[pyclass]` wrappers that translate between Python types and core Rust types. This separation ensures the core crate remains testable without Python.

### 2.3 Data Layout

#### ExperienceTable — Flat columnar arrays

```
ExperienceTable (obs_dim=4, act_dim=1, n=3 transitions):

observations: Vec<f32>  [o0_0, o0_1, o0_2, o0_3, o1_0, o1_1, o1_2, o1_3, o2_0, o2_1, o2_2, o2_3]
                         |---- transition 0 ----|---- transition 1 ----|---- transition 2 ----|
actions:      Vec<f32>  [a0, a1, a2]
rewards:      Vec<f32>  [r0, r1, r2]
terminated:   Vec<bool> [t0, t1, t2]
truncated:    Vec<bool> [u0, u1, u2]

Access pattern: obs[i] = observations[i * obs_dim .. (i+1) * obs_dim]
```

All scalar columns use `f32` (not `f64`) to match numpy `float32` and avoid conversion on the Python side. Rewards are stored as `f32` since RL rewards rarely need `f64` precision, and this halves memory bandwidth for the rewards column.

#### ReplayBuffer — Ring buffer with circular overwrite

```
ReplayBuffer (capacity=5, currently holding 5 transitions, write_pos=2):

Logical view:   [t3, t4, t0', t1', t2']
                              ^write_pos

Physical array: [t0', t1', t2', t3, t4]
                              ^write_pos=2 (next write goes to index 2)

When buffer is full, push overwrites at write_pos % capacity.
len = min(total_pushed, capacity)
```

The ring buffer does NOT use a second `ExperienceTable`. It stores data directly in flat `Vec<f32>` arrays of fixed capacity, overwriting in-place. This avoids reallocation and keeps cache lines hot.

#### VarLenStore — Offsets + flat data

```
VarLenStore containing [[10, 20, 30], [40, 50], [60, 70, 80, 90]]:

data:    Vec<u32>  [10, 20, 30, 40, 50, 60, 70, 80, 90]
offsets: Vec<u64>  [0,  3,  5,  9]
                    ^seq0 ^seq1 ^seq2 ^end

get(1) => &data[offsets[1]..offsets[2]] => &data[3..5] => [40, 50]
num_sequences() => offsets.len() - 1 => 3
total_elements() => offsets[last] => 9
```

Uses `u64` offsets (not `usize`) for deterministic cross-platform behavior and to match the test expectation of `(n_sequences + 1) * 8` bytes for offset storage.

---

## 3. Detailed Component Specifications

### 3.1 ExperienceTable

#### Rust types

```rust
// crates/rlox-core/src/buffer/mod.rs

/// A single experience record to push into the table.
/// Uses f32 throughout for numpy compatibility.
pub struct ExperienceRecord {
    pub obs: Vec<f32>,       // length must equal obs_dim
    pub action: f32,         // discrete actions stored as f32
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
}
```

```rust
// crates/rlox-core/src/buffer/columnar.rs

use crate::error::RloxError;
use super::ExperienceRecord;

/// Append-only columnar table for RL transitions.
///
/// All data is stored in flat `Vec<f32>` arrays for zero-copy export
/// to numpy. The table never reallocates existing data — it only
/// appends, preserving pointer stability for outstanding views.
pub struct ExperienceTable {
    obs_dim: usize,
    act_dim: usize,
    observations: Vec<f32>,    // [count * obs_dim] flat
    actions: Vec<f32>,         // [count * act_dim] flat
    rewards: Vec<f32>,         // [count]
    terminated: Vec<bool>,     // [count]
    truncated: Vec<bool>,      // [count]
    count: usize,
}

impl ExperienceTable {
    /// Create a new table with the given observation and action dimensions.
    pub fn new(obs_dim: usize, act_dim: usize) -> Self;

    /// Number of transitions stored.
    pub fn len(&self) -> usize;

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool;

    /// Append a single transition. Returns error on dimension mismatch.
    pub fn push(&mut self, record: ExperienceRecord) -> Result<(), RloxError>;

    /// Raw slice of all observation data. Shape: [count * obs_dim].
    /// The caller reshapes to [count, obs_dim] on the Python side.
    pub fn observations_raw(&self) -> &[f32];

    /// Raw slice of all action data. Shape: [count * act_dim].
    pub fn actions_raw(&self) -> &[f32];

    /// Raw slice of all rewards.
    pub fn rewards_raw(&self) -> &[f32];

    /// Slice of terminated flags.
    pub fn terminated(&self) -> &[bool];

    /// Slice of truncated flags.
    pub fn truncated(&self) -> &[bool];

    /// Drop all stored data.
    pub fn clear(&mut self);
}
```

**Why no `ExperienceSchema` struct?** The earlier draft used a separate schema type, but `obs_dim` and `act_dim` are two integers -- a dedicated struct adds indirection without value. The dimensions are stored directly on `ExperienceTable`.

**Why `Vec<bool>` instead of a bitset?** For zero-copy numpy export, we need byte-addressable storage. A packed bitset would require conversion to `np.bool_` arrays. `Vec<bool>` on most platforms stores one byte per element, which numpy can read directly. If memory becomes an issue for terminated/truncated columns (unlikely -- they are tiny vs observations), we can revisit.

#### Push implementation strategy

```rust
pub fn push(&mut self, record: ExperienceRecord) -> Result<(), RloxError> {
    if record.obs.len() != self.obs_dim {
        return Err(RloxError::ShapeMismatch {
            expected: format!("obs_dim={}", self.obs_dim),
            got: format!("obs.len()={}", record.obs.len()),
        });
    }
    self.observations.extend_from_slice(&record.obs);
    self.actions.push(record.action);
    self.rewards.push(record.reward);
    self.terminated.push(record.terminated);
    self.truncated.push(record.truncated);
    self.count += 1;
    Ok(())
}
```

`extend_from_slice` is critical for throughput -- it uses `memcpy` for `Copy` types like `f32`. For CartPole (4 floats = 16 bytes), this compiles to a single SSE move. For Atari (28224 floats = ~110KB), it triggers efficient large-copy paths in libc.

Pre-allocation: on construction, we can optionally accept a `capacity_hint` and call `Vec::with_capacity(capacity_hint * obs_dim)` to avoid early reallocations.

### 3.2 ReplayBuffer

```rust
// crates/rlox-core/src/buffer/ringbuf.rs

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use crate::error::RloxError;
use super::ExperienceRecord;

/// Fixed-capacity ring buffer with uniform random sampling.
///
/// Wraps flat columnar arrays with circular write semantics.
/// Oldest transitions are overwritten when capacity is reached (FIFO eviction).
pub struct ReplayBuffer {
    obs_dim: usize,
    act_dim: usize,
    capacity: usize,

    // Pre-allocated flat arrays of size [capacity * dim]
    observations: Vec<f32>,    // [capacity * obs_dim]
    actions: Vec<f32>,         // [capacity * act_dim]
    rewards: Vec<f32>,         // [capacity]
    terminated: Vec<bool>,     // [capacity]
    truncated: Vec<bool>,      // [capacity]

    write_pos: usize,          // next write index (mod capacity)
    count: usize,              // min(total_pushed, capacity)
}

/// A sampled batch of transitions. Owns its data (copied from the ring buffer).
pub struct SampledBatch {
    pub observations: Vec<f32>,  // [batch_size * obs_dim]
    pub actions: Vec<f32>,       // [batch_size * act_dim]
    pub rewards: Vec<f32>,       // [batch_size]
    pub terminated: Vec<bool>,   // [batch_size]
    pub truncated: Vec<bool>,    // [batch_size]
    pub obs_dim: usize,
    pub act_dim: usize,
    pub batch_size: usize,
}

impl ReplayBuffer {
    /// Create a ring buffer with fixed capacity.
    pub fn new(capacity: usize, obs_dim: usize, act_dim: usize) -> Self {
        Self {
            obs_dim,
            act_dim,
            capacity,
            observations: vec![0.0; capacity * obs_dim],
            actions: vec![0.0; capacity * act_dim],
            rewards: vec![0.0; capacity],
            terminated: vec![false; capacity],
            truncated: vec![false; capacity],
            write_pos: 0,
            count: 0,
        }
    }

    /// Number of valid transitions currently stored.
    pub fn len(&self) -> usize { self.count }

    pub fn is_empty(&self) -> bool { self.count == 0 }

    /// Push a transition, overwriting the oldest if at capacity.
    pub fn push(&mut self, record: ExperienceRecord) -> Result<(), RloxError>;

    /// Sample a batch of transitions uniformly at random.
    ///
    /// Uses ChaCha8Rng seeded with `seed` for deterministic cross-platform
    /// reproducibility. Returns owned `SampledBatch` (data is copied from
    /// the ring buffer since sampled indices are non-contiguous).
    pub fn sample(&self, batch_size: usize, seed: u64) -> Result<SampledBatch, RloxError>;
}
```

#### Push — O(1) amortized, zero allocation

```rust
pub fn push(&mut self, record: ExperienceRecord) -> Result<(), RloxError> {
    if record.obs.len() != self.obs_dim {
        return Err(RloxError::ShapeMismatch { /* ... */ });
    }
    let idx = self.write_pos;
    let obs_start = idx * self.obs_dim;
    self.observations[obs_start..obs_start + self.obs_dim]
        .copy_from_slice(&record.obs);
    self.actions[idx] = record.action;
    self.rewards[idx] = record.reward;
    self.terminated[idx] = record.terminated;
    self.truncated[idx] = record.truncated;

    self.write_pos = (self.write_pos + 1) % self.capacity;
    if self.count < self.capacity {
        self.count += 1;
    }
    Ok(())
}
```

`copy_from_slice` for the observation data -- same `memcpy` path as ExperienceTable. The fixed-size pre-allocated arrays mean zero heap allocation during push. This is key for >1M push/sec.

#### Sample — Gather by random indices

```rust
pub fn sample(&self, batch_size: usize, seed: u64) -> Result<SampledBatch, RloxError> {
    if batch_size > self.count {
        return Err(RloxError::BufferError(
            format!("batch_size {} > buffer len {}", batch_size, self.count)
        ));
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut batch = SampledBatch::with_capacity(batch_size, self.obs_dim, self.act_dim);

    for _ in 0..batch_size {
        let idx = rng.random_range(0..self.count);
        let obs_start = idx * self.obs_dim;
        batch.observations.extend_from_slice(
            &self.observations[obs_start..obs_start + self.obs_dim]
        );
        batch.actions.push(self.actions[idx]);
        batch.rewards.push(self.rewards[idx]);
        batch.terminated.push(self.terminated[idx]);
        batch.truncated.push(self.truncated[idx]);
    }
    batch.batch_size = batch_size;
    Ok(batch)
}
```

**Why copy on sample, not zero-copy?** Sampled indices are random and non-contiguous. A zero-copy view would require an index array + scatter gather, which numpy cannot represent as a contiguous view. The copy for batch_size=1024 with obs_dim=4 is only 1024 * 4 * 4 = 16KB -- well within L1 cache and trivially under 100us.

**Why ChaCha8Rng?** Cross-platform deterministic sampling. Given the same seed, ChaCha8Rng produces identical sequences on x86, ARM, and WASM. This is required for reproducible RL experiments.

### 3.3 VarLenStore

```rust
// crates/rlox-core/src/buffer/varlen.rs

/// Packed variable-length sequence storage.
///
/// Uses the Arrow ListArray pattern: a flat contiguous data array
/// plus an offsets array that marks sequence boundaries. This avoids
/// padding waste that occurs with fixed-size tensor storage.
///
/// Memory layout for sequences [[1,2,3], [4,5], [6,7,8,9]]:
///   data:    [1, 2, 3, 4, 5, 6, 7, 8, 9]
///   offsets: [0, 3, 5, 9]
pub struct VarLenStore {
    data: Vec<u32>,       // flat contiguous token storage
    offsets: Vec<u64>,    // sequence boundaries; offsets[0] is always 0
}

impl VarLenStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            offsets: vec![0],  // sentinel: offsets[0] = 0
        }
    }

    /// Append a variable-length sequence.
    pub fn push(&mut self, sequence: &[u32]) {
        self.data.extend_from_slice(sequence);
        self.offsets.push(self.data.len() as u64);
    }

    /// Number of sequences stored.
    pub fn num_sequences(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Total number of elements across all sequences.
    pub fn total_elements(&self) -> usize {
        self.data.len()
    }

    /// Get the i-th sequence as a slice.
    ///
    /// # Panics
    /// Panics if `index >= num_sequences()`.
    pub fn get(&self, index: usize) -> &[u32] {
        let start = self.offsets[index] as usize;
        let end = self.offsets[index + 1] as usize;
        &self.data[start..end]
    }

    /// Length of the i-th sequence.
    pub fn sequence_len(&self, index: usize) -> usize {
        let start = self.offsets[index] as usize;
        let end = self.offsets[index + 1] as usize;
        end - start
    }

    /// Raw flat data array (for inspection/export).
    pub fn flat_data(&self) -> &[u32] {
        &self.data
    }

    /// Raw offsets array (for inspection/export).
    pub fn offsets(&self) -> &[u64] {
        &self.offsets
    }
}
```

**Memory analysis:**
- Optimal storage: `total_elements * 4` bytes (raw u32 data)
- Offset overhead: `(num_sequences + 1) * 8` bytes
- For the test case (1000 Zipf sequences, ~20-30 elements avg): overhead = 8008 bytes / ~100KB data < 5%
- Vec capacity slack: `Vec` may allocate up to 2x the used length. Mitigation: call `shrink_to_fit()` after bulk loading, or use `Vec::with_capacity()` if total size is known.

### 3.4 Error Types

Add a new variant to `RloxError`:

```rust
// crates/rlox-core/src/error.rs

#[derive(Debug, Error)]
pub enum RloxError {
    // ... existing variants ...

    #[error("Buffer error: {0}")]
    BufferError(String),
}
```

### 3.5 PyO3 Bindings

```rust
// crates/rlox-python/src/buffer.rs

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyDict;

use rlox_core::buffer::columnar::ExperienceTable;
use rlox_core::buffer::ringbuf::{ReplayBuffer, SampledBatch};
use rlox_core::buffer::varlen::VarLenStore;
use rlox_core::buffer::ExperienceRecord;

// ---------- ExperienceTable ----------

#[pyclass(name = "ExperienceTable")]
pub struct PyExperienceTable {
    inner: ExperienceTable,
}

#[pymethods]
impl PyExperienceTable {
    #[new]
    fn new(obs_dim: usize, act_dim: usize) -> Self {
        Self { inner: ExperienceTable::new(obs_dim, act_dim) }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Push a single transition.
    #[pyo3(signature = (obs, action, reward, terminated, truncated))]
    fn push(
        &mut self,
        obs: PyReadonlyArray1<f32>,
        action: f32,
        reward: f32,
        terminated: bool,
        truncated: bool,
    ) -> PyResult<()> {
        let record = ExperienceRecord {
            obs: obs.as_slice()?.to_vec(),
            action,
            reward,
            terminated,
            truncated,
        };
        self.inner.push(record).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Return observations as a numpy array.
    /// Shape: (n, obs_dim). dtype: float32.
    ///
    /// This creates a numpy array that views the Rust-owned data.
    /// The PyExperienceTable must outlive the returned array.
    fn observations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let raw = self.inner.observations_raw();
        let n = self.inner.len();
        let obs_dim = self.inner.obs_dim();
        // Create a 1D array from the slice, then reshape to 2D
        let array_1d = PyArray1::from_slice(py, raw);
        let array_2d = array_1d.reshape([n, obs_dim])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(array_2d)
    }

    /// Return rewards as a numpy array. Shape: (n,). dtype: float32.
    fn rewards<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, self.inner.rewards_raw())
    }

    fn clear(&mut self) {
        self.inner.clear();
    }
}

// ---------- ReplayBuffer ----------

#[pyclass(name = "ReplayBuffer")]
pub struct PyReplayBuffer {
    inner: ReplayBuffer,
}

#[pymethods]
impl PyReplayBuffer {
    #[new]
    fn new(capacity: usize, obs_dim: usize, act_dim: usize) -> Self {
        Self { inner: ReplayBuffer::new(capacity, obs_dim, act_dim) }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[pyo3(signature = (obs, action, reward, terminated, truncated))]
    fn push(
        &mut self,
        obs: PyReadonlyArray1<f32>,
        action: f32,
        reward: f32,
        terminated: bool,
        truncated: bool,
    ) -> PyResult<()> {
        let record = ExperienceRecord {
            obs: obs.as_slice()?.to_vec(),
            action,
            reward,
            terminated,
            truncated,
        };
        self.inner.push(record).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Sample a batch. Returns dict with numpy arrays.
    #[pyo3(signature = (batch_size, seed))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        batch_size: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let batch = self.inner.sample(batch_size, seed)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);
        let obs_1d = PyArray1::from_vec(py, batch.observations);
        let obs_2d = obs_1d.reshape([batch.batch_size, batch.obs_dim])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        dict.set_item("obs", obs_2d)?;
        dict.set_item("actions", PyArray1::from_vec(py, batch.actions))?;
        dict.set_item("rewards", PyArray1::from_vec(py, batch.rewards))?;
        // terminated/truncated as bool arrays
        let term: Vec<u8> = batch.terminated.iter().map(|&b| b as u8).collect();
        let trunc: Vec<u8> = batch.truncated.iter().map(|&b| b as u8).collect();
        dict.set_item("terminated", PyArray1::from_slice(py, &term))?;
        dict.set_item("truncated", PyArray1::from_slice(py, &trunc))?;
        Ok(dict)
    }
}

// ---------- VarLenStore ----------

#[pyclass(name = "VarLenStore")]
pub struct PyVarLenStore {
    inner: VarLenStore,
}

#[pymethods]
impl PyVarLenStore {
    #[new]
    fn new() -> Self {
        Self { inner: VarLenStore::new() }
    }

    fn push(&mut self, seq: PyReadonlyArray1<u32>) -> PyResult<()> {
        self.inner.push(seq.as_slice()?);
        Ok(())
    }

    fn num_sequences(&self) -> usize {
        self.inner.num_sequences()
    }

    fn total_elements(&self) -> usize {
        self.inner.total_elements()
    }

    fn get<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyArray1<u32>>> {
        if index >= self.inner.num_sequences() {
            return Err(PyRuntimeError::new_err("index out of range"));
        }
        Ok(PyArray1::from_slice(py, self.inner.get(index)))
    }
}
```

#### Zero-copy strategy

The key challenge is returning a numpy array that references Rust-owned memory without copying. There are two approaches:

**Approach A: `PyArray::from_slice` (near-zero-copy)**
`numpy::PyArray1::from_slice(py, &slice)` copies the data into a new numpy-owned allocation. This is technically a copy, but it is a single `memcpy` -- O(n) with excellent cache behavior. For the `observations()` method on 10K transitions with obs_dim=4, this copies 160KB, which takes ~10us. This is fast enough to be "faster than explicit `np.array(..., copy=True)`" (which goes through Python iteration).

**Approach B: `unsafe` borrow via `PyArray::borrow_from_array_bound`**
True zero-copy: the numpy array points directly at the Rust `Vec<f32>` memory. However, this requires `unsafe` and careful lifetime management -- the Rust vec must not be reallocated while the numpy array is alive. Since `ExperienceTable` is append-only and Vec may reallocate on push, this is unsound unless we either:
1. Freeze the table (no pushes while a view is outstanding)
2. Use a custom allocator that never moves data
3. Use `Arc<[f32]>` for each column and hand out Arc clones

**Decision: Start with Approach A** (`from_slice` copy). It is safe, simple, and meets all benchmark targets. The copy overhead is dominated by the function call overhead across PyO3 for small buffers, and for large buffers the `memcpy` is still faster than any Python-side alternative. If profiling shows it is a bottleneck, upgrade to Approach B with `Arc`-backed columns.

The tests validate this works:
- `test_observations_returns_numpy_without_copy`: Checks shape and type -- passes with either approach
- `test_zero_copy_faster_than_explicit_copy`: `from_slice` (single memcpy) is faster than `np.array(arr, copy=True)` (Python-level copy) -- passes with Approach A

---

## 4. TDD Implementation Steps

Each step follows RED -> GREEN -> REFACTOR. The 12 xfail Python tests in `test_bench_buffer_ops.py` are the acceptance criteria.

### Step 1: ExperienceTable core (targets: `test_experience_table_importable`)

**RED:** Write Rust unit tests in `crates/rlox-core/src/buffer/columnar.rs`:
```rust
#[test]
fn empty_table_has_zero_len() {
    let table = ExperienceTable::new(4, 1);
    assert_eq!(table.len(), 0);
    assert!(table.is_empty());
}

#[test]
fn push_single_transition_increments_len() {
    let mut table = ExperienceTable::new(4, 1);
    table.push(sample_record(4)).unwrap();
    assert_eq!(table.len(), 1);
}

#[test]
fn push_many_transitions() {
    let mut table = ExperienceTable::new(4, 1);
    for _ in 0..1000 {
        table.push(sample_record(4)).unwrap();
    }
    assert_eq!(table.len(), 1000);
}

#[test]
fn observations_column_correct_length() {
    let mut table = ExperienceTable::new(4, 1);
    for _ in 0..10 {
        table.push(sample_record(4)).unwrap();
    }
    assert_eq!(table.observations_raw().len(), 40);
}

#[test]
fn rewards_column_correct_values() {
    let mut table = ExperienceTable::new(4, 1);
    let mut r = sample_record(4);
    r.reward = 42.0;
    table.push(r).unwrap();
    assert_eq!(table.rewards_raw()[0], 42.0);
}

#[test]
fn obs_dim_mismatch_returns_error() {
    let mut table = ExperienceTable::new(4, 1);
    let bad = sample_record(8); // wrong dim
    assert!(table.push(bad).is_err());
}

#[test]
fn clear_empties_all_columns() {
    let mut table = ExperienceTable::new(4, 1);
    for _ in 0..100 { table.push(sample_record(4)).unwrap(); }
    table.clear();
    assert_eq!(table.len(), 0);
    assert!(table.observations_raw().is_empty());
}
```

**GREEN:** Implement `ExperienceTable` struct with `new`, `push`, `len`, `is_empty`, `observations_raw`, `rewards_raw`, `terminated`, `truncated`, `clear`.

**REFACTOR:** Extract `sample_record()` test helper into a shared test util module.

### Step 2: ReplayBuffer core (targets: `test_replay_buffer_importable`)

**RED:** Write Rust unit tests in `crates/rlox-core/src/buffer/ringbuf.rs`:
```rust
#[test]
fn ring_buffer_respects_capacity() {
    let mut buf = ReplayBuffer::new(100, 4, 1);
    for _ in 0..200 { buf.push(sample_record(4)).unwrap(); }
    assert_eq!(buf.len(), 100);
}

#[test]
fn ring_buffer_overwrites_oldest() {
    let mut buf = ReplayBuffer::new(3, 4, 1);
    for i in 0..5 {
        let mut r = sample_record(4);
        r.reward = i as f32;
        buf.push(r).unwrap();
    }
    // Should contain rewards 2.0, 3.0, 4.0
    let batch = buf.sample(3, 42).unwrap();
    assert!(!batch.rewards.contains(&0.0));
    assert!(!batch.rewards.contains(&1.0));
}

#[test]
fn sample_returns_requested_size() {
    let mut buf = ReplayBuffer::new(1000, 4, 1);
    for _ in 0..1000 { buf.push(sample_record(4)).unwrap(); }
    let batch = buf.sample(64, 42).unwrap();
    assert_eq!(batch.batch_size, 64);
    assert_eq!(batch.observations.len(), 64 * 4);
}

#[test]
fn sample_errors_when_too_few() {
    let mut buf = ReplayBuffer::new(100, 4, 1);
    buf.push(sample_record(4)).unwrap();
    assert!(buf.sample(32, 42).is_err());
}

#[test]
fn sample_is_deterministic_with_same_seed() {
    let mut buf = ReplayBuffer::new(1000, 4, 1);
    for _ in 0..1000 { buf.push(sample_record(4)).unwrap(); }
    let b1 = buf.sample(32, 42).unwrap();
    let b2 = buf.sample(32, 42).unwrap();
    assert_eq!(b1.observations, b2.observations);
    assert_eq!(b1.rewards, b2.rewards);
}

#[test]
fn replay_buffer_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ReplayBuffer>();
}
```

**GREEN:** Implement `ReplayBuffer` with pre-allocated arrays, circular push, and ChaCha8Rng sampling.

**REFACTOR:** Ensure `sample_record` helper is reused from Step 1.

### Step 3: VarLenStore core (targets: `test_varlen_store_importable`)

**RED:** Write Rust unit tests in `crates/rlox-core/src/buffer/varlen.rs`:
```rust
#[test]
fn empty_store() {
    let store = VarLenStore::new();
    assert_eq!(store.num_sequences(), 0);
    assert_eq!(store.total_elements(), 0);
}

#[test]
fn push_and_retrieve() {
    let mut store = VarLenStore::new();
    store.push(&[10, 20, 30]);
    store.push(&[40, 50]);
    assert_eq!(store.num_sequences(), 2);
    assert_eq!(store.get(0), &[10, 20, 30]);
    assert_eq!(store.get(1), &[40, 50]);
}

#[test]
fn total_elements_correct() {
    let mut store = VarLenStore::new();
    store.push(&[1, 2, 3]);
    store.push(&[4, 5]);
    assert_eq!(store.total_elements(), 5);
}

#[test]
fn no_padding_waste() {
    let mut store = VarLenStore::new();
    store.push(&[1, 2, 3]);
    store.push(&[4, 5]);
    assert_eq!(store.flat_data(), &[1, 2, 3, 4, 5]);
    assert_eq!(store.offsets(), &[0, 3, 5]);
}

#[test]
fn sequence_len() {
    let mut store = VarLenStore::new();
    store.push(&[1, 2, 3]);
    store.push(&[4]);
    assert_eq!(store.sequence_len(0), 3);
    assert_eq!(store.sequence_len(1), 1);
}
```

**GREEN:** Implement `VarLenStore` as specified in section 3.3.

**REFACTOR:** Add `Default` impl for `VarLenStore`.

### Step 4: PyO3 bindings (targets: all 3 precondition tests)

**RED:** The 3 `TestBufferPreconditions` tests are already written in `test_bench_buffer_ops.py`.

**GREEN:**
1. Create `crates/rlox-python/src/buffer.rs` with `PyExperienceTable`, `PyReplayBuffer`, `PyVarLenStore`.
2. Register in `crates/rlox-python/src/lib.rs`:
   ```rust
   m.add_class::<buffer::PyExperienceTable>()?;
   m.add_class::<buffer::PyReplayBuffer>()?;
   m.add_class::<buffer::PyVarLenStore>()?;
   ```
3. Export in `python/rlox/__init__.py`:
   ```python
   from rlox._rlox_core import (
       CartPole, VecEnv, GymEnv,
       ExperienceTable, ReplayBuffer, VarLenStore,
   )
   ```

**REFACTOR:** Ensure error conversion is consistent with existing `rlox_err_to_py` pattern in `env.rs`.

### Step 5: Push throughput (targets: `test_rlox_push_throughput_cartpole`, `test_rlox_push_throughput_atari`)

**RED:** The benchmark tests are already written. They call `table.push(obs=obs, action=0, reward=1.0, terminated=False, truncated=False)` in a tight loop and measure throughput.

**GREEN:** The implementation from Steps 1+4 should already meet targets. Key optimizations if needed:
1. In `PyExperienceTable.push()`, avoid `to_vec()` by using `extend_from_slice` directly from the numpy slice.
2. Accept `action` as `f32` directly (the test passes `action=0`, an int, which PyO3 auto-converts).
3. Pre-allocate with `Vec::with_capacity()`.

**REFACTOR:** Profile with `cargo flamegraph` if throughput is under target. Most likely bottleneck: PyO3 function call overhead (~50-100ns per call), not the actual Rust push (~5-10ns for CartPole).

### Step 6: Sample latency (targets: `test_rlox_sample_latency` x4 batch sizes)

**RED:** Tests pre-fill 100K transitions and measure `buf.sample(batch_size, seed=42)` latency. Target: <100us.

**GREEN:** The `ReplayBuffer.sample()` from Step 2 handles this. For batch_size=1024, obs_dim=4:
- 1024 random index generations: ~1024 * 2ns = ~2us
- 1024 gather operations (4 floats each): ~1024 * 16B = 16KB copy, ~5us
- PyO3 dict construction + numpy array creation: ~10-20us
- Total estimate: ~30us, well under 100us.

**REFACTOR:** If needed, batch the index generation (generate all indices first, then gather), which is more cache-friendly.

### Step 7: Zero-copy validation (targets: `test_observations_returns_numpy_without_copy`, `test_zero_copy_faster_than_explicit_copy`)

**RED:** Tests check that `table.observations()` returns `np.ndarray` with shape `(n, 4)`, and that it is faster than `np.array(table.observations(), copy=True)`.

**GREEN:** `PyArray1::from_slice` + `reshape` produces a contiguous numpy array. The single `memcpy` is faster than Python-level `np.array(..., copy=True)` which goes through numpy's copy machinery.

**REFACTOR:** Consider adding `actions()`, `terminated()`, `truncated()` accessors for API completeness.

### Step 8: VarLenStore benchmarks (targets: `test_varlen_memory_efficiency`, `test_varlen_vs_padded_tensor_memory`)

**RED:** Tests push 1000 Zipf-distributed sequences and validate:
1. `store.total_elements() == sum_of_lengths`
2. `store.num_sequences() == 1000`
3. VarLen actual/padded ratio < 0.5

**GREEN:** The `VarLenStore` from Step 3 handles this directly. The memory efficiency test verifies our storage matches the theoretical minimum (data + offsets). The padding comparison is a pure math test -- Zipf(1.5) with min=16, max=4096 will have most sequences near 16-20 tokens, so mean/max ratio << 0.5.

**REFACTOR:** Add `with_capacity(n_sequences_hint, total_elements_hint)` constructor for bulk loading.

### Step 9: Trajectory segmentation (bonus, not in xfail tests)

**RED:** Write Rust tests for `segment_episodes()` and `compute_discounted_returns()`.

**GREEN:** Implement in `crates/rlox-core/src/buffer/trajectory.rs`. These are pure functions operating on slices.

**REFACTOR:** This prepares for Phase 3 (GAE computation).

---

## 5. Dependencies

### New dependencies for `rlox-core`

None required. The current dependency set is sufficient:

| Crate | Version | Used For |
|-------|---------|----------|
| `rand` | 0.9 | Already present. `Rng` trait for sampling. |
| `rand_chacha` | 0.9 | Already present. `ChaCha8Rng` for deterministic sampling. |
| `thiserror` | 2 | Already present. New `BufferError` variant. |

**Why not `arrow-rs`?** The earlier draft proposed `arrow-rs` for columnar storage. After analysis, raw `Vec<f32>` with manual indexing provides everything we need:
- Zero-copy export to numpy works via `PyArray::from_slice` on raw slices
- Arrow's `Float32Array` adds reference-counted buffer management that we don't need (our ownership model is simpler)
- Arrow adds ~40 transitive dependencies
- We can always wrap our flat arrays in Arrow arrays later (they share the same physical layout) if we need Arrow interop for downstream tools

### No changes to `rlox-python`

The existing `numpy = "0.23"` crate provides `PyArray1`, `PyArray2`, `PyReadonlyArray1` -- all we need for the bindings.

---

## 6. Performance Strategy

### Push throughput (>1M/sec for CartPole)

1. **Pre-allocate**: `ReplayBuffer` allocates all arrays upfront. `ExperienceTable` can use `with_capacity`.
2. **memcpy path**: `copy_from_slice` and `extend_from_slice` for `f32` slices compile to `memcpy`, which the CPU handles at memory bandwidth speed.
3. **Minimize PyO3 overhead**: The main bottleneck for per-transition push is crossing the Python->Rust boundary (~50-100ns per call). At 100ns overhead, theoretical max is 10M calls/sec. With 50ns of Rust work, we get ~6.6M/sec -- well above 1M target.
4. **Avoid allocation in push**: `ReplayBuffer.push()` does zero heap allocation (writes into pre-allocated arrays). `ExperienceTable.push()` may trigger Vec reallocation, but amortized cost is O(1).

### Push throughput (>10K/sec for Atari, obs_dim=28224)

Each push copies 28224 * 4 = ~110KB of observation data. At ~10GB/s memory bandwidth, one copy takes ~11us. Adding PyO3 overhead: ~12us per push = ~83K pushes/sec. Well above 10K target.

### Sample latency (<100us for batch_size <= 1024)

1. **Index generation**: ChaCha8Rng is fast (~2ns per u64). 1024 indices = ~2us.
2. **Gather**: For obs_dim=4, each gather copies 16 bytes. 1024 gathers = 16KB total, fits in L1 cache. ~5us.
3. **PyO3 return**: Constructing dict + 5 numpy arrays. `PyArray1::from_vec` donates the Vec allocation to numpy (no copy for owned data). ~10us.
4. **Total**: ~20-30us for batch_size=1024 with obs_dim=4.

### Zero-copy observations

`PyArray1::from_slice` performs a single `memcpy` which is faster than numpy's `np.array(..., copy=True)` because:
- No Python object creation per element
- No type checking per element
- Single continuous memory copy vs. potential strided copy

---

## 7. Risk Analysis

### Risk 1: `PyArray::from_slice` is technically a copy

**Severity:** Low
**Mitigation:** The benchmark test (`test_zero_copy_faster_than_explicit_copy`) compares `table.observations()` against `np.array(table.observations(), copy=True)`. Since `from_slice` does a bulk `memcpy` while `np.array(..., copy=True)` does a numpy-level copy of an already-created array, our approach wins. If true zero-copy is ever needed, upgrade to `Arc`-backed columns with `unsafe` PyArray construction.

### Risk 2: Vec reallocation invalidates outstanding numpy views

**Severity:** Medium (only if we move to true zero-copy)
**Mitigation:** With Approach A (`from_slice`), this is not an issue because the numpy array owns its own copy. If we later move to Approach B, we must either freeze the table or use stable-address storage (e.g., `Arc<[f32]>` per chunk).

### Risk 3: `action` type flexibility

**Severity:** Low
**Description:** The Python tests pass `action=0` (int). The Rust side accepts `f32`. PyO3 auto-converts Python `int` to `f32`. For multi-dimensional actions (`act_dim > 1`), we would need to accept arrays. Currently `act_dim=1` is all that's tested.
**Mitigation:** Accept `action` as `f32` for now. Add `push_batch(obs_array, actions_array, ...)` method later for vectorized push.

### Risk 4: `Vec<bool>` layout not guaranteed

**Severity:** Low
**Description:** Rust's `Vec<bool>` stores one byte per bool, but the exact representation (0x00/0x01 vs. 0x00/0xFF) is not formally guaranteed for all platforms.
**Mitigation:** For numpy export, convert `bool -> u8` explicitly (as already done in the existing `env.rs` bindings). This adds negligible overhead since terminated/truncated arrays are tiny.

### Risk 5: ChaCha8Rng `random_range` API changes

**Severity:** Low
**Description:** `rand` 0.9 changed some APIs. The method for generating a random number in a range may be `gen_range` or `random_range` depending on the exact version.
**Mitigation:** Check the actual `rand` 0.9 API. Likely `rng.random_range(0..self.count)` or `rng.gen_range(0..self.count)`. Pin to exact minor version if needed.

### Risk 6: Cross-platform determinism for sampling

**Severity:** Low
**Description:** ChaCha8Rng is deterministic, but the uniform distribution sampling could theoretically differ across platforms if the rejection sampling implementation changes.
**Mitigation:** `rand` 0.9 guarantees reproducible output for the same (RNG, distribution) pair. Lock the `rand` version in `Cargo.lock`.

---

## 8. Files to Create

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `crates/rlox-core/src/buffer/mod.rs` | Module exports, `ExperienceRecord` struct | ~30 |
| `crates/rlox-core/src/buffer/columnar.rs` | `ExperienceTable` implementation + tests | ~200 |
| `crates/rlox-core/src/buffer/ringbuf.rs` | `ReplayBuffer` + `SampledBatch` + tests | ~250 |
| `crates/rlox-core/src/buffer/varlen.rs` | `VarLenStore` implementation + tests | ~120 |
| `crates/rlox-core/src/buffer/trajectory.rs` | Episode segmentation + tests | ~100 |
| `crates/rlox-python/src/buffer.rs` | PyO3 bindings for all 3 types | ~180 |
| `tests/python/test_buffer.py` | Python integration tests | ~80 |

## Files to Modify

| File | Change |
|------|--------|
| `crates/rlox-core/src/lib.rs` | Add `pub mod buffer;` |
| `crates/rlox-core/src/error.rs` | Add `BufferError(String)` variant |
| `crates/rlox-python/src/lib.rs` | Add `mod buffer;`, register 3 new pyclass types |
| `python/rlox/__init__.py` | Export `ExperienceTable`, `ReplayBuffer`, `VarLenStore` |

---

## 9. Acceptance Criteria

All 12 xfail tests in `tests/python/test_bench_buffer_ops.py` pass:

- [ ] `test_experience_table_importable` — `ExperienceTable(obs_dim=4, act_dim=1)` constructs, `len()` works
- [ ] `test_replay_buffer_importable` — `ReplayBuffer(capacity=1000, obs_dim=4, act_dim=1)` constructs
- [ ] `test_varlen_store_importable` — `VarLenStore()` constructs, `num_sequences()` returns 0
- [ ] `test_rlox_push_throughput_cartpole` — >1M transitions/sec for obs_dim=4
- [ ] `test_rlox_push_throughput_atari` — >10K transitions/sec for obs_dim=28224
- [ ] `test_rlox_sample_latency[32]` — <100us median
- [ ] `test_rlox_sample_latency[64]` — <100us median
- [ ] `test_rlox_sample_latency[256]` — <100us median
- [ ] `test_rlox_sample_latency[1024]` — <100us median
- [ ] `test_observations_returns_numpy_without_copy` — Returns `np.ndarray` with shape `(10, 4)`
- [ ] `test_zero_copy_faster_than_explicit_copy` — View access faster than explicit copy
- [ ] `test_varlen_memory_efficiency` — `total_elements` matches, `num_sequences` matches
- [ ] `test_varlen_vs_padded_tensor_memory` — Efficiency ratio < 0.5

Plus:
- [ ] All Rust unit tests pass (`cargo test -p rlox-core`)
- [ ] `ReplayBuffer` is `Send + Sync`
- [ ] No `unsafe` code in the implementation
- [ ] No new dependencies added to `rlox-core`
