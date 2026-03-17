# Research Report: Should rlox Implement a Rust TensorDict?

## 1. Problem Statement

TorchRL's `TensorDict` is the universal data container for RL: environments produce them, buffers store them, policies consume them. It provides named key access, batch dimensions, device management, nested structure, and functional transforms. The question is whether rlox needs an equivalent, or whether the current fixed-struct approach is architecturally superior.

**Research question**: Does rlox's fixed-struct data model (ExperienceRecord, BatchTransition, RolloutBatch, SampledBatch, TensorData) leave critical use cases unserved, and if so, what is the minimal extension that addresses those gaps without sacrificing the 61-1635x performance advantages?

## 2. What TensorDict Actually Is (and What It Costs)

TensorDict [Bou et al., 2023, arXiv:2306.00577] is approximately:

```python
class TensorDict:
    _tensordict: Dict[str, Union[Tensor, TensorDict]]  # nested
    _batch_size: torch.Size                              # shared leading dims
    _device: torch.device
```

Every operation (indexing, slicing, `.to(device)`, `.clone()`, `.reshape()`) dispatches through a Python dictionary, checks batch dimension consistency, and optionally applies lazy transforms. This gives maximum flexibility at the cost of:

1. **Per-access overhead**: Every `td["obs"]` is a `dict.__getitem__` + batch shape validation
2. **Per-step overhead**: Environment wrappers create a new TensorDict per `step()` call
3. **Memory fragmentation**: Each tensor is a separate allocation (no columnar layout)
4. **GIL contention**: Python dict operations serialize parallel access

The benchmarks tell the story:
- GAE: rlox is **1635x** faster than TorchRL (TensorDict overhead per step in the GAE loop)
- Buffer push: rlox is **61x** faster (no dict construction, no batch size validation)
- E2E rollout: rlox is **40x** faster (no per-step TensorDict allocation)

## 3. Inventory of Current rlox Data Structures

| Structure | Location | Fields | Mutability | PyO3 Surface |
|-----------|----------|--------|------------|-------------|
| `ExperienceRecord` | buffer/mod.rs | obs, next_obs, action, reward, terminated, truncated | Owned, transient | Constructed from numpy args |
| `ReplayBuffer` | buffer/ringbuf.rs | 6 flat pre-allocated Vecs | Ring-write, random-read | `sample()` -> PyDict |
| `ExperienceTable` | buffer/columnar.rs | 6 flat append-only Vecs | Append-only | `observations()` -> numpy |
| `SampledBatch` | buffer/ringbuf.rs | 6 owned Vecs + dims | Owned, immutable after construction | Exposed as PyDict |
| `BatchTransition` | env/parallel.rs | obs, rewards, terminated, truncated, terminal_obs | Owned per step | Exposed as PyDict |
| `RolloutBatch` | pipeline/channel.rs | obs, actions, rewards, dones, advantages, returns + dims | Owned, sent through channel | PyRolloutBatch wrapper |
| `TensorData` | rlox-nn/tensor_data.rs | data: Vec<f32>, shape: Vec<usize> | Owned, consumed by NN traits | Constructed from numpy |
| `Transition` | env/mod.rs | obs, reward, terminated, truncated, info: HashMap<String,f64> | Per-step, single env | Internal |

Key observations:
1. **Every struct has known, fixed fields.** There is no dynamism in the Rust layer.
2. **The `info` field on `Transition` is already a HashMap** -- the one place where dynamism exists, and it is deliberately *not* propagated to batch structures.
3. **PyO3 boundary already uses PyDict** -- Python users already get `dict[str, ndarray]`.
4. **All hot-path structures (ReplayBuffer, ExperienceTable, RolloutBatch) are typed columnar Vecs.** This is why they are fast.

## 4. Use Cases TensorDict Addresses

### 4.1 Multi-Modal Observations (e.g., image + proprioception)

**Current gap**: `Observation(Vec<f32>)` is a single flat vector. An environment returning `{"image": [84,84,3], "joints": [17]}` must flatten everything into one Vec, losing semantic structure.

**How bad is this?** For Classic Control, MuJoCo, and LLM-RLHF (rlox's current targets), observations are flat vectors. Multi-modal is needed for:
- Robotics with vision + proprioception
- Multi-agent with heterogeneous observations
- Minecraft/NetHack-style environments

**Verdict**: Real gap, but not blocking current use cases.

### 4.2 Custom Fields (hidden_state, goal, log_prob, entropy)

**Current gap**: `ExperienceRecord` has exactly 6 fields. A user wanting to store `log_prob` or `hidden_state` alongside transitions cannot do so without modifying the Rust struct.

**How bad is this?** PPO already needs log_probs and values stored per-step. The Python `RolloutBatch` has these (obs, actions, rewards, dones, log_probs, values, advantages, returns). The Rust `RolloutBatch` is missing log_probs and values because they come from the NN forward pass, not the environment.

**Current workaround**: Users store extra fields in parallel Python arrays or use the Python-level RolloutBatch dataclass. This works.

**Verdict**: Moderate inconvenience, but the "compute in Rust, compose in Python" architecture handles this by design.

### 4.3 Nested Structure (td["next", "obs"])

**Current gap**: `ExperienceRecord` stores `next_obs` as a separate field. TensorDict's nesting lets you do `td["next"]["obs"]` and `td["next"]["reward"]` (for n-step returns).

**How bad is this?** The flat approach is fine for 1-step transitions. For n-step returns, rlox computes them as SIMD operations over flat arrays. Nesting would add indirection without benefit.

**Verdict**: Not a gap. Nesting is a Python API convenience, not a performance feature.

### 4.4 Device Management (.to(device))

**Current gap**: TensorData is always CPU Vec<f32>. Moving to GPU requires backend-specific conversion.

**How bad is this?** This is by design. rlox-nn traits accept TensorData and each backend (Burn, Candle) converts to its own tensor type internally. A `.to(device)` on a dict-like container would just be sugar for calling the backend's conversion, which is already one line of code.

**Verdict**: Not a gap. Device management belongs in the NN backend, not in the data container.

### 4.5 Lazy Operations (clone, reshape)

**Current gap**: All rlox operations are eager.

**How bad is this?** Laziness helps when you have deep transform chains (10+ stacked wrappers). rlox's architecture pushes transforms to SIMD kernel level, not wrapper chains. The TorchRL lazy approach is actually *slower* because it defers work to Python execution time.

**Verdict**: Not a gap. Eagerness is a feature.

## 5. Quantitative Cost Analysis of Adding a TensorDict

### Option A: Full TensorDict (HashMap<String, Column>)

```rust
enum Column {
    F32(Vec<f32>),
    F64(Vec<f64>),
    Bool(Vec<bool>),
    U32(Vec<u32>),
    Nested(TensorDict),
}

struct TensorDict {
    columns: HashMap<String, Column>,
    batch_size: usize,
}
```

**Cost analysis**:
- HashMap lookup: ~20-40ns per access (FxHashMap: ~10ns). Current struct field access: ~0ns (compiled to offset).
- Buffer push with 6 fields: 6 hash lookups + 6 branch predictions for Column enum = ~120ns overhead. Current: ~0ns overhead (direct memcpy).
- At 256 envs x 2048 steps = 524,288 pushes per rollout: **63ms wasted** on hash lookups alone.
- GAE loop accessing rewards[t], dones[t], values[t] per step: 3 hash lookups x 2048 steps = 6144 lookups = ~123us. Current: ~0us.

**Impact on benchmarks**:
- Buffer push: would go from 61x to ~30-40x vs TorchRL (still fast, but losing ~30% for no compute benefit)
- GAE: negligible impact (GAE is SIMD-bound, not access-bound)
- E2E rollout: ~0.5-1% regression

**Implementation cost**: ~2000 lines of Rust + redesigning all buffer/pipeline/env APIs. Breaking change to every consumer.

**Verdict**: Reject. The overhead is small in absolute terms but the benefit is zero for the hot path, and the implementation cost is enormous.

### Option B: Fixed Structs + Optional Extras (Recommended Hybrid)

```rust
/// Extends ExperienceRecord with user-defined f32 columns.
struct ExperienceRecordExt {
    // Core fields: compiled to fixed offsets, zero overhead
    pub obs: Vec<f32>,
    pub next_obs: Vec<f32>,
    pub action: Vec<f32>,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    // Optional extras: only pay for what you use
    pub extras: Option<SmallVec<[(CompactString, Vec<f32>); 4]>>,
}
```

**Why SmallVec<...; 4>?** Most users need 0-3 extra fields (log_prob, value, hidden_state). SmallVec avoids heap allocation for <= 4 entries. CompactString inlines strings <= 24 bytes (all common field names).

**Cost when extras is None**: Zero. `Option<SmallVec>` is a single pointer check.
**Cost when extras has 2 entries**: ~20ns per push (SmallVec scan + copy). Negligible.

**BUT**: This is solving a problem at the wrong layer. Let me explain why.

### Option C: Do Nothing in Rust. Solve It in Python. (Actual Recommendation)

The current architecture already handles extensibility:

```
Rust Layer (hot path)          Python Layer (flexibility)
---------------------------    --------------------------------
ExperienceRecord (6 fields)    dict with any keys
ReplayBuffer.sample() ->       -> PyDict with numpy arrays
  SampledBatch (6 fields)         + user can add arbitrary keys
RolloutBatch (6 fields)        Python RolloutBatch(@dataclass)
  via Pipeline channel             has 8 fields, easily extended
TensorData (flat f32 + shape)  numpy arrays with any dtype
```

The PyO3 boundary already returns `PyDict`. A Python user can do:

```python
batch = buffer.sample(256, seed=42)     # Returns dict of numpy arrays
batch["hidden_state"] = hidden_states   # Just add it
batch["log_probs"] = log_probs          # Just add it
```

For the Rust-only path (no Python), the user composes fixed structs:

```rust
let batch = buffer.sample(256, seed)?;
let values = critic.value(&TensorData::new(batch.observations.clone(), shape));
// Use batch.observations and values together -- they share the same index space
```

This is the Polars pattern. Polars has a typed DataFrame with a fixed schema. You don't add columns at runtime to a Polars Series -- you construct new columns and join them. The performance comes from knowing the schema at compile time.

## 6. Recommendation: Three Targeted Extensions (Not a TensorDict)

### Extension 1: Dict Observation Space

**What**: Allow `Observation` to be either flat or dict-structured.

```rust
pub enum Observation {
    Flat(Vec<f32>),
    Dict(SmallVec<[(CompactString, Vec<f32>); 4]>),
}
```

**Why**: This is the one place where the fixed-struct model genuinely constrains users. An environment returning `{"image": ..., "joints": ...}` cannot be represented today.

**When**: When rlox targets robotics or vision-based RL. Not blocking for Classic Control, MuJoCo, or LLM-RLHF.

**Performance impact**: Zero for `Flat` variant (current path). The `Dict` variant is only used at the env boundary, not in hot compute kernels.

**Implementation**: ~200 lines. Non-breaking (add variant to enum, update VecEnv to handle both).

### Extension 2: Typed Extra Columns on ReplayBuffer

**What**: Allow registering additional f32 columns at construction time.

```rust
impl ReplayBuffer {
    pub fn new(capacity: usize, obs_dim: usize, act_dim: usize) -> Self { ... }

    /// Register an additional column. Must be called before any push().
    pub fn register_column(&mut self, name: &str, dim: usize) -> Result<ColumnHandle, RloxError> {
        // Returns a typed handle for zero-cost access
    }
}

/// Opaque handle for O(1) column access (index into a Vec, not a HashMap).
#[derive(Copy, Clone)]
pub struct ColumnHandle(usize);
```

**Why**: Off-policy algorithms (SAC, TD3) may want to store `log_prob` or `alpha` alongside transitions. The handle pattern avoids HashMap overhead: `buffer.get_column(handle)` is a Vec index, not a string lookup.

**Performance impact**: Zero when no extra columns registered. When used, access is `O(1)` via handle (Vec index), not `O(1)` amortized (HashMap).

**Implementation**: ~300 lines. Non-breaking (existing API unchanged, new methods are additive).

### Extension 3: PyDict Builder for Custom Python Returns

**What**: A utility to build PyDict returns with arbitrary columns, replacing the current hand-rolled dict construction in buffer.rs.

```rust
/// Helper for building Python dict returns from Rust buffers.
struct BatchDictBuilder<'py> {
    dict: Bound<'py, PyDict>,
    batch_size: usize,
}

impl<'py> BatchDictBuilder<'py> {
    fn add_f32_matrix(&mut self, key: &str, data: Vec<f32>, dim: usize) -> PyResult<()>;
    fn add_f32_vector(&mut self, key: &str, data: Vec<f32>) -> PyResult<()>;
    fn add_bool_vector(&mut self, key: &str, data: Vec<bool>) -> PyResult<()>;
}
```

**Why**: The current PyReplayBuffer::sample() and PyPrioritizedReplayBuffer::sample() have ~30 lines of nearly identical dict construction code. This also makes it trivial for users extending the buffer to add custom columns to the Python return.

**Performance impact**: Zero (same operations, just factored out).

**Implementation**: ~100 lines. Internal refactor, no API change.

## 7. The Polars Analogy (Why This Is the Right Call)

Polars chose `DataFrame` (typed, schema-aware, columnar) over `dict[str, ndarray]` (untyped, schemaless). The result:

| Property | Polars DataFrame | pandas dict-of-arrays | TorchRL TensorDict |
|----------|-----------------|----------------------|-------------------|
| Schema | Compile-time known | Runtime | Runtime |
| Column access | O(1) by index | O(1) by hash | O(1) by hash |
| Validation | At construction | Never | Per-operation |
| Extension | `.with_column()` | `df["new"] = ...` | `td["new"] = ...` |
| Performance | 10-100x pandas | baseline | 0.7-1.0x PyTorch |

rlox's architecture maps to Polars:
- **Fixed structs** = Polars' typed schema (compile-time field access)
- **Flat Vec<f32>** = Polars' Apache Arrow columnar layout
- **PyDict at boundary** = Polars' `.to_pandas()` escape hatch
- **SIMD kernels** = Polars' expression engine

Adding a TensorDict would be like Polars adding a `dict[str, Any]` mode. It would be slower, untyped, and defeat the purpose.

## 8. Decision Matrix

| Criterion | Weight | A: Full TensorDict | B: Fixed + Extras | C: Do Nothing + 3 Extensions |
|-----------|--------|--------------------|--------------------|-------------------------------|
| Accuracy (correctness) | 0.20 | 5 (any schema) | 4 (fixed + opt.) | 4 (fixed + opt.) |
| Training cost (overhead) | 0.15 | 2 (hash overhead) | 4 (opt. overhead) | 5 (zero overhead) |
| Inference latency | 0.15 | 2 (dict dispatch) | 4 (handle dispatch) | 5 (direct access) |
| Data requirements | 0.10 | 5 | 4 | 4 |
| Interpretability | 0.10 | 3 (runtime keys) | 4 (typed + extras) | 5 (explicit structs) |
| Cold start (dev time) | 0.10 | 1 (2000+ lines) | 3 (500 lines) | 5 (600 lines, non-breaking) |
| Scalability | 0.05 | 3 | 4 | 5 |
| Implementation complexity | 0.05 | 1 | 3 | 5 |
| Maintenance burden | 0.05 | 1 | 3 | 5 |
| Production readiness | 0.05 | 2 | 4 | 5 |
| **Weighted Score** | **1.00** | **2.85** | **3.80** | **4.70** |

**Sensitivity**: Even doubling the Accuracy weight (0.40) and halving Training cost (0.075) and Inference latency (0.075), Option C scores 4.45 vs A at 3.40 and B at 3.75. The ranking is robust.

## 9. What NOT to Build

1. **Do not build a Rust HashMap-based TensorDict.** It provides zero performance benefit over Python dicts and adds maintenance burden.

2. **Do not add nesting.** `td["next"]["obs"]` is sugar for `batch.next_obs`. The flat struct is clearer and faster.

3. **Do not add lazy operations.** Laziness helps deep wrapper chains, which rlox does not have. Eagerness + SIMD is the right model.

4. **Do not add device management to data containers.** Device placement belongs in the NN backend (Burn/Candle), not in the buffer/pipeline layer.

5. **Do not make ExperienceRecord generic or trait-based.** The fixed 6-field struct compiles to known offsets. Making it generic would require either monomorphization (code bloat) or dynamic dispatch (overhead).

## 10. Implementation Priority

| Extension | Priority | Effort | Blocking Use Case |
|-----------|----------|--------|-------------------|
| 3. BatchDictBuilder (PyO3 refactor) | Now | 1 day | Code quality / DRY |
| 2. Typed Extra Columns | Phase 7+ | 2 days | SAC/TD3 log_prob storage |
| 1. Dict Observation Space | When needed | 2 days | Robotics / vision RL |

## 11. References

[1] A. Bou et al., "TorchRL: A Data-Driven Decision-Making Library for PyTorch," arXiv:2306.00577, 2023.
[2] R. Brockman et al., "OpenAI Gym," arXiv:1606.01540, 2016.
[3] J. Ritchie and R. Vink, "Polars: Blazingly Fast DataFrames in Rust," github.com/pola-rs/polars, 2024.
[4] S. Keshav, "How to Read a Paper," ACM SIGCOMM Computer Communication Review, vol. 37, no. 3, pp. 83-84, 2007.
