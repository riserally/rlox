# Benchmark: Buffer Operations

Replay buffer push throughput and sample latency — fundamental operations in every RL training loop.

## What is Measured

### Push Throughput
Insert 10,000 transitions into a buffer, one at a time (per-transition push, the common RL pattern).

| Framework | API | Buffer Type |
|-----------|-----|-------------|
| rlox `ExperienceTable` | `table.push(obs, action, reward, terminated, truncated)` | Flat `Vec<f32>` columnar store |
| TorchRL `ReplayBuffer` | `rb.add(td)` where `td` is a TensorDict | `LazyTensorStorage` |
| SB3 `ReplayBuffer` | `buf.add(obs, next_obs, action, reward, done, infos)` | Pre-allocated NumPy arrays |

Two observation sizes tested:
- **obs_dim=4** (CartPole) — overhead-dominated, stress-tests per-call cost
- **obs_dim=28,224** (Atari 84×84×4) — memcpy-dominated, tests memory bandwidth

### Sample Latency
Draw a random batch from a full buffer (100,000 transitions, obs_dim=4).

| Framework | API |
|-----------|-----|
| rlox `ReplayBuffer` | `buf.sample(batch_size, seed)` — ChaCha8 RNG, contiguous memory |
| TorchRL `ReplayBuffer` | `rb.sample()` — torch-based indexing |
| SB3 `ReplayBuffer` | `buf.sample(batch_size)` — `np.random.choice` + row indexing |

## Results

### Push Throughput

| obs_dim | rlox | TorchRL | SB3 | rlox throughput |
|---------|------|---------|-----|-----------------|
| 4 | 3.59 ms | 218.67 ms | 14.24 ms | 2,782,205 trans/s |
| 28,224 | 136.01 ms | 248.69 ms | 109.19 ms | 73,522 trans/s |

### Push Speedup

| obs_dim | vs TorchRL | 95% CI | vs SB3 | 95% CI |
|---------|-----------|--------|--------|--------|
| 4 | **60.8x** | [59.6, 62.3] | **4.0x** | [3.9, 4.0] |
| 28,224 | **1.8x** | [1.7, 2.0] | **0.8x** | [0.7, 0.9] |

### Sample Latency

| Batch Size | rlox (median) | rlox (p99) | TorchRL (median) | TorchRL (p99) | SB3 (median) | SB3 (p99) |
|-----------|---------------|------------|------------------|---------------|--------------|-----------|
| 32 | 1.5 us | 1.8 us | 17.2 us | 24.0 us | 17.2 us | 24.8 us |
| 64 | 1.5 us | 2.1 us | 20.1 us | 26.4 us | 20.6 us | 35.0 us |
| 256 | 4.2 us | 6.2 us | 22.2 us | 36.3 us | 29.3 us | 39.4 us |
| 1,024 | 10.1 us | 17.0 us | 65.0 us | 109.4 us | 61.0 us | 75.0 us |

### Sample Speedup

| Batch Size | vs TorchRL | 95% CI | vs SB3 | 95% CI |
|-----------|-----------|--------|--------|--------|
| 32 | **11.2x** | [10.9, 11.6] | **11.2x** | [10.8, 11.6] |
| 64 | **13.4x** | [12.9, 13.6] | **13.8x** | [13.2, 14.0] |
| 256 | **5.3x** | [5.0, 5.6] | **7.0x** | [6.7, 7.4] |
| 1,024 | **6.5x** | [6.0, 6.8] | **6.1x** | [5.8, 6.3] |

## Analysis

### Push: Why 61x faster than TorchRL for small observations

Each TorchRL `rb.add(td)` call:
1. Validates the TensorDict schema
2. Converts/copies tensor data into the storage backend
3. Updates internal metadata (indices, counters)

Each rlox `table.push()` call crosses the PyO3 boundary once, copies the observation via `extend_from_slice` (a single memcpy), and increments a counter. No Python object creation, no schema validation.

### Push: Why SB3 wins at Atari-sized observations

At obs_dim=28,224, each push copies ~110KB of data (28,224 × 4 bytes). The memcpy dominates total time, and both rlox and SB3 hit the same memory bandwidth ceiling. SB3's pre-allocated NumPy arrays avoid reallocation, while rlox's `Vec<f32>` may trigger occasional reallocations during growth.

### Sample: Predictable tail latency

rlox's p99 latency is remarkably low (17.0us for batch=1024 vs 75-109us for TorchRL/SB3). This comes from:
- **Pre-allocated ring buffer**: No heap allocation during sampling
- **ChaCha8 RNG**: Deterministic, cache-friendly random number generation
- **Contiguous memory layout**: Sequential reads from flat arrays, no pointer chasing

Source: [`benchmarks/bench_buffer_ops.py`](../../benchmarks/bench_buffer_ops.py)
