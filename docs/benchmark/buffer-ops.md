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
| 4 | 1.54 ms | 228.54 ms | 14.97 ms | 6,486,487 trans/s |
| 28,224 | 134.50 ms | 256.89 ms | 115.01 ms | 74,347 trans/s |

### Push Speedup

| obs_dim | vs TorchRL | 95% CI | vs SB3 | 95% CI |
|---------|-----------|--------|--------|--------|
| 4 | **148.2x** | [143.8, 157.6] | **9.7x** | [9.3, 10.3] |
| 28,224 | **1.9x** | [1.8, 2.0] | **0.9x** | [0.8, 0.9] |

### Sample Latency

| Batch Size | rlox (median) | rlox (p99) | TorchRL (median) | TorchRL (p99) | SB3 (median) | SB3 (p99) |
|-----------|---------------|------------|------------------|---------------|--------------|-----------|
| 32 | 1.5 us | 1.9 us | 20.1 us | 57.3 us | 17.5 us | 49.1 us |
| 64 | 2.0 us | 2.5 us | 20.9 us | 26.1 us | 22.1 us | 26.1 us |
| 256 | 4.2 us | 6.0 us | 28.5 us | 82.2 us | 40.9 us | 47.1 us |
| 1,024 | 9.2 us | 14.7 us | 95.9 us | 135.1 us | 74.7 us | 138.3 us |

### Sample Speedup

| Batch Size | vs TorchRL | 95% CI | vs SB3 | 95% CI |
|-----------|-----------|--------|--------|--------|
| 32 | **13.0x** | [13.0, 13.4] | **11.3x** | [11.1, 11.7] |
| 64 | **10.2x** | [10.1, 10.7] | **10.8x** | [10.7, 11.3] |
| 256 | **6.8x** | [6.6, 7.5] | **9.8x** | [9.5, 10.6] |
| 1,024 | **10.4x** | [9.7, 11.0] | **8.1x** | [7.6, 8.4] |

## Analysis

### Push: Why 148x faster than TorchRL for small observations

Each TorchRL `rb.add(td)` call:
1. Validates the TensorDict schema
2. Converts/copies tensor data into the storage backend
3. Updates internal metadata (indices, counters)

Each rlox `table.push()` call crosses the PyO3 boundary once, copies the observation via `extend_from_slice` (a single memcpy), and increments a counter. No Python object creation, no schema validation.

### Push: Why SB3 wins at Atari-sized observations

At obs_dim=28,224, each push copies ~110KB of data (28,224 × 4 bytes). The memcpy dominates total time, and both rlox and SB3 hit the same memory bandwidth ceiling. SB3's pre-allocated NumPy arrays avoid reallocation, while rlox's `Vec<f32>` may trigger occasional reallocations during growth.

### Sample: Predictable tail latency

rlox's p99 latency is remarkably low (14.7us for batch=1024 vs 135-138us for TorchRL/SB3). This comes from:
- **Pre-allocated ring buffer**: No heap allocation during sampling
- **ChaCha8 RNG**: Deterministic, cache-friendly random number generation
- **Contiguous memory layout**: Sequential reads from flat arrays, no pointer chasing

Source: [`benchmarks/bench_buffer_ops.py`](../../benchmarks/bench_buffer_ops.py)
