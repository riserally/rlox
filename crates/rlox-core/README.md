# rlox-core

Pure-Rust reinforcement learning primitives -- the data plane of [rlox](https://github.com/riserally/rlox).

## Key Types

### Environments

- `CartPole`, `Pendulum` -- built-in classic control environments
- `VecEnv` -- Rayon-parallel vectorized environment stepping
- `GymEnv` -- Gymnasium wrapper via PyO3

### Buffers

- `ReplayBuffer` -- fixed-capacity ring buffer with uniform sampling
- `PrioritizedReplayBuffer` -- SumTree-backed prioritized experience replay
- `MmapReplayBuffer` -- memory-mapped replay for large datasets
- `SequenceReplayBuffer` -- variable-length sequence storage (RNNs, transformers)
- `HERBuffer` -- Hindsight Experience Replay with configurable goal strategies
- `OfflineDatasetBuffer` -- offline RL dataset loading and sampling
- `ExperienceTable` -- columnar storage with extra columns
- `EpisodeTracker` -- episode boundary detection and metadata

### Training Ops

- `compute_gae` / `compute_gae_batched` -- Generalized Advantage Estimation (147x vs NumPy)
- `compute_vtrace` -- V-trace off-policy correction (IMPALA)
- `shape_rewards_pbrs` -- potential-based reward shaping
- `reptile_update`, `polyak_update` -- weight update utilities
- `RunningStats` -- Welford online mean/variance

### LLM Ops

- `compute_group_advantages` / `compute_batch_group_advantages` -- GRPO (35x vs NumPy)
- `compute_token_kl` / `compute_token_kl_schulman` -- token-level KL divergence
- `pack_sequences` -- sequence packing for efficient batching

### Pipeline

- `Pipeline` -- async rollout collector with crossbeam channels and backpressure

## Usage

```rust
use rlox_core::training::compute_gae;
use rlox_core::buffer::ReplayBuffer;
use rlox_core::env::{CartPole, VecEnv, RLEnv};

// GAE computation
let rewards = vec![1.0, 1.0, 0.0];
let values = vec![0.5, 0.6, 0.4];
let dones = vec![0.0, 0.0, 1.0];
let (advantages, returns) = compute_gae(&rewards, &values, &dones, 0.0, 0.99, 0.95);

// Replay buffer
let mut buffer = ReplayBuffer::new(10_000, 4, 1);
buffer.push_slices(
    &[0.1, 0.2, 0.3, 0.4],   // obs
    &[0.2, 0.3, 0.4, 0.5],   // next_obs
    &[1.0],                    // action
    1.0,                       // reward
    false,                     // done
);
let batch = buffer.sample(32, 42).unwrap();

// Vectorized environments
let envs: Vec<Box<dyn RLEnv>> = (0..8)
    .map(|i| Box::new(CartPole::new(i)) as Box<dyn RLEnv>)
    .collect();
let vec_env = VecEnv::new(envs);
```

## Part of rlox

This crate is the Rust data plane of the [rlox](https://github.com/riserally/rlox) reinforcement learning framework. See the main project for Python bindings, algorithms, and full documentation.

## License

Dual-licensed under [MIT](../../LICENSE-MIT) or [Apache 2.0](../../LICENSE-APACHE).
