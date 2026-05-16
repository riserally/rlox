# rlox-python

PyO3 bindings exposing rlox Rust primitives to Python.

This crate builds the `_rlox_core` native extension module that powers the `rlox` Python package. All heavy computation is done in Rust with the GIL released.

## Exposed Classes

### Environments
- `CartPole`, `VecEnv`, `GymEnv`

### Buffers
- `ReplayBuffer`, `PrioritizedReplayBuffer`, `MmapReplayBuffer`
- `SequenceReplayBuffer`, `HERBuffer`, `OfflineDatasetBuffer`
- `ExperienceTable`, `VarLenStore`, `EpisodeTracker`

### Neural Networks
- `ActorCritic` (Candle backend), `CandleCollector`

### Training
- `RunningStats`, `RunningStatsVec`
- `Pipeline`, `RolloutBatch`

## Exposed Functions

- `compute_gae`, `compute_gae_batched`, `compute_gae_batched_f32`
- `compute_vtrace`
- `shape_rewards_pbrs`, `compute_goal_distance_potentials`
- `reptile_update`, `polyak_update`, `average_weight_vectors`
- `random_shift_batch`, `pack_sequences`
- `compute_group_advantages`, `compute_batch_group_advantages`
- `compute_token_kl`, `compute_token_kl_schulman` (+ batch and f32 variants)

## Building

```bash
pip install maturin
maturin develop --release
```

## Part of rlox

This crate provides the Python bridge for [rlox](https://github.com/wojciechkpl/rlox). It is not intended to be used as a standalone Rust library.

## License

Dual-licensed under [MIT](../../LICENSE-MIT) or [Apache 2.0](../../LICENSE-APACHE).
