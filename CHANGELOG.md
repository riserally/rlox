# Changelog

All notable changes to rlox are documented here.

## [0.2.0] - 2026-03-16

### Added
- **Phase 7: Algorithm Completeness**
  - `GymVecEnv` wrapper for arbitrary Gymnasium environments (AutoresetMode.SAME_STEP)
  - `ContinuousPolicy` (Gaussian, orthogonal init) for on-policy continuous control
  - `BatchSteppable` trait for environment abstraction
  - Auto env detection: PPO/A2C auto-select Discrete/Continuous policy from action space
  - `reward_fn` parameter on `RolloutCollector` for reward shaping
  - Callbacks wired into all 7 algorithms (PPO, SAC, DQN, TD3, A2C, GRPO, DPO)
  - `save()`/`from_checkpoint()` on PPO, SAC, DQN, TD3, GRPO, DPO
  - `from_yaml()`/`to_yaml()` on PPOConfig, SACConfig, DQNConfig
  - GRPO batched advantages (eliminates Python loop, uses `compute_batch_group_advantages`)

- **Phase 8: Production Hardening**
  - Statistical evaluation toolkit: IQM, bootstrap CI, performance profiles, P(improvement)
  - `TrainingDiagnostics` callback: entropy collapse, KL spike, gradient explosion detection
  - Memory-mapped replay buffer (`MmapReplayBuffer`) for hot/cold architecture
  - CI workflows: GitHub Actions for tests + maturin wheel builds (4 platforms)
  - Experiment metadata capture + `save_experiment()`

- **Phase 9: Distributed & Scale**
  - Decoupled collection/training pipeline (crossbeam channels, `AsyncCollector`)
  - gRPC distributed env workers (`rlox-grpc` crate with tonic)
  - Multi-GPU training composition (PyTorch DDP wrapper)
  - vLLM, TGI, SGLang inference backends with factory
  - `RemoteEnvPool` Python client for gRPC workers
  - Transition provenance (`TransitionMeta` with serialize/deserialize)
  - MAPPO, DreamerV3, IMPALA algorithms with env auto-detection
  - API 1.0 freeze: comprehensive `__all__`, stability tests

- **Buffer Extensions**
  - Typed extra columns (`register_column`/`push_extra`) with O(1) ColumnHandle access
  - Dict observation space (`Observation::Dict`, `ObsSpace::Dict`)
  - `BatchDictBuilder` for deduplicated PyO3 dict construction

- **Infrastructure**
  - MIT OR Apache-2.0 dual license
  - Published to crates.io: rlox-core, rlox-nn, rlox-burn, rlox-candle
  - Tutorial: custom rewards and training loops (1,480 lines)
  - Logo and citation info (CITATION.cff)

### Fixed
- **Critical**: `PyVecEnv` silently fell back to CartPole for unknown env_ids — now raises `ValueError`
- **Critical**: Replay buffer missing `next_obs` — off-policy algorithms (SAC, TD3, DQN) computed wrong Bellman targets
- SAC: action scaling now multiplies by `act_high` (was only clipping)
- TD3: critic target updates moved outside `policy_delay` gate
- DQN: n-step flush uses actual termination flags (was hardcoded `terminated=True`)
- Config consolidation: single validated `PPOConfig` (was duplicated)

### Test Suite
- 313 Rust tests (was 255)
- 382 Python tests (was 85)
- Zero benchmark regressions

## [0.1.0] - 2026-03-14

### Added
- Phases 0-6: core Rust engine, environment stepping, buffers, GAE, V-trace
- LLM post-training: GRPO, DPO, token KL, sequence packing
- NN backend abstraction: rlox-nn traits, rlox-burn, rlox-candle
- Three-framework benchmark suite (rlox vs TorchRL vs SB3)
- Convergence benchmarks (rlox vs SB3 on Classic Control)
- 255 Rust tests, 85 Python tests
