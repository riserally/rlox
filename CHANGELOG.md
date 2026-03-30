# Changelog

All notable changes to rlox are documented here.

## [0.3.0] - 2026-03-30

### Added
- **VecNormalize environment wrapper** — obs/reward normalization at the
  environment boundary (SB3 architecture), replacing collector-level normalization
- **RunningStatsVec** — per-dimension Welford statistics in Rust (PyO3 exposed)
- **Native Pendulum-v1** — Rust environment with continuous action space
- **Polymorphic VecEnv.step_all** — accepts discrete (Vec<u32>) and continuous
  (ndarray float32) actions
- **VecEnv.action_space property** — typed dict for Python-side detection
- **VecEnv protocol** — formal protocol in `protocols.py`
- **A2CConfig, TD3Config** — dataclass configs with validation and YAML support
- **Offline RL**: TD3+BC, IQL, CQL, BC algorithms with `OfflineDatasetBuffer` (Rust)
- **Candle Hybrid Collection**: `CandleCollector` (180K SPS on CartPole), `HybridPPO` trainer
- **OffPolicyCollector**: Reusable multi-env collection for SAC, TD3, DQN (`n_envs` parameter)
- **`OfflineAlgorithm` base class** with `OfflineDataset` protocol for extensible offline RL
- **`SharedPolicy`** + weight sync for Candle/PyTorch interop
- **`RolloutBatch`** extended with `log_probs` and `values` fields
- **SB3 migration guide** at `docs/tutorials/migration-sb3.md`
- **API reference** pages with mkdocstrings autodoc
- **CONTRIBUTING.md** with development setup and guidelines
- **Cross-navigation** header across all documentation components
- Python 3.13 added to CI test matrix

### Fixed
- **Truncation bootstrap** — truncated episodes now bootstrap V(terminal_obs)
  instead of treating as deaths (value=0). Critical for MuJoCo time limits.
- **Per-dimension obs normalization** — replaced scalar mean/std with per-dim
  tracking, preserving observation structure across different scales
- **Return-based reward normalization** — std of discounted returns (SB3
  convention) instead of std of raw rewards
- **Train/collect obs mismatch** — consistent normalization during collection
  and training
- **A2C advantage normalization default** — changed to False, preventing gradient
  explosion with small batches (n_steps=5, batch=40)
- **log_std init** — 0.0 (std=1.0) matching SB3, was -0.5
- **GCS upload path** — absolute paths in convergence benchmark scripts
- IMPALA: V-trace now uses computed bootstrap value instead of hardcoded 0.0
- IMPALA: Auto-detects continuous envs, falls back to GymVecEnv for non-CartPole
- DreamerV3: World model frozen during actor-critic training (prevents gradient leakage)
- DreamerV3: Gradient clipping added to both world model and actor-critic updates
- MAPPO: NotImplementedError for n_agents > 1 (prevents silent dimension mismatch)
- MAPPO: Simplified critic input for single-agent case

### Changed
- Normalization moved from `RolloutCollector` to `VecNormalize` wrapper
- PPO auto-wraps env with VecNormalize when normalize flags set
- EvalCallback freezes normalization stats during evaluation

### Improved
- Landing page redesigned with quickstart, benchmarks, comparison table, algorithm grid
- Rust crate descriptions and lib.rs doc comments updated
- 80+ new Python tests (convergence fixes, VecNormalize, Pendulum, offline RL)
- 30+ new Rust tests (RunningStatsVec, Pendulum, OfflineDatasetBuffer)

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
