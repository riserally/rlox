# rlox Implementation Plans

Detailed phase-by-phase implementation plans for the rlox reinforcement learning framework.

Each phase document contains: objectives, feature mapping to PRD, TDD test specifications (write tests FIRST), implementation steps, technical reasoning, acceptance criteria, and current status.

## Phases

| Phase | Document | Focus | Status |
|-------|----------|-------|--------|
| 0 | [phase-0-skeleton.md](phase-0-skeleton.md) | Project scaffold, toolchain validation | COMPLETE |
| 1 | [phase-1-environment-engine.md](phase-1-environment-engine.md) | RLEnv trait, CartPole, parallel stepping, Gym bridge | COMPLETE |
| 2 | [phase-2-experience-storage.md](phase-2-experience-storage.md) | Columnar store, ring buffer, variable-length sequences, zero-copy bridge | COMPLETE |
| 3 | [phase-3-training-orchestrator.md](phase-3-training-orchestrator.md) | GAE, batch assembly, KL controller | COMPLETE |
| 4 | [phase-4-llm-post-training.md](phase-4-llm-post-training.md) | Token KL, GRPO advantages, DPO pairs | COMPLETE |
| 5 | [phase-5-polish-and-api.md](phase-5-polish-and-api.md) | Type stubs, proptest, API ergonomics | COMPLETE |
| 6 | [phase-6-three-framework-benchmark.md](phase-6-three-framework-benchmark.md) | rlox vs TorchRL vs SB3 benchmark suite | COMPLETE |
| 7 | [phase-7-algorithm-completeness.md](phase-7-algorithm-completeness.md) | PPO, GRPO, DPO end-to-end; Layer 1 API; wheels | NOT STARTED |
| 8 | [phase-8-production-hardening.md](phase-8-production-hardening.md) | SAC, TD3, DQN; Layer 2 trainers; configs; MuJoCo native | NOT STARTED |
| 9 | [phase-9-distributed-and-scale.md](phase-9-distributed-and-scale.md) | Multi-GPU, gRPC workers, MAPPO, DreamerV3, IMPALA, API 1.0 | NOT STARTED |

## Supporting Documents

| Document | Purpose |
|----------|---------|
| [rust_python_architecture_review.md](rust_python_architecture_review.md) | Rust/Python extensibility patterns, PyO3 design, critical bugs |

## TDD Methodology

Every phase follows strict Test-Driven Development:

1. **RED** — Write failing tests that define the expected behavior
2. **GREEN** — Implement the minimum code to pass those tests
3. **REFACTOR** — Clean up while keeping tests green

Each plan document specifies tests BEFORE implementation steps. Tests serve as the executable specification.

## Dependency Graph (Phases)

```
Phase 0 (Skeleton)
  └─> Phase 1 (Environment Engine)
        └─> Phase 2 (Experience Storage)
              └─> Phase 3 (Training Core)
                    └─> Phase 4 (LLM Post-Training)
                          └─> Phase 5 (Polish & API)
                                └─> Phase 6 (Benchmarks)
                                      └─> Phase 7 (Algorithm Completeness — v0.5)
                                            └─> Phase 8 (Production Hardening — v0.7)
                                                  └─> Phase 9 (Distributed & Scale — v1.0)
```

## Version Milestones

| Version | Phase | Target | Key Deliverables |
|---------|-------|--------|------------------|
| v0.5 | 7 | Q3 2026 | PPO + GRPO + DPO end-to-end, Layer 1 API, prebuilt wheels |
| v0.7 | 8 | Q4 2026 | SAC + TD3 + DQN, Layer 2 trainers, MuJoCo native, configs |
| v1.0 | 9 | Q2 2027 | Distributed training, MAPPO, DreamerV3, IMPALA, stable API |
