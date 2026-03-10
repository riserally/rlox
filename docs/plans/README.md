# rlox Implementation Plans

Detailed phase-by-phase implementation plans for the rlox reinforcement learning framework.

Each phase document contains: objectives, feature mapping to PRD, TDD test specifications (write tests FIRST), implementation steps, technical reasoning, acceptance criteria, and current status.

## Phases

| Phase | Document | Focus | Status |
|-------|----------|-------|--------|
| 0 | [phase-0-skeleton.md](phase-0-skeleton.md) | Project scaffold, toolchain validation | COMPLETE |
| 1 | [phase-1-environment-engine.md](phase-1-environment-engine.md) | RLEnv trait, CartPole, parallel stepping, Gym bridge | COMPLETE |
| 2 | [phase-2-experience-storage.md](phase-2-experience-storage.md) | Columnar store, lock-free buffer, variable-length sequences, zero-copy bridge | NOT STARTED |
| 3 | [phase-3-training-orchestrator.md](phase-3-training-orchestrator.md) | GAE, batch assembly, KL controller, PPO end-to-end | NOT STARTED |
| 4 | [phase-4-llm-post-training.md](phase-4-llm-post-training.md) | Token MDP, vLLM/TGI integration, DPO, GRPO | NOT STARTED |
| 5 | [phase-5-polish-and-api.md](phase-5-polish-and-api.md) | One-liner API, config system, error messages, correctness tests | NOT STARTED |

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
        └─> Phase 2 (Experience Storage)  ─────────┐
              └─> Phase 3 (Training + PPO)         │
                    ├─> Phase 4 (LLM Post-training) │
                    └─> Phase 5 (Polish & API) <────┘
```

Phase 4 and Phase 5 can proceed in parallel once Phase 3 is complete.
