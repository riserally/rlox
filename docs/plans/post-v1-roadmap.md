# Post-v1.0 Roadmap

**Date:** 2026-03-31
**Status:** Implementation starting

---

## Summary

| # | Item | Effort | Dependencies |
|---|------|--------|-------------|
| 9 | Asana cleanup | 0.5 day | None |
| 7 | MAPPO PettingZoo e2e test | 2-3 days | PettingZoo package |
| 5 | Distributed IMPALA | 1-2 weeks | gRPC server (exists) |
| 6 | DreamerV3 CNN | 1-2 weeks | None |
| 4 | MuJoCo FFI | 2-3 weeks | mujoco-sys crate |
| 8 | Multi-seed benchmarks | 1 week compute | Item 4 for full coverage |

```mermaid
graph TD
    I9["9. Asana Cleanup"] --> I7["7. MAPPO e2e"]
    I9 --> I5["5. Distributed IMPALA"]
    I9 --> I6["6. DreamerV3 CNN"]
    I9 --> I4["4. MuJoCo FFI"]
    I4 --> I8["8. Multi-seed Benchmarks"]

    style I9 fill:#51cf66
    style I7 fill:#51cf66
    style I5 fill:#4dabf7
    style I6 fill:#4dabf7
    style I4 fill:#ff922b
    style I8 fill:#ff922b
```

```mermaid
gantt
    title Post-v1.0 Timeline
    dateFormat YYYY-MM-DD
    section Housekeeping
        Asana cleanup           :i9, 2026-03-31, 1d
    section Quick Wins
        MAPPO e2e test          :i7, after i9, 3d
    section Core Features
        Distributed IMPALA      :i5, after i9, 14d
        DreamerV3 CNN           :i6, after i9, 14d
        MuJoCo FFI              :i4, after i9, 21d
    section Validation
        Multi-seed benchmarks   :i8, after i4, 7d
```
