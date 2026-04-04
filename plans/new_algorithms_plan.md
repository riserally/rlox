# New Algorithms Implementation Plan

## Algorithms: Decision Transformer, QMIX, Cal-QL

### Implementation Order (TDD)

1. Write all tests first (`tests/python/test_new_algorithms.py`)
2. Add config dataclasses to `python/rlox/config.py`
3. Implement Decision Transformer (`python/rlox/algorithms/decision_transformer.py`)
4. Implement QMIX (`python/rlox/algorithms/qmix.py`)
5. Implement Cal-QL (`python/rlox/algorithms/calql.py`) -- CQL + calibration (no existing CQL)
6. Register all in `trainer.py` and export from `__init__.py`
7. Run tests, iterate until green

```mermaid
flowchart TD
    A[Write Tests] --> B[Add Configs to config.py]
    B --> C1[Decision Transformer]
    B --> C2[QMIX]
    B --> C3[Cal-QL]
    C1 --> D[Register in trainer.py]
    C2 --> D
    C3 --> D
    D --> E[Export from __init__.py]
    E --> F[Run Tests - All Green]
```

### Architecture

```mermaid
flowchart LR
    subgraph Decision Transformer
        RTG[RTG Embedding] --> TF[Transformer Decoder]
        State[State Embedding] --> TF
        Action[Action Embedding] --> TF
        PE[Positional Encoding] --> TF
        TF --> ActionPred[Action Prediction Head]
    end

    subgraph QMIX
        Q1[Agent 1 Q-Net] --> MN[Mixing Network]
        Q2[Agent 2 Q-Net] --> MN
        Q3[Agent N Q-Net] --> MN
        GS[Global State] --> HN[Hypernetwork]
        HN --> MN
        MN --> QTot[Q_total]
    end

    subgraph Cal-QL
        CQL[CQL Base] --> CAL[Calibration Layer]
        QOffline[Offline Q] --> CAL
        QCurrent[Current Q] --> CAL
        CAL --> ScaledPenalty[Calibrated Penalty]
    end
```

### Sequence Diagram

```mermaid
sequenceDiagram
    participant Test as Test Suite
    participant Trainer as Trainer Registry
    participant DT as DecisionTransformer
    participant QM as QMIX
    participant CQ as CalQL

    Test->>DT: construct(env_id)
    DT-->>Test: instance
    Test->>DT: train(total_timesteps)
    DT-->>Test: metrics dict
    Test->>DT: save(path)
    Test->>DT: from_checkpoint(path)
    DT-->>Test: restored instance

    Test->>Trainer: Trainer("dt", env)
    Trainer->>DT: instantiate
    DT-->>Trainer: algo
    Trainer-->>Test: trainer

    Test->>QM: construct(env_id, n_agents=3)
    QM-->>Test: instance
    Test->>QM: verify monotonic mixing

    Test->>CQ: construct(env_id)
    CQ-->>Test: instance
    Test->>CQ: verify calibration scales penalty
```
