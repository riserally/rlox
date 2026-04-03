# Plan: Lock-free Concurrent Replay Buffer + Diffusion Policy

## Task 1: Lock-free Concurrent Replay Buffer (Rust)

### Architecture

```mermaid
flowchart LR
    subgraph Producers["Actor Threads (N)"]
        A1[Actor 1]
        A2[Actor 2]
        AN[Actor N]
    end

    subgraph Buffer["ConcurrentReplayBuffer"]
        WP[AtomicUsize write_pos]
        CF["AtomicBool[] commit_flags"]
        DATA["Pre-allocated flat arrays"]
        AC[AtomicUsize committed_count]
    end

    subgraph Consumer["Learner Thread"]
        S[sample batch_size, seed]
    end

    A1 -->|fetch_add| WP
    A2 -->|fetch_add| WP
    AN -->|fetch_add| WP
    WP -->|claim slot| DATA
    DATA -->|set flag| CF
    CF -->|read committed| S
```

### Sequence: Concurrent Push

```mermaid
sequenceDiagram
    participant T1 as Actor Thread 1
    participant T2 as Actor Thread 2
    participant WP as write_pos (Atomic)
    participant Data as Slot Arrays
    participant CF as commit_flags

    T1->>WP: fetch_add(1, Relaxed)
    Note right of WP: returns slot=0
    T2->>WP: fetch_add(1, Relaxed)
    Note right of WP: returns slot=1
    T1->>Data: write obs/act/reward to slot 0
    T2->>Data: write obs/act/reward to slot 1
    T1->>CF: store(true, Release) at slot 0
    T2->>CF: store(true, Release) at slot 1
```

### Key Design Decisions

- `fetch_add` on global `write_pos` for slot claiming (wraps via modulo)
- `AtomicBool` per slot as commit flag (Release on write, Acquire on read)
- `AtomicUsize` for `committed_count` incremented after commit flag set
- Sampling only reads committed slots
- On wrap-around, old commit flags get cleared when slot is reclaimed (set false before write)

## Task 2: Diffusion Policy (Python)

```mermaid
flowchart TB
    subgraph Training
        D[Dataset/Buffer] --> B[Sample Batch]
        B --> FD[Forward Diffusion: add noise at t]
        FD --> DN[Denoising Network predicts noise]
        DN --> L["Loss: MSE(eps, eps_pred)"]
        L --> OPT[Optimizer Step]
    end

    subgraph Inference
        N[Sample x_T ~ N 0,I] --> RD[Reverse Diffusion Loop]
        RD --> |t = T...1| PRED[Predict noise eps_theta]
        PRED --> STEP[DDPM/DDIM step]
        STEP --> |repeat| RD
        STEP --> A[Clean Action Sequence]
    end
```

### Components
- `DiffusionPolicyConfig(ConfigMixin)` dataclass
- `NoiseSchedule` class (linear + cosine)
- `DenoisingMLP` network (obs-conditioned noise predictor)
- `DiffusionPolicy` algorithm class satisfying `Algorithm` protocol
