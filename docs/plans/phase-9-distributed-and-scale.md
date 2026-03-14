# Phase 9 — Distributed and Scale (v1.0)

**Status: NOT STARTED**
**Target: Q2 2027**
**PRD Features: A11–A13, D1–D4, L2–L3, R6**
**Depends on: Phase 8 (production-hardened, full algorithm set)**

---

## Objective

Scale rlox to multi-GPU and multi-machine training. Add multi-agent RL (MAPPO, QMIX), world-model RL (DreamerV3), and async distributed RL (IMPALA). Integrate with LLM inference servers (vLLM, TGI). Ship stable API 1.0 with semver guarantees.

## PRD Feature Mapping

| PRD ID | Feature | Priority | Description |
|--------|---------|----------|-------------|
| D1 | Single-machine multi-GPU | P1 | PyTorch DDP/FSDP for gradient sync; rlox handles data collection |
| D2 | Decoupled collection/training | P1 | Async Rust tasks for parallel collect and learn |
| D3 | Distributed env workers (gRPC) | P2 | Environment stepping across multiple machines |
| D4 | Multi-node training | P2 | PyTorch DDP/FSDP for gradients; Rust handles data routing |
| A11 | MAPPO / QMIX | P2 | Multi-agent cooperative RL |
| A12 | DreamerV3 | P2 | World-model RL with Rust imagination rollouts |
| A13 | IMPALA / V-trace | P2 | Asynchronous distributed RL |
| L2 | vLLM / TGI / SGLang integration | P0 | Async Rust client for inference servers |
| L3 | Reward model serving | P0 | Batch RM scoring via REST/gRPC |
| R6 | Transition provenance | P2 | Full lineage tracking per transition |

## Architectural Decisions

### Distributed Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Coordinator (Python)                 │
│  Manages: training loop, checkpoints, logging        │
│  Owns: PyTorch model, optimizer, DDP/FSDP context    │
├─────────────────────────────────────────────────────┤
│              rlox Data Plane (Rust)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Env Worker 0  │  │ Env Worker 1  │  │ Env Worker N│ │
│  │ (local Rayon) │  │ (gRPC remote) │  │ (gRPC)     │ │
│  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘ │
│         └─────────────┬───┘                 │        │
│                       v                     │        │
│              ┌────────────────┐              │        │
│              │ Experience     │<─────────────┘        │
│              │ Aggregator     │                       │
│              │ (lock-free Q)  │                       │
│              └───────┬────────┘                       │
│                      v                               │
│              ┌────────────────┐                       │
│              │ Batch Builder  │                       │
│              │ (GAE, packing) │                       │
│              └───────┬────────┘                       │
│                      │ zero-copy                     │
├──────────────────────┼──────────────────────────────┤
│                      v                               │
│              PyTorch forward/backward                │
│              DDP/FSDP gradient sync                  │
└─────────────────────────────────────────────────────┘
```

Key principle: **rlox handles data movement; PyTorch handles gradient movement.** No reinventing DDP.

### Collection/Training Decoupling

```rust
// crates/rlox-dist/src/pipeline.rs

/// Decoupled collection and training pipeline.
/// Collection runs on Rayon thread pool.
/// Training batches flow through a lock-free channel.
pub struct Pipeline {
    collector_handle: JoinHandle<()>,
    batch_rx: crossbeam::channel::Receiver<RolloutBatch>,
    config: PipelineConfig,
}

impl Pipeline {
    pub fn new(
        env_factory: Box<dyn Fn() -> Box<dyn BatchSteppable>>,
        n_envs: usize,
        n_steps: usize,
        batch_size: usize,
    ) -> Self { ... }

    /// Non-blocking: get next batch if available.
    pub fn try_next_batch(&self) -> Option<RolloutBatch> {
        self.batch_rx.try_recv().ok()
    }

    /// Blocking: wait for next batch.
    pub fn next_batch(&self) -> RolloutBatch {
        self.batch_rx.recv().unwrap()
    }
}
```

## Implementation Plan

### Part A: Decoupled Collection/Training (Weeks 1–3)

The foundation for all distributed work. Collector and learner run as separate async tasks.

#### A.1 Lock-Free Experience Channel

```rust
// crates/rlox-dist/src/channel.rs
use crossbeam::channel::{bounded, Sender, Receiver};

/// Bounded channel for experience transfer.
/// Backpressure: if learner is slow, collector blocks (no memory blowup).
pub fn experience_channel(capacity: usize) -> (ExperienceSender, ExperienceReceiver) {
    let (tx, rx) = bounded(capacity);
    (ExperienceSender(tx), ExperienceReceiver(rx))
}
```

#### A.2 Async Collector Task

```rust
// crates/rlox-dist/src/collector.rs

/// Collector runs in its own thread, stepping envs and computing GAE.
/// Sends completed rollout batches to the learner via channel.
pub struct AsyncCollector {
    envs: Box<dyn BatchSteppable>,
    buffer: ExperienceTable,
    tx: ExperienceSender,
}

impl AsyncCollector {
    pub fn run(&mut self, policy_weights: Arc<RwLock<PolicyWeights>>) {
        loop {
            // 1. Get latest policy weights (reader lock — non-blocking)
            let weights = policy_weights.read().unwrap();

            // 2. Collect n_steps (Rayon parallel env stepping)
            let batch = self.collect_rollout(&weights);

            // 3. Compute GAE (Rust — 140x faster)
            let (advs, rets) = compute_gae(&batch.rewards, &batch.values, ...);

            // 4. Send to learner (blocks if channel full = backpressure)
            self.tx.send(RolloutBatch { batch, advs, rets });
        }
    }
}
```

### Part B: gRPC Distributed Env Workers (Weeks 3–5)

```rust
// crates/rlox-dist/src/grpc/env_service.rs
use tonic::{transport::Server, Request, Response, Status};

/// gRPC service for remote environment stepping.
/// Each worker runs N environments locally via Rayon.
#[tonic::async_trait]
impl EnvService for EnvWorker {
    async fn step_batch(
        &self,
        request: Request<StepRequest>,
    ) -> Result<Response<StepResponse>, Status> {
        let actions = request.into_inner().actions;
        let results = self.envs.step_batch(&actions)?;
        Ok(Response::new(StepResponse::from(results)))
    }

    async fn reset_batch(
        &self,
        request: Request<ResetRequest>,
    ) -> Result<Response<ResetResponse>, Status> { ... }
}
```

```protobuf
// crates/rlox-dist/proto/env.proto
service EnvService {
    rpc StepBatch(StepRequest) returns (StepResponse);
    rpc ResetBatch(ResetRequest) returns (ResetResponse);
    rpc GetSpaces(Empty) returns (SpacesResponse);
}
```

Python usage:

```python
from rlox.distributed import RemoteEnvPool

# Launch workers on remote machines
pool = RemoteEnvPool(
    workers=["gpu-node-1:50051", "gpu-node-2:50051"],
    envs_per_worker=64,
)

# Use exactly like local VecEnv
trainer = PPOTrainer(env=pool, config=PPOConfig(n_envs=128))
trainer.train(10_000_000)
```

### Part C: Multi-GPU Training (Week 4)

rlox doesn't reimplement DDP — it composes with PyTorch's existing distributed primitives:

```python
# python/rlox/distributed/multi_gpu.py
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class MultiGPUTrainer:
    """Wraps any rlox trainer for multi-GPU training."""

    def __init__(self, trainer_cls, env: str, config, **kwargs):
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")

        self.trainer = trainer_cls(env=env, config=config, device=device, **kwargs)
        self.trainer.policy = DDP(self.trainer.policy, device_ids=[rank])

        # Each GPU gets its own set of environments
        # rlox VecEnv runs on CPU; only model forward/backward on GPU
```

### Part D: vLLM / TGI Integration (Weeks 5–7)

```rust
// crates/rlox-dist/src/inference/mod.rs

/// Trait for LLM generation backends.
/// Implementations handle connection pooling, batching, backpressure.
#[async_trait]
pub trait GenerationBackend: Send + Sync {
    async fn generate(
        &self,
        prompts: &[String],
        params: GenerationParams,
    ) -> Result<Vec<GenerationResult>, RloxError>;

    async fn log_probs(
        &self,
        input_ids: &[Vec<u32>],
    ) -> Result<Vec<Vec<f32>>, RloxError>;
}

/// vLLM backend via OpenAI-compatible API.
pub struct VllmBackend {
    client: reqwest::Client,
    base_url: String,
    max_concurrent: usize,
}

/// TGI backend via HuggingFace Text Generation Inference API.
pub struct TgiBackend { ... }

/// SGLang backend.
pub struct SglangBackend { ... }
```

Python-side:

```python
# python/rlox/llm/generation.py
class LLMEnvironment:
    """Wraps an inference server as an RL environment."""

    def __init__(self, backend: str = "vllm", url: str = "http://localhost:8000"):
        self._backend = rlox.VllmBackend(url)  # Rust async client

    def generate(self, prompts: list[str], n: int = 1, **kwargs) -> list[Completion]:
        """Generate n completions per prompt. Returns token sequences + log_probs."""
        return self._backend.generate(prompts, n=n, **kwargs)

    def score(self, completions: list[Completion], reward_fn) -> list[float]:
        """Score completions with reward function."""
        ...
```

### Part E: Reward Model Serving (Week 7)

```python
# python/rlox/llm/reward.py
class RewardModelServer:
    """Batch reward model scoring. Supports ensemble and multi-objective."""

    def __init__(self, model: PreTrainedModel, batch_size: int = 32):
        self.model = model
        self.batch_size = batch_size

    def score_batch(self, prompts: list[str], completions: list[str]) -> np.ndarray:
        """Score prompt-completion pairs. Returns rewards array."""
        ...

class EnsembleRewardModel:
    """Average scores from multiple reward models for robustness."""
    def __init__(self, models: list[RewardModelServer], weights: list[float] | None = None): ...

class MultiObjectiveReward:
    """Weighted combination: helpfulness * w1 + safety * w2 + ..."""
    def __init__(self, objectives: dict[str, RewardModelServer], weights: dict[str, float]): ...
```

### Part F: MAPPO / QMIX (Weeks 7–9)

```python
# python/rlox/algorithms/mappo.py
class MAPPO:
    """Multi-Agent PPO with centralized critic, decentralized execution."""

    def __init__(self, env: str, n_agents: int, config: MAPPOConfig):
        # Each agent has own actor; shared critic sees global state
        self.actors = [ActorNetwork(obs_dim, act_dim) for _ in range(n_agents)]
        self.critic = CriticNetwork(global_state_dim)
        # Shared experience buffer — Rust handles agent-indexed storage
        self.buffer = rlox.MultiAgentBuffer(n_agents, capacity, obs_dim, act_dim)

    def train(self, total_timesteps: int):
        ...
```

```python
# python/rlox/algorithms/qmix.py
class QMIX:
    """QMIX: monotonic mixing of per-agent Q-values."""
    ...
```

### Part G: DreamerV3 (Weeks 8–10)

```python
# python/rlox/algorithms/dreamer.py
class DreamerV3:
    """World-model RL: learn dynamics, imagine rollouts, train policy in latent space."""

    def __init__(self, env: str, config: DreamerConfig):
        self.world_model = WorldModel(obs_dim, act_dim, config)
        self.actor = DreamerActor(latent_dim, act_dim)
        self.critic = DreamerCritic(latent_dim)
        self.buffer = rlox.ReplayBuffer(config.buffer_size, obs_dim, act_dim)

    def train(self, total_timesteps: int):
        for step in range(total_timesteps):
            # 1. Real environment interaction → buffer (Rust)
            # 2. World model training (PyTorch)
            # 3. Imagination rollouts (Rust-accelerated latent stepping)
            imagined = self._imagine(horizon=15)  # Rust latent env stepping
            # 4. Actor-critic training on imagined trajectories
```

Rust acceleration for imagination rollouts:

```rust
// crates/rlox-core/src/env/latent.rs

/// Latent space environment for DreamerV3 imagination.
/// Steps through learned dynamics model without real environment interaction.
pub struct LatentEnv {
    // Holds latent state as flat f32 vector
    // Forward pass delegates to Python model, but batched
}

impl BatchSteppable for LatentBatchStepper {
    fn step_batch(&mut self, actions: &[Action]) -> Result<Vec<StepResult>, RloxError> {
        // Batch all latent states → single Python model forward call
        // Avoids N separate Python calls for N imagination envs
        ...
    }
}
```

### Part H: IMPALA / V-trace (Weeks 9–10)

```rust
// crates/rlox-core/src/training/vtrace.rs

/// V-trace off-policy correction (Espeholt et al., 2018).
/// Corrects for stale policy in async actor-learner setup.
pub fn compute_vtrace(
    log_rhos: &[f32],      // log(pi / mu) importance ratios
    rewards: &[f32],
    values: &[f32],
    bootstrap_value: f32,
    gamma: f32,
    rho_bar: f32,          // clipping threshold (default 1.0)
    c_bar: f32,            // trace cutting threshold (default 1.0)
) -> Result<(Vec<f32>, Vec<f32>), RloxError> {
    // Returns (vs: corrected values, pg_advantages: policy gradient advantages)
    ...
}
```

```python
# python/rlox/algorithms/impala.py
class IMPALA:
    """Importance Weighted Actor-Learner Architecture."""

    def __init__(self, env: str, n_actors: int, config: IMPALAConfig):
        # Multiple actors collect asynchronously
        # Single learner applies V-trace correction
        self.pipeline = rlox.Pipeline(
            env_factory=lambda: rlox.VecEnv(env, config.envs_per_actor),
            n_actors=n_actors,
        )
```

### Part I: Transition Provenance (Week 10)

```rust
// crates/rlox-core/src/buffer/provenance.rs

/// Every transition tagged with full lineage.
#[derive(Clone, Debug)]
pub struct TransitionMeta {
    pub env_id: u32,
    pub episode_id: u64,
    pub step_in_episode: u32,
    pub global_step: u64,
    pub policy_version: u64,      // which checkpoint generated this
    pub reward_model_version: u64, // which RM scored this (LLM)
    pub timestamp_ns: u64,
}
```

Critical for LLM post-training debugging: "which policy version generated this completion? Which reward model scored it?"

### Part J: Stable API 1.0 (Week 11)

- Freeze all public Python API signatures
- Write migration guide from 0.x → 1.0
- Semver guarantees: breaking changes only in major versions
- Pre-deprecation: minimum 2 minor versions before removal
- Document all public API with docstrings + examples

## TDD Test Specifications

### Rust Tests

```rust
#[tokio::test]
async fn test_grpc_env_worker_step() {
    let worker = EnvWorker::new("CartPole-v1", 4);
    let server = Server::builder().add_service(EnvServiceServer::new(worker));
    // ... connect client, step, verify results
}

#[test]
fn test_vtrace_matches_reference() {
    // Compare against known-correct V-trace values from IMPALA paper
    ...
}

#[test]
fn test_pipeline_backpressure() {
    // Slow learner → collector blocks → no memory blowup
    let pipeline = Pipeline::new(..., channel_capacity=2);
    // Verify that after 2 batches, collector blocks
}

#[test]
fn test_transition_provenance_roundtrip() {
    let meta = TransitionMeta { env_id: 3, episode_id: 42, ... };
    let bytes = meta.serialize();
    let restored = TransitionMeta::deserialize(&bytes);
    assert_eq!(meta, restored);
}
```

### Python Tests

```python
def test_multi_gpu_ppo_converges():
    """PPO with 2 GPUs converges on CartPole."""
    # Uses torchrun for multi-GPU test
    ...

def test_remote_env_pool():
    """RemoteEnvPool connects to gRPC workers and steps correctly."""
    ...

def test_vllm_backend_generates():
    """VllmBackend generates completions from running vLLM server."""
    # Integration test — requires vLLM running
    ...

def test_mappo_cooperative_task():
    """MAPPO solves simple cooperative navigation."""
    ...

def test_dreamer_visual_control():
    """DreamerV3 learns from pixel observations on DMC Walker."""
    ...

def test_impala_async_training():
    """IMPALA with 4 actors achieves near-linear throughput scaling."""
    ...

def test_api_stability():
    """All public API signatures match 1.0 spec."""
    import inspect
    for name, obj in inspect.getmembers(rlox):
        if not name.startswith("_"):
            assert hasattr(obj, "__doc__"), f"{name} missing docstring"
```

## Success Criteria

- [ ] E2E distributed PPO across 4 machines with near-linear throughput scaling
- [ ] MAPPO solves cooperative navigation benchmark
- [ ] DreamerV3 achieves competitive results on DMC visual control
- [ ] IMPALA with 4 actors shows > 3.5x throughput vs single actor
- [ ] vLLM integration: GRPO training with remote generation works end-to-end
- [ ] Reward model serving: ensemble + multi-objective scoring works
- [ ] Transition provenance: full lineage queryable for any transition
- [ ] API 1.0: all public APIs documented, migration guide published
- [ ] 1,000+ GitHub stars
- [ ] rlox cited in ≥ 3 published RL papers
- [ ] Zero regressions on Phase 6/7/8 benchmarks
