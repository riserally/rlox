# rlox Python User Guide

rlox provides three levels of API for reinforcement learning in Python. Each level gives more control at the cost of more code.

| Level | What you write | What rlox handles |
|-------|----------------|-------------------|
| **High-level** (Trainer) | `Trainer("ppo", env="CartPole-v1").train(50_000)` | Everything |
| **Mid-level** (Algorithm) | Training loop, hyperparams | Network creation, collection, loss |
| **Low-level** (Primitives) | Full loop, custom networks | Fast env stepping, GAE, buffers |

---

## Installation

```bash
# Prerequisites: Python 3.10+, Rust toolchain
python3 -m venv .venv
source .venv/bin/activate

pip install maturin numpy gymnasium torch

# Build the Rust extension
maturin develop --release

# Verify
python -c "from rlox import CartPole; print('rlox ready')"
```

> Always use `--release` with maturin. Debug builds are 10-50x slower.

### CLI Quick Start

```bash
# Train from the command line (no Python script needed)
python -m rlox train --algo ppo --env CartPole-v1 --timesteps 100000
python -m rlox train --algo sac --env Pendulum-v1 --timesteps 50000 --save model.pt
python -m rlox eval --algo ppo --checkpoint model.pt --env CartPole-v1 --episodes 10
```

---

## High-Level: Trainer API

Three lines to a trained agent:

```python
from rlox import Trainer

trainer = Trainer("ppo", env="CartPole-v1", seed=42)
metrics = trainer.train(total_timesteps=50_000)
print(f"Mean reward: {metrics['mean_reward']:.1f}")
```

### Available Trainers

```python
from rlox import Trainer
```

**Trainer("ppo", ...)** -- On-policy, discrete or continuous actions.

```python
trainer = PPOTrainer(
    env="CartPole-v1",
    config={"n_envs": 16, "n_steps": 256, "learning_rate": 3e-4},
    seed=42,
)
```

**Trainer("sac", ...)** -- Off-policy, continuous actions (e.g. Pendulum, MuJoCo).

```python
trainer = SACTrainer(
    env="Pendulum-v1",
    config={"learning_rate": 3e-4, "buffer_size": 100_000},
    seed=42,
)
```

**Trainer("dqn", ...)** -- Off-policy, discrete actions with Rainbow extensions.

```python
trainer = DQNTrainer(
    env="CartPole-v1",
    config={"double_dqn": True, "dueling": True},
    seed=42,
)
```

**A2CTrainer** -- On-policy, single gradient step per rollout.

```python
from rlox import Trainer

trainer = A2CTrainer(
    env="CartPole-v1",
    config={"n_envs": 8, "learning_rate": 7e-4},
    seed=42,
)
```

**TD3Trainer** -- Off-policy, continuous actions with delayed policy updates.

```python
from rlox import Trainer

trainer = TD3Trainer(
    env="Pendulum-v1",
    config={"policy_delay": 2, "target_noise": 0.2},
    seed=42,
)
```

**Trainer("mappo", ...)** -- Multi-agent PPO with centralised critic and per-agent actors.

```python
from rlox import Trainer

trainer = MAPPOTrainer(
    env="spread_v3",   # PettingZoo environment
    n_agents=3,
    seed=42,
)
metrics = trainer.train(total_timesteps=500_000)
```

**Trainer("dreamer", ...)** -- World-model-based training (learns a latent dynamics model, trains the policy inside the learned world model).

```python
from rlox import Trainer

trainer = DreamerV3Trainer(
    env="Pendulum-v1",
    seed=42,
)
metrics = trainer.train(total_timesteps=200_000)
```

**Trainer("impala", ...)** -- Distributed actor-learner architecture with V-trace off-policy correction. Scales to many actors across machines via gRPC.

```python
from rlox import Trainer

trainer = IMPALATrainer(
    env="CartPole-v1",
    n_actors=8,
    seed=42,
)
metrics = trainer.train(total_timesteps=1_000_000)
```

### Callbacks

```python
from rlox.callbacks import (
    EarlyStoppingCallback,
    ProgressBarCallback,
    TimingCallback,
)

trainer = PPOTrainer(
    env="CartPole-v1",
    callbacks=[
        EarlyStoppingCallback(patience=20, min_delta=1.0),
        ProgressBarCallback(),   # tqdm progress bar
        TimingCallback(),         # phase-level profiling
    ],
)
metrics = trainer.train(total_timesteps=100_000)

# After training, see where time was spent
timing = trainer.callbacks[2]  # TimingCallback
print(timing.summary())
# {'env_step': 42.1, 'gae_compute': 8.3, 'gradient_update': 49.6}
```

| Callback | Purpose |
|----------|---------|
| `EarlyStoppingCallback` | Stop when reward plateaus for `patience` steps |
| `ProgressBarCallback` | tqdm progress bar with live reward display |
| `TimingCallback` | Wall-clock profiling of each training phase |
| `EvalCallback` | Periodic evaluation on a separate environment |
| `CheckpointCallback` | Save model weights at regular intervals |
| `Callback` | Base class for custom callbacks |

### Logging

```python
from rlox.logging import ConsoleLogger, WandbLogger, TensorBoardLogger

# Simple console output (no dependencies)
logger = ConsoleLogger(log_interval=500)
# Prints: step=500 | SPS=1234 | reward=45.20

# Weights & Biases
logger = WandbLogger(project="rlox-experiments", name="ppo-cartpole")

# TensorBoard
logger = TensorBoardLogger(log_dir="runs/ppo-cartpole")

trainer = Trainer("ppo", env="CartPole-v1", logger=logger)
trainer.train(total_timesteps=100_000)
```

Extend `LoggerCallback` for custom logging backends (CSV, MLflow, etc.):

```python
from rlox.logging import LoggerCallback

class CSVLogger(LoggerCallback):
    def on_train_step(self, step, metrics):
        # Write metrics to CSV
        ...
```

---

## Mid-Level: Algorithm API

The algorithm classes give you control over the training loop while handling network creation and loss computation:

### On-Policy (PPO, A2C)

```python
from rlox.algorithms import PPO, A2C

# PPO with custom hyperparameters
ppo = PPO(
    env_id="CartPole-v1",
    n_envs=8,
    seed=42,
    n_steps=128,
    n_epochs=4,
    clip_eps=0.2,
    learning_rate=2.5e-4,
)
metrics = ppo.train(total_timesteps=50_000)
```

```python
# A2C: single gradient step per rollout, shorter n_steps
a2c = A2C(
    env_id="CartPole-v1",
    n_envs=8,
    n_steps=5,
    learning_rate=7e-4,
    gae_lambda=1.0,  # full Monte Carlo returns
)
metrics = a2c.train(total_timesteps=50_000)
```

Both PPO and A2C use:
- `rlox.VecEnv` for parallel environment stepping
- `rlox.compute_gae` for advantage computation
- `RolloutCollector` for the collect-then-compute pattern

### Off-Policy (SAC, TD3, DQN)

```python
from rlox.algorithms import SAC, TD3, DQN

# SAC with automatic entropy tuning
sac = SAC(
    env_id="Pendulum-v1",
    buffer_size=1_000_000,
    learning_rate=3e-4,
    tau=0.005,
    gamma=0.99,
    auto_entropy=True,
)
metrics = sac.train(total_timesteps=20_000)
```

```python
# TD3 with delayed policy updates
td3 = TD3(
    env_id="Pendulum-v1",
    policy_delay=2,
    target_noise=0.2,
    noise_clip=0.5,
    exploration_noise=0.1,
)
metrics = td3.train(total_timesteps=20_000)
```

```python
# DQN with Rainbow extensions
dqn = DQN(
    env_id="CartPole-v1",
    double_dqn=True,
    dueling=True,
    n_step=3,
    prioritized=True,
    alpha=0.6,
    beta_start=0.4,
)
metrics = dqn.train(total_timesteps=50_000)
```

Off-policy algorithms use `rlox.ReplayBuffer` (or `PrioritizedReplayBuffer`) for storage, with Gymnasium for environment stepping.

### Multi-Environment Collection

All off-policy algorithms support parallel data collection via `OffPolicyCollector`. Use `n_envs` for automatic setup, or inject a custom collector:

```python
# Automatic: pass n_envs to any off-policy algorithm
sac = SAC(env_id="Pendulum-v1", n_envs=4, learning_starts=5000)
sac.train(total_timesteps=100_000)  # 4x collection throughput

td3 = TD3(env_id="Pendulum-v1", n_envs=4, learning_starts=5000)
dqn = DQN(env_id="CartPole-v1", n_envs=8, learning_starts=1000)
```

```python
# Manual: create and inject your own collector
from rlox.off_policy_collector import OffPolicyCollector
from rlox.exploration import GaussianNoise

buf = rlox.ReplayBuffer(1_000_000, obs_dim=3, act_dim=1)
collector = OffPolicyCollector(
    env_id="Pendulum-v1",
    n_envs=4,
    buffer=buf,
    exploration=GaussianNoise(sigma=0.1),
)
sac = SAC(env_id="Pendulum-v1", buffer=buf, collector=collector)
sac.train(total_timesteps=100_000)
```

The collector uses `GymVecEnv` internally and batch-inserts transitions via `push_batch` for efficiency. When `n_envs=1` (default), algorithms use the original single-env loop with zero overhead.

### Offline RL (TD3+BC, IQL, CQL, BC)

Train from static datasets without environment interaction. All offline algorithms
use `OfflineDatasetBuffer` (Rust-accelerated) and extend `OfflineAlgorithm` base class.

```python
import rlox
from rlox.algorithms.td3_bc import TD3BC

# Load dataset (D4RL, Minari, or custom numpy arrays)
buf = rlox.OfflineDatasetBuffer(
    obs.ravel(), next_obs.ravel(), actions.ravel(),
    rewards, terminated, truncated, normalize=True,
)
print(buf.stats())  # {'n_transitions': ..., 'n_episodes': ..., 'mean_return': ...}

# TD3+BC: TD3 with behavioral cloning regularization
algo = TD3BC(dataset=buf, obs_dim=17, act_dim=6, alpha=2.5)
algo.train(n_gradient_steps=100_000)
```

```python
# IQL: Implicit Q-Learning (avoids OOD action queries)
from rlox.algorithms.iql import IQL
algo = IQL(dataset=buf, obs_dim=17, act_dim=6, expectile=0.7)
```

```python
# CQL: Conservative Q-Learning (penalizes OOD Q-values)
from rlox.algorithms.cql import CQL
algo = CQL(dataset=buf, obs_dim=17, act_dim=6, cql_alpha=5.0)
```

```python
# BC: Behavioral Cloning (supervised learning on demonstrations)
from rlox.algorithms.bc import BC
algo = BC(dataset=buf, obs_dim=17, act_dim=6)
```

### Candle Hybrid Collection

`HybridPPO` runs policy inference entirely in Rust using Candle — zero Python
dispatch overhead during data collection. Collection takes only ~27% of wall
time vs ~50-60% with standard PyTorch inference.

```python
from rlox.algorithms.hybrid_ppo import HybridPPO

ppo = HybridPPO(env_id="CartPole-v1", n_envs=16, hidden=64)
metrics = ppo.train(total_timesteps=100_000)
print(ppo.timing_summary())
# {'collection_pct': 27.0, 'training_pct': 73.0}
```

### Inference with `predict()`

All algorithms provide a `predict()` method for evaluation:

```python
# SAC/TD3: returns numpy action array (scaled to env range)
action = sac.predict(obs, deterministic=True)

# DQN: returns int action
action = dqn.predict(obs)
```

### Custom Environments

Pass a pre-constructed Gymnasium env instead of an ID string:

```python
import gymnasium as gym

env = gym.make("Pendulum-v1", g=5.0)  # custom gravity
sac = SAC(env_id=env, learning_starts=1000)
sac.train(total_timesteps=50_000)
```

### LLM Post-Training (GRPO, DPO)

```python
from rlox.algorithms import GRPO, DPO

# GRPO: group-relative policy optimization
grpo = GRPO(
    model=my_lm,
    ref_model=ref_lm,
    reward_fn=reward_function,
    group_size=4,
    kl_coef=0.1,
    max_new_tokens=64,
)
metrics = grpo.train_step(prompt_batch)
```

```python
# DPO: direct preference optimization
dpo = DPO(
    model=my_lm,
    ref_model=ref_lm,
    beta=0.1,
)
loss, metrics = dpo.compute_loss(prompt, chosen, rejected)
```

---

## Low-Level: Rust Primitives

Import Rust primitives directly from `rlox`:

```python
import rlox
```

### Environment Stepping

```python
# Single CartPole
env = rlox.CartPole(seed=42)
obs = env.reset()                # shape: (4,)
result = env.step(1)             # push right
obs, reward = result["obs"], result["reward"]

# Vectorized CartPole (Rayon parallel)
vec = rlox.VecEnv(n=64, seed=0)
obs = vec.reset_all()            # shape: (64, 4)
result = vec.step_all([1] * 64)
next_obs = result["obs"]         # shape: (64, 4)
rewards = result["rewards"]      # shape: (64,)
terminated = result["terminated"] # shape: (64,), bool
truncated = result["truncated"]   # shape: (64,), bool

# Gymnasium wrapper
import gymnasium
gym_env = gymnasium.make("Acrobot-v1")
wrapped = rlox.GymEnv(gym_env)
```

### GAE Computation

```python
import numpy as np
import rlox

rewards = np.array([1.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float64)
values  = np.array([0.5, 0.6, 0.7, 0.3, 0.8], dtype=np.float64)
dones   = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)

advantages, returns = rlox.compute_gae(
    rewards=rewards,
    values=values,
    dones=dones,
    last_value=0.9,
    gamma=0.99,
    lam=0.95,
)
# advantages.shape == (5,), returns.shape == (5,)
# Invariant: returns == advantages + values

# Batched GAE: all environments in one call (Rayon-parallel)
rewards_flat = np.random.randn(8 * 2048)  # env-major: [env0_step0, env0_step1, ...]
values_flat = np.random.randn(8 * 2048)
dones_flat = np.zeros(8 * 2048)
last_vals = np.random.randn(8)

adv, ret = rlox.compute_gae_batched(
    rewards_flat, values_flat, dones_flat, last_vals,
    n_steps=2048, gamma=0.99, lam=0.95,
)

# f32 variant (1.5x faster at 64+ envs, avoids f64 conversion)
adv_f32, ret_f32 = rlox.compute_gae_batched_f32(
    rewards_flat.astype(np.float32), values_flat.astype(np.float32),
    dones_flat.astype(np.float32), last_vals.astype(np.float32),
    n_steps=2048, gamma=0.99, lam=0.95,
)
```

### V-trace

```python
log_rhos = np.array([0.2, -0.3, 0.8], dtype=np.float32)
rewards  = np.array([1.0, 2.0, 3.0], dtype=np.float32)
values   = np.array([0.5, 1.0, 1.5], dtype=np.float32)
dones    = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # episode boundaries

vs, pg_advantages = rlox.compute_vtrace(
    log_rhos=log_rhos,
    rewards=rewards,
    values=values,
    dones=dones,            # zeroes discount at episode boundaries
    bootstrap_value=2.0,
    gamma=0.99,
    rho_bar=1.0,
    c_bar=1.0,
)
```

### Replay Buffers

```python
# Uniform replay buffer (zero-copy push via Rust push_slices)
buf = rlox.ReplayBuffer(capacity=100_000, obs_dim=4, act_dim=1)
obs = np.zeros(4, dtype=np.float32)
next_obs = np.ones(4, dtype=np.float32)
buf.push(obs, action=np.array([0.5], dtype=np.float32), reward=1.0,
         terminated=False, truncated=False, next_obs=next_obs)
batch = buf.sample(batch_size=32, seed=0)
# batch keys: "obs", "next_obs", "actions", "rewards", "terminated", "truncated"

# Prioritized replay buffer (O(1) min via augmented min-tree)
pbuf = rlox.PrioritizedReplayBuffer(
    capacity=100_000, obs_dim=4, act_dim=1, alpha=0.6, beta=0.4
)
pbuf.push(obs, action=np.array([0.5], dtype=np.float32), reward=1.0,
          terminated=False, truncated=False, next_obs=next_obs, priority=1.0)
batch = pbuf.sample(batch_size=32, seed=0)
# Additional keys: "weights" (IS weights), "indices" (for priority update)
pbuf.update_priorities(batch["indices"], new_td_errors)
pbuf.set_beta(0.7)  # anneal toward 1.0

# Memory-mapped buffer (for Atari-scale observations)
mmap_buf = rlox.MmapReplayBuffer(
    hot_capacity=10_000,       # kept in memory
    total_capacity=1_000_000,  # overflow spills to disk
    obs_dim=84*84*4,
    act_dim=1,
    cold_path="/tmp/replay_cold.bin",
)
# Same push/sample API as ReplayBuffer
```

### LLM Operations

```python
# GRPO group-relative advantages (single group)
rewards = np.random.randn(16).astype(np.float64)
advantages = rlox.compute_group_advantages(rewards)

# Batched GRPO (Rayon-parallel for large batches)
all_rewards = np.random.randn(1024 * 8).astype(np.float64)  # 1024 prompts x 8 completions
all_advantages = rlox.compute_batch_group_advantages(all_rewards, group_size=8)

# Token-level KL divergence (single sequence)
log_p = np.random.randn(128).astype(np.float64)
log_q = np.random.randn(128).astype(np.float64)
kl = rlox.compute_token_kl(log_p, log_q)

# Batched KL (single Rust call for all sequences, Rayon-parallel)
log_p_flat = np.random.randn(32 * 2048).astype(np.float32)
log_q_flat = np.random.randn(32 * 2048).astype(np.float32)
kl_per_seq = rlox.compute_batch_token_kl_schulman_f32(log_p_flat, log_q_flat, seq_len=2048)
# kl_per_seq: (32,) array — 2-9x faster than TRL

# DPO preference pair
pair = rlox.DPOPair(
    prompt_tokens=np.array([1, 2, 3], dtype=np.uint32),
    chosen_tokens=np.array([4, 5], dtype=np.uint32),
    rejected_tokens=np.array([6, 7, 8], dtype=np.uint32),
)

# Variable-length sequence storage
store = rlox.VarLenStore()
store.push(np.array([1, 2, 3], dtype=np.uint32))
store.push(np.array([4, 5], dtype=np.uint32))
seq = store.get(0)  # array([1, 2, 3])

# Sequence packing for transformers
packed = rlox.pack_sequences(
    sequences=[np.array([1,2,3], dtype=np.uint32),
               np.array([4,5], dtype=np.uint32)],
    max_length=8,
)
```

### RunningStats

```python
stats = rlox.RunningStats()
stats.batch_update(np.array([1.0, 2.0, 3.0]))
print(stats.mean())   # 2.0
print(stats.std())     # ~0.816
print(stats.count())   # 3
```

---

## Configuration

Typed configuration with validation, merging, and serialisation:

```python
from rlox.config import PPOConfig, SACConfig, DQNConfig

# Create with defaults (CleanRL-matching)
cfg = PPOConfig()

# Create from dict (ignores unknown keys)
cfg = PPOConfig.from_dict({"n_envs": 16, "clip_eps": 0.1, "unknown_key": 42})

# Merge overrides into existing config
cfg2 = cfg.merge({"learning_rate": 1e-3})

# Serialise for logging
d = cfg.to_dict()

# Validation happens in __post_init__
try:
    PPOConfig(learning_rate=-1)  # raises ValueError
except ValueError as e:
    print(e)
```

### PPOConfig Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_envs` | 8 | Parallel environments |
| `n_steps` | 128 | Rollout length per env |
| `n_epochs` | 4 | SGD passes per rollout |
| `batch_size` | 256 | Minibatch size |
| `learning_rate` | 2.5e-4 | Adam LR |
| `clip_eps` | 0.2 | PPO clip range |
| `vf_coef` | 0.5 | Value loss coefficient |
| `ent_coef` | 0.01 | Entropy bonus coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `normalize_advantages` | True | Per-minibatch normalisation |
| `clip_vloss` | True | Clipped value loss |
| `anneal_lr` | True | Linear LR annealing |

---

## Config-Driven Training

Define your entire experiment in a YAML file and launch with `train_from_config`:

```yaml
# experiment.yaml
algorithm: ppo
env: CartPole-v1
total_timesteps: 100_000
seed: 42
config:
  n_envs: 16
  learning_rate: 3e-4
  n_steps: 128
  n_epochs: 4
logger:
  type: wandb
  project: rlox-experiments
```

```python
from rlox.runner import train_from_config
from rlox.config import TrainingConfig

# From a YAML file
metrics = train_from_config("experiment.yaml")

# Or build programmatically
cfg = TrainingConfig(
    algorithm="ppo",
    env="CartPole-v1",
    total_timesteps=100_000,
    seed=42,
    config={"n_envs": 16, "learning_rate": 3e-4},
)
metrics = train_from_config(cfg)
```

---

## VecNormalize

`VecNormalize` wraps a vectorised environment to apply running normalisation to observations and rewards. It uses `RunningStatsVec` (Rust) for efficient per-dimension statistics.

```python
from rlox import Trainer
from rlox.wrappers import VecNormalize

trainer = PPOTrainer(
    env="CartPole-v1",
    wrappers=[VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10.0)],
    seed=42,
)
metrics = trainer.train(total_timesteps=100_000)
```

`VecNormalize` is especially useful for environments with large or variable observation scales (MuJoCo, robotics).

---

## Diagnostics Dashboard

`MetricsCollector` aggregates training metrics in memory and feeds them to visualisation backends.

```python
from rlox.dashboard import MetricsCollector, HTMLReport, TerminalDashboard

# Collect metrics during training
collector = MetricsCollector()

from rlox import Trainer
trainer = PPOTrainer(
    env="CartPole-v1",
    callbacks=[collector],
    seed=42,
)
trainer.train(total_timesteps=50_000)

# Generate a static HTML report
report = HTMLReport(collector)
report.save("training_report.html")

# Or use the live terminal dashboard (Rich-based)
# Pass TerminalDashboard as a callback for real-time display:
from rlox import Trainer
trainer = PPOTrainer(
    env="CartPole-v1",
    callbacks=[TerminalDashboard()],
    seed=42,
)
trainer.train(total_timesteps=50_000)
```

---

## Custom Policies

### Discrete Actions (PPO/A2C)

```python
from rlox.policies import DiscretePolicy

policy = DiscretePolicy(obs_dim=4, n_actions=2, hidden=64)

# Required interface (called by PPOLoss / RolloutCollector):
actions, log_probs = policy.get_action_and_logprob(obs_tensor)
values = policy.get_value(obs_tensor)
log_probs, entropy = policy.get_logprob_and_entropy(obs_tensor, actions_tensor)
```

Architecture: separate actor and critic MLPs with orthogonal initialisation, Tanh activations, and reduced gain (0.01) on the policy head.

### Continuous Actions (SAC/TD3)

```python
from rlox.networks import SquashedGaussianPolicy, DeterministicPolicy, QNetwork

# SAC: squashed Gaussian policy
actor = SquashedGaussianPolicy(obs_dim=3, act_dim=1, hidden=256)
action, log_prob = actor.sample(obs_tensor)       # reparameterised
det_action = actor.deterministic(obs_tensor)       # mean through tanh

# TD3: deterministic policy
actor = DeterministicPolicy(obs_dim=3, act_dim=1, hidden=256, max_action=2.0)
action = actor(obs_tensor)  # scaled by max_action

# Shared Q-network for SAC/TD3
critic = QNetwork(obs_dim=3, act_dim=1, hidden=256)
q_value = critic(obs_tensor, action_tensor)  # scalar
```

### Discrete Q-Networks (DQN)

```python
from rlox.networks import SimpleQNetwork, DuelingQNetwork

# Standard DQN
q_net = SimpleQNetwork(obs_dim=4, act_dim=2, hidden=256)
q_values = q_net(obs_tensor)  # (B, n_actions)

# Dueling architecture: V(s) + A(s,a) - mean(A)
q_net = DuelingQNetwork(obs_dim=4, act_dim=2, hidden=256)
q_values = q_net(obs_tensor)  # same interface
```

---

## RolloutBatch and RolloutCollector

The collector orchestrates on-policy data collection:

```python
from rlox.collectors import RolloutCollector
from rlox.policies import DiscretePolicy

collector = RolloutCollector(
    env_id="CartPole-v1",
    n_envs=8,
    seed=0,
    gamma=0.99,
    gae_lambda=0.95,
    normalize_rewards=False,
    normalize_obs=False,
)

policy = DiscretePolicy(obs_dim=4, n_actions=2)
batch = collector.collect(policy, n_steps=128)

# batch is a RolloutBatch with shape (n_envs * n_steps, ...)
batch.obs.shape        # (1024, 4)
batch.actions.shape    # (1024,)
batch.advantages.shape # (1024,)
batch.returns.shape    # (1024,)
```

The collection pipeline:
1. Steps `n_envs` environments for `n_steps` using `rlox.VecEnv` or `GymVecEnv`
2. Evaluates the policy at each step (forward pass only)
3. Computes GAE using `rlox.compute_gae_batched` (single Rust call, Rayon-parallel)
4. Flattens and returns a `RolloutBatch`

### Minibatch Iteration

```python
for epoch in range(4):
    for mb in batch.sample_minibatches(batch_size=256, shuffle=True):
        # mb is a RolloutBatch with shape (256, ...)
        loss = compute_loss(mb)
        loss.backward()
```

---

## PPOLoss

Stateless loss calculator implementing the clipped surrogate objective:

```python
from rlox.losses import PPOLoss

loss_fn = PPOLoss(
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    clip_vloss=True,
)

total_loss, metrics = loss_fn(
    policy, obs, actions, old_log_probs,
    advantages, returns, old_values,
)
# metrics: policy_loss, value_loss, entropy, approx_kl, clip_fraction

total_loss.backward()
```

---

## Statistical Evaluation

Following Agarwal et al. (2021) for reliable deep RL evaluation:

```python
from rlox.evaluation import interquartile_mean, performance_profiles, stratified_bootstrap_ci

# IQM: robust central tendency (discards top/bottom 25%)
scores = [450, 480, 500, 200, 490]
iqm = interquartile_mean(scores)

# Bootstrap confidence interval
lower, upper = stratified_bootstrap_ci(scores, n_bootstrap=10_000, ci=0.95)

# Performance profiles: fraction of runs above threshold
profiles = performance_profiles(
    {"rlox": [450, 480, 500], "baseline": [300, 350, 400]},
    thresholds=[100, 200, 300, 400, 500],
)
```

---

## Using rlox with Non-CartPole Environments

`rlox.VecEnv` currently only supports CartPole natively. For other environments, use Gymnasium for stepping and rlox for the compute-heavy parts:

```python
import gymnasium
import numpy as np
import rlox

# Gymnasium for stepping
vec_env = gymnasium.vector.SyncVectorEnv(
    [lambda: gymnasium.make("Acrobot-v1") for _ in range(8)]
)
obs, _ = vec_env.reset(seed=42)

# rlox for storage
buffer = rlox.ExperienceTable(obs_dim=6, act_dim=1)

# rlox for GAE
rewards = np.ones(128, dtype=np.float64)
values = np.ones(128, dtype=np.float64) * 0.5
dones = np.zeros(128, dtype=np.float64)
advantages, returns = rlox.compute_gae(
    rewards, values, dones,
    last_value=0.5, gamma=0.99, lam=0.95,
)
```

See `benchmarks/convergence/rlox_runner.py` for a complete example of this pattern.

---

## Running Tests

```bash
# Rust tests
cargo test --package rlox-core

# Python tests
.venv/bin/python -m pytest tests/python/ -v

# Both
./scripts/test.sh
```

---

## torch.compile

Accelerate neural network inference with `torch.compile`:

```python
from rlox.compile import compile_policy

sac = SAC(env_id="Pendulum-v1")
compile_policy(sac)  # compiles actor, critic1, critic2
sac.train(total_timesteps=50_000)

# For on-policy policies (PPO/A2C), individual methods are compiled:
# get_action_and_logprob, get_value, get_logprob_and_entropy
ppo = PPO(env_id="CartPole-v1")
compile_policy(ppo)
```

---

## Cross-References

- [Examples](examples.md) -- comprehensive code examples
- [LLM Post-Training Guide](llm-post-training.md) -- DPO, GRPO, OnlineDPO
- [Rust User Guide](rust-guide.md) -- using `rlox-core` directly from Rust
- [Mathematical Reference](math-reference.md) -- algorithm derivations
- [References](references.md) -- academic papers
- [Getting Started](getting-started.md) -- tutorial walkthrough
