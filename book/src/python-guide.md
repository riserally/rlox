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
# Prerequisites: Python 3.12, Rust toolchain
python3.12 -m venv .venv
source .venv/bin/activate

pip install maturin numpy gymnasium torch

# Build the Rust extension
maturin develop --release

# Verify
python -c "from rlox import CartPole; print('rlox ready')"
```

> Always use `--release` with maturin. Debug builds are 10-50x slower.

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

### Callbacks

```python
from rlox.callbacks import EarlyStoppingCallback, EvalCallback, CheckpointCallback

trainer = PPOTrainer(
    env="CartPole-v1",
    callbacks=[
        EarlyStoppingCallback(patience=20, min_delta=1.0),
        CheckpointCallback(save_freq=10_000, save_path="checkpoints/"),
    ],
)
```

| Callback | Purpose |
|----------|---------|
| `EarlyStoppingCallback` | Stop when reward plateaus for `patience` steps |
| `EvalCallback` | Periodic evaluation on a separate environment |
| `CheckpointCallback` | Save model weights at regular intervals |
| `Callback` | Base class for custom callbacks |

### Logging

```python
from rlox.logging import WandbLogger, TensorBoardLogger

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
```

### V-trace

```python
log_rhos = np.array([0.2, -0.3, 0.8], dtype=np.float32)
rewards  = np.array([1.0, 2.0, 3.0], dtype=np.float32)
values   = np.array([0.5, 1.0, 1.5], dtype=np.float32)

vs, pg_advantages = rlox.compute_vtrace(
    log_rhos=log_rhos,
    rewards=rewards,
    values=values,
    bootstrap_value=2.0,
    gamma=0.99,
    rho_bar=1.0,
    c_bar=1.0,
)
```

### Replay Buffers

```python
# Uniform replay buffer
buf = rlox.ReplayBuffer(capacity=100_000, obs_dim=4, act_dim=1)
buf.push(obs=np.zeros(4, dtype=np.float32), action=0.5, reward=1.0,
         terminated=False, truncated=False)
batch = buf.sample(batch_size=32, seed=0)
# batch keys: "obs", "actions", "rewards", "terminated", "truncated"

# Prioritized replay buffer
pbuf = rlox.PrioritizedReplayBuffer(
    capacity=100_000, obs_dim=4, act_dim=1, alpha=0.6, beta=0.4
)
pbuf.push(obs, action=0.5, reward=1.0, terminated=False, truncated=False,
          priority=1.0)
batch = pbuf.sample(batch_size=32, seed=0)
# Additional keys: "weights" (IS weights), "indices" (for priority update)
pbuf.update_priorities(batch["indices"], new_td_errors)
pbuf.set_beta(0.7)  # anneal toward 1.0
```

### LLM Operations

```python
# GRPO group-relative advantages
rewards = np.random.randn(16).astype(np.float64)
advantages = rlox.compute_group_advantages(rewards)
# advantages = (rewards - mean) / std

# Token-level KL divergence
log_p = np.random.randn(128).astype(np.float64)
log_q = np.random.randn(128).astype(np.float64)
kl = rlox.compute_token_kl(log_p, log_q)

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
1. Steps `n_envs` environments for `n_steps` using `rlox.VecEnv`
2. Evaluates the policy at each step (forward pass only)
3. Computes GAE per environment using `rlox.compute_gae`
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

## Cross-References

- [Rust User Guide](rust-guide.md) -- using `rlox-core` directly from Rust
- [Mathematical Reference](math-reference.md) -- algorithm derivations
- [References](references.md) -- academic papers
- [Getting Started](getting-started.md) -- tutorial walkthrough
