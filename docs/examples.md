# rlox Examples

## Quick Start

### Train PPO on CartPole (2 lines)

```python
from rlox.trainers import PPOTrainer
metrics = PPOTrainer(env="CartPole-v1").train(50_000)
```

### CLI

```bash
python -m rlox train --algo ppo --env CartPole-v1 --timesteps 100000
python -m rlox train --algo sac --env Pendulum-v1 --timesteps 50000
python -m rlox eval --algo ppo --checkpoint model.pt --env CartPole-v1
```

## RL Algorithms

### PPO with Callbacks

```python
from rlox.algorithms.ppo import PPO
from rlox.callbacks import ProgressBarCallback, TimingCallback
from rlox.logging import ConsoleLogger

ppo = PPO(
    env_id="CartPole-v1",
    n_envs=8,
    learning_rate=3e-4,
    n_steps=128,
    n_epochs=4,
    callbacks=[ProgressBarCallback(), TimingCallback()],
    logger=ConsoleLogger(log_interval=1000),
    seed=42,
)
metrics = ppo.train(total_timesteps=100_000)

# Check training phase breakdown
timing_cb = ppo.callbacks.callbacks[1]
print(timing_cb.summary())
# {'env_step': 45.2, 'gae_compute': 5.1, 'gradient_update': 49.7}
```

### SAC on MuJoCo

```python
from rlox.algorithms.sac import SAC

sac = SAC(
    env_id="HalfCheetah-v4",
    buffer_size=1_000_000,
    learning_rate=3e-4,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    learning_starts=10_000,
    hidden=256,
    seed=42,
)
metrics = sac.train(total_timesteps=1_000_000)

# Get actions for evaluation
import numpy as np
obs = np.zeros(17, dtype=np.float32)
action = sac.predict(obs, deterministic=True)  # scaled to env range

# Save/load
sac.save("sac_halfcheetah.pt")
```

### SAC with Custom Environment

```python
import gymnasium as gym
from rlox.algorithms.sac import SAC

# Pass a pre-constructed environment
env = gym.make("Pendulum-v1", g=5.0)  # custom gravity
sac = SAC(env_id=env, learning_starts=1000)
sac.train(total_timesteps=50_000)
```

### Multi-Environment Off-Policy Training

All off-policy algorithms support parallel data collection with `n_envs`:

```python
from rlox.algorithms.sac import SAC
from rlox.algorithms.td3 import TD3
from rlox.algorithms.dqn import DQN

# SAC with 4 parallel environments
sac = SAC(env_id="HalfCheetah-v4", n_envs=4, learning_starts=10_000)
metrics = sac.train(total_timesteps=1_000_000)

# TD3 with 4 parallel environments
td3 = TD3(env_id="Pendulum-v1", n_envs=4, learning_starts=1000)
metrics = td3.train(total_timesteps=50_000)

# DQN with 8 parallel environments
dqn = DQN(env_id="CartPole-v1", n_envs=8, learning_starts=1000)
metrics = dqn.train(total_timesteps=100_000)
```

### Custom Collector with Exploration

```python
import rlox
from rlox.algorithms.sac import SAC
from rlox.off_policy_collector import OffPolicyCollector
from rlox.exploration import GaussianNoise

# Share buffer between collector and algorithm
buf = rlox.ReplayBuffer(1_000_000, obs_dim=3, act_dim=1)
collector = OffPolicyCollector(
    env_id="Pendulum-v1",
    n_envs=4,
    buffer=buf,
    exploration=GaussianNoise(sigma=0.1, clip=0.3),
)

sac = SAC(env_id="Pendulum-v1", buffer=buf, collector=collector)
metrics = sac.train(total_timesteps=50_000)
```

### TD3 on Pendulum

```python
from rlox.algorithms.td3 import TD3

td3 = TD3(
    env_id="Pendulum-v1",
    learning_rate=3e-4,
    policy_delay=2,
    target_noise=0.2,
    noise_clip=0.5,
    exploration_noise=0.1,
)
metrics = td3.train(total_timesteps=50_000)
action = td3.predict(obs)
```

### DQN with Prioritized Experience Replay

```python
from rlox.algorithms.dqn import DQN

dqn = DQN(
    env_id="CartPole-v1",
    double_dqn=True,
    dueling=True,
    prioritized=True,
    n_step=3,
    buffer_size=100_000,
    learning_starts=1000,
)
metrics = dqn.train(total_timesteps=100_000)
action = dqn.predict(obs)  # returns int
```

### A2C

```python
from rlox.algorithms.a2c import A2C

a2c = A2C(env_id="CartPole-v1", n_envs=8, learning_rate=7e-4)
metrics = a2c.train(total_timesteps=50_000)
```

### A2CTrainer (High-Level)

```python
from rlox.trainers import A2CTrainer

metrics = A2CTrainer(env="CartPole-v1").train(50_000)
```

### TD3Trainer (High-Level)

```python
from rlox.trainers import TD3Trainer

metrics = TD3Trainer(env="Pendulum-v1").train(50_000)
```

### MAPPOTrainer (Multi-Agent)

Multi-agent PPO with centralised critic. Works with PettingZoo environments.

```python
from rlox.trainers import MAPPOTrainer

trainer = MAPPOTrainer(env="spread_v3", n_agents=3, seed=42)
metrics = trainer.train(total_timesteps=500_000)
```

### DreamerV3Trainer (World Model)

Learns a latent dynamics model and trains the policy inside the learned world model. Particularly effective for image-based and sparse-reward environments.

```python
from rlox.trainers import DreamerV3Trainer

trainer = DreamerV3Trainer(env="Pendulum-v1", seed=42)
metrics = trainer.train(total_timesteps=200_000)
```

### IMPALATrainer (Distributed Actors)

Distributed actor-learner architecture with V-trace off-policy correction. Actors collect experience in parallel (optionally across machines via gRPC) while a central learner trains the policy.

```python
from rlox.trainers import IMPALATrainer

trainer = IMPALATrainer(env="CartPole-v1", n_actors=8, seed=42)
metrics = trainer.train(total_timesteps=1_000_000)
```

### Config-Driven Training

Define experiments in YAML and launch without writing Python scripts:

```yaml
# experiment.yaml
algorithm: ppo
env: CartPole-v1
total_timesteps: 100_000
seed: 42
config:
  n_envs: 16
  learning_rate: 3e-4
```

```python
from rlox.runner import train_from_config

metrics = train_from_config("experiment.yaml")
```

### VecNormalize

Running observation and reward normalisation for vectorised environments:

```python
from rlox.trainers import PPOTrainer
from rlox.wrappers import VecNormalize

trainer = PPOTrainer(
    env="HalfCheetah-v4",
    wrappers=[VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10.0)],
    seed=42,
)
metrics = trainer.train(total_timesteps=1_000_000)
```

### Diagnostics Dashboard & HTML Report

```python
from rlox.dashboard import MetricsCollector, HTMLReport, TerminalDashboard
from rlox.trainers import PPOTrainer

# Collect metrics and generate an HTML report after training
collector = MetricsCollector()
trainer = PPOTrainer(env="CartPole-v1", callbacks=[collector], seed=42)
trainer.train(total_timesteps=50_000)

report = HTMLReport(collector)
report.save("training_report.html")

# Or use a live terminal dashboard during training
trainer = PPOTrainer(
    env="CartPole-v1",
    callbacks=[TerminalDashboard()],
    seed=42,
)
trainer.train(total_timesteps=50_000)
```

## LLM Post-Training

See [llm-post-training.md](llm-post-training.md) for the full guide.

### DPO Quick Example

```python
from rlox.algorithms.dpo import DPO

dpo = DPO(model=model, ref_model=ref_model, beta=0.1)
for prompt, chosen, rejected in dataset:
    metrics = dpo.train_step(prompt, chosen, rejected)
```

### GRPO Quick Example

```python
from rlox.algorithms.grpo import GRPO

grpo = GRPO(
    model=model, ref_model=ref_model,
    reward_fn=my_reward_fn, group_size=4,
)
for prompts in dataloader:
    metrics = grpo.train_step(prompts)
```

## Rust Primitives (Low-Level)

### Fast GAE Computation

```python
import rlox
import numpy as np

# Single environment
rewards = np.random.randn(2048)
values = np.random.randn(2048)
dones = np.zeros(2048)
adv, ret = rlox.compute_gae(rewards, values, dones, last_value=0.0, gamma=0.99, lam=0.95)

# Batched (8 envs, Rayon-parallel)
rewards_flat = np.random.randn(8 * 2048)  # env-major layout
values_flat = np.random.randn(8 * 2048)
dones_flat = np.zeros(8 * 2048)
last_vals = np.random.randn(8)
adv, ret = rlox.compute_gae_batched(rewards_flat, values_flat, dones_flat, last_vals, 2048, 0.99, 0.95)

# f32 variant (1.5x faster at 64 envs)
adv, ret = rlox.compute_gae_batched_f32(
    rewards_flat.astype(np.float32), values_flat.astype(np.float32),
    dones_flat.astype(np.float32), last_vals.astype(np.float32),
    2048, 0.99, 0.95,
)
```

### Replay Buffers

```python
import rlox
import numpy as np

# Standard ring buffer
buf = rlox.ReplayBuffer(capacity=100_000, obs_dim=4, act_dim=1)

obs = np.zeros(4, dtype=np.float32)
action = np.zeros(1, dtype=np.float32)
buf.push(obs, action, reward=1.0, terminated=False, truncated=False, next_obs=obs)

batch = buf.sample(batch_size=256, seed=42)
# batch["obs"], batch["actions"], batch["rewards"], batch["next_obs"], etc.

# Prioritized replay
pbuf = rlox.PrioritizedReplayBuffer(100_000, obs_dim=4, act_dim=1, alpha=0.6, beta=0.4)
pbuf.push(obs, action, 1.0, False, False, obs, priority=1.0)
batch = pbuf.sample(256, seed=42)
# batch["weights"], batch["indices"] for importance sampling

# Memory-mapped buffer (for Atari-scale observations)
mmap_buf = rlox.MmapReplayBuffer(
    hot_capacity=10_000,      # in-memory
    total_capacity=1_000_000, # total (hot + disk)
    obs_dim=84*84*4,
    act_dim=1,
    cold_path="/tmp/replay_cold.bin",
)
```

### VecEnv Stepping

```python
import rlox

# Native Rust CartPole (fastest)
env = rlox.VecEnv(n=1024, seed=42)
result = env.step_all(actions)  # actions: numpy uint32 array
# result["obs"]: (1024, 4) float32 — contiguous flat buffer

# Gymnasium wrapper (any env)
from rlox import GymVecEnv
env = GymVecEnv("HalfCheetah-v4", n_envs=8)
result = env.step_all(actions)
```

### KL Divergence

```python
import rlox
import numpy as np

# f32 batched (fastest for LLM workloads)
log_p = np.random.randn(32 * 2048).astype(np.float32)
log_q = np.random.randn(32 * 2048).astype(np.float32)
kl = rlox.compute_batch_token_kl_schulman_f32(log_p, log_q, seq_len=2048)
# kl: (32,) array of per-sequence KL values
```

## Monitoring & Profiling

### Console Logger

```python
from rlox.logging import ConsoleLogger
logger = ConsoleLogger(log_interval=500)
# Prints: step=500 | SPS=1234 | reward=45.20
```

### W&B Integration

```python
from rlox.logging import WandbLogger
logger = WandbLogger(project="my-rl-project", config={"algo": "ppo"})
```

### TensorBoard

```python
from rlox.logging import TensorBoardLogger
logger = TensorBoardLogger(log_dir="runs/ppo_cartpole")
```

### Profiling with TimingCallback

```python
from rlox.callbacks import TimingCallback

timing = TimingCallback()
ppo = PPO(env_id="CartPole-v1", callbacks=[timing])
ppo.train(50_000)
print(timing.summary())
# {'env_step': 42.1, 'gae_compute': 8.3, 'gradient_update': 49.6}
```

## torch.compile

```python
from rlox.compile import compile_policy
from rlox.algorithms.sac import SAC

sac = SAC(env_id="Pendulum-v1")
compile_policy(sac)  # compiles actor, critic1, critic2
sac.train(50_000)
```
