# rlox Examples

## Quick Start

### Train PPO on CartPole (2 lines)

```python
from rlox import Trainer
metrics = Trainer("ppo", env="CartPole-v1").train(50_000)
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
from rlox import Trainer

metrics = Trainer("a2c", env="CartPole-v1").train(50_000)
```

### TD3Trainer (High-Level)

```python
from rlox import Trainer

metrics = Trainer("td3", env="Pendulum-v1").train(50_000)
```

### MAPPO (Multi-Agent)

Multi-agent PPO with centralised critic. Works with PettingZoo environments.

```python
from rlox import Trainer

trainer = Trainer("mappo", env="spread_v3", config={"n_agents": 3}, seed=42)
metrics = trainer.train(total_timesteps=500_000)
```

### DreamerV3Trainer (World Model)

Learns a latent dynamics model and trains the policy inside the learned world model. Particularly effective for image-based and sparse-reward environments.

```python
from rlox import Trainer

trainer = Trainer("dreamer", env="Pendulum-v1", seed=42)
metrics = trainer.train(total_timesteps=200_000)
```

### IMPALATrainer (Distributed Actors)

Distributed actor-learner architecture with V-trace off-policy correction. Actors collect experience in parallel (optionally across machines via gRPC) while a central learner trains the policy.

```python
from rlox import Trainer

trainer = Trainer("impala", env="CartPole-v1", n_actors=8, seed=42)
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
from rlox import Trainer
from rlox.wrappers import VecNormalize

trainer = Trainer(
    "ppo", env="HalfCheetah-v4",
    config={"normalize_obs": True, "normalize_rewards": True},
    seed=42,
)
metrics = trainer.train(total_timesteps=1_000_000)
```

### Diagnostics Dashboard & HTML Report

```python
from rlox.dashboard import MetricsCollector, HTMLReport, TerminalDashboard
from rlox import Trainer

# Collect metrics and generate an HTML report after training
collector = MetricsCollector()
trainer = Trainer("ppo", env="CartPole-v1", callbacks=[collector], seed=42)
trainer.train(total_timesteps=50_000)

report = HTMLReport(collector)
report.save("training_report.html")

# Or use a live terminal dashboard during training
trainer = Trainer(
    "ppo", env="CartPole-v1",
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

## Core Primitives

### Environment Stepping

=== "Python"

    ```python
    import rlox

    # Native Rust CartPole (fastest)
    env = rlox.VecEnv(n=1024, seed=42)
    result = env.step_all(actions)  # actions: numpy uint32 array
    # result["obs"]: (1024, 4) float32

    # Gymnasium wrapper (any env)
    from rlox import GymVecEnv
    env = GymVecEnv("HalfCheetah-v4", n_envs=8)
    result = env.step_all(actions)
    ```

=== "Rust"

    ```rust
    use rlox_core::env::builtins::CartPole;
    use rlox_core::env::parallel::VecEnv;
    use rlox_core::env::spaces::Action;
    use rlox_core::env::RLEnv;
    use rlox_core::seed::derive_seed;

    // Create 64 parallel CartPole environments
    let envs: Vec<Box<dyn RLEnv>> = (0..64)
        .map(|i| Box::new(CartPole::new(Some(derive_seed(42, i)))) as _)
        .collect();
    let mut vec_env = VecEnv::new(envs);

    let observations = vec_env.reset_all(Some(42)).unwrap();
    let actions: Vec<Action> = (0..64)
        .map(|i| Action::Discrete((i % 2) as u32))
        .collect();
    let batch = vec_env.step_all(&actions).unwrap();
    assert_eq!(batch.obs.len(), 64);
    ```

### Continuous Actions (Pendulum)

=== "Python"

    ```python
    import rlox
    import numpy as np

    env = rlox.VecEnv(n=4, seed=42, env_id="Pendulum-v1")
    obs = env.reset_all()  # (4, 3) — cos(θ), sin(θ), ω

    actions = np.array([[0.5], [-1.0], [2.0], [0.0]], dtype=np.float32)
    result = env.step_all(actions)
    ```

=== "Rust"

    ```rust
    use rlox_core::env::builtins::Pendulum;
    use rlox_core::env::spaces::Action;
    use rlox_core::env::RLEnv;

    let mut env = Pendulum::new(Some(42));
    let obs = env.reset(Some(42)).unwrap();
    println!("obs: {:?}", obs.as_slice()); // [cos θ, sin θ, ω]

    let t = env.step(&Action::Continuous(vec![1.5])).unwrap();
    println!("reward: {:.2}", t.reward);
    ```

### GAE Computation

=== "Python"

    ```python
    import rlox
    import numpy as np

    rewards = np.random.randn(2048)
    values = np.random.randn(2048)
    dones = np.zeros(2048)
    adv, ret = rlox.compute_gae(rewards, values, dones,
                                 last_value=0.0, gamma=0.99, lam=0.95)

    # Batched (8 envs, Rayon-parallel)
    rewards_flat = np.random.randn(8 * 2048)
    values_flat = np.random.randn(8 * 2048)
    dones_flat = np.zeros(8 * 2048)
    last_vals = np.random.randn(8)
    adv, ret = rlox.compute_gae_batched(rewards_flat, values_flat,
                                         dones_flat, last_vals, 2048, 0.99, 0.95)
    ```

=== "Rust"

    ```rust
    use rlox_core::training::gae::compute_gae;

    let rewards = &[1.0, 1.0, 1.0, 0.0, 1.0];
    let values  = &[0.5, 0.6, 0.7, 0.3, 0.8];
    let dones   = &[0.0, 0.0, 0.0, 1.0, 0.0];

    let (advantages, returns) = compute_gae(
        rewards, values, dones,
        0.9,   // last_value
        0.99,  // gamma
        0.95,  // gae_lambda
    );

    // Invariant: returns[t] == advantages[t] + values[t]
    for t in 0..5 {
        assert!((returns[t] - (advantages[t] + values[t])).abs() < 1e-10);
    }
    ```

### Replay Buffers

=== "Python"

    ```python
    import rlox
    import numpy as np

    buf = rlox.ReplayBuffer(capacity=100_000, obs_dim=4, act_dim=1)
    obs = np.zeros(4, dtype=np.float32)
    buf.push(obs, np.zeros(1), reward=1.0, terminated=False,
             truncated=False, next_obs=obs)

    batch = buf.sample(batch_size=256, seed=42)
    # batch["obs"], batch["actions"], batch["rewards"], ...

    # Prioritized replay
    pbuf = rlox.PrioritizedReplayBuffer(100_000, 4, 1, alpha=0.6, beta=0.4)
    pbuf.push(obs, np.zeros(1), 1.0, False, False, obs, priority=1.0)
    batch = pbuf.sample(256, seed=42)
    # batch["weights"], batch["indices"]
    ```

=== "Rust"

    ```rust
    use rlox_core::buffer::ringbuf::ReplayBuffer;
    use rlox_core::buffer::priority::PrioritizedReplayBuffer;
    use rlox_core::buffer::ExperienceRecord;

    // Uniform replay buffer
    let mut buf = ReplayBuffer::new(100_000, 4, 1);
    buf.push(ExperienceRecord {
        obs: vec![0.1, 0.2, 0.3, 0.4],
        next_obs: vec![0.2, 0.3, 0.4, 0.5],
        action: vec![0.0],
        reward: 1.0,
        terminated: false,
        truncated: false,
    }).unwrap();

    let batch = buf.sample(32, 42).unwrap();
    assert_eq!(batch.batch_size, 32);

    // Prioritized replay
    let mut per = PrioritizedReplayBuffer::new(100_000, 4, 1, 0.6, 0.4);
    per.push(ExperienceRecord {
        obs: vec![0.1; 4], next_obs: vec![0.2; 4],
        action: vec![0.0], reward: 1.0,
        terminated: false, truncated: false,
    }, 1.0).unwrap();
    ```

### Reward Shaping (PBRS)

=== "Python"

    ```python
    import rlox
    import numpy as np

    rewards = np.array([1.0, 1.0, 1.0])
    phi = np.array([0.5, 0.6, 0.7])       # potential of current state
    phi_next = np.array([0.6, 0.7, 0.8])   # potential of next state
    dones = np.array([0.0, 0.0, 1.0])

    shaped = rlox.shape_rewards_pbrs(rewards, phi, phi_next, gamma=0.99, dones=dones)
    ```

=== "Rust"

    ```rust
    use rlox_core::training::reward_shaping::shape_rewards_pbrs;

    let rewards = &[1.0, 1.0, 1.0];
    let phi = &[0.5, 0.6, 0.7];
    let phi_next = &[0.6, 0.7, 0.8];
    let dones = &[0.0, 0.0, 1.0];

    let shaped = shape_rewards_pbrs(rewards, phi, phi_next, 0.99, dones).unwrap();
    // done=1 → raw reward only (no shaping at episode boundary)
    assert!((shaped[2] - 1.0).abs() < 1e-10);
    ```

### Weight Operations (Meta-Learning)

=== "Python"

    ```python
    import rlox
    import numpy as np

    meta = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    task = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    rlox.reptile_update(meta, task, lr=0.1)  # in-place
    # meta is now [1.3, 2.3, 3.3]
    ```

=== "Rust"

    ```rust
    use rlox_core::training::weight_ops::{reptile_update, polyak_update};

    let mut meta = vec![1.0f32, 2.0, 3.0];
    let task = vec![4.0f32, 5.0, 6.0];
    reptile_update(&mut meta, &task, 0.1);
    // meta is now [1.3, 2.3, 3.3]

    let mut target = vec![0.0f32; 3];
    let source = vec![1.0f32; 3];
    polyak_update(&mut target, &source, 0.005);
    ```

### KL Divergence

=== "Python"

    ```python
    import rlox
    import numpy as np

    log_p = np.random.randn(32 * 2048).astype(np.float32)
    log_q = np.random.randn(32 * 2048).astype(np.float32)
    kl = rlox.compute_batch_token_kl_schulman_f32(log_p, log_q, seq_len=2048)
    # kl: (32,) array of per-sequence KL values
    ```

=== "Rust"

    ```rust
    use rlox_core::training::kl::compute_token_kl;

    let log_p = &[-1.0, -2.0, -0.5];
    let log_q = &[-1.0, -2.0, -0.5];
    let kl = compute_token_kl(log_p, log_q).unwrap();
    assert!(kl.abs() < 1e-15); // identical distributions → KL = 0
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
