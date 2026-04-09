# Migrating from Stable-Baselines3

## Side-by-Side Comparison

| Task | SB3 | rlox |
|------|-----|------|
| **Create PPO** | `PPO("MlpPolicy", "CartPole-v1")` | `PPO(env_id="CartPole-v1")` |
| **Train** | `model.learn(50_000)` | `ppo.train(50_000)` |
| **Predict** | `model.predict(obs)` | `ppo.predict(obs)` |
| **Save** | `model.save("ppo")` | `ppo.save("ppo.pt")` |
| **Load** | `PPO.load("ppo")` | `PPO.from_checkpoint("ppo.pt")` |
| **Eval callback** | `EvalCallback(eval_env)` | `EvalCallback(env_id=..., eval_freq=...)` |
| **Multi-env** | `make_vec_env("CartPole-v1", 8)` | `PPO(env_id="CartPole-v1", n_envs=8)` |
| **VecNormalize** | `VecNormalize(env)` | `VecNormalize(norm_obs=True, norm_reward=True)` wrapper |
| **Config-driven** | N/A (kwargs only) | `train_from_config("experiment.yaml")` |

## What's Different

### Faster by Default
rlox uses Rust for env stepping, GAE computation, and replay buffers. No code changes needed — just use rlox and get 3-50x speedup on data-plane operations.

### No Policy Strings
SB3 uses `"MlpPolicy"` strings. rlox auto-detects the policy from the environment's action space:
```python
# SB3
model = PPO("MlpPolicy", "CartPole-v1")      # discrete
model = PPO("MlpPolicy", "Pendulum-v1")       # continuous

# rlox — auto-detects
ppo = PPO(env_id="CartPole-v1")               # uses DiscretePolicy
ppo = PPO(env_id="Pendulum-v1")               # uses ContinuousPolicy
```

### Configs are Dataclasses
SB3 uses kwargs. rlox supports both kwargs and YAML configs:
```python
# rlox kwargs (same as SB3)
ppo = PPO(env_id="CartPole-v1", learning_rate=3e-4, n_steps=128)

# rlox YAML config
from rlox.config import PPOConfig
config = PPOConfig.from_yaml("ppo_config.yaml")
```

### Additional Algorithms
rlox includes algorithms SB3 doesn't have:

| Algorithm | Description |
|-----------|-------------|
| `TD3BC` | Offline RL: TD3 + behavioral cloning |
| `IQL` | Offline RL: implicit Q-learning |
| `CQL` | Offline RL: conservative Q-learning |
| `BC` | Behavioral cloning |
| `GRPO` | LLM post-training |
| `DPO` | Direct preference optimization |
| `HybridPPO` | Candle hybrid collection (180K SPS) |
| `IMPALA` | Distributed actor-learner with V-trace |
| `MAPPO` | Multi-agent PPO with centralised critic |
| `DreamerV3` | Model-based RL with learned world model |

## Full Migration Example

### SB3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

env = DummyVecEnv([lambda: gym.make("CartPole-v1")] * 8)
eval_env = gym.make("CartPole-v1")

model = PPO(
    "MlpPolicy", env,
    learning_rate=2.5e-4,
    n_steps=128,
    n_epochs=4,
    batch_size=256,
    verbose=1,
)

eval_callback = EvalCallback(
    eval_env, eval_freq=5000, best_model_save_path="./best/"
)

model.learn(total_timesteps=100_000, callback=eval_callback)
model.save("ppo_cartpole")

# Evaluate
obs, _ = eval_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    if terminated or truncated:
        obs, _ = eval_env.reset()
```

### rlox

```python
from rlox.algorithms.ppo import PPO
from rlox.callbacks import EvalCallback, ProgressBarCallback

ppo = PPO(
    env_id="CartPole-v1",
    n_envs=8,
    learning_rate=2.5e-4,
    n_steps=128,
    n_epochs=4,
    batch_size=256,
    callbacks=[
        EvalCallback(env_id="CartPole-v1", eval_freq=5000),
        ProgressBarCallback(),
    ],
)

metrics = ppo.train(total_timesteps=100_000)
ppo.save("ppo_cartpole.pt")

# Evaluate
import gymnasium as gym
import numpy as np
env = gym.make("CartPole-v1")
obs, _ = env.reset()
for _ in range(1000):
    action = ppo.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
```

## SAC Migration

```python
# SB3
from stable_baselines3 import SAC
model = SAC("MlpPolicy", "Pendulum-v1", learning_starts=1000)
model.learn(50_000)

# rlox
from rlox.algorithms.sac import SAC
sac = SAC(env_id="Pendulum-v1", learning_starts=1000)
sac.train(50_000)

# rlox with multi-env collection (SB3 doesn't support this for SAC)
sac = SAC(env_id="Pendulum-v1", n_envs=4, learning_starts=1000)
sac.train(50_000)
```

## VecNormalize Migration

```python
# SB3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
env = DummyVecEnv([lambda: gym.make("HalfCheetah-v4")])
env = VecNormalize(env, norm_obs=True, norm_reward=True)
model = PPO("MlpPolicy", env)
model.learn(1_000_000)

# rlox
from rlox.algorithms.ppo import PPO
from rlox.wrappers import VecNormalize
ppo = PPO(
    env_id="HalfCheetah-v4",
    n_envs=8,
    wrappers=[VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10.0)],
)
ppo.train(1_000_000)
```

## Config-Driven Training (rlox only)

SB3 requires writing Python scripts for every experiment. rlox supports YAML configs:

```python
from rlox.runner import train_from_config
metrics = train_from_config("experiment.yaml")
```

## Known Algorithmic Asymmetries

When migrating from SB3, be aware that rlox intentionally differs in a few implementation details. These differences have been validated via multi-seed convergence benchmarks and do not prevent parity, but they can cause confusion when comparing loss values or debugging.

| Asymmetry | rlox | SB3 | Impact |
|-----------|------|-----|--------|
| **PPO value loss** | `0.5 * MSE` with `clip_vloss=True` (CleanRL convention) | `F.mse_loss` (no inner 0.5) with `clip_range_vf=None` | `vf_coef=0.5` in rlox produces an effective weight of 0.25, vs 0.5 in SB3. Convergence is equivalent in practice. |
| **DQN loss function** | MSE loss `(Q - y)^2` | Huber loss `F.smooth_l1_loss` | MSE can produce larger gradients on outlier TD errors. Consider setting `max_grad_norm=10.0` for stability. |
| **SAC critic loss factor** | No `0.5` factor on critic loss | `0.5 * F.mse_loss` on each critic | The effective critic learning rate differs by 2x. rlox's defaults are calibrated for this. |
| **DQN/SAC train_freq** | `train_freq=1, gradient_steps=1` (default) | `train_freq=1, gradient_steps=1` (default) | Identical defaults. Override via config or kwargs. |

See the [PPO algorithm page](../algorithms/ppo.md#value-loss-formulation) for the full rationale behind the value loss choice, including the Hopper-v4 A/B test that led to the current defaults.

## Advanced Algorithms (rlox only)

rlox includes MAPPO (multi-agent), DreamerV3 (world-model-based), and IMPALA (distributed actor-learner). These have no SB3 equivalent.

## Offline RL (rlox only)

SB3 doesn't support offline RL. With rlox:

```python
import rlox
from rlox.algorithms.td3_bc import TD3BC

# Load D4RL dataset
import d4rl, gymnasium
env = gymnasium.make("halfcheetah-medium-v2")
dataset = env.get_dataset()

buf = rlox.OfflineDatasetBuffer(
    dataset["observations"].ravel().astype("float32"),
    dataset["next_observations"].ravel().astype("float32"),
    dataset["actions"].ravel().astype("float32"),
    dataset["rewards"].astype("float32"),
    dataset["terminals"].astype("uint8"),
    dataset["timeouts"].astype("uint8"),
    normalize=True,
)

algo = TD3BC(dataset=buf, obs_dim=17, act_dim=6, alpha=2.5)
algo.train(n_gradient_steps=100_000)
```
