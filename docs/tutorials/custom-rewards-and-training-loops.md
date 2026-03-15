# Custom Reward Functions and Training Loops

This tutorial covers how to extend rlox beyond the built-in algorithms by writing
custom reward functions, composing low-level primitives into bespoke training loops,
and implementing custom components in Rust. It assumes you have installed rlox and
are familiar with the basics from the getting-started tutorial.

**Table of contents**

- [Part 1: Custom Reward Functions (Python)](#part-1-custom-reward-functions-python)
- [Part 2: Custom Training Loops (Python)](#part-2-custom-training-loops-python)
- [Part 3: Custom Components in Rust](#part-3-custom-components-in-rust)
- [Part 4: Integration Patterns](#part-4-integration-patterns)

---

## Part 1: Custom Reward Functions (Python)

### 1.1 Reward Shaping for Simulation RL

The `RolloutCollector` accepts an optional `reward_fn` parameter that transforms
raw environment rewards before they are stored. The signature is:

```python
def reward_fn(obs: np.ndarray, actions: np.ndarray, raw_rewards: np.ndarray) -> np.ndarray
```

- `obs` -- observations **before** the step, shape `(n_envs, obs_dim)`
- `actions` -- actions taken, shape `(n_envs,)` (discrete) or `(n_envs, act_dim)` (continuous)
- `raw_rewards` -- the environment's original rewards, shape `(n_envs,)`
- Returns a new reward array of the same shape as `raw_rewards`.

The function is called once per time step with batched data across all environments.

**Example: Penalise large pole angles in CartPole**

CartPole gives a constant reward of 1.0 per step. We can add a penalty
proportional to the pole angle to encourage the agent to keep the pole
more upright, not just alive:

```python
import numpy as np
from rlox import RolloutCollector
from rlox.policies import DiscretePolicy

def angle_penalty_reward(obs: np.ndarray, actions: np.ndarray, raw_rewards: np.ndarray) -> np.ndarray:
    """Penalise large pole angles. CartPole obs = [x, x_dot, theta, theta_dot]."""
    theta = obs[:, 2]  # pole angle (radians)
    penalty = 0.5 * theta ** 2  # quadratic penalty
    return raw_rewards - penalty

collector = RolloutCollector(
    "CartPole-v1",
    n_envs=8,
    seed=42,
    reward_fn=angle_penalty_reward,
)

policy = DiscretePolicy(obs_dim=4, n_actions=2)
batch = collector.collect(policy, n_steps=128)

# The batch.rewards now contain the shaped rewards
print(f"Mean shaped reward per step: {batch.rewards.mean():.3f}")
```

**Example: Curiosity-based reward (state visitation count)**

A simple intrinsic motivation approach: give a bonus inversely proportional to
how many times a discretised state has been visited.

```python
import numpy as np
from collections import defaultdict

class CuriosityReward:
    """Count-based exploration bonus added to the extrinsic reward."""

    def __init__(self, n_bins: int = 20, bonus_scale: float = 0.1):
        self.n_bins = n_bins
        self.bonus_scale = bonus_scale
        self.visit_counts: dict[tuple, int] = defaultdict(int)

    def _discretise(self, obs: np.ndarray) -> list[tuple]:
        """Bin each observation into a coarse grid."""
        clipped = np.clip(obs, -5.0, 5.0)
        binned = np.floor(clipped * self.n_bins / 10.0).astype(int)
        return [tuple(row) for row in binned]

    def __call__(self, obs: np.ndarray, actions: np.ndarray, raw_rewards: np.ndarray) -> np.ndarray:
        keys = self._discretise(obs)
        bonuses = np.zeros(len(keys), dtype=np.float64)
        for i, key in enumerate(keys):
            self.visit_counts[key] += 1
            bonuses[i] = self.bonus_scale / np.sqrt(self.visit_counts[key])
        return raw_rewards + bonuses


curiosity = CuriosityReward(n_bins=20, bonus_scale=0.5)

collector = RolloutCollector(
    "CartPole-v1",
    n_envs=8,
    seed=42,
    reward_fn=curiosity,
)
```

Because `reward_fn` is any callable matching the signature, you can use a class
with `__call__` to maintain state across steps (as shown above with visit counts).


### 1.2 Custom Reward Functions for LLM Post-Training (GRPO)

The `GRPO` trainer accepts a `reward_fn` with signature:

```python
def reward_fn(completions: list[torch.Tensor], prompts: torch.Tensor) -> list[float]
```

- `completions` -- list of token ID tensors, one per completion
- `prompts` -- the expanded prompt tensor (repeated `group_size` times)
- Returns a list of scalar rewards, one per completion

rlox computes group advantages from these rewards using the Rust-accelerated
`compute_batch_group_advantages` function.

**Example: Math correctness reward**

```python
import re
import torch
from rlox.algorithms.grpo import GRPO

def math_correctness_reward(
    completions: list[torch.Tensor], prompts: torch.Tensor
) -> list[float]:
    """Score 1.0 if the completion contains the correct answer, else 0.0.

    Assumes a tokenizer is available and answers are in '\\boxed{...}' format.
    """
    rewards = []
    for completion in completions:
        text = tokenizer.decode(completion.tolist(), skip_special_tokens=True)
        # Extract answer from \boxed{...}
        match = re.search(r"\\boxed\{([^}]+)\}", text)
        if match and match.group(1).strip() == expected_answer:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


grpo = GRPO(
    model=policy_model,
    ref_model=ref_model,
    reward_fn=math_correctness_reward,
    group_size=4,
    kl_coef=0.1,
)
metrics = grpo.train_step(prompt_batch)
```

**Example: Format compliance reward (JSON)**

```python
import json

def json_format_reward(
    completions: list[torch.Tensor], prompts: torch.Tensor
) -> list[float]:
    """Reward completions that produce valid JSON with required keys."""
    required_keys = {"answer", "reasoning"}
    rewards = []
    for completion in completions:
        text = tokenizer.decode(completion.tolist(), skip_special_tokens=True)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and required_keys.issubset(parsed.keys()):
                rewards.append(1.0)
            else:
                rewards.append(0.3)  # valid JSON but missing keys
        except json.JSONDecodeError:
            rewards.append(0.0)
    return rewards
```

**Example: Multi-objective reward**

Use `MultiObjectiveReward` from `rlox.llm.reward_models` to combine
multiple objectives with configurable weights:

```python
import numpy as np
from rlox.llm.reward_models import MultiObjectiveReward

def helpfulness_scorer(prompts: list[str], completions: list[str]) -> np.ndarray:
    """Score how helpful each completion is (stub -- replace with a real model)."""
    return np.array([len(c) / 500.0 for c in completions])  # proxy: longer = more helpful

def safety_scorer(prompts: list[str], completions: list[str]) -> np.ndarray:
    """Score safety (stub -- replace with a real classifier)."""
    blocked_words = {"hack", "exploit", "malware"}
    scores = []
    for c in completions:
        if any(w in c.lower() for w in blocked_words):
            scores.append(0.0)
        else:
            scores.append(1.0)
    return np.array(scores)

multi_reward = MultiObjectiveReward(
    objectives={
        "helpfulness": helpfulness_scorer,
        "safety": safety_scorer,
    },
    weights={
        "helpfulness": 0.6,
        "safety": 0.4,
    },
)

# Use in a GRPO-style loop
scores = multi_reward.score_batch(prompts=["Explain X"], completions=["..."])
```

Each objective callable has signature `(prompts: list[str], completions: list[str]) -> np.ndarray`.
The `MultiObjectiveReward.score_batch` method returns the weighted sum.

**Ensemble reward models** work similarly. `EnsembleRewardModel` takes a list of
`nn.Module` reward models and optional weights, serving as a drop-in replacement:

```python
from rlox.llm.reward_models import EnsembleRewardModel

ensemble = EnsembleRewardModel(
    models=[reward_model_a, reward_model_b, reward_model_c],
    weights=[0.5, 0.3, 0.2],  # optional; defaults to uniform
)
scores = ensemble.score_batch(prompts=["..."], completions=["..."])
```

### 1.3 Reward Model Training (Bradley-Terry)

A quick sketch of training a Bradley-Terry reward model that can then be used
with `RewardModelServer`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    """Simple MLP reward model for scoring sequences."""

    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return scalar reward per sequence."""
        # Mean-pool embeddings across the sequence length
        embeds = self.embed(input_ids)  # (B, T, D)
        pooled = embeds.mean(dim=1)     # (B, D)
        return self.net(pooled).squeeze(-1)  # (B,)


def bradley_terry_loss(
    reward_model: nn.Module,
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry pairwise loss: -log(sigmoid(r_chosen - r_rejected))."""
    r_chosen = reward_model(chosen_ids)
    r_rejected = reward_model(rejected_ids)
    return -F.logsigmoid(r_chosen - r_rejected).mean()


# Training loop
reward_model = RewardModel(vocab_size=32000)
optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-4)

for chosen, rejected in preference_dataloader:
    loss = bradley_terry_loss(reward_model, chosen, rejected)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Serve the trained model
from rlox.llm.reward_models import RewardModelServer
server = RewardModelServer(reward_model, batch_size=64)
```

---

## Part 2: Custom Training Loops (Python)

rlox is designed in layers. The high-level `PPO` class is convenient, but you
can drop down to **Layer 1 primitives** -- `RolloutCollector`, `compute_gae`,
`PPOLoss` -- and compose them yourself.

### 2.1 Using Layer 1 Primitives

**Example: PPO with custom logging and per-rollout metrics**

```python
import torch
import torch.nn as nn
import numpy as np

import rlox
from rlox import RolloutCollector, compute_gae
from rlox.losses import PPOLoss
from rlox.policies import DiscretePolicy

# Setup
policy = DiscretePolicy(obs_dim=4, n_actions=2)
optimizer = torch.optim.Adam(policy.parameters(), lr=2.5e-4, eps=1e-5)
loss_fn = PPOLoss(clip_eps=0.2, vf_coef=0.5, ent_coef=0.01)

collector = RolloutCollector(
    "CartPole-v1",
    n_envs=8,
    seed=42,
    gamma=0.99,
    gae_lambda=0.95,
)

# Custom training loop
n_updates = 100
n_steps = 128
batch_size = 256
n_epochs = 4

for update in range(n_updates):
    # 1. Collect rollout (calls compute_gae internally)
    batch = collector.collect(policy, n_steps=n_steps)

    # 2. Custom logging -- compute whatever you want from the batch
    mean_reward = batch.rewards.sum().item() / 8
    mean_advantage = batch.advantages.mean().item()
    advantage_std = batch.advantages.std().item()
    print(f"[update {update:3d}] reward={mean_reward:.1f}  "
          f"adv_mean={mean_advantage:.3f}  adv_std={advantage_std:.3f}")

    # 3. SGD updates
    for epoch in range(n_epochs):
        for mb in batch.sample_minibatches(batch_size, shuffle=True):
            adv = (mb.advantages - mb.advantages.mean()) / (mb.advantages.std() + 1e-8)
            loss, metrics = loss_fn(
                policy, mb.obs, mb.actions, mb.log_probs,
                adv, mb.returns, mb.values,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

    # 4. Custom stopping condition
    if mean_reward > 450:
        print(f"Solved at update {update}!")
        break
```

**Example: PPO with curriculum learning**

Change environment difficulty as training progresses. Since `RolloutCollector`
wraps a `VecEnv` or `GymVecEnv`, you can replace the collector mid-training:

```python
from rlox import RolloutCollector
from rlox.policies import ContinuousPolicy
from rlox.losses import PPOLoss

# Start with a simpler environment
envs = ["Pendulum-v1", "InvertedPendulum-v5", "InvertedDoublePendulum-v5"]
policy = ContinuousPolicy(obs_dim=3, act_dim=1)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
loss_fn = PPOLoss()

for stage, env_id in enumerate(envs):
    print(f"\n--- Stage {stage}: {env_id} ---")
    collector = RolloutCollector(env_id, n_envs=8, seed=42 + stage)

    for update in range(50):
        batch = collector.collect(policy, n_steps=256)
        mean_r = batch.rewards.sum().item() / 8

        for _ in range(4):
            for mb in batch.sample_minibatches(256, shuffle=True):
                adv = (mb.advantages - mb.advantages.mean()) / (mb.advantages.std() + 1e-8)
                loss, _ = loss_fn(
                    policy, mb.obs, mb.actions, mb.log_probs,
                    adv, mb.returns, mb.values,
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        if update % 10 == 0:
            print(f"  update {update}: mean_reward={mean_r:.1f}")
```

### 2.2 Implementing REINFORCE from Scratch

A full example of building a new algorithm using rlox primitives. REINFORCE
is the simplest policy-gradient method -- no critic, no clipping, just
`-log_prob * return`.

```python
import torch
import torch.nn as nn
import numpy as np

import rlox
from rlox import GymVecEnv, compute_gae
from rlox.policies import DiscretePolicy
from rlox.callbacks import Callback, CallbackList, EarlyStoppingCallback


class REINFORCE:
    """Vanilla REINFORCE with optional baseline (GAE with lambda=1.0)."""

    def __init__(
        self,
        env_id: str,
        n_envs: int = 8,
        seed: int = 42,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        use_baseline: bool = True,
        callbacks: list[Callback] | None = None,
    ):
        self.env_id = env_id
        self.n_envs = n_envs
        self.gamma = gamma
        self.use_baseline = use_baseline

        # Use GymVecEnv for generality (works with any Gymnasium env)
        self.env = GymVecEnv(env_id, n_envs=n_envs, seed=seed)

        import gymnasium as gym
        tmp = gym.make(env_id)
        obs_dim = int(np.prod(tmp.observation_space.shape))
        n_actions = int(tmp.action_space.n)
        tmp.close()

        self.policy = DiscretePolicy(obs_dim=obs_dim, n_actions=n_actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.callbacks = CallbackList(callbacks)
        self._obs = None
        self._global_step = 0

    @torch.no_grad()
    def _collect_rollout(self, n_steps: int):
        """Collect a rollout and compute returns via GAE (lambda=1 for MC returns)."""
        if self._obs is None:
            self._obs = self.env.reset_all()

        all_obs, all_actions, all_log_probs, all_rewards, all_dones, all_values = (
            [], [], [], [], [], [],
        )

        for _ in range(n_steps):
            obs_t = torch.as_tensor(self._obs, dtype=torch.float32)
            actions, log_probs = self.policy.get_action_and_logprob(obs_t)
            values = self.policy.get_value(obs_t) if self.use_baseline else torch.zeros(self.n_envs)

            actions_np = actions.cpu().numpy().astype(np.uint32).tolist()
            result = self.env.step_all(actions_np)

            all_obs.append(obs_t)
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_values.append(values)
            all_rewards.append(torch.as_tensor(result["rewards"].astype(np.float32)))

            dones = result["terminated"].astype(bool) | result["truncated"].astype(bool)
            all_dones.append(torch.as_tensor(dones.astype(np.float32)))
            self._obs = result["obs"].copy()

        # Bootstrap value
        last_obs_t = torch.as_tensor(self._obs, dtype=torch.float32)
        last_values = self.policy.get_value(last_obs_t) if self.use_baseline else torch.zeros(self.n_envs)

        # Compute returns per env using rlox.compute_gae (lambda=1.0 gives MC returns)
        all_returns = []
        for env_idx in range(self.n_envs):
            rewards_env = torch.stack([r[env_idx] for r in all_rewards])
            values_env = torch.stack([v[env_idx] for v in all_values])
            dones_env = torch.stack([d[env_idx] for d in all_dones])

            advantages, returns = compute_gae(
                rewards=rewards_env.numpy().astype(np.float64),
                values=values_env.numpy().astype(np.float64),
                dones=dones_env.numpy().astype(np.float64),
                last_value=float(last_values[env_idx]),
                gamma=self.gamma,
                lam=1.0,  # lambda=1 makes GAE equivalent to Monte Carlo returns
            )
            all_returns.append(torch.as_tensor(returns, dtype=torch.float32))

        # Flatten
        obs = torch.stack(all_obs).reshape(-1, all_obs[0].shape[-1])
        actions = torch.stack(all_actions).reshape(-1)
        log_probs = torch.stack(all_log_probs).reshape(-1)
        values = torch.stack(all_values).reshape(-1)
        returns = torch.stack(all_returns).T.reshape(-1)  # (n_envs, n_steps) -> (n_steps, n_envs) -> flat
        rewards = torch.stack(all_rewards).reshape(-1)

        return obs, actions, log_probs, values, returns, rewards

    def train(self, total_timesteps: int, n_steps: int = 128) -> dict[str, float]:
        """Train with REINFORCE."""
        steps_per_rollout = self.n_envs * n_steps
        n_updates = max(1, total_timesteps // steps_per_rollout)

        self.callbacks.on_training_start()
        all_rewards = []

        for update in range(n_updates):
            obs, actions, old_log_probs, values, returns, rewards = self._collect_rollout(n_steps)

            # REINFORCE loss: -log_prob * advantage
            if self.use_baseline:
                advantage = returns - values.detach()
            else:
                advantage = returns

            # Normalise advantages
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # Recompute log_probs with gradient
            new_log_probs, entropy = self.policy.get_logprob_and_entropy(obs, actions)
            loss = -(new_log_probs * advantage.detach()).mean() - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            mean_reward = rewards.sum().item() / self.n_envs
            all_rewards.append(mean_reward)
            self._global_step += 1

            should_continue = self.callbacks.on_step(
                reward=mean_reward, step=self._global_step
            )
            if not should_continue:
                break

        self.callbacks.on_training_end()
        return {"mean_reward": float(np.mean(all_rewards))}


# Usage
reinforce = REINFORCE(
    "CartPole-v1",
    n_envs=8,
    seed=42,
    callbacks=[EarlyStoppingCallback(patience=20)],
)
metrics = reinforce.train(total_timesteps=200_000)
print(f"REINFORCE mean reward: {metrics['mean_reward']:.1f}")
```

Key points:
- `GymVecEnv` provides the same `step_all`/`reset_all` interface as the native `VecEnv`
- `compute_gae` with `lam=1.0` gives Monte Carlo returns (no bias, high variance)
- The `Callback` system integrates naturally -- just pass callbacks to your constructor
  and call the hooks at appropriate points

### 2.3 Custom Off-Policy Loop with ReplayBuffer

Using the Rust-backed `ReplayBuffer` with a custom update rule:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rlox
from rlox import ReplayBuffer, GymVecEnv

class SimpleQNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def custom_dqn_loop(
    env_id: str = "CartPole-v1",
    total_steps: int = 50_000,
    buffer_capacity: int = 10_000,
    batch_size: int = 64,
    gamma: float = 0.99,
    learning_starts: int = 1_000,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 10_000,
    seed: int = 42,
):
    """Minimal DQN with custom epsilon schedule and Rust replay buffer."""

    import gymnasium as gym
    tmp = gym.make(env_id)
    obs_dim = int(np.prod(tmp.observation_space.shape))
    n_actions = int(tmp.action_space.n)
    tmp.close()

    env = GymVecEnv(env_id, n_envs=1, seed=seed)
    buffer = ReplayBuffer(capacity=buffer_capacity, obs_dim=obs_dim, act_dim=1)
    q_net = SimpleQNetwork(obs_dim, n_actions)
    target_net = SimpleQNetwork(obs_dim, n_actions)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)

    obs = env.reset_all()
    sample_seed = 0

    for step in range(total_steps):
        # Epsilon-greedy with linear decay
        frac = min(1.0, step / eps_decay_steps)
        epsilon = eps_start + frac * (eps_end - eps_start)

        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                q_vals = q_net(obs_t)
                action = q_vals.argmax(dim=-1).item()

        result = env.step_all([action])
        next_obs = result["obs"]
        reward = float(result["rewards"][0])
        terminated = bool(result["terminated"][0])
        truncated = bool(result["truncated"][0])

        # Push into Rust ReplayBuffer
        buffer.push(obs[0].tolist(), [float(action)], reward, terminated, truncated)
        obs = next_obs

        # Train after warmup
        if step >= learning_starts and buffer.len() >= batch_size:
            sample_seed += 1
            batch = buffer.sample(batch_size, sample_seed)

            # batch fields are flat numpy arrays; reshape as needed
            b_obs = torch.as_tensor(
                np.array(batch.observations).reshape(batch_size, obs_dim),
                dtype=torch.float32,
            )
            b_actions = torch.as_tensor(
                np.array(batch.actions).reshape(batch_size).astype(int),
                dtype=torch.long,
            )
            b_rewards = torch.as_tensor(
                np.array(batch.rewards), dtype=torch.float32,
            )
            b_terminated = torch.as_tensor(
                np.array(batch.terminated), dtype=torch.float32,
            )

            # Double DQN target
            with torch.no_grad():
                next_q = target_net(b_obs)  # simplified: using same obs
                max_next_q = next_q.max(dim=-1).values
                targets = b_rewards + gamma * max_next_q * (1.0 - b_terminated)

            current_q = q_net(b_obs).gather(1, b_actions.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(current_q, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Hard target update every 500 steps
            if step % 500 == 0:
                target_net.load_state_dict(q_net.state_dict())

        if step % 5000 == 0:
            print(f"step={step}, epsilon={epsilon:.3f}, buffer_size={buffer.len()}")

    return q_net


q_net = custom_dqn_loop()
```

The `ReplayBuffer` is backed by a pre-allocated Rust ring buffer. Key methods:
- `buffer.push(obs, action, reward, terminated, truncated)` -- zero-allocation write
- `buffer.sample(batch_size, seed)` -- deterministic uniform sampling via ChaCha8Rng
- `buffer.len()` -- number of valid transitions

### 2.4 Mixing Algorithms: Online DPO

Combine generation, scoring, preference construction, and DPO updates in a
single loop. This pattern is useful for online RLHF:

```python
import torch
from rlox.algorithms.dpo import DPO
import rlox

def online_dpo_loop(
    model,
    ref_model,
    reward_fn,       # (completions, prompts) -> list[float]
    prompt_dataset,  # iterable of prompt tensors
    n_epochs: int = 3,
    group_size: int = 4,
    beta: float = 0.1,
    learning_rate: float = 1e-5,
):
    """Online DPO: generate, score, select preferences, then DPO update."""
    dpo = DPO(model=model, ref_model=ref_model, beta=beta, learning_rate=learning_rate)

    for epoch in range(n_epochs):
        for prompts in prompt_dataset:
            # 1. Generate multiple completions per prompt
            expanded = prompts.repeat_interleave(group_size, dim=0)
            with torch.no_grad():
                completions = model.generate(expanded, max_new_tokens=64)

            # 2. Score completions
            comp_list = [completions[i] for i in range(completions.shape[0])]
            rewards = reward_fn(comp_list, expanded)

            # 3. Construct preferences: best vs worst in each group
            n_prompts = prompts.shape[0]
            for i in range(n_prompts):
                group_start = i * group_size
                group_rewards = rewards[group_start : group_start + group_size]
                best_idx = group_start + int(max(range(group_size), key=lambda j: group_rewards[j]))
                worst_idx = group_start + int(min(range(group_size), key=lambda j: group_rewards[j]))

                chosen = completions[best_idx].unsqueeze(0)
                rejected = completions[worst_idx].unsqueeze(0)
                prompt = prompts[i].unsqueeze(0)

                # 4. DPO gradient step
                metrics = dpo.train_step(prompt, chosen, rejected)

            print(f"Epoch {epoch}: loss={metrics['loss']:.4f}  "
                  f"chosen_reward={metrics['chosen_reward']:.3f}  "
                  f"rejected_reward={metrics['rejected_reward']:.3f}")
```

---

## Part 3: Custom Components in Rust

### 3.1 Implementing a Custom Environment

Implement the `RLEnv` trait to create a new Rust-native environment.

**Full example: GridWorld**

```rust
// crates/rlox-core/src/env/gridworld.rs  (hypothetical)

use std::collections::HashMap;

use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::env::spaces::{Action, ActionSpace, ObsSpace, Observation};
use crate::env::{RLEnv, Transition};
use crate::error::RloxError;
use crate::seed::rng_from_seed;

/// A simple 5x5 grid world with a goal in the corner.
///
/// Actions: 0=up, 1=right, 2=down, 3=left
/// Observation: [row, col] normalised to [0, 1]
/// Reward: -0.01 per step, +1.0 on reaching the goal
pub struct GridWorld {
    rows: usize,
    cols: usize,
    pos: (usize, usize),
    goal: (usize, usize),
    max_steps: u32,
    steps: u32,
    rng: ChaCha8Rng,
    done: bool,
    action_space: ActionSpace,
    obs_space: ObsSpace,
}

impl GridWorld {
    pub fn new(rows: usize, cols: usize, seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or(0);
        let mut env = Self {
            rows,
            cols,
            pos: (0, 0),
            goal: (rows - 1, cols - 1),
            max_steps: (rows * cols * 4) as u32,
            steps: 0,
            rng: rng_from_seed(seed),
            done: true,
            action_space: ActionSpace::Discrete(4),
            obs_space: ObsSpace::Box {
                low: vec![0.0, 0.0],
                high: vec![1.0, 1.0],
                shape: vec![2],
            },
        };
        let _ = env.reset(Some(seed));
        env
    }

    fn obs(&self) -> Observation {
        Observation(vec![
            self.pos.0 as f32 / (self.rows - 1) as f32,
            self.pos.1 as f32 / (self.cols - 1) as f32,
        ])
    }
}

impl RLEnv for GridWorld {
    fn step(&mut self, action: &Action) -> Result<Transition, RloxError> {
        if self.done {
            return Err(RloxError::EnvError(
                "Environment is done. Call reset() first.".into(),
            ));
        }

        let dir = match action {
            Action::Discrete(a) => *a,
            _ => return Err(RloxError::InvalidAction("Expected Discrete action".into())),
        };

        if dir > 3 {
            return Err(RloxError::InvalidAction(format!(
                "Action {dir} out of range for Discrete(4)"
            )));
        }

        // Move
        match dir {
            0 => { if self.pos.0 > 0 { self.pos.0 -= 1; } }               // up
            1 => { if self.pos.1 < self.cols - 1 { self.pos.1 += 1; } }    // right
            2 => { if self.pos.0 < self.rows - 1 { self.pos.0 += 1; } }    // down
            3 => { if self.pos.1 > 0 { self.pos.1 -= 1; } }               // left
            _ => unreachable!(),
        }

        self.steps += 1;
        let terminated = self.pos == self.goal;
        let truncated = !terminated && self.steps >= self.max_steps;
        self.done = terminated || truncated;

        let reward = if terminated { 1.0 } else { -0.01 };

        Ok(Transition {
            obs: self.obs(),
            reward,
            terminated,
            truncated,
            info: HashMap::new(),
        })
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<Observation, RloxError> {
        if let Some(s) = seed {
            self.rng = rng_from_seed(s);
        }
        // Random start position (not on the goal)
        loop {
            self.pos = (
                self.rng.random_range(0..self.rows),
                self.rng.random_range(0..self.cols),
            );
            if self.pos != self.goal {
                break;
            }
        }
        self.steps = 0;
        self.done = false;
        Ok(self.obs())
    }

    fn action_space(&self) -> &ActionSpace { &self.action_space }
    fn obs_space(&self) -> &ObsSpace { &self.obs_space }

    fn render(&self) -> Option<String> {
        let mut grid = String::new();
        for r in 0..self.rows {
            for c in 0..self.cols {
                if (r, c) == self.pos {
                    grid.push('A');
                } else if (r, c) == self.goal {
                    grid.push('G');
                } else {
                    grid.push('.');
                }
            }
            grid.push('\n');
        }
        Some(grid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gridworld_reset_not_on_goal() {
        let env = GridWorld::new(5, 5, Some(42));
        let obs = env.obs();
        // Should not start at (1.0, 1.0) which is the goal
        assert!(!(obs.0[0] == 1.0 && obs.0[1] == 1.0));
    }

    #[test]
    fn gridworld_reaches_goal() {
        let mut env = GridWorld::new(2, 2, Some(0));
        env.pos = (0, 0); // force start
        env.done = false;
        // Navigate to (1, 1): down then right
        let t = env.step(&Action::Discrete(2)).unwrap(); // down
        assert!(!t.terminated);
        let t = env.step(&Action::Discrete(1)).unwrap(); // right
        assert!(t.terminated);
        assert!((t.reward - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn gridworld_wall_clipping() {
        let mut env = GridWorld::new(3, 3, Some(0));
        env.pos = (0, 0);
        env.done = false;
        // Try moving up from top row -- should stay
        let _ = env.step(&Action::Discrete(0)).unwrap();
        assert_eq!(env.pos.0, 0);
    }

    #[test]
    fn gridworld_truncates() {
        let mut env = GridWorld::new(2, 2, Some(0));
        env.pos = (0, 0);
        env.done = false;
        // Bounce left repeatedly -- never reaches goal
        for _ in 0..200 {
            match env.step(&Action::Discrete(3)) {
                Ok(t) if t.truncated => return, // success: truncation detected
                Ok(_) => {}
                Err(_) => { env.reset(Some(0)).unwrap(); }
            }
        }
    }
}
```

**Using it with VecEnv:**

```rust
use rlox_core::env::parallel::VecEnv;
use rlox_core::env::RLEnv;
use rlox_core::seed::derive_seed;

fn make_gridworld_vec_env(n: usize, seed: u64) -> VecEnv {
    let envs: Vec<Box<dyn RLEnv>> = (0..n)
        .map(|i| {
            Box::new(GridWorld::new(5, 5, Some(derive_seed(seed, i)))) as Box<dyn RLEnv>
        })
        .collect();
    VecEnv::new(envs)
}
```

**Exposing via PyO3:**

To make your custom environment available from Python, add a PyO3 wrapper in
`crates/rlox-python/src/lib.rs`:

```rust
use pyo3::prelude::*;

#[pyclass]
struct PyGridWorld {
    inner: GridWorld,
}

#[pymethods]
impl PyGridWorld {
    #[new]
    fn new(rows: usize, cols: usize, seed: Option<u64>) -> Self {
        Self { inner: GridWorld::new(rows, cols, seed) }
    }

    fn step(&mut self, action: u32) -> PyResult<(Vec<f32>, f64, bool, bool)> {
        let t = self.inner
            .step(&Action::Discrete(action))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok((t.obs.into_inner(), t.reward, t.terminated, t.truncated))
    }

    fn reset(&mut self, seed: Option<u64>) -> PyResult<Vec<f32>> {
        let obs = self.inner
            .reset(seed)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(obs.into_inner())
    }
}
```

### 3.2 Implementing a Custom Buffer

Example: a buffer that only samples recent transitions (a "recency-biased" buffer).

```rust
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use rlox_core::buffer::{ExperienceRecord, ringbuf::SampledBatch};
use rlox_core::error::RloxError;

/// Replay buffer that samples with exponential recency bias.
///
/// More recent transitions are exponentially more likely to be sampled.
/// This can help in non-stationary environments where old data is stale.
pub struct RecencyBiasedBuffer {
    inner: Vec<ExperienceRecord>,
    capacity: usize,
    write_pos: usize,
    count: usize,
    /// Decay rate: higher = more bias toward recent data. Range (0, 1).
    decay: f64,
}

impl RecencyBiasedBuffer {
    pub fn new(capacity: usize, decay: f64) -> Self {
        assert!((0.0..1.0).contains(&decay), "decay must be in (0, 1)");
        Self {
            inner: Vec::with_capacity(capacity),
            capacity,
            write_pos: 0,
            count: 0,
            decay,
        }
    }

    pub fn push(&mut self, record: ExperienceRecord) {
        if self.inner.len() < self.capacity {
            self.inner.push(record);
        } else {
            self.inner[self.write_pos] = record;
        }
        self.write_pos = (self.write_pos + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }

    /// Sample with exponential recency bias.
    ///
    /// The probability of sampling index `i` (where `i=0` is the most recent)
    /// is proportional to `decay^i`.
    pub fn sample(&self, batch_size: usize, seed: u64) -> Result<Vec<&ExperienceRecord>, RloxError> {
        if batch_size > self.count {
            return Err(RloxError::BufferError(format!(
                "batch_size {batch_size} > buffer len {}",
                self.count
            )));
        }

        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Build cumulative weights
        let weights: Vec<f64> = (0..self.count)
            .map(|i| self.decay.powi(i as i32))
            .collect();
        let total: f64 = weights.iter().sum();

        let mut samples = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let r: f64 = rng.random::<f64>() * total;
            let mut cumsum = 0.0;
            let mut selected = 0;
            for (i, &w) in weights.iter().enumerate() {
                cumsum += w;
                if cumsum >= r {
                    selected = i;
                    break;
                }
            }
            // Convert recency index to ring buffer index
            let buf_idx = (self.write_pos + self.capacity - 1 - selected) % self.capacity;
            samples.push(&self.inner[buf_idx]);
        }

        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(reward: f32) -> ExperienceRecord {
        ExperienceRecord {
            obs: vec![0.0; 4],
            action: vec![0.0],
            reward,
            terminated: false,
            truncated: false,
        }
    }

    #[test]
    fn recency_buffer_respects_capacity() {
        let mut buf = RecencyBiasedBuffer::new(10, 0.9);
        for i in 0..20 {
            buf.push(make_record(i as f32));
        }
        assert_eq!(buf.len(), 10);
    }

    #[test]
    fn recency_buffer_biases_toward_recent() {
        let mut buf = RecencyBiasedBuffer::new(100, 0.5);
        for i in 0..100 {
            buf.push(make_record(i as f32));
        }
        // Sample many times and check that average reward is closer to 99 than 50
        let samples = buf.sample(1000, 42).unwrap();
        let mean: f32 = samples.iter().map(|s| s.reward).sum::<f32>() / 1000.0;
        assert!(mean > 70.0, "mean reward should be biased high, got {mean}");
    }
}
```

### 3.3 Custom Advantage Computation

The built-in `compute_gae` in `crates/rlox-core/src/training/gae.rs` follows a
simple pattern: take slices, iterate, return vectors. Here is an example of
implementing V-trace advantages (used in IMPALA) in the same style:

```rust
/// Compute V-trace advantages for off-policy correction.
///
/// V-trace clips importance weights to reduce variance from stale data,
/// while maintaining an unbiased fixed-point.
///
/// # Arguments
/// - `rewards`: per-step rewards [T]
/// - `values`: per-step value estimates [T]
/// - `dones`: done flags (0.0 or 1.0) [T]
/// - `log_probs_policy`: log-probs under the current policy [T]
/// - `log_probs_behavior`: log-probs under the behavior policy [T]
/// - `last_value`: bootstrap value V(T)
/// - `gamma`: discount factor
/// - `rho_bar`: clipping threshold for importance weights (default 1.0)
/// - `c_bar`: clipping threshold for trace coefficients (default 1.0)
pub fn compute_vtrace(
    rewards: &[f64],
    values: &[f64],
    dones: &[f64],
    log_probs_policy: &[f64],
    log_probs_behavior: &[f64],
    last_value: f64,
    gamma: f64,
    rho_bar: f64,
    c_bar: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = rewards.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    // Compute clipped importance weights
    let rhos: Vec<f64> = log_probs_policy
        .iter()
        .zip(log_probs_behavior.iter())
        .map(|(&lp, &lb)| (lp - lb).exp().min(rho_bar))
        .collect();

    let cs: Vec<f64> = log_probs_policy
        .iter()
        .zip(log_probs_behavior.iter())
        .map(|(&lp, &lb)| (lp - lb).exp().min(c_bar))
        .collect();

    // Backward pass
    let mut advantages = vec![0.0; n];
    let mut last_v_correction = 0.0;

    for t in (0..n).rev() {
        let next_non_terminal = 1.0 - dones[t];
        let next_value = if t == n - 1 { last_value } else { values[t + 1] };

        let delta = rhos[t] * (rewards[t] + gamma * next_value * next_non_terminal - values[t]);
        last_v_correction = delta + gamma * next_non_terminal * cs[t] * last_v_correction;
        advantages[t] = last_v_correction;
    }

    let returns: Vec<f64> = advantages
        .iter()
        .zip(values.iter())
        .map(|(a, v)| a + v)
        .collect();

    (advantages, returns)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vtrace_on_policy_matches_gae_lambda_one() {
        // When policy == behavior, rho=1, c=1, so V-trace
        // with rho_bar=inf, c_bar=inf degenerates to GAE(lambda=1)
        let rewards = &[1.0, 1.0, 1.0];
        let values = &[0.0, 0.0, 0.0];
        let dones = &[0.0, 0.0, 1.0];
        let log_probs = &[-1.0, -1.0, -1.0]; // same policy and behavior

        let (adv, _) = compute_vtrace(
            rewards, values, dones,
            log_probs, log_probs,
            0.0, 0.99,
            f64::INFINITY, f64::INFINITY,
        );

        // Compare with GAE(lambda=1)
        let (gae_adv, _) = crate::training::gae::compute_gae(
            rewards, values, dones, 0.0, 0.99, 1.0,
        );

        for (a, b) in adv.iter().zip(gae_adv.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn vtrace_clips_large_importance_weights() {
        let rewards = &[1.0];
        let values = &[0.0];
        let dones = &[1.0];
        // policy much more likely than behavior -> large ratio
        let log_p = &[0.0];   // prob = 1.0
        let log_b = &[-10.0]; // prob ~ 4.5e-5

        let (adv_clipped, _) = compute_vtrace(
            rewards, values, dones, log_p, log_b,
            0.0, 0.99, 1.0, 1.0, // rho_bar=1
        );

        let (adv_unclipped, _) = compute_vtrace(
            rewards, values, dones, log_p, log_b,
            0.0, 0.99, f64::INFINITY, f64::INFINITY,
        );

        // Clipped advantage should be smaller
        assert!(adv_clipped[0] < adv_unclipped[0]);
        // Clipped advantage = min(rho, 1) * delta = 1.0 * 1.0 = 1.0
        assert!((adv_clipped[0] - 1.0).abs() < 1e-10);
    }
}
```

The pattern to follow:
1. Accept `&[f64]` slices for all numeric inputs (matches the `compute_gae` contract)
2. Return `(Vec<f64>, Vec<f64>)` for (advantages, returns)
3. Iterate backwards for temporal-difference style computation
4. Test edge cases: empty input, single step, episode boundaries

---

## Part 4: Integration Patterns

### 4.1 Mixing Rust and Python

rlox follows the **Polars architecture**: Rust data plane, Python control plane.
Use this decision framework:

| Component | Language | Why |
|-----------|----------|-----|
| Environment stepping | Rust | Called millions of times per training run |
| Buffer push/sample | Rust | Hot inner loop, cache-friendly flat arrays |
| GAE / advantage computation | Rust | Pure numeric, no Python object overhead |
| Policy forward/backward | Python (PyTorch) | Needs autograd, GPU, ecosystem |
| Training loop orchestration | Python | Flexible, easy to iterate |
| Reward functions | Python (usually) | Often calls external models, APIs, or parsers |
| Reward functions (hot path) | Rust | When called > 1M times (e.g., per-token scoring) |

**The PyO3 boundary pattern:**

When crossing the Rust/Python boundary, data passes through numpy arrays (zero-copy
when possible). The key types at the boundary are:

```
Python side          PyO3 boundary         Rust side
-----------------------------------------------------------
np.ndarray    <-->   PyReadonlyArrayN  --> &[f64] or &[f32]
np.ndarray    <-->   PyArray1           <- Vec<f64>
list[float]   <-->   Vec<f64>          --> Vec<f64>
int           <-->   u64 / usize       --> u64 / usize
```

If a reward function is called once per training step on a batch, Python is fine.
If it is called per-token or per-environment-step in a tight loop, consider
implementing it in Rust and exposing it via PyO3.

### 4.2 Testing Custom Components

**Testing reward functions (Python):**

```python
import numpy as np

def test_angle_penalty_reward():
    """Reward function should reduce reward when angle is large."""
    obs = np.array([
        [0.0, 0.0, 0.0, 0.0],     # theta = 0 (upright)
        [0.0, 0.0, 0.2, 0.0],     # theta = 0.2 rad
    ])
    actions = np.array([0, 1])
    raw_rewards = np.array([1.0, 1.0])

    shaped = angle_penalty_reward(obs, actions, raw_rewards)

    assert shaped[0] == 1.0, "No penalty when theta=0"
    assert shaped[1] < 1.0, "Should penalise large angle"
    assert shaped[1] > 0.0, "Penalty should not overwhelm reward"
    np.testing.assert_allclose(shaped[1], 1.0 - 0.5 * 0.2**2, rtol=1e-6)

def test_reward_fn_preserves_shape():
    """reward_fn must return the same shape as raw_rewards."""
    obs = np.random.randn(16, 4)
    actions = np.random.randint(0, 2, size=16)
    raw_rewards = np.ones(16)

    shaped = angle_penalty_reward(obs, actions, raw_rewards)
    assert shaped.shape == raw_rewards.shape
```

**Testing custom training loops:**

Test convergence on a simple environment with a known solution:

```python
def test_reinforce_learns_cartpole():
    """REINFORCE should achieve > 200 mean reward on CartPole within 100k steps."""
    reinforce = REINFORCE("CartPole-v1", n_envs=8, seed=42)
    metrics = reinforce.train(total_timesteps=100_000)
    assert metrics["mean_reward"] > 200, (
        f"Expected > 200 reward, got {metrics['mean_reward']:.1f}"
    )
```

**Testing Rust environments:**

Follow the existing test patterns in `crates/rlox-core/src/env/builtins.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_reset_produces_valid_obs() {
        let env = GridWorld::new(5, 5, Some(42));
        let obs = env.obs();
        assert_eq!(obs.as_slice().len(), 2);
        for &v in obs.as_slice() {
            assert!((0.0..=1.0).contains(&v), "obs should be normalised, got {v}");
        }
    }

    #[test]
    fn env_step_after_done_errors() {
        let mut env = GridWorld::new(2, 2, Some(0));
        env.pos = (0, 0);
        env.done = false;
        let _ = env.step(&Action::Discrete(2)).unwrap(); // down
        let t = env.step(&Action::Discrete(1)).unwrap();  // right -> goal
        assert!(t.terminated);
        // Now stepping should error
        assert!(env.step(&Action::Discrete(0)).is_err());
    }

    #[test]
    fn env_seeded_determinism() {
        let run = |seed| {
            let mut env = GridWorld::new(5, 5, Some(seed));
            let obs = env.obs().into_inner();
            let t = env.step(&Action::Discrete(1)).unwrap();
            (obs, t.obs.into_inner())
        };
        assert_eq!(run(42), run(42));
        assert_ne!(run(42), run(99));
    }
}
```

**Testing advantage computation:**

Use the property that `returns[t] == advantages[t] + values[t]` (this holds for
any correct advantage estimator):

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn vtrace_returns_equal_advantages_plus_values(n in 1..200usize) {
        let rewards: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
        let values: Vec<f64> = (0..n).map(|i| (i as f64) * 0.05).collect();
        let dones: Vec<f64> = (0..n).map(|i| if i % 10 == 9 { 1.0 } else { 0.0 }).collect();
        let log_probs: Vec<f64> = vec![-1.0; n]; // on-policy

        let (adv, ret) = compute_vtrace(
            &rewards, &values, &dones,
            &log_probs, &log_probs,
            0.0, 0.99, 1.0, 1.0,
        );

        for i in 0..n {
            let diff = (ret[i] - (adv[i] + values[i])).abs();
            prop_assert!(diff < 1e-10, "mismatch at {i}: ret={}, adv+val={}",
                ret[i], adv[i] + values[i]);
        }
    }
}
```

---

## Source code reference

| Component | Path |
|-----------|------|
| `RolloutCollector` | `python/rlox/collectors.py` |
| `PPOLoss` | `python/rlox/losses.py` |
| `RolloutBatch` | `python/rlox/batch.py` |
| `DiscretePolicy` / `ContinuousPolicy` | `python/rlox/policies.py` |
| `PPO` algorithm | `python/rlox/algorithms/ppo.py` |
| `GRPO` algorithm | `python/rlox/algorithms/grpo.py` |
| `DPO` algorithm | `python/rlox/algorithms/dpo.py` |
| `GymVecEnv` | `python/rlox/gym_vec_env.py` |
| `MultiObjectiveReward` / `EnsembleRewardModel` | `python/rlox/llm/reward_models.py` |
| Callbacks | `python/rlox/callbacks.py` |
| Config dataclasses | `python/rlox/config.py` |
| `RLEnv` trait | `crates/rlox-core/src/env/mod.rs` |
| `BatchSteppable` trait | `crates/rlox-core/src/env/batch.rs` |
| `CartPole` (reference env) | `crates/rlox-core/src/env/builtins.rs` |
| `VecEnv` | `crates/rlox-core/src/env/parallel.rs` |
| `compute_gae` | `crates/rlox-core/src/training/gae.rs` |
| `ReplayBuffer` | `crates/rlox-core/src/buffer/ringbuf.rs` |
| `ExperienceTable` | `crates/rlox-core/src/buffer/columnar.rs` |
| LLM ops (`compute_group_advantages`, `compute_token_kl`) | `crates/rlox-core/src/llm/ops.rs` |
| `Pipeline` / Rust `RolloutBatch` | `crates/rlox-core/src/pipeline/channel.rs` |
| NN traits (`ActorCritic`, `QFunction`) | `crates/rlox-nn/src/traits.rs` |
| Action/Observation types | `crates/rlox-core/src/env/spaces.rs` |
