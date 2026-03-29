# Buffers & Primitives API

## Rust-Accelerated Buffers

These classes are implemented in Rust and exposed via PyO3. All sampling operations release the GIL.

### ReplayBuffer

Uniform ring buffer with O(1) push and O(batch_size) sample.

```python
import rlox

buf = rlox.ReplayBuffer(capacity=100_000, obs_dim=4, act_dim=1)
buf.push(obs, action, reward, terminated, truncated, next_obs)
batch = buf.sample(batch_size=256, seed=42)
# Returns dict: obs, next_obs, actions, rewards, terminated, truncated
```

### PrioritizedReplayBuffer

SumTree-based priority sampling with O(log N) sample and O(1) min-priority.

```python
pbuf = rlox.PrioritizedReplayBuffer(
    capacity=100_000, obs_dim=4, act_dim=1, alpha=0.6, beta=0.4
)
pbuf.push(obs, action, reward, terminated, truncated, next_obs, priority=1.0)
batch = pbuf.sample(batch_size=256, seed=42)
# Additional keys: weights (importance sampling), indices (for priority update)
pbuf.update_priorities(batch["indices"], new_td_errors)
pbuf.set_beta(0.7)  # Anneal toward 1.0
```

### OfflineDatasetBuffer

Read-only buffer for offline RL. Loaded once from static datasets.

```python
buf = rlox.OfflineDatasetBuffer(
    obs.ravel(), next_obs.ravel(), actions.ravel(),
    rewards, terminated, truncated,
    normalize=True,  # Compute obs normalization stats
)

# Uniform transition sampling (for TD3+BC, IQL, CQL, BC)
batch = buf.sample(batch_size=256, seed=42)

# Trajectory subsequence sampling (for Decision Transformer)
traj = buf.sample_trajectories(batch_size=8, seq_len=20, seed=42)

# Dataset statistics
print(buf.stats())  # n_transitions, n_episodes, mean_return, ...
print(buf.n_episodes())
```

### MmapReplayBuffer

Hot/cold tiered buffer with memory-mapped disk spillover.

```python
buf = rlox.MmapReplayBuffer(
    hot_capacity=10_000, total_capacity=1_000_000,
    obs_dim=84*84*4, act_dim=1,
    cold_path="/tmp/replay_cold.bin",
)
# Same push/sample API as ReplayBuffer
buf.close()  # Cleans up cold file
```

### CandleCollector

Rust-native rollout collector using Candle for policy inference. Zero Python overhead during collection.

```python
collector = rlox.CandleCollector(
    env_id="CartPole-v1", n_envs=16,
    obs_dim=4, n_actions=2, n_steps=128,
    hidden=64, seed=42,
)

# Receive completed rollout batch (blocks until ready)
batch = collector.recv()
# Returns dict: observations, actions, rewards, dones,
#               log_probs, values, advantages, returns

# Sync weights from PyTorch after training
collector.sync_weights(flat_f32_params)

# Extract Candle weights for PyTorch init
weights = collector.get_weights()

collector.stop()
```

## Computation Primitives

### GAE

```python
# Single-trajectory GAE
advantages, returns = rlox.compute_gae(rewards, values, dones, last_value, gamma, lam)

# Batched GAE (Rayon-parallel, all envs in one call)
adv, ret = rlox.compute_gae_batched(rewards_flat, values_flat, dones_flat, last_vals, n_steps, gamma, lam)

# f32 variant (faster for 64+ envs)
adv_f32, ret_f32 = rlox.compute_gae_batched_f32(...)
```

### V-trace

```python
vs, pg_advantages = rlox.compute_vtrace(log_rhos, rewards, values, dones, bootstrap_value, gamma, rho_bar, c_bar)
```

### LLM Operations

```python
# GRPO group advantages
advantages = rlox.compute_batch_group_advantages(rewards, group_size)

# Token-level KL divergence
kl = rlox.compute_batch_token_kl(log_p, log_q, seq_len)
kl_schulman = rlox.compute_batch_token_kl_schulman(log_p, log_q, seq_len)

# Sequence packing
packed, masks, positions = rlox.pack_sequences(lengths, max_len)
```
