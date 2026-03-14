# rlox Evaluation Plan: Latency & Convergence Benchmarks

## 1. Motivation

Phase 6 established rlox's **infrastructure-level speedups** (GAE 140×, buffer 148×, E2E 3.9–53×) via microbenchmarks. What's missing is the question that matters to practitioners: **does rlox train RL agents faster wall-clock and does it converge to the same (or better) final performance?**

This plan defines a rigorous comparative evaluation of rlox against **Stable-Baselines3 (SB3)** — the de facto standard for single-machine RL — across standard environments and algorithms, measuring both **latency** (wall-clock time, throughput) and **convergence** (sample efficiency, cumulative reward).

### Why SB3 as the primary baseline?

| Criterion | SB3 | TorchRL | CleanRL |
|-----------|-----|---------|---------|
| Established baseline in papers | ✓ | Partial | ✓ |
| Apples-to-apples algorithm parity | ✓ (PPO, SAC, TD3, DQN, A2C) | Different API patterns | Single-file, no library |
| Community adoption / citations | ~5K GitHub stars, 3K+ papers | ~3K stars, newer | ~4K stars, reference impl |
| Hyperparameter defaults documented | ✓ (rl-zoo3) | Partial | ✓ |

**Decision**: SB3 primary, CleanRL secondary reference for hyperparameter validation.

---

## 2. Evaluation Axes

### Axis A — Wall-Clock Training Efficiency
- **Metric**: Wall-clock time to reach reward threshold T (seconds)
- **Metric**: Steps per second (SPS) during training
- **Metric**: Total wall-clock for N total timesteps

### Axis B — Sample Efficiency & Convergence
- **Metric**: Episodic return vs. environment steps (learning curve)
- **Metric**: Steps to threshold (number of env interactions to reach reward T)
- **Metric**: Final performance (mean return over last 100 episodes)

### Axis C — Statistical Rigor (following Agarwal et al., NeurIPS 2021)
- **Metric**: Interquartile Mean (IQM) across seeds — robust to outliers
- **Metric**: Bootstrap 95% confidence intervals (10K resamples)
- **Metric**: Performance profiles — P(algorithm reaches τ fraction of reference score)
- **Metric**: Probability of improvement — P(rlox > SB3) via stratified bootstrap

---

## 3. Environment × Algorithm Matrix

### 3.1 Classic Control (fast iteration, sanity check)

| Environment | Algorithm | Threshold T | Max Steps | Rationale |
|-------------|-----------|-------------|-----------|-----------|
| CartPole-v1 | PPO | 475 (avg 100 ep) | 100K | Canonical on-policy sanity check |
| CartPole-v1 | DQN | 475 | 100K | Canonical off-policy sanity check |
| CartPole-v1 | A2C | 475 | 100K | Baseline on-policy |
| Acrobot-v1 | PPO | -100 | 500K | Sparse reward, harder exploration |
| MountainCar-v0 | DQN | -110 | 500K | Hard exploration, delayed reward |
| Pendulum-v1 | SAC | -200 | 50K | Continuous control entry point |
| Pendulum-v1 | TD3 | -200 | 50K | Continuous control entry point |

### 3.2 MuJoCo Continuous Control (the standard benchmark)

| Environment | Algorithm | Threshold T | Max Steps | Rationale |
|-------------|-----------|-------------|-----------|-----------|
| HalfCheetah-v4 | PPO | 3000 | 1M | Standard locomotion |
| HalfCheetah-v4 | SAC | 8000 | 1M | Off-policy efficiency |
| HalfCheetah-v4 | TD3 | 8000 | 1M | Deterministic policy baseline |
| Hopper-v4 | PPO | 2000 | 1M | Contact-rich, termination |
| Hopper-v4 | SAC | 2500 | 1M | Sample efficiency comparison |
| Walker2d-v4 | PPO | 2500 | 2M | Bipedal stability |
| Walker2d-v4 | SAC | 3000 | 2M | Off-policy on harder task |
| Ant-v4 | PPO | 3000 | 3M | High-dimensional (111-dim obs) |
| Humanoid-v4 | SAC | 4000 | 5M | Hardest standard benchmark |

### 3.3 Atari (optional, stretch goal)

| Environment | Algorithm | Threshold T | Max Steps | Rationale |
|-------------|-----------|-------------|-----------|-----------|
| BreakoutNoFrameskip-v4 | DQN | 30 | 10M | Visual DQN canonical benchmark |
| PongNoFrameskip-v4 | PPO | 18 | 10M | Visual PPO canonical benchmark |

> Atari requires frame-stacking wrappers, CNN policies, and significant GPU time. Defer to Phase 2 of evaluation unless resources permit.

---

## 4. Hyperparameter Protocol

### Principle: Use SB3's tuned defaults (rl-zoo3) for both frameworks.

This ensures we measure **framework overhead**, not hyperparameter tuning skill. Both frameworks run identical hyperparameters.

### PPO Defaults (MuJoCo)
```yaml
n_envs: 8
n_steps: 2048
batch_size: 64
n_epochs: 10
learning_rate: 3e-4
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.0
vf_coef: 0.5
max_grad_norm: 0.5
normalize_obs: true
normalize_reward: true
policy: MlpPolicy [64, 64]
```

### SAC Defaults (MuJoCo)
```yaml
buffer_size: 1_000_000
batch_size: 256
learning_rate: 3e-4
gamma: 0.99
tau: 0.005
learning_starts: 10_000
train_freq: 1
gradient_steps: 1
ent_coef: "auto"
policy: MlpPolicy [256, 256]
```

### TD3 Defaults (MuJoCo)
```yaml
buffer_size: 1_000_000
batch_size: 256  # was 100 in original, SB3 uses 256
learning_rate: 1e-3
gamma: 0.99
tau: 0.005
policy_delay: 2
target_policy_noise: 0.2
target_noise_clip: 0.5
learning_starts: 10_000
train_freq: 1
policy: MlpPolicy [400, 300]
```

### DQN Defaults (CartPole)
```yaml
buffer_size: 100_000
batch_size: 64
learning_rate: 1e-4
gamma: 0.99
exploration_fraction: 0.1
exploration_final_eps: 0.05
target_update_interval: 10_000
train_freq: 4
gradient_steps: 1
policy: MlpPolicy [64, 64]
```

---

## 5. Experimental Protocol

### 5.1 Seeds & Repetitions

- **10 independent seeds** per (env, algorithm, framework) tuple
- Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` — shared across frameworks for reproducibility
- Each seed controls: environment reset, weight initialization, action sampling, buffer sampling

### 5.2 Evaluation Schedule

- **Evaluate every `eval_freq` steps** (not episodes)
  - Classic Control: every 1K steps
  - MuJoCo: every 10K steps
- **Evaluation episodes**: 10 deterministic episodes per checkpoint
- **Record**: mean return, std return, episode length, wall-clock timestamp

### 5.3 Logging

Each run produces a structured JSON log:
```json
{
  "framework": "rlox",
  "algorithm": "PPO",
  "environment": "HalfCheetah-v4",
  "seed": 42,
  "hyperparameters": { ... },
  "hardware": { "cpu": "...", "gpu": "...", "ram_gb": ... },
  "evaluations": [
    {"step": 10000, "wall_clock_s": 12.3, "mean_return": 450.2, "std_return": 35.1, "ep_length": 1000.0, "sps": 8125.0}
  ],
  "training_metrics": {
    "total_wall_clock_s": 320.5,
    "total_steps": 1000000,
    "mean_sps": 3120.0,
    "peak_memory_mb": 512
  }
}
```

### 5.4 Hardware Control

- **Same machine**, sequential execution (no resource contention)
- Report: CPU model, GPU model, RAM, PyTorch version, CUDA version
- **Warmup run** (1 seed, discarded) before measured runs to stabilize JIT/caching

---

## 6. Analysis Plan

### 6.1 Learning Curves

For each (env, algorithm):
- Plot mean ± std return vs. environment steps (sample efficiency)
- Plot mean ± std return vs. wall-clock time (wall-clock efficiency)
- Shaded regions: IQM ± bootstrapped 95% CI across 10 seeds
- Both rlox and SB3 on same axes

### 6.2 Aggregate Metrics Table

| Env | Algo | Framework | IQM Return | Steps to T | Wall-clock to T | SPS |
|-----|------|-----------|------------|------------|----------------|-----|
| ... | ...  | rlox      | ...        | ...        | ...            | ... |
| ... | ...  | SB3       | ...        | ...        | ...            | ... |

### 6.3 Performance Profiles (Agarwal et al.)

- Normalize scores: `score_normalized = (score - random) / (expert - random)`
- Plot fraction of runs achieving ≥ τ of reference score, τ ∈ [0, 1]
- Aggregate across all environments per algorithm
- Higher area under curve = more reliable algorithm

### 6.4 Probability of Improvement

For each (env, algorithm):
- P(rlox final return > SB3 final return) via paired bootstrap
- P(rlox wall-clock to T < SB3 wall-clock to T) via paired bootstrap
- Report with 95% CI

### 6.5 Speedup Decomposition

Break down wall-clock differences into components:
1. **Environment step time** (already measured in Phase 6)
2. **Buffer operations** (already measured)
3. **GAE / advantage computation** (already measured)
4. **Neural network forward/backward** (should be identical — same PyTorch)
5. **Python overhead / data transfer**

This identifies where rlox's Rust core delivers the most value and where the bottleneck shifts to PyTorch.

---

## 7. Implementation Structure

```
benchmarks/
  convergence/
    run_experiment.py       # Main entry: runs one (framework, algo, env, seed)
    configs/
      ppo_cartpole.yaml
      ppo_halfcheetah.yaml
      sac_halfcheetah.yaml
      ...
    sb3_runner.py           # SB3 training + evaluation harness
    rlox_runner.py          # rlox training + evaluation harness
    analyze.py              # Load JSON logs → tables, plots, profiles
    plot_learning_curves.py # Matplotlib/seaborn learning curves
    plot_profiles.py        # Performance profiles (Agarwal et al.)
    requirements.txt        # stable-baselines3, gymnasium[mujoco], rliable
```

### Key design decisions:

1. **Separate runner files** per framework — avoids import contamination, makes profiling clean
2. **YAML configs** — one file per (algo, env) pair, shared between frameworks
3. **`rliable`** library for statistical analysis (Agarwal et al. reference implementation)
4. **JSON logs** — machine-readable, version-controlled results

---

## 8. Expected Outcomes & Hypotheses

### H1: rlox achieves higher SPS than SB3
- **Expected**: 2–5× SPS improvement on Classic Control (env-stepping dominated)
- **Expected**: 1.3–2× SPS improvement on MuJoCo (NN forward/backward dominates)
- **Rationale**: rlox Rust core eliminates Python overhead in env stepping, GAE, buffer ops; but PyTorch NN operations are identical

### H2: rlox matches SB3 sample efficiency
- **Expected**: Identical learning curves vs. environment steps (same algorithm, same hyperparameters)
- **Any divergence indicates a bug** in rlox's algorithm implementation
- This is a correctness test, not a performance test

### H3: rlox reaches threshold T faster wall-clock
- **Expected**: Proportional to SPS improvement
- **Key insight**: The wall-clock advantage is greatest on tasks where env stepping + data pipeline is the bottleneck (many envs, short episodes, small policies)

### H4: Advantage diminishes on GPU-heavy workloads
- **Expected**: On Humanoid-v4 with large policies, NN compute dominates and rlox advantage shrinks toward 1.1–1.3×
- This motivates future work on Rust-native NN inference

---

## 9. Execution Phases

### Phase E1: Classic Control (1–2 days)
- CartPole + Pendulum + Acrobot + MountainCar
- PPO, DQN, A2C, SAC, TD3
- Quick iteration, validate framework correctness
- **Gate**: rlox and SB3 reach same final reward within 1 std

### Phase E2: MuJoCo Core (3–5 days)
- HalfCheetah, Hopper, Walker2d
- PPO, SAC, TD3
- 10 seeds × 3 envs × 3 algos × 2 frameworks = 180 runs
- **Gate**: Learning curves overlap on sample efficiency axis

### Phase E3: MuJoCo Extended + Analysis (2–3 days)
- Ant, Humanoid
- Full statistical analysis, performance profiles
- Generate publication-quality figures
- **Gate**: All hypotheses tested, results documented

### Phase E4 (Optional): Atari
- BreakoutNoFrameskip-v4, PongNoFrameskip-v4
- DQN, PPO with CNN policies
- Requires significant GPU time

---

## 10. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| rlox algorithm has a bug → divergent learning curves | Phase E1 gate: must match SB3 within 1 std on CartPole before proceeding |
| MuJoCo installation issues | Use `gymnasium[mujoco]` (built-in MuJoCo 3.x, no license needed) |
| Run time too long (180+ experiments) | Start with 5 seeds, expand to 10 only for final results |
| SB3 hyperparameter mismatch | Use `rl-zoo3` defaults verbatim, document any deviations |
| Hardware variance between runs | Sequential execution, report hardware, discard warmup run |
| rlox VecEnv wrapper adds overhead vs SB3's native SubprocVecEnv | Benchmark both rlox native CartPole and GymEnv-wrapped MuJoCo separately |

---

## 11. Deliverables

1. **Results JSON archive** — all raw experiment logs
2. **Learning curve plots** — per (env, algo), both vs steps and vs wall-clock
3. **Aggregate table** — IQM, steps-to-threshold, SPS, wall-clock-to-threshold
4. **Performance profiles** — aggregate reliability across environments
5. **Speedup decomposition** — where rlox wins and where it's equal
6. **Summary report** — 2-page findings document suitable for README or blog post
