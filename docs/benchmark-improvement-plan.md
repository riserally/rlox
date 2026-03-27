# Benchmark Improvement Plan — Publication Readiness

## Context

AI researcher review identified systematic methodological issues that make current benchmark results unsuitable for publication. This plan addresses all gaps, ordered by priority.

---

## P0 — Blocking for Publication

### P0.1: Fix normalization asymmetry between rlox and SB3

**Problem:** rlox implements `_RunningMeanStd` for obs/reward normalization when configs specify `normalize_obs: true`, but SB3 runner ignores these flags entirely. This creates an unfair comparison — rlox trains with normalization, SB3 trains without.

**Impact:** Explains PPO Hopper gap (449 vs 3,494). Invalidates all MuJoCo PPO comparisons where normalization flags are set.

**Affected configs:**
- `ppo_halfcheetah.yaml`: `normalize_obs: true, normalize_reward: true`
- `ppo_hopper.yaml`: `normalize_obs: true, normalize_reward: true`
- `ppo_walker2d.yaml`: likely same
- `ppo_ant.yaml`: likely same

**Fix — Option A (preferred): Add VecNormalize to SB3 runner**

File: `benchmarks/convergence/sb3_runner.py`

When config specifies `normalize_obs` or `normalize_reward`, wrap the SB3 environment:
```python
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

env = DummyVecEnv([lambda: gym.make(env_id)])
if hp.get("normalize_obs", False) or hp.get("normalize_reward", False):
    env = VecNormalize(
        env,
        norm_obs=hp.get("normalize_obs", False),
        norm_reward=hp.get("normalize_reward", False),
    )
```

Update the SB3 evaluation to use the same normalization stats (SB3's `VecNormalize` tracks this automatically via `training=False` mode).

**Fix — Option B: Remove normalization from configs**

Remove `normalize_obs` and `normalize_reward` from all configs. Both frameworks train on raw observations. This is simpler but may hurt PPO MuJoCo convergence for both frameworks.

**Verification:** After fix, PPO Hopper should converge to similar levels for both frameworks (both ~1,000-3,000+ range).

**Effort:** Medium (Option A), Trivial (Option B)

---

### P0.2: Fix hidden size handling in off-policy runners

**Problem:** The refactored rlox runner passes `hidden=policy_cfg["hidden_sizes"][0]` to SAC/TD3/DQN, using only the first element. TD3 HalfCheetah config specifies `[400, 300]` — rlox creates `[400, 400]` while SB3 creates `[400, 300]`.

**Affected files:**
- `benchmarks/convergence/rlox_runner.py` lines ~440, ~475, ~511

**Current code:**
```python
hidden = policy_cfg.get("hidden_sizes", [256, 256])[0]
sac = SAC(env_id=env_id, hidden=hidden, ...)
```

**Fix:** The standalone algorithm classes (SAC, TD3, DQN) accept `hidden: int` which creates two layers of that width. To match SB3's `net_arch`, either:

1. Change algorithm constructors to accept `hidden_sizes: list[int]` and build networks accordingly
2. Or pass `hidden=hidden_sizes[0]` and document this as a known approximation (not ideal)

**Recommended:** Option 1 — update `QNetwork`, `SquashedGaussianPolicy`, `DeterministicPolicy`, `SimpleQNetwork` to accept `hidden_sizes: list[int]`. This is a larger refactor but ensures architectural parity.

**Interim fix:** For configs where `hidden_sizes = [X, X]` (both layers same), the current code is correct. Only TD3 HalfCheetah `[400, 300]` is affected. Change that config to `[400, 400]` or `[256, 256]` for now.

**Effort:** Low (interim), High (full refactor)

---

### P0.3: Match vectorized environment type

**Problem:** SB3 uses `SubprocVecEnv` (parallel subprocesses) while rlox uses `SyncVectorEnv` (sequential). SPS comparisons are unfair.

**File:** `benchmarks/convergence/sb3_runner.py`

**Current SB3 code:**
```python
env = make_vec_env(env_id, n_envs=n_envs, seed=seed)  # creates SubprocVecEnv
```

**Fix:** Use `DummyVecEnv` for SB3 to match rlox's sequential stepping:
```python
from stable_baselines3.common.vec_env import DummyVecEnv
env = DummyVecEnv([lambda i=i: gym.make(env_id) for i in range(n_envs)])
```

Or document both and report SPS for each configuration.

**Effort:** Trivial

---

### P0.4: Run multi-seed MuJoCo experiments (10 seeds)

**Problem:** MuJoCo results are single-seed. Not publishable per Agarwal et al. (2021).

**Fix:** Update the GCP startup script or run command to use `--seeds 0-9`:
```bash
docker compose run --rm benchmark-convergence python run_experiment.py --phase e2 --seeds 0-9
```

The `run_experiment.py` already supports `--seeds "0-9"` syntax. Just need to update the Docker compose entrypoint or startup script.

**File to update:** `scripts/gcp-convergence-bench.sh` or `docker-compose.yml` benchmark-convergence command.

**Effort:** Trivial (infrastructure exists)

**Compute estimate:**
- Phase E1 (7 configs): ~30 min per config x 10 seeds x 2 frameworks = ~70 hours
- Phase E2 (7 configs): ~3 hours per config x 10 seeds x 2 frameworks = ~420 hours
- Total: ~490 compute-hours on e2-standard-8 (~$20 at $0.04/hr)
- Can parallelize across multiple VMs

---

## P1 — Required for Paper

### P1.1: Increase evaluation episodes for MuJoCo

**Problem:** `eval_episodes: 10` in all configs. MuJoCo environments have high variance.

**Fix:** Update YAML configs:
```yaml
# Classic Control (low variance) — keep 10
eval_episodes: 10

# MuJoCo (high variance) — increase to 30
eval_episodes: 30
```

**Files:** All `benchmarks/convergence/configs/ppo_halfcheetah.yaml`, `sac_halfcheetah.yaml`, etc.

**Effort:** Trivial

---

### P1.2: Add Agarwal et al. aggregate metrics

**Problem:** Current analysis uses IQM and bootstrap CIs per-task, but missing aggregate visualizations.

**Required visualizations:**
1. **IQM bar chart with 95% CI** — one bar per framework, aggregated across all tasks
2. **Performance profile** — probability of achieving X% of reference score, across all tasks
3. **Probability of improvement** — P(rlox > SB3) with CI
4. **Optimality gap** — how far below optimal each run falls

**Current state:** `plot_profiles.py` implements performance profiles and probability of improvement. Missing: IQM bar chart and optimality gap.

**Fix:** Add to `analyze.py`:
```python
def compute_aggregate_iqm(results_by_task, normalize_scores=True):
    """Compute IQM across all tasks with bootstrap CI."""
    # Normalize each task's scores to [0, 1] using min/max reference
    # Compute IQM across the flattened normalized scores
    # Bootstrap for CI
```

Add to a new `plot_aggregate.py`:
```python
def plot_iqm_barplot(rlox_iqm, sb3_iqm, rlox_ci, sb3_ci):
    """Bar chart with error bars showing IQM +/- 95% CI."""
```

**Effort:** Medium

---

### P1.3: Add wall-clock-to-threshold plots

**Problem:** The most compelling metric for rlox (faster training wall-clock) is not visualized.

**Data available:** Each `EvalRecord` has `wall_clock_s` and `mean_return`. The threshold crossing point can be computed from the learning curve.

**Fix:** Add `plot_wall_clock_efficiency.py`:
```python
def wall_clock_to_threshold(evaluations, threshold):
    """Find first wall_clock_s where mean_return >= threshold."""
    for e in evaluations:
        if e["mean_return"] >= threshold:
            return e["wall_clock_s"]
    return float("inf")
```

Plot as grouped bar chart: rlox vs SB3 wall-clock to reach threshold per environment.

**Thresholds:**
- CartPole: 475
- Acrobot: -100
- HalfCheetah: 5000
- Hopper: 1000
- Pendulum: -200

**Effort:** Low

---

### P1.4: Report SPS excluding eval time

**Problem:** SPS = total_steps / total_wall_clock includes evaluation time. Faster frameworks do more evals per unit time, biasing the SPS downward.

**Fix:** Track training-only wall clock separately. In `_BenchmarkEvalCallback.on_step()`, accumulate eval time and subtract:
```python
eval_start = time.monotonic()
# ... run eval ...
self._total_eval_time += time.monotonic() - eval_start
```

Report both `sps_inclusive` and `sps_training_only`.

**Effort:** Low

---

## P2 — Strengthens Paper

### P2.1: Add Walker2d, Ant experiments

**Problem:** Configs exist but no results. These are standard MuJoCo benchmarks.

**Fix:** Already included in Phase E2/E3. Just need to run the experiments (included in P0.4).

**Effort:** Zero (already in run_experiment.py phase configs)

---

### P2.2: Add LunarLander-v3

**Problem:** Gap between Classic Control (trivial) and MuJoCo (requires separate install). LunarLander bridges this gap.

**Fix:** Add configs:
```yaml
# configs/ppo_lunarlander.yaml
algorithm: PPO
environment: LunarLander-v3
max_steps: 500000
eval_freq: 5000
eval_episodes: 10
hyperparameters:
  n_envs: 8
  n_steps: 2048
  # ... standard PPO defaults
```

Add `dqn_lunarlander.yaml` as well.

**Effort:** Low

---

### P2.3: Consider CleanRL as third baseline

**Problem:** Comparing only against SB3 limits credibility. CleanRL is widely used as a reference.

**Fix:** Add a `cleanrl_runner.py` that runs CleanRL's single-file implementations. CleanRL uses a different training loop structure but produces comparable results.

**Alternative:** Cite CleanRL's published numbers from their benchmark paper rather than re-running. Less convincing but zero effort.

**Effort:** High (full runner), Zero (citation only)

---

### P2.4: Create convergence documentation page

**Problem:** `docs/benchmark/convergence/` has PNG plots but no markdown report synthesizing results.

**Fix:** Create `docs/benchmark/convergence/README.md` with:
- Results table with IQM + CI
- Performance profile plot
- Wall-clock efficiency plot
- Discussion of key findings
- Methodology section (seeds, eval protocol, hardware)

**Effort:** Medium (after results are generated)

---

### P2.5: Investigate DQN throughput

**Problem:** rlox DQN is 0.7x SB3's SPS. Unlike on-policy algorithms, off-policy DQN doesn't benefit from Rust GAE. The training loop overhead may dominate.

**Investigation steps:**
1. Profile rlox DQN with `TimingCallback` to identify bottleneck
2. Check if the `_BenchmarkEvalCallback` adds per-step overhead
3. Compare buffer `push()` and `sample()` times between rlox and SB3
4. Check if SB3 DQN uses C++ extensions for buffer operations

**Potential fixes:**
- Use `sample_into()` for allocation-free sampling
- Batch multiple env steps before updating (train_freq > 1)
- Use VecEnv for parallel collection in DQN

**Effort:** Medium

---

### P2.6: Document known limitations

**Problem:** DQN MountainCar (-200) is a known limitation, not a bug. Needs documentation.

**Fix:** Add to convergence README:
```markdown
### Known Limitations

- **DQN MountainCar**: Both rlox and SB3 fail to consistently solve this environment
  with epsilon-greedy exploration. The sparse reward structure requires directed
  exploration (e.g., curiosity, count-based) which is not implemented.

- **DQN Throughput**: rlox DQN is ~0.7x SB3's SPS because off-policy algorithms
  do not benefit from Rust-accelerated GAE computation. The throughput advantage
  of rlox is concentrated in on-policy algorithms (PPO/A2C: 1.3x-3.1x).
```

**Effort:** Trivial

---

## Implementation Order

### Week 1: Fix comparison fairness (before running experiments)
1. P0.1: Fix normalization asymmetry (choose Option A or B)
2. P0.2: Fix hidden size handling (interim: match configs)
3. P0.3: Match vectorized env type
4. P1.1: Increase eval episodes for MuJoCo configs
5. P1.4: Add training-only SPS tracking

### Week 2: Run experiments
6. P0.4: Run 10-seed experiments on GCP (Phase E1 + E2)
   - Multiple VMs for parallelism
   - Classic Control: ~7 hours per VM
   - MuJoCo: ~42 hours per VM

### Week 3: Analysis and visualization
7. P1.2: Compute Agarwal et al. aggregate metrics
8. P1.3: Create wall-clock-to-threshold plots
9. P2.4: Write convergence documentation page
10. P2.6: Document known limitations

### Week 4: Polish
11. P2.2: Add LunarLander experiments
12. P2.5: Investigate DQN throughput
13. Final review and paper draft

---

## Hardware Requirements

| Phase | Configs | Seeds | Frameworks | Est. Hours | Est. Cost |
|-------|---------|-------|------------|------------|-----------|
| E1 (Classic Control) | 7 | 10 | 2 | 70 | $3 |
| E2 (MuJoCo Core) | 7 | 10 | 2 | 420 | $17 |
| E3 (Complex MuJoCo) | 2 | 10 | 2 | 120 | $5 |
| LunarLander | 2 | 10 | 2 | 20 | $1 |
| **Total** | **18** | **10** | **2** | **630** | **~$26** |

Can be parallelized across 4-8 VMs to finish in 2-3 days.

---

## References

- [1] Agarwal et al., "Deep RL at the Edge of the Statistical Precipice," NeurIPS 2021
- [2] Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993
- [3] Raffin et al., "Stable-Baselines3: Reliable RL Implementations," JMLR 2021
- [4] Huang et al., "CleanRL: High-quality Single-file Implementations of Deep RL Algorithms," JMLR 2022
