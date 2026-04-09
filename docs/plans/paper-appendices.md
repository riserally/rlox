# Appendices — rlox paper draft

**Draft status**: v1 stubs, 2026-04-08. Every appendix promised by
§3 Methods, §4 Results, §5 Discussion, and §2 Related Work is
scaffolded here. Prose that does not depend on the v2 sweep numbers
is fully written; numeric tables are `⏳` placeholders with the
exact structure the final data will populate.

**Pairs with**: all four main-text draft files.
`paper-methods-section.md` forward-references Appendices A, B, C, D.
`paper-results-section.md` forward-references Appendices C, D, E.
`paper-discussion.md` forward-references Appendix A (for the
"publish your translation layer" argument). `paper-related-work.md`
references Appendix F (Humanoid single-seed numbers).

**Target length total**: 8–12 pages including tables. Appendices
are where venue page limits are more forgiving, so err on the side
of completeness.

---

## Appendix A. Hyperparameter translation table

Forward-referenced from: §3.3, §3.7, §3.8, §5.3.2.

This appendix reproduces the complete `multi_seed_runner_sb3.py`
translation layer as of the commit used for the §4 sweeps. The
table documents every hyperparameter name that appears in any
`benchmarks/convergence/configs/*.yaml` file and how the SB3
harness consumes it.

### A.1 Identity passthrough keys

These keys have the same name and semantics in both frameworks. The
translation layer forwards them unchanged to the SB3 algorithm
constructor.

| Key | rlox location | SB3 location | Notes |
|---|---|---|---|
| `learning_rate` | `PPOConfig`, `SACConfig`, `TD3Config`, `DQNConfig`, `A2CConfig` | All algorithm constructors | Constant by default; see `anneal_lr` for schedules |
| `batch_size` | All algo configs | All algorithm constructors | Minibatch size for SGD |
| `buffer_size` | `SACConfig`, `TD3Config`, `DQNConfig` | SAC/TD3/DQN constructors | Off-policy replay buffer capacity |
| `gamma` | All | All | Discount factor |
| `gae_lambda` | `PPOConfig`, `A2CConfig` | PPO, A2C | GAE parameter λ |
| `ent_coef` | `PPOConfig`, `A2CConfig`, `SACConfig` | PPO, A2C, SAC | Entropy bonus coefficient; `'auto'` in SAC |
| `vf_coef` | `PPOConfig`, `A2CConfig` | PPO, A2C | Value loss coefficient |
| `max_grad_norm` | `PPOConfig`, `A2CConfig`, `DQNConfig` | PPO, A2C, DQN | Gradient norm clipping; rlox DQN default is `inf` (opt-in) |
| `n_steps` | `PPOConfig`, `A2CConfig` | PPO, A2C | On-policy rollout length |
| `n_epochs` | `PPOConfig` | PPO | Minibatch passes per update |
| `tau` | `SACConfig`, `TD3Config` | SAC, TD3 | Polyak averaging for target networks |
| `learning_starts` | `SACConfig`, `TD3Config`, `DQNConfig` | SAC, TD3, DQN | Random exploration steps before training |
| `train_freq` | `SACConfig`, `TD3Config`, `DQNConfig` | SAC, TD3, DQN | Env steps between training rounds |
| `gradient_steps` | `SACConfig`, `TD3Config`, `DQNConfig` | SAC, TD3, DQN | SGD steps per training round |
| `policy_delay` | `TD3Config` | TD3 | Delayed policy updates (TD3-specific) |
| `target_policy_noise` | `TD3Config` | TD3 | Target action smoothing std |
| `target_noise_clip` | `TD3Config` | TD3 | Target noise clipping range |
| `exploration_fraction` | `DQNConfig` | DQN | Fraction of training for ε decay |
| `exploration_final_eps` | `DQNConfig` | DQN | Final ε value after decay |

### A.2 Renamed keys

These keys have the same semantics but different names between
frameworks. The translation layer maps them at construction time.

| rlox key | SB3 key | Applies to |
|---|---|---|
| `clip_eps` (or alias `clip_range`) | `clip_range` | PPO |
| `target_update_freq` | `target_update_interval` | DQN |

### A.3 SB3-specific override injections

These keys are present in rlox-authored preset YAMLs but require
additional SB3-specific keys that the translation layer injects
from an override table (`_SB3_OVERRIDES`). Without these overrides
SB3 would use its defaults for the missing keys, which in some
cells produces degenerate training (see §3.8.2 DQN CartPole
collapse).

| Cell | rlox preset (relevant keys) | SB3 override injected |
|---|---|---|
| DQN / CartPole-v1 | `target_update_freq: 10, hidden: 256, learning_rate: 2.3e-3` | `train_freq: 256, gradient_steps: 128` |
| DQN / MountainCar-v0 | `target_update_freq: 600, hidden: 256, learning_rate: 4e-3` | `train_freq: 16, gradient_steps: 8` |

Both override entries are transcribed directly from
rl-baselines3-zoo's `hyperparams/dqn.yml` and represent the SB3
community's consensus recipe for each cell.

### A.4 Keys handled specially (not forwarded to SB3 constructor)

These keys affect harness setup (environment construction, policy
network shape) rather than being passed directly to the SB3
algorithm constructor.

| Key | Handling | Rationale |
|---|---|---|
| `n_envs` | Controls the length of the `DummyVecEnv` lambda list in `_make_env` | SB3 takes `n_envs` implicitly via `VecEnv` construction, not as an algorithm kwarg |
| `normalize_obs` | Toggles `VecNormalize(norm_obs=True)` wrapping in `_make_env` | Wrapper-level configuration |
| `normalize_rewards` (alias `normalize_reward`) | Toggles `VecNormalize(norm_reward=True)` | Wrapper-level configuration |
| `hidden` | Mapped to `policy_kwargs={"net_arch": [hidden, hidden]}` | SB3 MlpPolicy takes network arch via `policy_kwargs` |
| `anneal_lr` | **Dropped** on the SB3 side (see A.6) | SB3 linear-schedule requires a callable; translation is non-trivial |
| `normalize_advantages` | Dropped; both frameworks default `True` and rl-zoo3 presets do not override | Accidental alignment we rely on; flagged in §3.8 |
| `clip_vloss` | Dropped (see A.6) | Intentionally NOT mirrored into SB3's `clip_range_vf` — see §3.8.1 and §5.2.2 |
| `double_dqn` | Dropped; rlox DQN defaults to True, SB3 DQN is vanilla | Divergence preserved rather than aligned; see §3.8.2 |
| `dueling` | Dropped; rlox DQN extension not in SB3 core | rlox-only feature |
| `n_step` | Dropped; rlox DQN extension | rlox-only feature |
| `prioritized` | Dropped; rlox DQN extension | rlox-only feature |
| `alpha`, `beta_start` | Dropped; PER hyperparameters | Only used if `prioritized=True` |
| `exploration_initial_eps` | Dropped; rlox default 1.0, SB3 default 1.0 | Accidental alignment |
| `exploration_noise` | TD3-specific; translated into `action_noise=NormalActionNoise(...)` | Handled in `_build_sb3_model` at construction time |

### A.5 Device control

The translation layer forces `device="cpu"` on the SB3 side to
match rlox's CPU-default and to eliminate the CUDA-availability
confound described in §3.4. Without this override SB3 would
auto-select CUDA on any host where `torch.cuda.is_available()`
returns True, giving it a wall-clock (but not sample-efficiency)
advantage that is unrelated to the framework architecture being
measured.

```python
# Excerpt from _build_sb3_model:
sb3_kwargs.setdefault("device", "cpu")
```

### A.6 Keys intentionally NOT translated

Two keys are present in the rlox preset YAMLs but intentionally
dropped on the SB3 side:

**`anneal_lr`**. rlox supports linear learning-rate annealing via
a built-in schedule; SB3 supports the same via a callable schedule
passed to the optimizer. Translating the boolean to a functional
schedule is possible but requires careful handling of
`total_timesteps` — which is not known at algorithm-construction
time. We verified that none of the rl-zoo3 MuJoCo presets enable
`anneal_lr` by default, so dropping it is a no-op in practice. If
a future preset enables it, the translation layer will emit a
warning via the "unknown preset key" fallback.

**`clip_vloss`**. As discussed in §3.8.1, rlox's CleanRL-style
`clip_vloss=True` default should NOT be mirrored into SB3's
`clip_range_vf`. Mirroring would force SB3 to abandon its native
plain-MSE formulation, which is empirically SB3's best-performing
recipe. The translation layer preserves this asymmetry by dropping
`clip_vloss` on the SB3 side.

---

## Appendix B. n=10 seed subset for MuJoCo

Forward-referenced from: §3.6, §4.1.3.

### B.1 Motivation

At n=5 seeds, the Interquartile Mean (IQM) from Agarwal et al.
[@agarwal2021precipice] reduces to the trimmed mean
`(x_1 + x_2 + x_3) / 3` after sorting, which is numerically close
to the median and has substantially less outlier-robustness than
the statistic was designed for. At n=10 the IQM trims the two
extreme seeds on each side, leaving six seeds in the aggregate,
which is where the statistic's noise-reduction properties pay
their bits.

We re-ran the four PPO MuJoCo headline cells at n=10 seeds where
compute budget permitted. SAC and TD3 MuJoCo cells are omitted
from the n=10 subset because the primary story in §4.1 uses n=5
for the full table and extending to n=10 for only some cells
creates non-comparable entries.

### B.2 Results

| Algo | Environment | Timesteps | **rlox IQM (n=10)** | **rlox 95% CI** | **SB3 IQM (n=10)** | **SB3 95% CI** |
|---|---|---:|---:|---|---:|---|
| PPO | Hopper-v4 | 1M | ⏳ | [⏳, ⏳] | ⏳ | [⏳, ⏳] |
| PPO | HalfCheetah-v4 | 1M | ⏳ | [⏳, ⏳] | ⏳ | [⏳, ⏳] |
| PPO | Walker2d-v4 | 2M | ⏳ | [⏳, ⏳] | ⏳ | [⏳, ⏳] |

**Table B.1.** PPO MuJoCo multi-seed convergence at n=10 seeds.
Both rlox and SB3 cells computed from
`benchmarks/multi_seed_runner*.py` at commit `cd8cbb9` using the
eval protocol of §3.5. CIs are 95% stratified bootstrap with
10 000 resamples per cell.

### B.3 Comparison to n=5 in Table 1

| Cell | n=5 rlox IQM | n=10 rlox IQM | Δ | n=5 CI width | n=10 CI width |
|---|---:|---:|---:|---:|---:|
| PPO Hopper-v4 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| PPO HalfCheetah-v4 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| PPO Walker2d-v4 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

**Table B.2.** n=5 vs n=10 comparison for the three PPO MuJoCo
cells in common. Smaller CI widths at n=10 are expected; any cell
where the n=10 IQM falls outside the n=5 CI indicates that the
n=5 estimate was driven by outliers and that the n=10 number is
the one to trust.

### B.4 Computational cost

The n=10 subset required an additional ⏳ core-hours of GCP
compute beyond the main n=5 sweep. The marginal cost per
additional seed is approximately ⏳ core-hours for Hopper,
⏳ for HalfCheetah, and ⏳ for Walker2d (which uses 2M training
steps instead of 1M).

> *Note to authors*: if the n=10 sweep did not complete before
> submission, cut this entire appendix and modify §3.6 to note
> that n=10 was planned but out of compute budget. Do not
> report n=10 for some cells and n=5 for others in the same
> table — that introduces a confound the reviewer cannot
> correct for.

---

## Appendix C. Wall-clock reporting

Forward-referenced from: §3.4, §4.2, §5.2.

Wall-clock comparisons between frameworks are treacherous because
the two frameworks do different amounts of work per step (§3.11
and §5.2.4). This appendix reports per-framework steps-per-second
(SPS) measurements rather than ratios, with a description of the
hardware and the measurement protocol so readers can reproduce.

### C.1 Hardware

All measurements were collected on two reference systems:

1. **M-series Apple Silicon**: M3 Pro, 12-core CPU (6 performance +
   6 efficiency), 36 GB unified memory, macOS 14. This is the
   author's primary development machine.
2. **GCP e2-standard-8**: Intel Cascade Lake, 8 vCPUs @ 2.8 GHz, 32
   GB memory, Ubuntu 22.04. This is the instance type used for the
   multi-seed sweep in §4.1.

Both systems have no GPU attached; both frameworks are forced to
CPU (§3.4). Temperature and thermal throttling were not explicitly
controlled; each measurement is the median of at least 30 timed
runs, which averages out short-term thermal noise.

### C.2 Per-cell SPS

| Cell | rlox SPS (M3) | SB3 SPS (M3) | rlox SPS (GCP) | SB3 SPS (GCP) |
|---|---:|---:|---:|---:|
| PPO / CartPole-v1 (8 envs, n_steps=128) | ⏳ | ⏳ | ⏳ | ⏳ |
| PPO / Hopper-v4 (8 envs, n_steps=2048) | ⏳ | ⏳ | ⏳ | ⏳ |
| PPO / HalfCheetah-v4 (8 envs, n_steps=2048) | ⏳ | ⏳ | ⏳ | ⏳ |
| PPO / Walker2d-v4 (8 envs, n_steps=2048) | ⏳ | ⏳ | ⏳ | ⏳ |
| SAC / Pendulum-v1 (1 env) | ⏳ | ⏳ | ⏳ | ⏳ |
| SAC / HalfCheetah-v4 (1 env) | ⏳ | ⏳ | ⏳ | ⏳ |
| TD3 / HalfCheetah-v4 (1 env) | ⏳ | ⏳ | ⏳ | ⏳ |
| DQN / CartPole-v1 (1 env) | ⏳ | ⏳ | ⏳ | ⏳ |

**Table C.1.** Steps-per-second throughput per framework per cell.
Numbers are wall-clock environment steps divided by total
training time, including algorithm updates and evaluation
overhead.

### C.3 Why no "X× faster" summary

We deliberately do not report a single "rlox is X× faster than
SB3" number for the training pipeline because the ratio varies
substantially by cell: ⏳× on small-observation CartPole, ⏳× on
Hopper, ⏳× on HalfCheetah with its larger observation space, and
⏳× on Walker2d with longer episodes. Citing a single ratio would
require choosing a reference cell, which is a methodological
choice the reader should make themselves based on which
environment most resembles their workload.

The component microbenchmarks in §4.2 (Tables 3–5) are more
directly comparable across hardware because they isolate individual
operations whose cost is proportional to the data size rather than
to environment-specific physics complexity.

### C.4 Sample efficiency is not wall-clock efficiency

One final reminder: the rlox-vs-SB3 convergence table in §4.1 is
normalized by **environment steps**, not wall-clock time. Both
frameworks train for the same number of environment interactions,
so the convergence comparison is fair regardless of per-step
speed. A reader who wants to interpret the §4.1 table as
"rlox reaches reward X in wall-clock time Y faster than SB3"
should combine the §4.1 convergence numbers with the §C.2
throughput numbers to get the composite wall-clock claim.

---

## Appendix D. Excluded cells

Forward-referenced from: §3.11, §4.1.5, §4.4.2.

Three cells were excluded from the §4.1 main comparison table for
reasons that do not reflect framework quality. We document each
here with its reasoning and a separate reference measurement.

### D.1 DQN / MountainCar-v0

**Reason for exclusion**: neither framework's default recipe
reaches the goal flag. Both rlox DQN and SB3 DQN with the rl-zoo3
`dqn_mountaincar.yaml` preset using `target_update_freq: 600,
hidden: 256, learning_rate: 4e-3` require the additional cadence
keys `train_freq: 16, gradient_steps: 8` to converge. Without
those keys the agent never explores to the goal position in 120k
training steps and the eval reward saturates at -200 (the
time-limit cap).

**Reference measurement with the full recipe**:

| Framework | Recipe | Seeds | IQM | 95% CI |
|---|---|---:|---:|---|
| rlox DQN | `train_freq=16, gradient_steps=8` | 5 | ⏳ | [⏳, ⏳] |
| SB3 DQN | Same recipe + rl-zoo3 `_SB3_OVERRIDES` | 5 | ⏳ | [⏳, ⏳] |

With the full recipe, both frameworks converge to the zoo
reference (~-110); the delta between frameworks is within
statistical noise and the cell would belong in §4.1. We exclude
it from the main table because the story about "when rlox
default recipe diverges from SB3 default recipe" is clearer as a
separate appendix than as a footnote in Table 1.

### D.2 PPO / Humanoid-v4

**Reason for exclusion**: the recommended training budget is
10 000 000 environment steps, which at 5 seeds is 50M total env
steps per framework, which is outside the compute budget of the
v2 multi-seed sweep. Running Humanoid at 1M or 2M steps would
produce an unconverged policy whose reported reward is not
representative of either framework's capability on the cell.

**Reference from single-seed run**: a single-seed rlox PPO run on
Humanoid-v4 to 10M steps at commit ⏳ reached ⏳ mean reward,
which is within ⏳% of SB3-zoo's published Humanoid-v4 reference of
6251. This is reported in Appendix F with full caveats about
single-seed interpretation.

### D.3 Atari (all algorithms)

**Reason for exclusion**: a same-harness Atari comparison requires
matching the AtariWrapper preprocessing pipeline (frame stacking,
resize, grayscale, reward clipping, noop max, episode life loss)
across both frameworks and verifying that the wrapper parameters
are identical. This is a finite amount of work but the
infrastructure was not ready at the time of the v2 multi-seed run.

rlox supports Atari training via `gymnasium.wrappers.AtariWrapper`
and single-seed runs on Breakout-v4 and Pong-v4 with the rl-zoo3
recipes reach zoo-reference rewards within ⏳% at the end of
training. A proper multi-seed comparison is deferred to a future
update.

---

## Appendix E. Reproducibility manifest

Forward-referenced from: §3.10, §4, §5.4.2.

### E.1 Repository and commit

- **Repository**: `github.com/riserally/rlox`
- **Commit used for §4 results**: `cd8cbb9` (anonymized as `ANON`
  for double-blind submission venues)
- **Tag**: `paper-v1` (to be created at time of submission)

All figures, tables, and performance claims in §4 trace to this
specific commit. Readers reproducing the results should
`git checkout cd8cbb9` before running any benchmark script.

### E.2 Dependencies

Pinned via `pyproject.toml` and `Cargo.lock`. Primary dependencies:

| Package | Version | Role |
|---|---|---|
| Python | 3.12.x | Interpreter |
| PyTorch | 2.1.x | Policy networks, autograd |
| numpy | 1.26.x | Array interop |
| gymnasium | 0.29.x | Environment interface |
| mujoco | 3.0.x | MuJoCo physics |
| stable-baselines3 | 2.7.1 | Comparison target |
| mkdocs-static-i18n | 1.3.x | Documentation i18n (not runtime) |
| Rust | 1.75 stable | Rust toolchain |
| PyO3 | 0.20.x | Python bindings |
| ndarray | 0.15.x | Rust data arrays |
| rayon | 1.8.x | Rust parallelism |

Exact versions are in `pyproject.toml` and `Cargo.lock` in the
repository at commit `cd8cbb9`.

### E.3 Hardware

**Primary**: GCP e2-standard-8, us-central1-f, 8 vCPUs, 32 GB RAM,
Ubuntu 22.04 LTS. Total cost per full multi-seed sweep ≈ ⏳ USD.

**Secondary (development)**: Apple M3 Pro, 36 GB, macOS 14. Used
for the component microbenchmarks in §4.2 and for local
development. Component benchmarks were also rerun on the GCP
instance for consistency.

### E.4 One-command reproduction

The complete multi-seed sweep is reproducible with:

```bash
git clone https://github.com/riserally/rlox
cd rlox
git checkout cd8cbb9
bash ../rlox-priv/scripts/run-multi-seed.sh --gcp
```

(where `rlox-priv` contains the GCP launch script).

Individual cells can be re-run locally with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin numpy gymnasium torch stable-baselines3
maturin develop --release

# rlox side
python benchmarks/multi_seed_runner.py --algo ppo --env Hopper-v4 \
  --timesteps 1000000 --seeds 5 --output results/my-reproduction

# SB3 side (same eval protocol, same config YAML)
python benchmarks/multi_seed_runner_sb3.py --algo ppo --env Hopper-v4 \
  --timesteps 1000000 --seeds 5 --output results/my-reproduction
```

### E.5 Per-seed artifacts

Every seed in every cell writes a JSON file to the output directory:

```
results/multi-seed/
├── ppo_CartPole-v1_seeds5.json       # rlox aggregate
├── sb3_ppo_CartPole-v1_seeds5.json   # SB3 aggregate
├── ppo_Hopper-v4_seeds5.json
├── sb3_ppo_Hopper-v4_seeds5.json
├── ...
├── versions.json                     # Environment/lib version capture
└── <per-seed debug logs>
```

Each aggregate JSON contains the per-seed `mean_reward`,
`std_reward`, `wall_time_s`, and `sps` fields, plus the aggregated
`iqm`, `ci_low`, `ci_high`, `mean`, `std`, `min`, `max`, and
`mean_sps` summary statistics. The `per_seed` list preserves the
individual seed records for readers who want to compute their own
statistics.

### E.6 Random seed convention

Seeds are deterministic and reproducible across runs:

- **Training seed for seed index i**: `seed_i = i * 1000 + 42`,
  so n=5 seeds are `{42, 1042, 2042, 3042, 4042}`.
- **Evaluation seed for episode ep of seed_i**: `seed_i + 1000 + ep`,
  so the first seed's eval episodes use seeds `{1042, 1043, ..., 1071}`.
- **Environment thunks** pass the same seed to
  `gym.make(env_id).reset(seed=...)`.
- **PyTorch CPU threading** is non-deterministic and can produce
  small run-to-run variations even at identical seeds. We do not
  set `torch.use_deterministic_algorithms(True)` because it
  disables some PyTorch kernels and materially slows training.
  The observed run-to-run variation is smaller than the seed-to-seed
  variation and does not affect the IQM-level comparisons in §4.1.

---

## Appendix F. Single-seed reference measurements

Forward-referenced from: Appendix D.2 (Humanoid-v4), §4.4 (qualitative).

Several cells that are outside the v2 multi-seed sweep compute
budget were run single-seed for reference. These numbers should
not be interpreted as framework claims; they are included only so
a reader can compare against published numbers in the algorithm
and zoo literatures for context.

| Algo | Environment | Timesteps | Seed | rlox reward | SB3 reward | zoo reference |
|---|---|---:|---:|---:|---:|---:|
| PPO | Humanoid-v4 | 10 M | 42 | ⏳ | ⏳ | 6251 (v3) |
| PPO | Ant-v4 | 1 M | 42 | ⏳ | ⏳ | 2865 (v3) |
| SAC | Humanoid-v4 | 2 M | 42 | ⏳ | ⏳ | 6251 (v3) |

**Table F.1.** Single-seed reference runs for cells outside the
multi-seed compute budget. These should not be read as framework
comparisons — a single seed is inadequate for statistical claims
(§5.3.1). They are provided so readers checking rlox against
published numbers for these cells have a direct reference.

The zoo reference numbers are on `-v3` MuJoCo environments; our
single-seed runs use `-v4`. The version difference is documented
in §3.2 and contributes ⏳–⏳% to observed deltas.

---

## Writing-time notes for author(s)

### Appendix A scope check

Appendix A reproduces the full translation layer as a table. This
is intentionally comprehensive because §5.3.2 argues that "future
framework comparisons should publish their translation layer as a
first-class artifact". Publishing it in an appendix rather than as
a separate document is the concrete implementation of that
recommendation. Do not trim.

### Appendix B (n=10) — cut or keep

If the n=10 sweep does not complete before submission:

1. **Cut Appendix B entirely** — remove it from the appendix list.
2. **Update §3.6** — remove the "n=10 where budget permitted"
   sentence and replace with "n=5 throughout; an n=10 subset is
   planned as future work".
3. **Update §4.1.3** — delete the subsection and renumber.

Do NOT leave Appendix B with partially-filled tables. A reviewer
will flag the inconsistency immediately.

### Appendix C wall-clock story

The wall-clock numbers in Appendix C are the ones most likely to
be cited by practitioners evaluating rlox for adoption. Make sure
they are reproducible and that the methodology (median of 30 runs,
no thermal control, forced CPU) is documented clearly enough that
a reader can repeat the measurement. The "why no X× summary"
subsection (§C.3) is deliberately pedantic — delete it if space
is tight, but its removal should be a conscious choice.

### Appendix E reproducibility manifest

The manifest has three versions depending on venue:

1. **Double-blind** (NeurIPS main track, ICML main track): replace
   `riserally` with `ANONYMIZED` everywhere, strip the commit hash
   until publication, include a `supplementary.zip` instead of a
   git URL.
2. **Open / post-publication**: current text is correct; paste the
   exact commit hash and tag once submitted.
3. **Systems venue** (MLSys, OSDI, SOSP): expand §E.3 with full
   CPU topology (`lscpu` output), memory bandwidth measurements,
   and exact `RUSTFLAGS` / `target-cpu` settings. These venues
   expect more hardware detail than ML venues.

### Appendix F single-seed disclaimer

The single-seed table in Appendix F is load-bearing for "did you
benchmark Humanoid" reviewer questions. Make sure the disclaimer
(§F opening paragraph) is large and unmissable, so a reviewer
skimming the table does not accidentally cite a single-seed number
as if it were a multi-seed result.

### Missing appendices to consider

Four additional appendices were discussed during the draft:

- **Appendix G (PPO ablation on Hopper full curves)**: learning
  curves showing how the two PPO loss formulations diverge over
  training length. Would be a complement to §4.3.1's single-point
  A/B. Kept out of the current draft because the plot is not yet
  generated; add as a follow-up if space permits.
- **Appendix H (Failure mode gallery)**: screenshots or plots of
  cells where rlox or SB3 diverges, useful for debugging. Kept
  out because this is more of a tutorial appendix than a paper
  appendix.
- **Appendix I (Algorithm-by-algorithm pseudocode)**: pseudocode
  for each of the 22 rlox algorithms. Way too long for an
  appendix; defer to the repository's docs site.
- **Appendix J (Build system and PyO3 boundary code)**: showing
  the actual PyO3 function signatures for a few representative
  hot-path operations (e.g. `compute_gae_batched`). Useful for
  systems venues; cut from the ML-venue draft.

Decide with co-authors whether any of G/H/I/J is worth the space
based on the target venue.

### Cross-linking checklist

Every forward-reference in the main text should resolve to an
appendix entry above, and vice versa. Final pass:

- [ ] §3.3 → Appendix A ✅
- [ ] §3.7 → Appendix A ✅
- [ ] §3.8 → Appendix A (for the "full catalog" reference) ✅
- [ ] §3.10 → Appendix E ✅
- [ ] §3.11 → Appendix D ✅
- [ ] §4.1.3 → Appendix B ✅
- [ ] §4.1.5 → Appendix D ✅
- [ ] §4.2 → Appendix C ✅
- [ ] §4.4.2 → Appendix D ✅
- [ ] §5.3.2 → Appendix A (for the "publish translation layer" argument) ✅
- [ ] §5.4.2 → Appendix E (for installation instructions) ✅
- [ ] Appendix D.2 → Appendix F ✅

If any of these fails to resolve after the final copy-edit, fix
the reference or delete the target.
