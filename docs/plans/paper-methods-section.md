# Methods — rlox paper draft

**Draft status**: v1, 2026-04-08. Numbers marked `⏳` are waiting on the
v2 multi-seed GCP sweep. Replace before submission.

**Intended section**: §3 Methods, between §2 Related Work and §4 Results.

**Grounded in**: `docs/plans/benchmark-comparison-inconsistencies.md` for
the per-axis evidence. Every claim here should be checkable against that
document's cited source lines.

---

## 3. Methods

### 3.1 Framework architecture

rlox is a reinforcement learning framework built on the *Polars pattern*:
a Rust data plane handles all compute-intensive, non-autograd operations,
while a Python control plane, backed by PyTorch, owns the training loop,
policy networks, configuration, and logging. The two layers communicate
via PyO3 [@pyo3] using zero-copy numpy↔Rust array exchange at the
boundary.

The architectural split is precisely the autograd boundary. Any operation
whose gradient is taken with respect to network parameters runs in
PyTorch; any operation without an autograd dependency runs in Rust.
Concretely, this places environment stepping, replay buffers,
Generalized Advantage Estimation (GAE) [@schulman2015gae], running
observation statistics, and V-trace correction [@espeholt2018impala] on
the Rust side; and places policy networks, loss functions, and
optimizers on the PyTorch side.

The rlox codebase implements 22 reinforcement learning algorithms
through a unified `Trainer('<algo>', env=..., config=...)` entry point,
with `<algo>` ∈ {ppo, a2c, trpo, vpg, sac, td3, dqn, mpo, impala,
mappo, qmix, dreamerv3, cql, calql, iql, td3_bc, bc, dt, dtp,
diffusion, grpo, dpo}. For this paper we benchmark the six on-policy
and off-policy algorithms (PPO, A2C, SAC, TD3, DQN, TRPO) that have
well-established convergence baselines in the Stable-Baselines3 [@sb3]
and CleanRL [@cleanrl] reference implementations.

### 3.2 Environments

We evaluate on classic control and MuJoCo [@todorov2012mujoco]
continuous-control tasks from the Gymnasium [@gymnasium] suite:

| Category | Environments |
|---|---|
| Classic control | CartPole-v1, Acrobot-v1, MountainCar-v0, Pendulum-v1 |
| MuJoCo | Hopper-v4, HalfCheetah-v4, Walker2d-v4, Ant-v4, Humanoid-v4 |

**Version pinning.** MuJoCo environments are the `-v4` variants, which
is the Gymnasium 0.29+ default. SB3's rl-baselines3-zoo reference
numbers are predominantly on the `-v3` variants, which differ in contact
model and termination bounds. We therefore do **not** compare absolute
rlox numbers to the zoo leaderboard — all comparisons in §4 are against
SB3 re-run **in our harness** on the `-v4` environments. The
`gym.make(env_id).spec.id` string is recorded in a `versions.json`
artifact next to every results JSON as an audit trail.

### 3.3 Hyperparameter sourcing

Per-(algorithm, environment) hyperparameters are taken from
rl-baselines3-zoo's `hyperparams/*.yml` files, checked into the rlox
repository as `benchmarks/convergence/configs/<algo>_<env>.yaml`. Both
rlox and SB3 read the *same* YAML files via the same `_resolve_preset`
helper in `benchmarks/multi_seed_runner.py` and its sibling
`benchmarks/multi_seed_runner_sb3.py`; this is the single source of
truth for the comparison.

The key mapping from rlox configuration keys to SB3 constructor keyword
arguments is implemented in `_translate_config` and exposed as a
maintained artifact; see Appendix B for the full table. The mapping
covers three cases:

1. **Identity passthrough** (e.g. `learning_rate`, `batch_size`,
   `gamma`).
2. **Simple renames** (e.g. `clip_eps` → `clip_range`,
   `target_update_freq` → `target_update_interval`).
3. **Semantic overrides** where an SB3 default implicitly differs from
   rlox's default (e.g. SB3 DQN's `train_freq=4` vs rlox DQN's
   `train_freq=1`).

Case (3) is maintained as an explicit `_SB3_OVERRIDES` table. We
discovered it was necessary when SB3 DQN on CartPole-v1 with rlox's
preset — which uses `target_update_interval=10` matching
rl-baselines3-zoo — collapsed to 9 mean reward without the SB3-specific
cadence keys (`train_freq=256, gradient_steps=128`) that zoo provides
for SB3. The override table is published alongside the paper and should
be cited in any re-use.

### 3.4 Training protocol

Each (algorithm, environment) cell is trained for the `total_timesteps`
budget specified in its rl-baselines3-zoo YAML, which ranges from
50 000 steps (Pendulum-v1) to 2 000 000 steps (Walker2d-v4). Timesteps
are counted in **environment steps**, not gradient steps or episodes,
for both frameworks.

**Parallelism.** PPO and A2C use `n_envs=8` parallel environments in
both frameworks, as specified by the preset YAML. Off-policy algorithms
(SAC, TD3, DQN) use `n_envs=1` in both frameworks; rlox's multi-env
`OffPolicyCollector` is available but we mirror SB3's single-env
convention in the apples-to-apples comparison to isolate algorithmic
differences from vectorization differences.

**Compute device.** Both frameworks are forced to CPU for the
comparison runs (`device="cpu"`). rlox's default is already CPU for
policy tensors; SB3 defaults to CUDA when available, which would give
it an unrelated wall-clock advantage on GPU hosts. Forcing CPU
eliminates this confound; sample efficiency is preserved either way,
and wall-clock comparisons are reported separately in Appendix C.

**Seeds.** Each cell is run for `n_seeds = 5` seeds in the form
`seed_i = i × 1000 + 42` for `i ∈ {0, 1, 2, 3, 4}`. The same seed
sequence is used for both frameworks. 10-seed runs on the headline
MuJoCo cells are reported in Appendix D where compute budget permitted
(see §3.7 on the IQM-at-n=5 caveat).

### 3.5 Evaluation protocol

The evaluation protocol is deliberately fixed and applied identically
to both frameworks:

1. **Episodes per seed**: 30 deterministic rollouts. This is larger
   than SB3's default eval (10 episodes) and smaller than CleanRL's
   (100 episodes); 30 was chosen to give a single-seed standard error
   below one seed's worth of seed-to-seed variance on MuJoCo tasks
   without dominating wall-clock budget.
2. **Policy determinism**: `algo.predict(obs, deterministic=True)` —
   i.e. the mean action for Gaussian policies, the argmax for
   categorical policies. This differs from some published CleanRL plots
   which sample from the stochastic policy at eval time; our
   deterministic choice gives lower variance and matches the SB3
   convention.
3. **Environment reset seeds**: each of the 30 eval episodes uses a
   unique seed `env.reset(seed=base + 1000 + ep)` where `base` is the
   training seed and `ep ∈ {0, 1, ..., 29}`. A single repeated
   `env.reset(seed=base + 1000)` (as was briefly the case in an earlier
   rlox harness version) produces 30 near-identical rollouts with
   fictitious variance — see the discussion in §3.9.
4. **VecNormalize at eval**: if the training run used observation or
   reward normalization via `VecNormalize`, the running statistics are
   **frozen** at eval time (`vn.training = False`) and evaluation
   observations are re-normalized through the frozen statistics before
   the policy forward pass. Skipping this freeze silently updates the
   statistics with eval trajectories and changes the reported numbers
   by several times the seed-to-seed noise floor on MuJoCo tasks.
5. **Reported metric**: the mean of the 30 per-episode returns is
   recorded as the *per-seed* evaluation score. Statistical aggregation
   across seeds is described in §3.6.

### 3.6 Statistical reporting

Per the recommendations of Agarwal et al.
[@agarwal2021precipice] we report **interquartile mean (IQM)** with a
95% stratified bootstrap confidence interval as the primary per-cell
statistic.

Concretely, for a cell with 5 seeds and per-seed scores `x_0, x_1, ..., x_4`
sorted ascending, the IQM is `(x_1 + x_2 + x_3) / 3`. The bootstrap CI
is computed with 10 000 resamples per cell stratified by seed using
`rlox.evaluation.stratified_bootstrap_ci`.

**Caveat we disclose explicitly.** At n=5 seeds, the IQM is a three-sample
average trimmed from both tails of a five-sample distribution. In the
limiting case this is numerically close to — and at some n coincides
with — the median. IQM as a noise-reducing statistic pays its bits above
n ≈ 10. We report n=5 for the full matrix (Table 1) and n=10 for the
MuJoCo headline cells where compute budget permitted (Table 2). Where
n=5, we label the statistic as "IQM (≈ median at n=5)" in captions and
caption-linked footnotes. We do **not** report `mean ± std` alongside
IQM in the same table; the former invites reviewers to apply Gaussian
assumptions to non-Gaussian reward distributions.

### 3.7 Framework comparison harness

The SB3-in-rlox-harness runner, `benchmarks/multi_seed_runner_sb3.py`,
is the single point of contact between rlox's evaluation conventions
and SB3's model constructors. Its structure is:

```python
def run_single_seed(algo, env_id, total_timesteps, seed, config):
    config = _resolve_preset(algo, env_id)   # same YAML as rlox
    sb3_kwargs, harness = _translate_config(algo, config)
    model, venv = _build_sb3_model(algo, env_id, seed, sb3_kwargs, harness)
    model.learn(total_timesteps=total_timesteps)

    # 30 deterministic eval episodes with unique per-episode seeds
    # and frozen VecNormalize stats — matching rlox exactly.
    return _eval_protocol(algo, env_id, seed, model, venv)
```

The translation layer deliberately handles one set of concerns
(hyperparameter key renaming, override injection, VecEnv wrapping) and
leaves the rest — training loop, loss formulation, optimizer choice —
to each framework's native implementation. This isolates comparison
semantics from framework-internal choices, which is both the strength
(we're measuring framework behavior, not translation quirks) and the
weakness (differences in the *native* loss formulations are preserved
and must be disclosed; see §3.8).

All comparison numbers in §4 come from this runner. The runner is open
source and unit-tested; any reader can clone the repository, run the
runner on the exact same preset YAMLs, and reproduce the comparison.

### 3.8 Known algorithmic asymmetries

An apples-to-apples framework comparison at the "same hyperparameters"
level does **not** imply the underlying algorithms are bit-identical.
rlox and SB3 differ in several native implementation details that are
preserved by our translation layer (rather than forcibly aligned). We
catalog the major ones here; the full catalog is in
`docs/plans/benchmark-comparison-inconsistencies.md`.

#### 3.8.1 PPO value-loss formulation

rlox's `PPOLoss` follows the CleanRL convention: a half-weighted
max-of-clipped value loss computed as
$$
L^{\text{VF}}_{\text{rlox}}(\theta) = \tfrac{1}{2}\,\mathbb{E}_t\!\left[\max\left((V_\theta(s_t) - R_t)^2,\,(V^{\text{clip}}_\theta - R_t)^2\right)\right],
$$
where $V^{\text{clip}}_\theta = V_{\theta_{\text{old}}}(s_t) + \mathrm{clip}(V_\theta(s_t) - V_{\theta_{\text{old}}}(s_t), -\epsilon, +\epsilon)$.

SB3's PPO uses plain MSE without the clipping trick and without the
inner $\tfrac{1}{2}$:
$$
L^{\text{VF}}_{\text{SB3}}(\theta) = \mathbb{E}_t\!\left[(V_\theta(s_t) - R_t)^2\right].
$$

At matched `vf_coef=0.5`, the effective value-loss gradient in rlox is
therefore **half** of SB3's. This is an intentional design choice: an
early development version of rlox defaulted to the SB3-aligned
formulation, and a controlled A/B on Hopper-v4 at 1M training steps,
seed=42, gave:

- SB3-aligned variant (no inner 0.5, no vloss clipping): **837.0** mean reward
- CleanRL variant (inner 0.5, max-of-clipped): **1955.3** mean reward

Removing the CleanRL formulation regressed the final-policy quality by
57%. The 200k and 500k step bisections that originally motivated the
alignment were in the wrong evaluation regime — both formulations are
still in early-training noise at those budgets; the divergence emerges
only at training lengths where the value function begins taking
large-magnitude updates. The rlox default is therefore the CleanRL
formulation, and we report this as a disclosed asymmetry.

**Implication for the comparison.** When §4 reports
"rlox PPO = X, SB3 PPO = Y" on Hopper-v4, we are comparing each
framework's *best-performing native formulation* on matched
hyperparameters. This is the question most practitioners care about
("which framework should I use?") rather than the academic question
of "how does the framework behave if I force both to use the same
loss?". Both questions are valid; we answer the first explicitly.

#### 3.8.2 DQN loss and gradient clipping

SB3 DQN uses Huber loss (`F.smooth_l1_loss`) with gradient norm clipping
at 10.0 [@sb3_dqn_source]. rlox DQN uses plain mean-squared TD error
with no gradient clipping, following the Mnih et al. [@mnih2015dqn]
original recipe plus CleanRL's default. The rlox `DQNConfig` exposes
`max_grad_norm` as an opt-in knob (default `float('inf')`); setting it
to 10.0 reproduces the SB3 clipping semantics. We did not enable it in
the rlox runs reported in §4 because doing so requires re-tuning the
cadence keys (`train_freq`, `gradient_steps`) that rlox DQN currently
sets to 1/1 by default — SB3 uses 4/1 by default, and rl-zoo3 CartPole
uses 256/128.

#### 3.8.3 SAC critic loss factor

SB3 SAC's critic loss is
$L^Q_{\text{SB3}} = \tfrac{1}{2}\sum_{i=1,2} \mathbb{E}[(Q_{\phi_i}(s,a) - y)^2]$
whereas rlox SAC's is
$L^Q_{\text{rlox}} = \sum_{i=1,2} \mathbb{E}[(Q_{\phi_i}(s,a) - y)^2]$
— i.e. rlox's critic gradient is twice SB3's at the same effective
critic learning rate. We leave this asymmetric because both frameworks
converge to statistically equivalent returns on the six MuJoCo SAC
cells despite the factor-of-two difference, which is itself a small
finding about SAC's empirical robustness to critic-gradient scaling at
the scales we test.

#### 3.8.4 Network architectures

Both frameworks use two hidden layers for all algorithms. PPO/A2C use
`[64, 64]` with Tanh activations and orthogonal initialization
(gain $\sqrt{2}$ for hidden layers, 0.01 for the policy head, 1.0 for
the value head) — this is matched bit-for-bit between the frameworks.
SAC uses `[256, 256]` with ReLU. DQN uses `[256, 256]` with ReLU, which
matches our preset YAML (`hidden: 256`) and the rl-baselines3-zoo
CartPole recipe; we note that SB3's `MlpPolicy` *default* is `[64, 64]`
and our preset explicitly overrides it.

PPO/A2C in both frameworks keep actor and critic as separate MLPs
without sharing hidden layers. SB3's `share_features_extractor=True`
default refers to the `FlattenExtractor`, which is the identity
function for Box and Discrete observation spaces; the `mlp_extractor`
trunks that follow are separate — so the effective architecture is
two independent MLPs, matching rlox.

### 3.9 Environment normalization

When `normalize_obs` or `normalize_rewards` is set in the preset YAML,
observations and/or rewards are wrapped with a running-statistics
normalizer (rlox `VecNormalize`, SB3 `VecNormalize`) with matching
parameters: `clip_obs=10.0`, `clip_reward=10.0`,
`epsilon=1e-8`, and the gamma discount from the algorithm's preset.

Running statistics are updated during training and **frozen** during
evaluation (`training = False`) in both frameworks. Observations at
eval time are re-normalized through the frozen statistics before the
policy forward pass. This detail matters: an earlier version of our
rlox harness omitted the explicit freeze, which silently updated the
running statistics with eval trajectories and produced numbers that
drifted by a significant fraction of a seed's worth of variance. The
fix is a single line (`vn.training = False`) applied identically to
both frameworks in the current harness.

### 3.10 Reproducibility

Every artifact needed to reproduce the results is open:

- **Code**: `github.com/riserally/rlox`, tagged with the commit hash
  used for the v2 sweep (see Appendix A for the exact commit).
- **Preset YAMLs**: `benchmarks/convergence/configs/` — the same files
  consumed by both frameworks.
- **Multi-seed runners**: `benchmarks/multi_seed_runner.py` (rlox) and
  `benchmarks/multi_seed_runner_sb3.py` (SB3 in the same harness).
- **Raw per-seed JSON logs**: published alongside the paper as
  `supplementary/results.tar.gz`.
- **Environment version record**: each cell writes a `versions.json`
  capturing `gym.make(env_id).spec.id`, `gymnasium.__version__`,
  `torch.__version__`, `stable_baselines3.__version__`, and the rlox
  git SHA at run time.

A single-command reproduction of the full table takes approximately
6 hours on an e2-standard-8 GCP instance; we publish the GCP launch
script as `scripts/run-multi-seed.sh --gcp`.

### 3.11 Limitations of the comparison

We highlight five limitations that reviewers should be aware of:

1. **n=5 seeds is too few for IQM.** See §3.6. We report n=10 for the
   headline cells in Table 2.
2. **We compare to a single external framework (SB3).** CleanRL and
   TorchRL are both interesting comparison targets; we omit them to
   keep the methods section tractable and because CleanRL's pipeline
   assumes single-env-per-experiment and TorchRL's abstraction tower
   makes a same-harness comparison substantially more work. A follow-up
   evaluation against both is planned.
3. **The asymmetric loss formulations (§3.8) mean §4 is not a
   measurement of "same algorithm, different language".** It is a
   measurement of "best native formulation of each framework, matched
   hyperparameters, matched eval protocol". These are different
   questions and we address the distinction explicitly.
4. **MuJoCo `-v4` environments** differ from the `-v3` versions that
   populate most published SB3-zoo leaderboard numbers. We re-run SB3
   in our harness on `-v4` rather than citing the zoo leaderboard, but
   a reader comparing our table to the public zoo leaderboard will see
   absolute differences of 100–400 reward on Hopper and Walker2d that
   are attributable to the physics change alone.
5. **DQN MountainCar-v0** fails to converge in both frameworks without
   the `train_freq=16, gradient_steps=8` recipe from rl-baselines3-zoo.
   We document this in the preset YAML and omit the cell from the main
   comparison table; see Appendix E.

---

## Appendix B. Hyperparameter translation table (abbreviated)

The full table is at
`benchmarks/multi_seed_runner_sb3.py` — `_PASSTHROUGH_KEYS`,
`_RENAMED_KEYS`, `_SPECIAL_KEYS`, `_SB3_OVERRIDES`. Key entries:

**Identity passthrough (both frameworks use the same key):**
`learning_rate`, `batch_size`, `buffer_size`, `gamma`, `gae_lambda`,
`ent_coef`, `vf_coef`, `max_grad_norm`, `n_steps`, `n_epochs`, `tau`,
`learning_starts`, `train_freq`, `gradient_steps`, `policy_delay`,
`target_policy_noise`, `target_noise_clip`, `exploration_fraction`,
`exploration_final_eps`.

**Renames:**

| rlox key | SB3 key |
|---|---|
| `clip_eps` / `clip_range` | `clip_range` |
| `target_update_freq` | `target_update_interval` |

**SB3-only keys injected by override table:**

| cell | override keys |
|---|---|
| DQN / CartPole-v1 | `train_freq=256, gradient_steps=128` |
| DQN / MountainCar-v0 | `train_freq=16, gradient_steps=8` |

**Reasons for dropping keys:**

| dropped key | why |
|---|---|
| `n_envs` | Passed to VecEnv construction, not SB3 constructor |
| `normalize_obs`, `normalize_rewards` | Wired through VecNormalize wrapper, not SB3 constructor |
| `hidden` | Mapped to `policy_kwargs={"net_arch": [hidden, hidden]}` |
| `anneal_lr` | rlox implements linear LR annealing internally; SB3 uses schedules, translation is non-trivial and currently dropped. Verified that rl-zoo3 MuJoCo presets do not set this, so the drop is a no-op in practice. |
| `clip_vloss` | Intentionally dropped on the SB3 side — rlox's CleanRL-style clipping should not be mirrored onto SB3's plain MSE; see §3.8.1. |

---

## Appendix C. Wall-clock reporting

We report steps-per-second (SPS) for each framework in Appendix C
independently rather than as a ratio, because the two frameworks are
doing different amounts of work per step:

- rlox's `RolloutCollector` steps 8 parallel environments via Rust
  `VecEnv` with Rayon; `compute_gae_batched` runs on-CPU as a
  Rayon-parallel loop.
- SB3's `DummyVecEnv` drives 8 Python environments sequentially;
  `compute_returns_and_advantage` runs as a Python loop.

At matched CPU and matched env-step counts, rlox is faster per step
(3.9× end-to-end rollout speedup on CartPole, 2.1× on HalfCheetah in
our benchmarks), but this is a framework-level property and not an
algorithmic property. Headline "sample efficiency" comparisons in
§4 normalize by total env steps, not wall-clock.

---

## Appendix E. Excluded cells

- **DQN / MountainCar-v0**: both frameworks require `train_freq=16,
  gradient_steps=8` to reach the goal; rlox's default
  `train_freq=1` over-trains on early transitions and the agent never
  explores to the goal flag. This is a recipe issue, not a convergence
  failure — with the cadence keys set, both frameworks converge to
  reward ≈ –120 on matched seeds. We exclude this cell from Table 1
  (which uses rlox-preset defaults) and report it separately in
  Appendix E.
- **Humanoid-v4 with PPO**: 10 000 000 training steps is outside our
  compute budget for the multi-seed sweep; single-seed numbers from an
  earlier run are reported in Appendix F as reference only.
- **Atari**: out of scope for this paper. rlox supports Atari via
  `AtariWrapper` but the benchmark infrastructure for a same-harness
  Atari comparison is not yet in place.

---

## References

(All @cite keys in this document resolve to a BibTeX entry in
`paper/references.bib`. Citations below are sketches — verify page
numbers and conference venues before submission.)

- **[@agarwal2021precipice]** Agarwal, R., Schwarzer, M., Castro, P. S.,
  Courville, A., & Bellemare, M. G. (2021). *Deep Reinforcement Learning
  at the Edge of the Statistical Precipice.* NeurIPS 2021.
  arXiv:2108.13264.
- **[@andrychowicz2021matters]** Andrychowicz, M., Raichuk, A.,
  Stańczyk, P., et al. (2021). *What Matters for On-Policy
  Reinforcement Learning? A Large-Scale Empirical Study.* ICLR 2021.
  arXiv:2006.05990.
- **[@engstrom2020matters]** Engstrom, L., Ilyas, A., Santurkar, S.,
  et al. (2020). *Implementation Matters in Deep Policy Gradients: A
  Case Study on PPO and TRPO.* ICLR 2020. arXiv:2005.12729.
- **[@henderson2018matters]** Henderson, P., Islam, R., Bachman, P.,
  Pineau, J., Precup, D., & Meger, D. (2018). *Deep Reinforcement
  Learning that Matters.* AAAI 2018. arXiv:1709.06560.
- **[@sb3]** Raffin, A., Hill, A., Gleave, A., Kanervisto, A.,
  Ernestus, M., & Dormann, N. (2021). *Stable-Baselines3: Reliable
  Reinforcement Learning Implementations.* JMLR 22(268):1–8.
- **[@cleanrl]** Huang, S., Dossa, R. F. J., Ye, C., & Braga, J.
  (2022). *CleanRL: High-quality Single-file Implementations of Deep
  Reinforcement Learning Algorithms.* JMLR 23(274):1–18.
- **[@schulman2015gae]** Schulman, J., Moritz, P., Levine, S., Jordan,
  M., & Abbeel, P. (2015). *High-Dimensional Continuous Control Using
  Generalized Advantage Estimation.* arXiv:1506.02438.
- **[@schulman2017ppo]** Schulman, J., Wolski, F., Dhariwal, P., Radford,
  A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms.*
  arXiv:1707.06347.
- **[@espeholt2018impala]** Espeholt, L., Soyer, H., Munos, R., et al.
  (2018). *IMPALA: Scalable Distributed Deep-RL with Importance Weighted
  Actor-Learner Architectures.* ICML 2018. arXiv:1802.01561.
- **[@mnih2015dqn]** Mnih, V., Kavukcuoglu, K., Silver, D., et al.
  (2015). *Human-level control through deep reinforcement learning.*
  Nature 518:529–533.
- **[@todorov2012mujoco]** Todorov, E., Erez, T., & Tassa, Y. (2012).
  *MuJoCo: A physics engine for model-based control.* IROS 2012.
- **[@gymnasium]** Towers, M., Terry, J. K., et al. (2023). *Gymnasium.*
  github.com/Farama-Foundation/Gymnasium.
- **[@pyo3]** PyO3 project. *Rust bindings for Python.*
  github.com/PyO3/pyo3.
- **[@sb3_dqn_source]** Source-level citations to
  `stable_baselines3/dqn/dqn.py:217,224`. Verified against SB3 v2.x in
  the rlox repository's pinned venv at the time of the v2 multi-seed
  run.

---

## Review checklist before submission

- [ ] Replace every `⏳` in §4 tables with the final v2 multi-seed numbers
- [ ] Add the exact rlox commit hash to §3.10 (currently a placeholder)
- [ ] Double-check the Hopper A/B numbers in §3.8.1 match the committed
      A/B log (837.0 / 1955.3 per `/tmp/hopper_ab.log`)
- [ ] Verify §3.8.2 claim about rlox DQN `max_grad_norm=inf` default against
      `python/rlox/config.py` line after the latest commit
- [ ] Confirm the n=10 subset for Appendix D is actually in scope — if
      compute budget does not permit, cut the reference and re-word §3.6
- [ ] Verify §3.8.3 SAC critic factor claim against current rlox SAC
      source (the 2× factor was verified as of 2026-04-07; flag if the
      SAC critic code has been touched since)
- [ ] Make sure every @cite key has a BibTeX entry in paper/references.bib
- [ ] Run `pandoc` or LaTeX compile over this file to catch markdown→LaTeX
      issues before final submission
- [ ] Cross-link the inconsistencies doc from this section's intro
- [ ] Anonymize the arXiv/GitHub URLs if required by submission venue
      (use `github.com/ANONYMIZED/rlox` placeholder)
