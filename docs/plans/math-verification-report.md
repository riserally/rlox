# Mathematical Verification Report

**Date**: 2026-03-29
**Scope**: All RL definitions, equations, pseudocode, hyperparameters, and code examples across rlox educational content.

---

## Verification Summary

| File | Status | Issues Found |
|------|--------|-------------|
| `tutorials/policy-gradient-fundamentals.md` | Fixed | 1 issue (Rust GAE API mismatch) |
| `algorithms/vpg.md` | Verified correct | 0 |
| `algorithms/ppo.md` | Verified correct | 0 |
| `algorithms/sac.md` | Verified correct | 0 |
| `algorithms/dqn.md` | Verified correct | 0 |
| `algorithms/td3.md` | Verified correct | 0 |
| `algorithms/a2c.md` | Verified correct | 0 |
| `algorithms/trpo.md` | Verified correct | 0 |
| `algorithms/impala.md` | Fixed | 1 issue (double rho_t in policy loss) |
| `algorithms/dreamer.md` | Verified correct | 0 |
| `algorithms/mappo.md` | Verified correct | 0 |
| `algorithms/index.md` | Verified correct | 0 |
| `math-reference.md` | Verified correct | 0 |
| `learning-path.md` | Verified correct | 0 |
| `rust-guide.md` | Fixed | 1 issue (V-trace missing dones param + wrong types) |

**Total issues found**: 3 (all fixed)
- CRITICAL: 0
- HIGH: 2
- MEDIUM: 1
- LOW: 0

---

## Issues Found and Fixed

### Issue 1: IMPALA Pseudocode Double rho_t Weighting

- **File**: `docs/algorithms/impala.md`, line 57
- **Severity**: HIGH (misleading pseudocode)
- **What was wrong**: The policy loss in the pseudocode was:
  ```
  L_policy = -mean(rho_t * log pi_theta(a|s) * A_t)
  ```
  But `A_t` was already defined (line 55) as `rho_t * (r_t + gamma * v_{t+1} - V(x_t))`, which already includes the importance weight `rho_t`. Multiplying by `rho_t` again in the loss produces a double-weighting.
- **Correct version**:
  ```
  L_policy = -mean(log pi_theta(a|s) * A_t)
  ```
  Since `A_t` already contains the `rho_t` clipped importance weight per the V-trace policy gradient formula (Espeholt et al., 2018, Eq. 2).
- **Cross-reference**: `math-reference.md` Eq. 37 is correct: `A_t^pg = rho_bar_t * (r_t + gamma * v_{t+1} - V(x_t))`, and the policy gradient (Eq. in IMPALA section of the algorithms page, line 30) correctly shows `rho_t * nabla log pi * (r_t + gamma * v_{t+1} - V(x_t))` without double-counting.
- **Status**: FIXED

### Issue 2: Tutorial Rust GAE Signature Mismatch

- **File**: `docs/tutorials/policy-gradient-fundamentals.md`, lines 342-369
- **Severity**: HIGH (code example won't compile against actual API)
- **What was wrong**: The Rust code example showed:
  ```rust
  pub fn compute_gae(
      rewards: &[f64],
      values: &[f64],   // len = T + 1
      dones: &[f64],
      gamma: f64,
      gae_lambda: f64,
  ) -> (Vec<f64>, Vec<f64>)
  ```
  The actual implementation in `crates/rlox-core/src/training/gae.rs` takes `last_value: f64` as a separate parameter, and `values` has length T (not T+1):
  ```rust
  pub fn compute_gae(
      rewards: &[f64],
      values: &[f64],    // len = T
      dones: &[f64],
      last_value: f64,   // V(s_{T+1}) bootstrap
      gamma: f64,
      gae_lambda: f64,
  ) -> (Vec<f64>, Vec<f64>)
  ```
  The loop logic also differed: the actual code peels the last step to avoid a branch, whereas the tutorial had a simpler but API-incompatible version.
- **Correct version**: Updated to match actual implementation signature and logic.
- **Status**: FIXED

### Issue 3: Rust Guide V-trace Missing `dones` Parameter and Wrong Types

- **File**: `docs/rust-guide.md`, lines 302-317
- **Severity**: MEDIUM (code example won't compile against actual API)
- **What was wrong**: The code example showed:
  ```rust
  let (vs, pg_advantages) = compute_vtrace(
      log_rhos,
      rewards,
      values,
      2.0,   // bootstrap_value
      0.99,  // gamma
      1.0,   // rho_bar
      1.0,   // c_bar
  ).unwrap();
  ```
  The actual implementation in `crates/rlox-core/src/training/vtrace.rs` takes 8 parameters including a `dones` slice, and operates on `f32` (not `f64`):
  ```rust
  pub fn compute_vtrace(
      log_rhos: &[f32],
      rewards: &[f32],
      values: &[f32],
      dones: &[f32],
      bootstrap_value: f32,
      gamma: f32,
      rho_bar: f32,
      c_bar: f32,
  ) -> Result<(Vec<f32>, Vec<f32>), RloxError>
  ```
- **Correct version**: Added `dones` parameter and explicit `f32` type annotations.
- **Status**: FIXED

---

## Detailed Verification Results

### 1. Mathematical Correctness

#### Policy Gradient Theorem
- **tutorial/policy-gradient-fundamentals.md**: Verified correct. The score function trick derivation (lines 52-76), trajectory factorization (lines 80-93), and REINFORCE estimator (lines 107-119) are all standard and correct per Williams (1992).
- **math-reference.md**: Eq. 10 matches standard form.

#### GAE Formula
- **tutorial/policy-gradient-fundamentals.md**: Eq at line 277-281 is correct: `A_t^GAE = sum (gamma*lambda)^l * delta_{t+l}`. Recursive form (line 296-298) is correct.
- **math-reference.md**: Eqs. 1-4 are correct, including the termination mask product in Eq. 2 (more general than the tutorial's simplified form, both valid).
- **algorithms/ppo.md**: GAE formula (lines 24-27) correct.
- **algorithms/a2c.md**: GAE formula (lines 17-19) correct.
- **Actual Rust implementation**: Matches the recursive formula exactly.

#### PPO Clipped Objective
- **tutorial/policy-gradient-fundamentals.md**: Line 425-429, correctly uses `min(r*A, clip(r,1-eps,1+eps)*A)`.
- **algorithms/ppo.md**: Line 12, correct.
- **math-reference.md**: Eq. 5, correct.
- The `min` operation is correct (not `max`). When advantage is positive, `min` prevents the ratio from exceeding `1+eps`; when negative, it prevents the ratio from going below `1-eps`.

#### PPO Total Loss Sign Convention
- **algorithms/ppo.md**: Line 45: `loss = -L_clip + vf_coef * L_vf + ent_coef * L_ent` where `L_ent = -H[pi]`. This means `ent_coef * L_ent = -ent_coef * H`, so entropy is effectively subtracted from loss (= added to objective). Correct.
- **math-reference.md**: Eq. 8: `L = -L^CLIP + c_v * L^V - c_h * L^H`. Since `L^H = E[H[pi]]` (positive), `-c_h * L^H` subtracts entropy from loss = adds to objective. Correct.
- **tutorial/policy-gradient-fundamentals.md**: Line 483: `loss = policy_loss + vf_coef * vf_loss - ent_coef * entropy.mean()` where `policy_loss = -torch.min(...)` (already negated). So entropy is subtracted from the total loss. Correct and consistent.

#### SAC Objective
- **algorithms/sac.md**: Line 11-13, `J(pi) = sum E[r + alpha * H(pi)]`. Entropy is added to return (maximized). Correct.
- **math-reference.md**: Eq. 12, identical. Correct.
- **SAC Bellman backup** (sac.md line 18): `Q_target = r + gamma * (min Q' - alpha * log pi)`. The `-alpha * log pi` is correct since `H = -E[log pi]`, so `V(s) = E[Q - alpha*log pi]`. Correct.
- **SAC actor loss** (sac.md line 26): `L_pi = E[alpha * log pi - min Q]`. Minimizing this maximizes Q while maximizing entropy. Correct.
- **Auto-entropy** (sac.md line 32): `L(alpha) = -alpha * E[log pi + H_bar]`. Correct per Haarnoja et al. (2018b).

#### DQN Target
- **algorithms/dqn.md**: Standard target (line 12) and Double DQN (line 18) are both correct.
- **math-reference.md**: Eqs. 25-27 correct.

#### TD3 Target Smoothing
- **algorithms/td3.md**: Line 18: `a' = pi_target(s') + clip(eps, -c, c)` where `eps ~ N(0, sigma^2)`. Correct per Fujimoto et al. (2018).
- **math-reference.md**: Eq. 21-22 correct.

#### V-trace
- **algorithms/impala.md**: V-trace target (lines 11-15), TD (lines 17-19), clipped weights (lines 22-24) all correct per Espeholt et al. (2018).
- **math-reference.md**: Eqs. 33-37 correct. Backward recursion Eq. 36 matches the actual Rust implementation.

#### TRPO KL Constraint
- **algorithms/trpo.md**: Constrained optimization (lines 11-16) correct. Natural gradient step (lines 20-23) correct: `Delta_theta = sqrt(2*delta / (g^T F^{-1} g)) * F^{-1} g`. This is the standard TRPO update.
- **math-reference.md**: Not covered separately (TRPO section defers to algorithm page).

#### DreamerV3 RSSM
- **algorithms/dreamer.md**: Prior/posterior distributions (lines 12-21) correctly described. The posterior `q(z|h,o)` conditions on observation, prior `p(z|h)` does not. KL balancing formula (lines 31-33) with stop-gradient correct per Hafner et al. (2023).

#### MAPPO Centralized Critic
- **algorithms/mappo.md**: Per-agent clipped objective (lines 14-15) correct. Centralized advantage using global state `V_phi(s)` (line 22) correct. The reward `r_t^i` is per-agent while V is global, which is the standard CTDE formulation.

### 2. Notation Consistency

| Symbol | Expected | Consistent? |
|--------|----------|------------|
| theta | Policy parameters | Yes -- used consistently across all files |
| phi | Value/critic parameters | Yes -- used for V_phi and Q_phi |
| pi | Policy | Yes |
| V | State value function | Yes |
| Q | Action-value function | Yes |
| gamma | Discount factor | Yes |
| lambda | GAE lambda | Yes -- `lambda` in equations, `gae_lambda` in code |
| alpha | SAC temperature / PER exponent | Context-dependent but consistent within each algorithm |
| tau | Polyak coefficient | Yes |
| epsilon (eps) | PPO clip range / DQN exploration | Context-dependent but clear in context |

- **No notation inconsistencies found.** The notation table in `math-reference.md` is accurate and all algorithm pages follow it.

### 3. Pseudocode Accuracy

#### VPG (vpg.md)
- Steps in correct order: collect, compute returns, compute advantages, policy update, value update. Verified correct.

#### PPO (ppo.md)
- Correct: collect, GAE, normalize, multi-epoch minibatch SGD with clipping. Matches implementation pattern.

#### SAC (sac.md)
- Correct order: collect, critic update, actor update, entropy tuning, target update. Matches standard SAC.

#### DQN (dqn.md)
- Correct: epsilon-greedy, replay, Double DQN selection, target update. Matches implementation.

#### TD3 (td3.md)
- Correct: exploration noise, target smoothing, twin critic, delayed actor update. Matches Fujimoto et al.

#### A2C (a2c.md)
- Correct: single gradient step, no minibatches/epochs. Matches config defaults.

#### TRPO (trpo.md)
- Correct: CG solve, backtracking line search, separate VF update. Matches standard TRPO.

#### IMPALA (impala.md)
- **Fixed**: policy loss had double rho_t weighting (see Issue 1 above). Now correct.

#### DreamerV3 (dreamer.md)
- Correct: world model train, imagination, actor-critic in latent space.

#### MAPPO (mappo.md)
- Correct: per-agent PPO with centralized critic.

### 4. Common RL Errors Check

| Check | Result |
|-------|--------|
| Advantage normalized before PPO update? | Yes -- `ppo.md` line 37: "normalize advantages (if enabled)", config default `normalize_advantages=True` |
| Entropy bonus sign correct? | Yes -- subtracted from loss = added to objective (verified in PPO, A2C, IMPALA) |
| Value loss coefficient applied? | Yes -- `vf_coef * L_vf` in all on-policy pseudocode |
| Gradient clipping mentioned? | Yes -- all on-policy algorithms mention `max_grad_norm` |
| Terminated vs truncated distinction? | Partially -- `rust-guide.md` documents `Transition` with both `terminated` and `truncated` fields. The GAE code uses a combined `dones` flag (standard practice). Tutorial does not discuss the distinction explicitly. |
| Bootstrap at truncation? | The `VecEnv` section in `rust-guide.md` (line 111) documents `terminal_obs` for bootstrapping. Correct. |

### 5. Code Example Accuracy (Rust Guide)

| Example | Import Paths | Signatures | Assert Values | Status |
|---------|-------------|------------|---------------|--------|
| CartPole | `rlox_core::env::builtins::CartPole` | `new(Some(42))`, `reset(Some(42))` | obs len=4, reward=1.0 | Verified correct |
| VecEnv | `rlox_core::env::parallel::VecEnv` | `new(envs)`, `reset_all`, `step_all` | Batch fields correct | Verified correct |
| ExperienceRecord | `rlox_core::buffer::ExperienceRecord` | Field names match | -- | Verified correct |
| ExperienceTable | `rlox_core::buffer::columnar::ExperienceTable` | `new(4, 1)` | -- | Verified correct |
| ReplayBuffer | `rlox_core::buffer::ringbuf::ReplayBuffer` | `new(100_000, 4, 1)`, `sample(32, 42)` | len=100_000 | Verified correct |
| PrioritizedReplayBuffer | `rlox_core::buffer::priority::PrioritizedReplayBuffer` | `new(cap, obs, act, alpha, beta)` | -- | Verified correct |
| GAE | `rlox_core::training::gae::compute_gae` | `(rewards, values, dones, last_value, gamma, lambda)` | returns invariant | Verified correct |
| V-trace | `rlox_core::training::vtrace::compute_vtrace` | **Was missing `dones` param and using f64** | -- | **FIXED** |
| RunningStats | `rlox_core::training::normalization::RunningStats` | `new()`, `update`, `mean`, `std`, `count` | mean=2.0, std~0.8165, count=3 | Verified correct |
| GRPO | `rlox_core::llm::ops::compute_group_advantages` | `(rewards)` | sum~0, constant->0 | Verified correct |
| Token KL | `rlox_core::llm::ops::compute_token_kl` | `(log_p, log_q)` | identical->0, mismatch->Err | Verified correct |

### 6. Hyperparameter Tables vs Config Defaults

All hyperparameter tables were verified against the actual `config.py` dataclass defaults.

#### PPO (ppo.md vs PPOConfig)
All 16 parameters match exactly. Verified correct.

#### SAC (sac.md vs SACConfig)
All 9 parameters match. Verified correct.

#### DQN (dqn.md vs DQNConfig)
All 17 parameters match. Verified correct.

#### TD3 (td3.md vs TD3Config)
All 11 parameters match. Verified correct.

#### A2C (a2c.md vs A2CConfig)
All 10 parameters match. Verified correct.

#### TRPO (trpo.md vs TRPOConfig)
All 10 parameters match. Verified correct.

#### IMPALA (impala.md vs IMPALAConfig)
All 11 parameters match. Verified correct.

#### DreamerV3 (dreamer.md vs DreamerV3Config)
All 12 parameters match. Verified correct.

#### MAPPO (mappo.md vs MAPPOConfig)
All 12 parameters match. Verified correct.

---

## Files Modified

1. `docs/algorithms/impala.md` -- Removed duplicate `rho_t` in policy loss pseudocode
2. `docs/tutorials/policy-gradient-fundamentals.md` -- Updated Rust GAE signature to match actual API (separate `last_value` parameter, `values` length T not T+1)
3. `docs/rust-guide.md` -- Added missing `dones` parameter to V-trace example, corrected to `f32` types, added dones documentation

---

## Cross-Consistency Matrix

The following equations appear in multiple files. All are now consistent:

| Equation | Tutorial | Algorithm Page | Math Reference | Status |
|----------|----------|---------------|----------------|--------|
| Policy gradient | Lines 107-119 | vpg.md L9, a2c.md L11 | Eq. 10 | Consistent |
| GAE | Lines 277-281 | ppo.md L24, a2c.md L17 | Eqs. 1-4 | Consistent |
| PPO clip | Lines 425-429 | ppo.md L12 | Eq. 5 | Consistent |
| PPO total loss | Line 483 | ppo.md L20 | Eq. 8 | Consistent |
| SAC objective | -- | sac.md L11 | Eq. 12 | Consistent |
| SAC Bellman | -- | sac.md L18 | Eq. 16 | Consistent |
| DQN target | -- | dqn.md L12 | Eq. 26 | Consistent |
| Double DQN | -- | dqn.md L18 | Eq. 27 | Consistent |
| V-trace | -- | impala.md L11-24 | Eqs. 33-37 | Consistent |
| Polyak update | -- | sac.md L69, td3.md L64 | Eq. 49 | Consistent |
