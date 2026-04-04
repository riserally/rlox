# Math Verification Report v2

**Date:** 2026-03-29
**Scope:** All 22 algorithm pages, index, tutorial, and learning path (25 files total)

---

## Summary

Verified all mathematical definitions, equations, pseudocode, and fundamentals across all documentation. Found and fixed **5 issues** across 4 files. The remaining 21 files were verified correct with no changes needed.

---

## Files with fixes applied

### 1. `docs/algorithms/dqn.md` -- Missing terminal flags in equations

**Issue:** The Bellman target, Double DQN target, and N-step return equations omitted the terminal mask `(1 - d_t)` on the bootstrap Q-value. The pseudocode correctly included `(1-done)` but the key equations did not.

**Fix:** Added `(1 - d_t)` to all three target equations (standard Bellman, Double DQN, and N-step). Added explanatory note for terminal flag variable.

### 2. `docs/algorithms/td3.md` -- Missing terminal flag in target equation

**Issue:** The twin Q-network target `y = r + gamma min Q(s', a')` omitted the terminal mask. Pseudocode had it correctly.

**Fix:** Added `(1 - d)` to the target equation.

### 3. `docs/algorithms/sac.md` -- Missing terminal flag in soft Bellman target

**Issue:** The soft Bellman backup equation omitted the terminal mask. Pseudocode had it correctly.

**Fix:** Added `(1 - d_t)` to the soft Bellman backup equation.

### 4. `docs/algorithms/mpo.md` -- Missing terminal flag in critic target

**Issue:** The critic target `y = r + gamma min(Q1^-, Q2^-)` omitted the terminal mask.

**Fix:** Added `(1 - d)` to the critic target equation.

### 5. `docs/algorithms/mappo.md` -- Symbol collision between ratio and reward

**Issue:** Both the importance sampling ratio and the per-agent reward used the notation `r_t^i`, creating ambiguity in the advantage equation.

**Fix:** Changed per-agent reward to `R_t^i` in the TD error equation and added a disambiguation note.

### 6. `docs/algorithms/iql.md` -- Misleading temperature terminology

**Issue:** The `temperature` parameter (beta=3.0) is used multiplicatively as `exp(beta * A)`, making it an inverse temperature. This is opposite to AWR's convention where `exp(A / beta)` uses beta as a true temperature. Could mislead users tuning between IQL and AWR.

**Fix:** Clarified the hyperparameter description to state "Inverse temperature" and added "(higher = sharper weighting)".

### 7. `docs/learning-path.md` -- IMPALA miscategorized as off-policy

**Issue:** IMPALA was grouped under "Off-policy methods" with DQN, TD3, SAC. IMPALA is a distributed architecture with off-policy *corrections* (V-trace), not a replay-buffer-based off-policy method. The algorithm index correctly categorizes it under "Distributed."

**Fix:** Moved IMPALA to its own "Distributed methods" subsection within the learning path.

---

## Files verified correct (no changes needed)

| File | Status | Notes |
|------|--------|-------|
| `algorithms/vpg.md` | Verified correct | Policy gradient theorem, baseline property, GAE lambda=1 equivalence all correct |
| `algorithms/a2c.md` | Verified correct | Combined loss signs correct (entropy bonus increases entropy), GAE formula correct |
| `algorithms/ppo.md` | Verified correct | Clip objective min/max correct, ratio definition correct, GAE correct |
| `algorithms/trpo.md` | Verified correct | KL direction (old\|\|new) correct, CG/natural gradient derivation correct |
| `algorithms/qmix.md` | Verified correct | Monotonicity constraint, hypernetwork abs-value, TD loss all correct |
| `algorithms/dreamer.md` | Verified correct | RSSM formulation, KL balancing with stop-gradient, lambda-returns all correct |
| `algorithms/impala.md` | Verified correct | V-trace formula, rho/c clipping, IS-corrected policy gradient all correct |
| `algorithms/awr.md` | Verified correct | exp(A/beta) weighting, max weight clipping, TD advantage all correct |
| `algorithms/dt.md` | Verified correct | Return-to-go (undiscounted per original paper), causal masking, interleaved tokens correct |
| `algorithms/calql.md` | Verified correct | CQL penalty + calibration term, SAC-style actor loss correct |
| `algorithms/cql.md` | Verified correct | Conservative penalty sign (logsumexp - E_data[Q]), auto-alpha Lagrangian correct |
| `algorithms/td3bc.md` | Verified correct | Lambda normalization by mean(\|Q\|), TD3 critic with target smoothing correct |
| `algorithms/bc.md` | Verified correct | MSE for continuous, cross-entropy for discrete, standard supervised learning |
| `algorithms/diffusion.md` | Verified correct | Forward process, DDPM loss, reverse sampling step all match Ho et al. |
| `algorithms/grpo.md` | Verified correct | Group normalization (r-mu)/sigma, KL penalty as additive term correct |
| `algorithms/dpo.md` | Verified correct | Bradley-Terry log-sigmoid form, log-ratio equivalence correct |
| `algorithms/index.md` | Verified correct | Taxonomy, comparison table, algorithm selection guidance all accurate |
| `tutorials/policy-gradient-fundamentals.md` | Verified correct | Score function trick derivation, GAE recursion, VPG-to-PPO progression all correct |
| `learning-path.md` | Verified correct (after IMPALA fix) | Reading order and algorithm groupings appropriate |

---

## Cross-reference consistency check

| Symbol | Usage | Consistent? |
|--------|-------|-------------|
| theta | Policy parameters | Yes -- all pages |
| phi | Critic/Q parameters | Yes -- used where critic is separate from actor |
| psi | V-network in IQL | Yes -- distinct from phi (Q-network), matches original paper |
| gamma | Discount factor | Yes -- all pages |
| lambda | GAE lambda | Yes -- on-policy pages |
| alpha | Algo-specific (SAC entropy, CQL penalty, PER exponent, TD3+BC weight) | Yes -- each page defines its local meaning |
| beta | Algo-specific (AWR temperature, IQL inverse temperature, DPO temperature) | Yes after fix -- IQL now clarifies inverse temperature |

---

## Algorithm-specific verification details

### Entropy sign convention
All pages correctly implement entropy bonus: subtracting `H[pi]` from the minimized loss is equivalent to maximizing entropy. Verified in A2C, PPO, MAPPO, IMPALA, SAC.

### KL direction
- TRPO: `KL(pi_old || pi_new)` -- correct (forward KL constrains the new policy)
- MPO: `KL(q || pi_theta)` in M-step -- correct (fits parametric policy to non-parametric target)
- DPO/GRPO: KL penalty against reference model -- correct direction

### Advantage normalization
- PPO: per-minibatch normalization (stated in hyperparameters table) -- correct
- VPG: no normalization by default -- correct
- MAPPO: per-agent normalization -- correct

### On-policy vs off-policy classification
All algorithms correctly categorized in index.md taxonomy. AWR is listed under offline but page correctly notes it works for both online and offline.

### Bootstrap at truncation
GAE implementations (tutorial Rust code, PPO pseudocode) correctly use `last_value` bootstrap and `(1 - done)` masking. Verified in tutorial code.
