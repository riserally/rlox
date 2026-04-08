# rlox — agent quick reference

> Read this FIRST before any code-writing or review task. It replaces ~8
> discovery calls (Grep/Glob/Read of layout files) with one read. If a fact
> here is stale, fix the fact before writing new code.
>
> This file is **machine-oriented** — terse, factual, no marketing.
> For a human-facing overview, read `README.md`.

---

## What this is

Rust-accelerated reinforcement learning framework. Python control plane +
Rust data plane. The Polars pattern: **any data transform without autograd
belongs in Rust**. Autograd (policies, losses, optimizers) lives in PyTorch.

22 RL algorithms share a unified `Trainer('algo', env=..., config=...)` API.
Convergence parity with Stable-Baselines3 on the core six (PPO, SAC, TD3,
DQN, A2C, TRPO).

---

## Critical commands

| purpose | command |
|---|---|
| Install (editable, rebuild Rust) | `maturin develop --release` |
| Run unit tests | `./.venv/bin/python -m pytest tests/python/ -q` |
| Run slow / integration tests | `./.venv/bin/python -m pytest -m slow` |
| Run Rust tests | `cargo test --workspace` |
| Python lint | `ruff check python/ tests/` |
| Rust lint | `cargo clippy --workspace -- -D warnings` |
| Format | `ruff format python/ tests/ && cargo fmt --all` |
| Run a single benchmark cell | `./.venv/bin/python benchmarks/multi_seed_runner.py --algo ppo --env Hopper-v4 --timesteps 100000 --seeds 1` |
| Multi-seed convergence (local) | `cd ../rlox-priv && bash scripts/run-multi-seed.sh --local` |
| Multi-seed convergence (GCP) | `cd ../rlox-priv && bash scripts/run-multi-seed.sh --gcp` |
| Aggregate historic results | `./.venv/bin/python scripts/inspect_results.py` |

Python interpreter: always invoke via `./.venv/bin/python`, never bare
`python3` (the venv has the editable `rlox` install and MuJoCo).

---

## Directory map (1-line each — keep stable)

### Python control plane (`python/rlox/`)
- `trainer.py`                  — unified `Trainer('ppo'|'sac'|...)` entrypoint + `ALGORITHM_REGISTRY`
- `config.py`                   — `PPOConfig` / `SACConfig` / ... dataclasses with YAML alias support
- `losses.py`                   — `PPOLoss` (SB3-aligned, see "Non-obvious facts")
- `collectors.py`               — `RolloutCollector` (on-policy rollouts, GAE via Rust)
- `vec_normalize.py`            — SB3-compatible running obs/reward normalization
- `gym_vec_env.py`              — Gymnasium `SyncVectorEnv` wrapper matching `rlox.VecEnv` interface
- `policies.py`                 — `DiscretePolicy`, `ContinuousPolicy` (orthogonal init, Tanh)
- `checkpoint.py`               — `safe_torch_load` with `allow_unsafe` opt-in
- `algorithms/`                 — 22 self-contained algo files (`ppo.py`, `sac.py`, `dqn.py`, ...)
- `callbacks.py`, `logging.py`  — callback protocol + console/TB/MLflow loggers

### Rust data plane (`crates/`)
- `rlox-core/`      — GAE, replay buffer, running stats, env physics; pure Rust, no PyO3
- `rlox-python/`    — PyO3 bindings (`rlox._rlox_core`)
- `rlox-candle/`    — Candle NN backend (experimental)
- `rlox-burn/`      — Burn NN backend (experimental)
- `rlox-grpc/`      — distributed actor/learner via tonic
- `rlox-bench/`     — microbenchmarks (criterion)
- `rlox-nn/`        — shared NN primitives

### Benchmarks, tests, docs
- `benchmarks/multi_seed_runner.py`    — 5-seed IQM + bootstrap CI harness (auto-resolves preset YAMLs)
- `benchmarks/convergence/configs/`    — per-(algo, env) YAML presets (`ppo_hopper.yaml`, ...)
- `benchmarks/convergence/rlox_runner.py` — legacy single-seed runner (kept for equivalence tests)
- `tests/python/`                      — pytest suite, `-m slow` for long integration runs
- `docs/plans/`                        — live plan + review markdown (read these before proposing changes)
- `scripts/inspect_results.py`         — aggregator across v5/v6/v7/v8 single-seed JSON logs

### Companion repository
- `../rlox-priv/`                      — GCP launch scripts, non-public benchmark data, paper drafts
- `../rlox-priv/scripts/run-multi-seed.sh` — GCP sweep launcher (`--local` or `--gcp`)
- `../rlox-priv/results/`              — archived benchmark runs

---

## Conventions (enforced)

- **TDD**: Red-Green-Refactor. No new code without a test. No test, no commit.
- **Never attribute code changes to Claude** in commit messages or file headers.
- **No mocking the database** in integration tests (we got burned once).
- **Type hints**: PEP 484/604/695. No bare `Any` except at FFI boundaries.
- **Async**: Tokio in Rust, `asyncio` in Python. No blocking calls inside async functions.
- **Error handling**: `thiserror` per-module in Rust, typed exceptions in Python.
  Never use bare `except:`.
- **Data plane rule**: if it's data transformation without autograd, it belongs in Rust.
- **Configurability**: prefer configurable parameters over hardcoded seeds, rep counts, magic numbers.

---

## Non-obvious facts (common gotchas)

- **`PPOLoss` has NO inner 0.5 factor** on the value loss. We follow SB3's
  `F.mse_loss` convention, not CleanRL's `0.5 * MSE`. `vf_coef=0.5` here
  matches `vf_coef=0.5` in SB3. See commit `9a7d628` and
  `docs/plans/results-inspection-2026-04-06.md`.
- **`PPOConfig.clip_vloss` default is `False`** (matches SB3 `clip_range_vf=None`).
  Flip to `True` for the CleanRL max-of-clipped formulation.
- **DQN defaults `train_freq=1, gradient_steps=1`** (1 grad step per env step).
  Set `train_freq=16, gradient_steps=8` for MountainCar or the agent over-trains
  on early transitions and never explores to the goal. See commit `17cdfeb`.
- **`Pendulum-v1` and `CartPole-v1` have native Rust envs** (see `_NATIVE_ENV_IDS`
  in `collectors.py`). When wrapped with `VecNormalize`, discreteness MUST be
  detected from the actual `action_space` object, not the env_id — see commit
  `1d882e8`.
- **`multi_seed_runner.py` auto-resolves per-(algo, env) preset YAMLs** from
  `benchmarks/convergence/configs/<algo>_<env_short>.yaml`. Without this,
  Trainer defaults (CleanRL CartPole-tuned) silently ran on MuJoCo.
- **Eval episode seeding**: use `seed=base+ep`, not `seed=base`, in eval loops.
  Identical resets across episodes inflate pseudo-precision of per-seed std.
- **Truncation bootstrap**: `RolloutCollector` adds `gamma * V(terminal_obs)`
  to the reward on truncation (time-limit), passing only `terminated` (not
  `done`) to GAE. This matters a lot for HalfCheetah (always truncates at 1000).
- **Checkpoints are secure by default**: `safe_torch_load(path)` uses
  `weights_only=True`. Pass `allow_unsafe=True` explicitly for legacy checkpoints.

---

## Where "the rules" live

| scope | file |
|---|---|
| User global prefs | `~/.claude/CLAUDE.md` |
| Workspace-wide | `/Users/wojciechkowalinski/Sync/work/rlox-workspace/CLAUDE.md` |
| Auto-memory index | `~/.claude/projects/-Users-wojciechkowalinski-Sync-work-rlox-workspace/memory/MEMORY.md` |
| Live plans | `docs/plans/*.md` (read before proposing changes) |

---

## Load-bearing plan docs

Read these before proposing architectural or experimental changes:

- `docs/plans/results-inspection-2026-04-06.md` — current experimental state, SB3 alignment evidence
- `docs/plans/convergence-gap-investigation.md` — PPO vs SB3 MuJoCo history
- `docs/plans/master-improvement-plan.md` — the north-star roadmap
- `docs/plans/paper-implementation.md` — paper-track work queue
- `../rlox-priv/docs/plans/multi-seed-pre-flight-review-2026-04-06.md` — latest methodology critique

---

## Current state snapshot (update when it changes)

- **PPO/SAC/TD3/A2C/DQN** all converge with default configs on their representative envs.
- **DQN MountainCar** requires `train_freq=16, gradient_steps=8`; still ~35 reward behind SB3 at single seed, multi-seed pending.
- **Multi-seed GCP sweep** runs via `rlox-priv/scripts/run-multi-seed.sh --gcp`. Auto-resolves preset YAMLs.
- **Trainer↔runner PPO equivalence** pinned by `tests/python/test_ppo_path_equivalence.py` (slow marker, Hopper 24k, 100-reward tolerance).
- **Known weak spot**: DQN MountainCar single-seed -158 vs SB3 -123. Multi-seed should smooth this.
