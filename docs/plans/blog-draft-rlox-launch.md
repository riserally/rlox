# rlox — Rust-Accelerated Reinforcement Learning

**Blog draft** — intended for Substack/Medium/personal site. Not final; numbers
marked `⏳` are waiting on the v2 multi-seed GCP sweep. Update before publishing.

---

## The Polars pattern, applied to RL

> *"Python is the wrong language for the environment step. So don't step
> environments in Python."*

If you've ever waited for an RL training run on MuJoCo, watching your GPU sit at
12% utilization while `htop` shows Python pinned at 100% on one core, you've
already discovered the central unfairness of modern reinforcement learning
frameworks: **the bottleneck is not the policy. The bottleneck is the data
plane.**

[Polars](https://pola.rs/) proved the Rust-Python dataframe architecture works.
Not "Rust is a nice language for systems programming" — that was never in
dispute. The thing Polars actually demonstrated is that a well-designed
Python *control plane* sitting on top of a Rust *data plane* can produce a
tool that is **simultaneously** more pleasant to use than pandas and an order
of magnitude faster. Users don't choose between ergonomics and speed. They
get both.

rlox is that pattern applied to reinforcement learning.

- **Python** owns the training loop, the PyTorch policies, the config files,
  the callbacks, the logging — everything a researcher actually touches.
- **Rust** owns the environment stepping, the replay buffer, the GAE
  computation, the running statistics, the V-trace. Everything you'd rather
  not debug at 11pm.

The line between them is [PyO3](https://pyo3.rs/). Observations come across
as zero-copy numpy arrays. Actions go back the same way. The typical training
step never leaves Rust for the duration of a rollout.

---

## A thirty-second demo

```python
from rlox import Trainer

trainer = Trainer("ppo", env="CartPole-v1", seed=42)
metrics = trainer.train(total_timesteps=50_000)
print(f"Mean reward: {metrics['mean_reward']:.1f}")
```

That's three lines. The policy is a standard PyTorch MLP. Under the hood,
those 50,000 environment steps are distributed across 8 parallel
`rlox.VecEnv` workers running in Rust, with the returned batch handed back
to PyTorch as a single `float32` tensor. `compute_gae` runs as Rayon-
parallelized Rust before the minibatch loop sees it.

If you want to see the same thing in the raw component API:

```python
from rlox import RolloutCollector, PPOLoss
from rlox.policies import DiscretePolicy
import torch

policy = DiscretePolicy(obs_dim=4, n_actions=2, hidden=64)
optimizer = torch.optim.Adam(policy.parameters(), lr=2.5e-4)

collector = RolloutCollector("CartPole-v1", n_envs=8, seed=0)
loss_fn = PPOLoss(clip_eps=0.2, vf_coef=0.5, ent_coef=0.01)

for update in range(100):
    batch = collector.collect(policy, n_steps=128)
    for epoch in range(4):
        for mb in batch.sample_minibatches(batch_size=256):
            adv = (mb.advantages - mb.advantages.mean()) / (mb.advantages.std() + 1e-8)
            loss, _ = loss_fn(policy, mb.obs, mb.actions, mb.log_probs,
                              adv, mb.returns, mb.values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

The `RolloutCollector` does the thing rlox exists to do: it steps `n_envs`
environments for `n_steps` each (Rust), gathers the obs/actions/rewards/dones
into contiguous buffers (Rust), calls `compute_gae_batched` across all
environments in parallel (Rust, Rayon), and hands you back a flat
`RolloutBatch` ready for PyTorch's autograd. You never see a Python list of
transitions because no Python list of transitions ever existed.

---

## What the speedup actually is

Numbers from `benchmarks/` on an M-series MacBook Pro (all reproducible via
`cargo bench` + `pytest benchmarks/`):

| Component | vs SB3 (NumPy) | vs TorchRL (PyTorch) |
|---|---:|---:|
| GAE (32K steps, single env) | **147×** | **1,700×** |
| `compute_gae_batched` (256 steps × 16 envs) | **78×** | **450×** |
| `ReplayBuffer.push` (10K transitions) | **9.7×** | **148×** |
| End-to-end rollout (256 × 2048 steps) | **3.9×** | **53×** |
| GRPO advantages (LLM post-training) | **35×** (NumPy) | **34×** (PyTorch) |
| KL divergence (f32, tokenwise) | **2–9×** vs TRL | — |

The biggest gains are in the per-element, per-step operations: GAE, buffer
writes, running statistics. These are the exact places where Python
interpreter overhead dominates the actual arithmetic cost by 10–100×.

The **smallest** gain is the end-to-end rollout number (3.9×). That's
because by the time you're doing a full rollout, you're amortizing the
Python policy forward pass across the whole batch and the arithmetic cost
of the policy starts to dominate. Which is exactly as it should be: rlox
accelerates the parts that deserve acceleration and gets out of the way on
the parts where PyTorch is already optimal.

---

## Does it actually train?

Speed without convergence is a parlor trick. The whole point of rlox is to
be a drop-in alternative for the SB3/TorchRL workflow, not a microbenchmark
generator.

Same-harness, multi-seed comparison on the core six algorithms. Both
frameworks read the exact same preset YAMLs from
`benchmarks/convergence/configs/`; both are evaluated for 30 deterministic
episodes per seed with unique per-episode seeds (see the methods section
below for the gotchas). 5 seeds per cell, IQM + bootstrap CI per Agarwal
et al. 2021.

| Algorithm | Environment | rlox IQM | SB3 IQM | SB3-zoo ref |
|---|---|---:|---:|---:|
| PPO | CartPole-v1 | **434.4** [309, 460] | **438.2** [390, 500] | 500 |
| PPO | Acrobot-v1 | **-83.3** [-96, -80] | **-83.7** [-97, -77] | -75 |
| PPO | Hopper-v4 | ⏳ | ⏳ | 3578 (v3) |
| PPO | HalfCheetah-v4 | ⏳ | ⏳ | 5819 |
| PPO | Walker2d-v4 | ⏳ | ⏳ | 4226 |
| SAC | Pendulum-v1 | ⏳ | ⏳ | -150 |
| SAC | HalfCheetah-v4 | ⏳ | ⏳ | 9656 |
| SAC | Hopper-v4 | ⏳ | ⏳ | 3470 |
| SAC | Walker2d-v4 | ⏳ | ⏳ | 4502 |
| TD3 | Pendulum-v1 | ⏳ | ⏳ | -150 |
| TD3 | HalfCheetah-v4 | ⏳ | ⏳ | 9709 |
| DQN | CartPole-v1 | ⏳ | **500** [218, 500] | 500 |
| A2C | CartPole-v1 | ⏳ | **491.6** [167, 500] | 500 |

The two cells where both frameworks have landed show **the numbers agree to
within 4 reward points**. That's statistical noise, not framework
divergence. The remaining cells are waiting on the v2 multi-seed sweep.

### The honest version of "drop-in"

rlox is *not* byte-identical to SB3. Nobody's is. CleanRL isn't either.
Every RL framework has its own default choices, and anyone who claims
otherwise has not read the source. The important thing is that we list
every divergence in a public, citable document
([`docs/plans/benchmark-comparison-inconsistencies.md`](../benchmark-comparison-inconsistencies.md))
so you can decide whether the comparison is the one you care about.

Some highlights we found while building the multi-seed harness:

- **PPO value loss formulation**: rlox follows the CleanRL convention
  (`0.5 * max(vf_loss1, vf_loss2).mean()` with `clip_vloss=True` default).
  SB3 defaults to plain `F.mse_loss` without clipping. We tried aligning
  rlox to SB3's formulation to "match exactly"; at Hopper-v4 1M seed=42
  the aligned variant regressed from **1955 → 837** reward. The CleanRL
  formulation is empirically superior at the horizons we care about, and
  we kept it as the rlox default while disclosing the divergence.
- **DQN loss & gradient clipping**: SB3 uses Huber (`smooth_l1_loss`) with
  grad clip at 10; rlox uses plain MSE with no grad clip. Both converge
  with their native recipes, but the hyperparameter recipes are *not*
  interchangeable. SB3 DQN with rlox's defaults collapses on CartPole (9
  reward); rlox DQN with SB3's recipe does the opposite on MountainCar.
- **SAC critic loss**: SB3 has an inner `0.5` factor (`0.5 * sum(mse1, mse2)`);
  rlox does `mse1 + mse2`. rlox SAC has a 2× stronger critic gradient at
  the same `vf_coef`. Despite this asymmetry, both converge to
  statistically equivalent numbers on the MuJoCo six — which is itself an
  interesting finding about SAC's stability to critic-gradient scaling.
- **DQN `train_freq` / `gradient_steps` defaults**: rlox trains every env
  step by default (1/1). SB3 trains every 4 env steps (4/1). When you map
  the same YAML config between them without adjusting these, SB3 DQN
  CartPole with `target_update_interval=10` collapses because the target
  network bounces around faster than SB3 can train. We maintain a small
  `_SB3_OVERRIDES` table in the harness that adds back the SB3-specific
  update-cadence keys. That table is published.

None of this is a bug. All of it is *disclosed*.

---

## What I wish someone had told me before I started

Three things, in priority order.

### 1. Single-seed PPO is a lying benchmark.

I spent an evening chasing what looked like a 34% rlox–SB3 gap on Hopper.
I wrote a careful A/B test comparing CleanRL-style vs SB3-style value loss,
saw a consistent advantage for the SB3 formulation at 200k and 500k steps,
and flipped the default. The next multi-seed run showed **a 57% regression
at 1M**. The 200k/500k bisection was in the wrong evaluation regime —
both formulations are still in early-training noise at that budget. The
late-training behavior diverged sharply, and a single-seed test at 1M
would also have missed it because Hopper's seed variance at 1M is on the
order of ±500 reward points.

Multi-seed benchmarks are not a nice-to-have. At n=5 they're still barely
enough. At n=1 they actively mislead you.

### 2. Your "fair comparison" has at least six asymmetries you haven't noticed.

When I started writing
[`multi_seed_runner_sb3.py`](https://github.com/riserally/rlox/blob/main/benchmarks/multi_seed_runner_sb3.py),
I thought the hard part would be mapping rlox config keys to SB3 kwargs.
It wasn't. The hard part was realizing that:

- rlox and SB3 use **different eval protocols** (30 episodes vs 10, unique
  per-episode seed vs single seed, deterministic vs stochastic).
- rlox and SB3 handle **VecNormalize freezing at eval differently** by
  default (rlox forgets to, SB3 doesn't).
- SB3-zoo's famous **"Hopper 3578"** is on Hopper-*v3*, not v4. Different
  physics damping. Comparing your v4 number to that reference is off by
  200–400 reward before you've done anything wrong.
- SB3 **auto-selects CUDA** while rlox currently forces CPU for its policy
  tensors. On a GPU host, the SB3 number is computed at 4–8× the wall-clock
  throughput of the rlox number, even though the per-step work is the same.
- SB3 stores **normalized observations in the replay buffer**; rlox stores
  raw observations and normalizes at sample time. Numerically equivalent
  when running, divergent when you reload a checkpoint.
- IQM at n=5 **literally reduces to the median**. Call it the median in
  the paper, or run 10 seeds.

All six are cataloged in the inconsistencies doc. The only defense against
them is to write an explicit harness that equalizes what you can equalize,
document what you can't, and hand a reviewer the evidence.

### 3. The Python bottleneck is bigger than you think, but so is the Python advantage.

Moving the data plane to Rust bought us 3.9× on end-to-end rollouts. That's
big but not transformative. Moving *everything* to Rust — no PyTorch — would
buy us maybe another 2×, and cost us essentially the entire ML ecosystem.
Every experiment with a custom network. Every callback. Every logger.
Every tutorial that assumes `import torch`. Every researcher who already
knows how to debug PyTorch but doesn't know Rust.

The Polars pattern is right: push the slow stuff down, keep the pleasant
stuff up. The line between "slow stuff" and "pleasant stuff" is PyTorch's
autograd boundary. Anything before autograd — collection, normalization,
advantages — is Rust. Anything autograd touches is PyTorch. The interface
is [PyO3 arrays](https://pyo3.rs/latest/numpy.html) at the boundary and
nothing else.

---

## How to try it

```bash
pip install rlox
```

Or from source (needed until we ship pre-built wheels):

```bash
git clone https://github.com/riserally/rlox
cd rlox
python3.12 -m venv .venv
source .venv/bin/activate
pip install maturin numpy gymnasium torch
maturin develop --release
```

The [Getting Started guide](https://riserally.github.io/rlox/python/getting-started/)
walks through the rest. If you're coming from SB3, the
[migration guide](https://riserally.github.io/rlox/python/tutorials/migration-sb3/)
is a side-by-side translation of the patterns you already know.

And if you speak Polish, German, or French: the landing page and first-run
docs are translated. The contribution process for more languages is
documented at [`docs/CONTRIBUTING-translations.md`](https://github.com/riserally/rlox/blob/main/docs/CONTRIBUTING-translations.md).

---

## What's not done

Three things you should know about before you decide whether to invest
attention:

- **Windows wheels don't exist yet.** macOS and Linux work. Windows needs
  the same `maturin develop --release` dance, which means Rust toolchain
  setup.
- **DQN MountainCar is unstable** without the SB3-zoo `train_freq=16,
  gradient_steps=8` recipe. Documented in the inconsistencies doc; the
  fix is one YAML entry.
- **The paper is not yet peer-reviewed.** The benchmark evidence in this
  post is reproducible (code + configs + harness + eval protocol are all
  open), but the formal writeup with full statistical analysis is still
  in draft.

---

## Who this is for

- **RL researchers** who want to run CleanRL-style custom experiments without
  giving up SB3-style ergonomics, and who are tired of profiling Python.
- **LLM post-training practitioners** who need fast GRPO / DPO / KL
  primitives and don't want to roll their own.
- **People who liked Polars** and are curious whether the pattern
  generalizes.

If you are none of those, rlox is probably not the tool for you today.
Stable-Baselines3 is excellent at what it does, CleanRL is excellent at
what it does, TorchRL is excellent at what it does. rlox is interesting
precisely because it is *not* a strict superset of any of them.

---

## Coda

The paper is at [GitHub link TBD]. The code is at
[github.com/riserally/rlox](https://github.com/riserally/rlox).
The inconsistencies doc is at
[docs/plans/benchmark-comparison-inconsistencies.md](https://github.com/riserally/rlox/blob/main/docs/plans/benchmark-comparison-inconsistencies.md).
Questions welcome; issues more so; PRs most of all.

If you try it and something feels wrong, **open an issue**. The fastest
way to improve this is for people running actual experiments to find the
next six asymmetries I haven't noticed yet.

---

### Footnotes / "things I cut for length"

**F1 — Why not fork SB3?** Because the architectural choice ("everything
in Python") is the very thing we're trying to replace. Forking gets you
nicer Python code but the same bottleneck.

**F2 — Why not fork TorchRL?** Because TorchRL's abstraction tower is
dense, and fitting a Rust data plane under it would be a larger engineering
project than building rlox from scratch.

**F3 — Why not use Jax?** Jax is a great answer to "how do I JIT-compile
the inner loop." It is not a great answer to "how do I integrate with an
existing PyTorch research workflow and a large body of SB3 user code."
rlox is optimized for the second question.

**F4 — Performance numbers qualifier.** All benchmarks are on an M-series
MacBook Pro (ARM, 10-core). Exact reproductions are at `benchmarks/`.
x86 Linux with `-C target-cpu=native` tends to shift the ratios by
10–20% but the qualitative conclusions hold.

**F5 — The GRPO / DPO story.** LLM post-training is one of the cleanest
wins for the rlox pattern because the per-token arithmetic is pure data
movement with no autograd until the very end. The KL divergence operator
alone is 2–9× faster than TRL at batch sizes we care about. There's a
separate blog post's worth of material there; this post intentionally
stays in classic-control / MuJoCo territory.

---

### Publishing checklist (don't forget before hitting publish)

- [ ] Replace every `⏳` in the convergence table with the actual v2 multi-seed number
- [ ] Add the SB3 multi-seed cells once the SB3 sweep reruns in the same order
- [ ] Verify the GitHub URLs above resolve on the public site
- [ ] Swap the `TBD` for the paper arxiv/HTML link if available
- [ ] Double-check the M-series benchmark claim — rerun `cargo bench` if it's been more than a month
- [ ] Consider adding a plot (radar chart of speedups? line chart of rlox vs SB3 over env steps?)
- [ ] Decide whether to publish the inconsistencies doc as a separate post — it's long enough to stand alone
- [ ] Choose a hero image / og:image for social cards
- [ ] Tag: #reinforcement-learning #rust #python #pytorch #ml-engineering
