# Related Work — rlox paper draft

**Draft status**: v1, 2026-04-08. Section 2 of the paper.

**Pairs with**: `paper-introduction.md` (the short per-framework
summary in §1.4 leads into this section). Citations here should be
the fullest version; §1.4 and §3.8 citations are deliberately
narrower and forward-reference here.

**Target length**: 2.5–3.5 pages depending on venue (shorter for
systems venues that demand a Background section, longer for
general-ML venues).

---

## 2. Related Work

We situate rlox in four overlapping bodies of prior work: (§2.1)
reinforcement learning frameworks targeting Python workflows, (§2.2)
the emerging Rust-for-ML ecosystem that makes the Polars pattern
viable at all, (§2.3) the methodological literature on RL
benchmarking reliability, and (§2.4) the infrastructure for
hyperparameter sourcing and reproducibility that rlox builds on.

### 2.1 Reinforcement learning frameworks

The RL framework landscape splits cleanly along two axes:
*abstraction weight* (how much scaffolding the framework asks the
user to adopt) and *execution model* (pure Python, Python + C++
tensor backend, JIT-compiled, or hybrid). We summarize the
most-cited frameworks, grouped by execution model.

#### Pure Python + PyTorch

**Stable-Baselines3** [@sb3] is the de facto standard for
production-oriented RL work. It provides unified APIs for PPO, SAC,
TD3, DQN, A2C, DDPG, and HER across a single `BaseAlgorithm`
hierarchy; strong, well-tested hyperparameters via the companion
rl-baselines3-zoo repository [@sb3zoo]; and a mature callback /
logger ecosystem that integrates with MLflow, TensorBoard, and
Weights & Biases. SB3's implementation is entirely Python: the
`VecEnv` abstractions, the `RolloutBuffer` and `ReplayBuffer`, the
`VecNormalize` wrapper, and the `compute_returns_and_advantage`
method all run as interpreted Python code with numpy-backed
storage. This is the architectural limitation rlox is designed to
address. SB3 is also the primary comparison target for §4.1: every
multi-seed cell is run in both frameworks under the matched-harness
protocol of §3.7.

**CleanRL** [@cleanrl] takes the opposite position on abstraction:
each algorithm is a single self-contained file, typically 200–500
lines, with the full training loop, policy network, loss function,
and replay buffer in one place. CleanRL's explicit goal is
research readability — a PhD student should be able to read a
CleanRL PPO file and understand every line. The cost is that code
reuse across algorithms is by copy-paste rather than by shared
modules, and that the data plane inherits all of Python's
performance limitations. rlox shares CleanRL's transparency
philosophy — each rlox algorithm file (e.g.
`python/rlox/algorithms/ppo.py`) is self-contained and reads
linearly — but lifts the data-plane operations into a shared Rust
kernel that all algorithms call. We read CleanRL as the canonical
"understandable and slow" implementation in the same way that SB3
is the canonical "featureful and slow" implementation.

**TorchRL** [@torchrl] provides a comprehensive type system for RL
built around `TensorDict`, `EnvBase`, `Collector`, and `LossModule`
abstractions. TorchRL's strength is compositional power: every
component is a drop-in replaceable module, which makes sophisticated
research workflows (e.g. combining off-policy actor-critic with
model-based rollouts) implementable without inheritance trees.
Its weakness from the rlox perspective is abstraction weight:
reading TorchRL source code requires understanding the full module
hierarchy, and the runtime data path passes through several
abstraction layers that each add Python overhead. Our §4.2
microbenchmarks against TorchRL show the cumulative effect:
`compute_gae` in TorchRL is approximately three orders of magnitude
slower than `rlox.compute_gae_batched` at 32 k rollout steps.

**Tianshou** [@tianshou] is similar in spirit to SB3 and CleanRL
(PyTorch-based, single-machine focused, unified API) and targets the
research community directly. It is not included in our comparison
because its hyperparameter defaults differ from the rl-zoo3
conventions we use as ground truth and a same-harness comparison
would require an additional translation layer equivalent in scope to
the SB3 one.

**Ray RLlib** [@rllib] is the largest and most battle-tested
distributed RL framework. RLlib targets multi-node, multi-GPU
training at scale and provides algorithm implementations designed
around the Ray task graph rather than around single-machine
efficiency. We consider RLlib a complementary tool rather than a
direct comparison target: a researcher who needs to train across
100 machines should reach for RLlib; a researcher who needs to
train on one machine faster should reach for rlox. The two
priorities trade off against each other — distributing work
across machines necessarily introduces serialization overheads
that a single-machine framework can avoid.

#### JAX-based

**Brax** [@brax] is the canonical JAX-based RL framework with
JIT-compiled physics environments. Brax achieves its headline
billion-step-per-second throughput numbers by putting the entire
environment simulation inside the JIT graph and running on GPU or
TPU, which eliminates Python overhead from every inner loop. The
trade-off is that environments must be JIT-compatible — Brax
provides its own MuJoCo-alternative environments written in JAX,
and cannot drop into existing Gymnasium MuJoCo or Atari workflows.
rlox targets the opposite constituency: users who already have
Gymnasium-based environments and PyTorch-based policies and want
to keep them both.

**purejaxrl** [@purejaxrl] takes the CleanRL philosophy of
single-file implementations into JAX, producing extremely fast
PPO and SAC implementations that run the full training loop inside
a single `jax.jit`-compiled function. Again, the performance
story is impressive but conditional on environments being
JIT-compatible and on the user being willing to work in JAX.

**Sample Factory** [@sample_factory] achieves very high throughput
through an asynchronous actor-learner architecture with shared
replay buffers; it is oriented toward single-machine high-throughput
training and is closer in spirit to rlox's architectural goals
than to a framework comparison. We consider Sample Factory a
good reference for "what does peak single-machine throughput look
like" on asynchronous workflows, though its implementation model
and ours differ in the choice of where to put the
synchronization boundary.

#### C++ and hybrid

**Tianshou with C++ env backend** and **Acme** [@acme] both
incorporate C++ code paths for environment stepping or buffers,
but neither is architected around the Polars pattern: they use
C++ as a performance optimization for specific components rather
than as the primary home of the data plane. Acme also targets
DeepMind's internal research workflows, where the primary design
pressure is supporting novel algorithms (distributional methods,
model-based approaches) rather than maximizing throughput on
standard ones.

#### Summary table

| Framework | Backend | Env model | API weight | Primary target |
|---|---|---|---|---|
| Stable-Baselines3 | PyTorch + Python | Gymnasium | Medium | Production / applied |
| CleanRL | PyTorch + Python | Gymnasium | Minimal | Research readability |
| TorchRL | PyTorch + Python | Gymnasium / custom | Heavy | Compositional research |
| Tianshou | PyTorch + Python | Gymnasium | Medium | Research |
| Ray RLlib | Various + Ray | Gymnasium | Heavy | Distributed / scale |
| Brax | JAX + XLA | JAX-native | Medium | JIT research |
| purejaxrl | JAX + XLA | JAX-native | Minimal | JIT single-file |
| Sample Factory | PyTorch + C++ | Gymnasium | Medium | Async throughput |
| Acme | JAX + TF | Custom | Heavy | DeepMind research |
| **rlox** | **PyTorch + Rust** | **Gymnasium + native Rust** | **Medium** | **Single-machine speed with existing PyTorch workflows** |

**Table (related work)**. Positioning of rlox in the RL framework
landscape. rlox's distinguishing position is "PyTorch + Rust,
Gymnasium-compatible, single-machine, medium abstraction weight" —
a quadrant that is not otherwise occupied.

### 2.2 The Rust-for-ML ecosystem

Five years ago, proposing a Rust data plane for PyTorch workflows
would have been architecturally naive: the bindings, the array
interoperability, and the tensor libraries did not exist. The
landscape has changed materially, and rlox is built on top of
several projects that independently matured to the point where this
architectural choice is now straightforward.

**PyO3** [@pyo3] is the foundation. It provides Rust bindings to
the Python interpreter with automatic reference counting, zero-copy
buffer protocol support via the `numpy` crate [@rust_numpy], and
a build system (`maturin` [@maturin]) that produces Python wheels
directly from a `Cargo.toml` + `pyproject.toml` pair. PyO3 is the
reason the rlox Rust↔Python boundary is narrow and typed rather
than needing a custom FFI layer. At the time of writing PyO3 is
used in production by Polars [@polars], Pydantic Core, Ruff, and
a growing number of scientific Python packages. The `maturin`
workflow we use — `maturin develop --release` during development,
built wheels for distribution — is the standard path; the rlox
build system is a direct application of the pattern.

**tch-rs** [@tch_rs] provides Rust bindings to the LibTorch C++
API, enabling Rust code to construct and run PyTorch tensor
operations. We considered using tch-rs for the rlox data plane —
it would allow the backward scan in GAE to run against the same
torch::Tensor objects that PyTorch uses for autograd — but
ultimately rejected this choice because the tch-rs binding layer
adds overhead comparable to PyO3's numpy boundary, and because
depending on LibTorch's ABI makes rlox's build system substantially
more fragile. Our data plane uses `ndarray` [@rust_ndarray]
internally and exposes numpy arrays at the boundary.

**candle** [@candle] is a lightweight Rust tensor library
developed by HuggingFace, aimed primarily at LLM inference. Candle
provides automatic differentiation, a CUDA backend, and a smaller
dependency footprint than LibTorch. We experimented with candle
as the underlying tensor library for an all-Rust version of rlox
(see `crates/rlox-candle`) but decided that users are better
served by PyTorch on the autograd side — the ecosystem advantages
of PyTorch (logging, checkpointing, existing model zoo, community
familiarity) outweigh the cleanliness of an all-Rust stack at this
stage of rlox's development. The `rlox-candle` crate remains in
the repository as an experimental path and as a testbed for pure-
Rust training on specific environments.

**burn** [@burn] is a Rust-native deep learning framework that
emphasizes backend polymorphism (a single model can run on
different compute backends without code changes) and compile-time
graph optimization. Burn is architecturally the closest existing
framework to "what if rlox had no PyTorch dependency at all" and
is worth watching; our experimental `crates/rlox-burn` explores
the same question but is not the primary development path.

**linfa** [@linfa] is the Rust analog of scikit-learn, providing
classical machine learning algorithms (linear models, clustering,
decomposition). It is not directly relevant to rlox but is part
of the broader Rust-for-ML trend: the ecosystem is now dense enough
that picking Rust for ML infrastructure is a reasonable choice
rather than a quixotic one.

**Polars** [@polars] is the canonical demonstration of the Polars
pattern in the data-processing space: a Rust-native dataframe with
zero-copy Python bindings that simultaneously outperforms pandas
and provides a more ergonomic API. Polars is both an architectural
inspiration for rlox and, in a narrower technical sense, a proof
of concept that the specific performance claims of the Polars
pattern are achievable rather than hypothetical. The architectural
split between Polars' Python frontend (`polars.DataFrame`) and its
Rust backend (`polars-core`, `polars-lazy`) is directly analogous
to the rlox Python control plane / Rust data plane split.

### 2.3 Reliability of RL benchmarks

A substantial literature exists on the difficulty of making
defensible RL benchmark claims. This literature directly motivates
the same-harness protocol of §3 and the inconsistencies catalog of
§3.8.

**Henderson et al. 2018** [@henderson2018matters]. The foundational
paper on RL reproducibility. Henderson et al. ran the same nominal
algorithm (PPO, DDPG, ACKTR) on the same environments using different
reference implementations and found reward disparities of up to 2×
between implementations of "the same" algorithm. They attributed
the disparities to a mixture of seed variance, hyperparameter
tuning, and implementation details that were not disclosed in the
papers describing the algorithms. Their recommendations — run
multiple seeds, report confidence intervals, publish hyperparameters
— are the baseline that any modern RL benchmark paper is expected
to meet. The rlox benchmark protocol (§3.4–§3.6) adopts all of their
recommendations and extends them with a same-harness eval protocol
that removes one of the confounds they identified.

**Engstrom et al. 2020** [@engstrom2020matters]. A focused follow-up
that isolated the specific implementation details affecting PPO
and TRPO convergence: value-function clipping, reward scaling,
orthogonal initialization, Adam epsilon, learning rate annealing,
observation normalization, and gradient clipping. They found that
enabling or disabling these "implementation details" could swing
the final reward by more than the algorithm choice itself on
MuJoCo benchmarks. rlox's inconsistencies catalog (§3.8) extends
this work by making the full set of defaults observable and by
documenting where rlox and SB3 agree and disagree on each axis
Engstrom et al. identified.

**Andrychowicz et al. 2021** [@andrychowicz2021matters]. The
comprehensive large-scale empirical study of on-policy RL
hyperparameter sensitivity. Andrychowicz et al. trained over 250,000
agents across 5 environments and 25 design choices, producing
posterior distributions over which design choices matter and which
do not. Two findings are directly relevant to rlox: first, their
observation that the *distribution* of hyperparameter settings
that work well is narrow and unevenly distributed, which is why
rl-zoo3's per-environment tuned hyperparameters work and generic
defaults often don't; second, their observation that value-function
clipping has a large effect on late-training performance even when
early-training curves look identical, which is exactly the finding
we independently reproduce in §4.3.1 with the Hopper loss-formulation
A/B.

**Agarwal et al. 2021** [@agarwal2021precipice]. The "Deep RL at the
Edge of the Statistical Precipice" paper, which introduced the
rliable library and the Interquartile Mean (IQM) as the preferred
aggregation statistic for multi-seed RL benchmarks. Agarwal et al.
demonstrated that commonly-used statistics (mean, median with
confidence bands) give substantially different pictures of which
algorithm is "better" on standard benchmarks, and argued for IQM
with stratified bootstrap confidence intervals as the least
distorted summary statistic. rlox adopts the Agarwal et al. protocol
directly — our `rlox.evaluation.interquartile_mean` and
`stratified_bootstrap_ci` are drop-in replacements for the rliable
functions — and we cite them in every reporting context (§3.6,
§4.1, the n=5 caveat).

**Colas et al. 2018** [@colas2018how] examined the statistical
significance of single-seed and small-n RL comparisons and
concluded that 5 seeds is inadequate for detecting the kinds of
small-to-moderate effect sizes that separate competitive RL
algorithms. Their recommendations inform our choice to run n=10
on the headline PPO MuJoCo cells (§4.1.3) where compute budget
permitted.

### 2.4 Hyperparameter sourcing and reproducibility infrastructure

**rl-baselines3-zoo** [@sb3zoo] is the open-source repository of
tuned hyperparameters for Stable-Baselines3 algorithms across
standard Gymnasium environments. The tuning process documented in
the zoo README involves hundreds to thousands of training runs
per (algorithm, environment) cell using Bayesian optimization,
producing YAML files that the community has converged on as the
reference "good" hyperparameters for each setting. rlox's preset
YAMLs in `benchmarks/convergence/configs/` are sourced from
rl-zoo3 with minimal modification, and our same-harness comparison
deliberately uses these preset YAMLs for both frameworks — this
removes hyperparameter-tuning quality as a confound in the
framework comparison.

**RL Zoo3** extends the hyperparameter repository with trained
model checkpoints, learning curves, and a standard interface for
running training with a single command (`python train.py --algo
ppo --env Hopper-v4`). We consider rl-zoo3 complementary to rlox:
the zoo provides the "what are the right hyperparameters" answer
that rlox then runs against the "what is the right data plane"
question.

**sacred** [@sacred] and **hydra** [@hydra] are generic Python
experiment configuration frameworks that many RL projects use for
hyperparameter management. rlox uses neither — our configuration
is a simple YAML + dataclass pipeline (`rlox.config`) with
validation — but the choice of a single source-of-truth YAML file
per (algorithm, environment) cell is directly inspired by the
rl-zoo3 conventions built on top of hydra.

**MLflow** [@mlflow] and **Weights & Biases** [@wandb] are the two
most common experiment tracking systems in the Python ML ecosystem.
rlox provides first-class callbacks for both (`rlox.callbacks.mlflow`
and `rlox.callbacks.wandb`) and the multi-seed runner can log all
per-seed metrics to either backend with a single configuration
flag. Neither system is required — the default behavior is to log
JSON files to the `--output` directory — but they are the standard
tools for the production-oriented workflows that rlox is designed
to serve.

### 2.5 Summary of relationships

rlox is distinct from prior work in the intersection of four
positions: (a) it targets PyTorch workflows rather than JAX,
(b) it is single-machine rather than distributed, (c) it uses
Rust as the data plane language rather than Python with C++
escape hatches, and (d) it adopts the same-harness comparison
protocol from the reproducibility literature [@henderson2018matters;
@agarwal2021precipice] rather than citing framework-native
published numbers. No other framework in our survey occupies all
four positions simultaneously, and the combination is
architecturally distinctive: it is the specific combination that
yields the Polars-pattern-style 3–50× data-plane speedup on
PyTorch-native RL workflows without asking the user to change
their training loop, their policy network, or their environment.

We emphasize that "distinct" is not the same as "better". rlox is
a better choice than SB3 for single-machine PyTorch users who
care about training speed; it is a worse choice than Brax for
users who can afford to port their environments to JAX; it is a
worse choice than Ray RLlib for users who need to distribute
across more than one machine; and it is a worse choice than
CleanRL for students who are trying to understand how PPO works
by reading the source code. The choice of framework depends on
which of these constraints dominates in a particular workflow.
The contribution of this paper is to establish that, for the
PyTorch-single-machine quadrant, rlox provides a factor-of-several
speedup over the best prior art on data-plane operations without
sacrificing convergence correctness or the ergonomics of existing
PyTorch research code.

---

## Writing-time notes for author(s)

*These notes are for the author(s) and should be deleted before
compilation.*

### Citation discipline

- Every paragraph in §2 should have at least one citation. Flag
  any paragraphs without one during copy-edit — unbalanced citation
  density is the top signal reviewers use to identify a related-work
  section that was written in a hurry.
- The citations in §2.3 (reliability literature) are load-bearing
  for the methodology claims in §3 and §4. Double-check that every
  claim in §3.6 (statistical reporting) and §4.1.1 (IQM caveat)
  traces back to Agarwal et al. 2021 or Colas et al. 2018.
- Check that Haarnoja et al. 2018 (SAC) and Fujimoto et al. 2018
  (TD3) are cited in §2.1 or §2.3 — the algorithm citations are
  typically expected by reviewers even when the framework paper
  is about implementation rather than algorithm design.

### What I cut for length

- A full paragraph on **DeepMind's acme** design decisions and how
  they influenced TorchRL. Cut because acme is not the primary
  comparison target and the details were not load-bearing.
- A section on **Spinning Up** [@spinningup] as an educational
  reference. It is not a framework in the production sense and
  the comparison would be unfair.
- A discussion of **MushroomRL** and other smaller frameworks.
  Cut because the related-work table is already long and the
  marginal reviewer is unlikely to care about frameworks outside
  the top tier.

### Venue-specific trims

- **NeurIPS / ICML**: current length is about right. Keep §2.1–2.4.
- **MLSys / OSDI / SOSP**: compress §2.1 (they know what SB3 is)
  and expand §2.2 (the Rust-ML landscape is the genuinely
  novel-for-systems-venue content). Consider adding a paragraph
  on the `tch-rs` vs PyO3 design trade-off that explicitly
  justifies why we chose PyO3 over direct LibTorch bindings.
- **AAAI / IJCAI**: compress both §2.2 and §2.4 to half their
  current length; expand §2.3 (the reliability literature is
  better-received at AI venues).

### Open questions

1. **Do we include a paragraph on non-RL applications of the Polars
   pattern?** Polars itself is the obvious one but there are at
   least three others: Pydantic Core (validation), Ruff (linting),
   and Datafusion (SQL). Including them would reinforce the
   architectural argument but might feel off-topic in an RL paper.
   Currently I cite Polars only; reconsider if reviewers push back
   on the novelty of the Polars pattern.
2. **Should §2.3 cite papers about RL benchmarks specifically, or
   also about ML reproducibility more broadly?** The current draft
   stays narrowly in RL. Broadening to the ML reproducibility
   literature (e.g. Pineau et al. 2021, "Improving Reproducibility
   in Machine Learning Research") would make the section more
   general but is arguably off-topic for a framework paper.
3. **Is there any framework we are missing that a reviewer will
   flag as "how can you not mention X"?** Candidates I considered
   and rejected: DI-engine, MushroomRL, ChainerRL (defunct),
   MBRL-Lib (model-based specific), rl_games (GPU-focused, similar
   in spirit to Sample Factory). Sanity-check this list with a
   co-author before submission.
