# Introduction — rlox paper draft

**Draft status**: v1, 2026-04-08. Section 1 of the paper. Written to
pair with `docs/plans/paper-methods-section.md` and
`docs/plans/paper-results-section.md`. No `⏳` placeholders in this
section — the Introduction should stand on its own regardless of the
final results numbers, with forward references to §4 for the data.

**Target length**: 3–4 pages including the architecture figure. This
draft is ~650 lines of prose + diagram, which should compress to
~3.5 pages after LaTeX typesetting.

---

## 1. Introduction

### 1.1 The Python data-plane tax

Modern reinforcement learning research has settled into a comfortable
architectural plateau: the policy network is a PyTorch or JAX module,
the environment is a Gymnasium instance wrapped in a vectorized
runner, the training loop is a for-loop in Python, and the data
structures — replay buffers, advantage estimators, running statistics
— are numpy arrays that the Python interpreter shuffles around on
every step. This arrangement is ergonomic, well-understood, and
supported by a robust ecosystem [@sb3; @cleanrl; @torchrl]. It is
also, at scale, grossly inefficient for reasons that have nothing to
do with the underlying algorithms.

Consider a PPO training run on a MuJoCo [@todorov2012mujoco] locomotion
task. Per training update the framework must: step eight parallel
environments for two thousand steps each, compute Generalized
Advantage Estimation [@schulman2015gae] over the rollout, normalize
observations and rewards against running statistics, shuffle
sixteen thousand transitions into minibatches, and apply PPO's
[@schulman2017ppo] clipped surrogate objective for ten epochs. The
policy forward/backward pass runs on the GPU in under a hundred
microseconds per minibatch. The *data shuffling* around it runs on
the CPU, interprets Python bytecode at every step, and spends
between two and thirty times longer than the GPU work. Profiles of
Stable-Baselines3 [@sb3] and TorchRL [@torchrl] on Hopper-v4 show
the same pattern: environment stepping and advantage computation are
where training time goes, not gradient computation.

This is the **Python data-plane tax**: the overhead that modern RL
frameworks pay for the convenience of writing their entire data plane
in the same language as their training loop. It is not a problem of
algorithm design, nor of hyperparameter tuning, nor of GPU
utilization. It is a problem of architecture.

The dataframe community has already confronted and solved an
analogous problem. Polars [@polars] demonstrated that a Rust-native
data structure with zero-copy Python bindings can produce a tool
that is both more ergonomic than pandas *and* an order of magnitude
faster on the same workloads. The critical architectural insight is
that the Python layer should own user-facing concerns — query
construction, type checking, interactive REPL use — while the Rust
layer owns the compute-intensive operations that dominate cost. The
boundary between them is a zero-copy array interface. Neither layer
is compromised to support the other, and users never need to know
which side of the boundary a particular operation runs on.

**rlox applies the Polars pattern to reinforcement learning.** The
Python layer owns the training loop, policy networks (via PyTorch),
configuration, logging, callbacks, and everything else a researcher
touches during development. The Rust layer owns environment stepping,
replay buffers, GAE computation, running statistics, and the tokenwise
operations used by LLM post-training algorithms. The boundary is a
PyO3 [@pyo3] zero-copy array interface that hands observation tensors
to Python at the start of each step and receives action tensors back
at the end. A typical training loop never needs to instantiate a
Python list of transitions, because no Python list of transitions
ever exists.

### 1.2 Contributions

This paper presents rlox, an open-source reinforcement learning
framework implementing the Polars pattern for the RL data plane.
Concretely we contribute:

1. **An architectural demonstration** that the Polars split applies
   directly to RL: the environment-step / advantage-compute / buffer
   operations can be moved into a Rust data plane without disrupting
   the PyTorch-based training loop, at speedups of 3–50× on data-plane
   operations vs Stable-Baselines3 (§4.2).

2. **A convergence-parity benchmark** on the six most-used RL
   algorithms (PPO, SAC, TD3, DQN, A2C, TRPO) across thirteen
   environments from classic control and MuJoCo, using a
   same-harness multi-seed protocol that removes the per-framework
   evaluation-protocol differences which typically contaminate such
   comparisons. Every cell reports IQM + 95% bootstrap CI per Agarwal
   et al. [@agarwal2021precipice] and is reproducible from the
   published preset YAMLs (§3.3, §3.4, §4.1).

3. **A catalog of benchmark-comparison pitfalls** enumerating the
   hyperparameter, loss-formulation, evaluation-protocol, and
   environment-version differences between rlox, Stable-Baselines3,
   and CleanRL, with cited source lines from each upstream
   implementation. This catalog is published as a separate document
   and can be cited independently of the main paper (§3.8, Appendix B).

4. **A demonstration that single-seed RL benchmarks are
   systematically misleading** — we present a case study in which an
   apparently-correct 200k-step A/B test motivated a change that
   regressed the 1M-step convergence by 57% on Hopper-v4, and
   discuss what this implies for the design of future framework
   benchmarks (§4.3.1).

5. **An open-source implementation** of 22 RL algorithms sharing a
   unified `Trainer('<algo>', env=..., config=...)` API, with Rust
   kernels for GAE, replay buffers, running statistics, V-trace,
   GRPO advantages, and tokenwise KL divergence. The implementation
   is available at `github.com/riserally/rlox` under the MIT license.

### 1.3 Architecture at a glance

Figure 1 shows the rlox architecture. The Python control plane sits
atop a Rust data plane, with PyO3 providing the zero-copy array
interface at the autograd boundary.

```mermaid
graph TB
    subgraph Python["Python Control Plane (user-facing)"]
        direction TB
        T[Trainer.train]
        P["Policy (PyTorch nn.Module)<br/>• Actor + critic<br/>• Autograd live here"]
        L["Loss fn (PyTorch)<br/>• PPOLoss, SACLoss, ...<br/>• Clipping, entropy, KL"]
        O["Optimizer (torch.optim)<br/>• Adam, AdamW, ..."]
        Cb["Callbacks (Python)<br/>• Logging, checkpoint, eval<br/>• MLflow, W&B, stdout"]
        Cfg["Config (YAML → dataclass)<br/>• Per-algo, per-env presets<br/>• Validated at load time"]
    end

    subgraph PyO3["PyO3 zero-copy boundary<br/>(numpy ↔ ndarray, no serialization)"]
        direction TB
        B1["Observation batch (float32)"]
        B2["Action batch (float32 | int64)"]
    end

    subgraph Rust["Rust Data Plane (hot path)"]
        direction TB
        VE["VecEnv<br/>• Rayon-parallel step_all<br/>• Native CartPole/Pendulum<br/>• Gymnasium passthrough"]
        Buf["ReplayBuffer / RolloutBuffer<br/>• Ring + mmap backends<br/>• Prioritized variant<br/>• Zero-copy sample"]
        GAE["compute_gae_batched<br/>• Backward scan<br/>• Truncation bootstrap<br/>• 1 700× vs TorchRL"]
        Stats["Running Mean/Std<br/>• Welford online update<br/>• Obs + return norm"]
        VTr["V-trace correction<br/>• IMPALA / off-policy"]
        LLM["GRPO adv + tokenwise KL<br/>• f32 / f64 paths<br/>• LLM post-training"]
    end

    T --> P
    P --> L
    L --> O
    O -.gradient step.-> P
    T --> Cb
    T --> Cfg

    T ==collect rollout==> B1
    B1 ==obs batch==> VE
    VE ==step==> Buf
    VE ==next obs==> B1

    P ==forward==> B2
    B2 ==action==> VE

    Buf --> GAE
    GAE --> L
    VE --> Stats
    Stats -.normalize.-> B1
    VE --> VTr
    VTr --> L
    Buf --> LLM

    classDef py fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000
    classDef rs fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    classDef bd fill:#f3e5f5,stroke:#6a1b9a,stroke-width:1px,stroke-dasharray:5 5,color:#000

    class T,P,L,O,Cb,Cfg py
    class VE,Buf,GAE,Stats,VTr,LLM rs
    class B1,B2 bd
```

**Figure 1.** rlox architecture. The **Python control plane** (blue)
owns the training loop, policy networks, loss functions, optimizers,
configuration, and callbacks — everything a researcher edits or
debugs during a typical workflow. The **Rust data plane** (orange)
owns environment stepping, buffers, advantage estimation, running
statistics, V-trace, and LLM-specific primitives — the
compute-intensive operations that traditionally bottleneck
Python-based RL frameworks. The two layers communicate through a
**PyO3 zero-copy boundary** (purple, dashed): observation and action
batches are numpy arrays that share memory with Rust `ndarray`
instances, with no serialization or data copy at the boundary. The
architectural rule is strict: if an operation takes gradients, it
runs in Python/PyTorch; if it does not, it runs in Rust.

The key architectural implications visible in Figure 1:

- **Autograd stays in PyTorch.** Policy networks, losses, and
  optimizers are unmodified `torch.nn.Module` and `torch.optim`
  objects. Any custom policy, any custom loss, any custom optimizer
  that works in vanilla PyTorch works in rlox without modification.
  Researchers do not learn a new differentiable-programming DSL.

- **Data transforms stay in Rust.** The backward scan in GAE, the
  ring-buffer writes in the replay buffer, the Welford updates in
  running statistics, and the Rayon-parallel environment stepping
  are all pure Rust, compiled with `-O3` at `maturin develop
  --release`. The Python interpreter never steps through any of
  these operations.

- **The boundary is narrow and typed.** PyO3 exposes exactly three
  array types at the interface: `PyArray2<f32>` for observations,
  `PyArray1<f32>` or `PyArray1<i64>` for actions, and a small set
  of scalar types for configuration. The API surface of the Rust
  layer is deliberately small — approximately twenty-five public
  functions — so that the translation cost between Python and Rust
  is low for both users and maintainers.

- **Configuration is shared.** The same YAML preset files drive both
  the unified Python `Trainer` and the SB3 comparison harness (§3.7),
  ensuring that framework comparisons cannot silently drift on
  hyperparameters.

### 1.4 Relationship to prior frameworks

rlox sits deliberately between Stable-Baselines3 [@sb3] and CleanRL
[@cleanrl] on the ergonomics-vs-transparency axis, and between both
of them and TorchRL [@torchrl] on the abstraction-weight axis.

- **Stable-Baselines3** is rlox's closest comparison target and the
  framework rlox most closely tries to be compatible with. SB3
  provides a unified algorithm API, strong default hyperparameters
  via rl-baselines3-zoo [@sb3zoo], and production-grade reliability
  on a wide range of environments. SB3's architectural limitation is
  that everything — environment stepping, buffers, GAE, VecNormalize
  — runs in Python, and the resulting overhead is unavoidable for
  small-observation MuJoCo tasks. rlox inherits SB3's API
  sensibilities (unified `Trainer`, preset-driven configs, opt-in
  complexity) and replaces its data plane with Rust.

- **CleanRL** provides single-file, research-readable implementations
  of each algorithm with minimal abstraction. CleanRL's strength is
  that a researcher can read and modify a complete PPO implementation
  in a single 300-line file; its weakness is that each algorithm
  has its own rollout buffer, its own advantage computation, its
  own seeding conventions, and reuse across algorithms is by
  copy-paste. rlox inherits CleanRL's transparency (every rlox
  algorithm file is self-contained and reads like a CleanRL
  implementation) but lifts the data-plane operations into a shared
  Rust kernel that all algorithms call into.

- **TorchRL** provides a comprehensive abstraction tower
  (TensorDict, EnvBase, Collector, LossModule, DataCollector...)
  that unifies RL workflows across on-policy, off-policy, and
  model-based algorithms inside a single type system. TorchRL's
  strength is compositional power — every component is a drop-in
  replaceable module — but the abstraction weight is substantial:
  reading TorchRL source requires understanding the full module
  hierarchy. rlox is narrower: it does not try to unify model-based
  and model-free into a single type system, and it accepts some
  code duplication across algorithms in exchange for implementation
  files that fit on one screen.

- **JAX-based frameworks** (Brax [@brax], Anakin [@anakin], Sample
  Factory [@sample_factory]) achieve their speedups through JIT
  compilation of the inner training loop and often through
  accelerated environments that run inside the JIT graph. This is
  an excellent approach but requires environments to be
  JIT-compatible, which rules out most Gymnasium environments
  (including the MuJoCo suite as exposed by `gymnasium.envs.mujoco`,
  which calls into a C++ library outside the JIT graph). rlox is
  optimized for the case where the researcher wants to keep their
  existing PyTorch workflow and Gymnasium environments.

The distinguishing claim of rlox is not that it is the fastest RL
framework — Brax on TPUs can step billions of environment steps per
second on environments that fit its JIT model. The claim is that
**rlox is the fastest RL framework that does not ask you to
abandon your existing PyTorch research workflow**. For researchers
with PyTorch codebases, Gymnasium environments, and SB3-style
expectations about hyperparameter portability, rlox is a drop-in
replacement that makes training several times faster while
preserving convergence parity.

### 1.5 Scope and non-goals

We are explicit about what rlox does not try to be:

- **Not a JAX framework.** rlox is PyTorch-first. Users who want
  JIT-compiled inner loops are better served by Brax [@brax] or
  purejaxrl [@purejaxrl]. We target the much larger population of
  researchers who have existing PyTorch codebases.

- **Not a distributed framework first.** rlox provides an IMPALA
  [@espeholt2018impala] implementation and gRPC-based actor/learner
  code in `crates/rlox-grpc`, but the primary design target is a
  single-machine workflow. Ray [@ray] and acme [@acme] remain better
  choices for distributed training at scale.

- **Not a model-based-only framework.** rlox includes a DreamerV3
  [@dreamerv3] implementation for completeness, but the primary
  focus is on model-free on-policy and off-policy methods where
  environment-step throughput is the dominant cost.

- **Not a library that hides Rust.** Users writing custom
  environments can write them in Python (and most should); users
  writing custom loss functions write them in PyTorch; but users
  writing custom buffer primitives or custom advantage estimators
  should expect to edit Rust, not Python. We consider this an
  acceptable cost because the set of people who need to customize
  data-plane operations is small, and those who do are generally
  comfortable with systems-level work.

- **Not a replacement for careful benchmarking.** The benchmarks in
  §4 control for the variables we can control (hyperparameters,
  seeds, eval protocol, environment version, compute device), and
  §3.8 catalogs the variables we can't force into agreement without
  forking the comparison framework. A reader who wants the last 1%
  of comparison rigor should treat §4 as the best we can currently
  do and reproduce the sweep in their own environment.

### 1.6 Paper roadmap

The rest of the paper is organized as follows. **Section 2 (Related
Work)** surveys the Rust-for-ML landscape (tch-rs, candle, burn,
huggingface/candle), the prior art in RL framework design (SB3,
CleanRL, TorchRL, Brax, Ray RLlib [@rllib]), and the reliability
literature in RL benchmarking (Henderson et al. [@henderson2018matters],
Andrychowicz et al. [@andrychowicz2021matters], Engstrom et al.
[@engstrom2020matters]) that informs our same-harness protocol.

**Section 3 (Methods)** describes the rlox architecture in depth, the
same-harness comparison protocol, the evaluation procedure, and the
catalog of known algorithmic asymmetries between rlox and SB3 that
our comparison preserves rather than hides.

**Section 4 (Results)** presents the multi-seed convergence parity
table, component-level microbenchmark speedups, ablation studies on
the most consequential design decisions, and a qualitative discussion
of the three cells where framework-native recipes diverge.

**Section 5 (Discussion)** reflects on what the benchmark results
imply for the broader RL-framework ecosystem, what they do *not*
imply (framework-agnostic claims about algorithm choice), and the
three lessons about single-seed testing, evaluation protocol
sensitivity, and hyperparameter defaults that we wish we had
internalized before building rlox.

**Section 6 (Conclusion)** closes with the architectural takeaway
and pointers to the open-source release.

**Appendices** cover: the full hyperparameter translation table (A),
the n=10 MuJoCo subset results (B), wall-clock reporting per framework
(C), excluded cells with reasoning (D), and the reproducibility
manifest including exact commit hashes, pinned dependency versions,
and a one-command reproduction script (E).

All code, preset configurations, per-seed result JSONs, benchmark
scripts, and the comparison harness are open-source at
`github.com/riserally/rlox`.

---

## Writing-time notes for author(s)

*These notes are for the author(s) and should be deleted before
compilation. They live here because the introduction is the piece
of the paper most sensitive to authorial voice and the one most
likely to be rewritten multiple times.*

### Tone targets

- **Confident but not triumphalist.** The Polars analogy is doing a
  lot of rhetorical work; make sure the reader understands we're
  saying "the same architectural pattern applies", not "we're as
  important as Polars". Re-read after first full draft and soften
  anything that sounds self-congratulatory.
- **Specific but not pedantic.** Numbers like "3–50× faster" belong
  in §4; the introduction should motivate the architecture and leave
  quantitative claims for later. The one exception is the 1 700× GAE
  speedup, which is distinctive enough to mention once in §1.2 as
  evidence that the architectural choice pays off.
- **Honest about scope.** §1.5 (Scope and non-goals) is load-bearing.
  A reader who opens the paper thinking rlox is a JAX framework and
  only realizes it's not in §4 will be annoyed; one who reads §1.5
  and decides the scope is wrong for them will close the paper with
  a good impression. Optimize for the second.

### Architecture diagram notes

- The Mermaid diagram in §1.3 should be replaced with a real TikZ or
  matplotlib figure for the final submission. Mermaid renders
  acceptably in GitHub but LaTeX venues want a `.pdf` or
  `.eps` figure with consistent typography.
- Colors in the current Mermaid diagram (#e3f2fd blue, #fff3e0
  orange, #f3e5f5 purple) are paper-appropriate pastels. If the
  venue is monochrome-only, swap for greyscale patterns
  (solid/dotted/dashed).
- The Python block is drawn larger than the Rust block in the
  current layout because there are more nodes (6 vs 6). Adjust so
  the visual weight reflects which layer is conceptually
  foreground — typically Python, since that is what the user sees.
- Consider adding a second figure in §1.3 showing a concrete data
  flow for a single PPO training update (rollout → GAE → minibatch
  → gradient step → repeat). The current Figure 1 is architectural;
  a data-flow figure would complement it. Only add if space
  permits.

### Related work breadth vs depth

- §1.4 should summarize relationships; §2 (Related Work) should
  go deep. Do not duplicate citations between them unless the
  introduction claim is strictly narrower than the full related-work
  paragraph.
- **Mandatory citations for reviewer trust**:
  - Stable-Baselines3 [@sb3]
  - CleanRL [@cleanrl]
  - TorchRL [@torchrl]
  - Brax [@brax]
  - Henderson et al. 2018 [@henderson2018matters] (reproducibility)
  - Agarwal et al. 2021 [@agarwal2021precipice] (IQM / CI)
  - Schulman et al. 2017 PPO [@schulman2017ppo]
  - Mnih et al. 2015 DQN [@mnih2015dqn]
- **Strongly recommended**:
  - Haarnoja et al. 2018 SAC
  - Fujimoto et al. 2018 TD3
  - Schulman et al. 2015 GAE [@schulman2015gae]
  - Andrychowicz et al. 2021 [@andrychowicz2021matters] (what matters in on-policy RL)
  - Engstrom et al. 2020 [@engstrom2020matters] (PPO implementation details)

### Figure 1 alternatives I considered

- **Option A** (current): Mermaid graph with subgraphs for Python /
  PyO3 / Rust layers, nodes inside each. Pros: captures
  bidirectional data flow; reviewers can see which operations live
  where. Cons: busy; may be hard to parse on a phone screen.
- **Option B**: simpler two-box diagram with "Python" and "Rust"
  labels and an arrow between them. Pros: clean. Cons: loses all
  specificity; reader has to take our word for what's in each box.
- **Option C**: swimlane diagram showing a single PPO training
  update as a sequence of Python↔Rust crossings. Pros: concrete;
  answers "what does this look like in practice". Cons: the
  swimlane convention is less common in ML papers and may confuse
  reviewers expecting a block diagram.

I went with Option A because I think specificity is worth the
busy-ness. Reconsider if reviewers complain; Option B is ~45 minutes
of rework.

### Things I deliberately did NOT put in the introduction

- Performance numbers beyond the single 1 700× GAE reference. The
  introduction is motivation; §4 is evidence. Mixing them bloats §1.
- The name "rlox" etymology. (It is not important.)
- The list of 22 algorithms. §3.1 covers it; §1 should not.
- Any direct comparison to a specific framework's specific
  benchmark number. Those comparisons live in §4 with proper
  caveats.
- Discussion of the LLM post-training primitives (GRPO, DPO, KL).
  They deserve a paragraph but the RL paper is already tight on
  space; leave them for a §4.2.4 mention and an Appendix.

### Open questions for co-author discussion

1. **Should §1.2 claim 5 contributions or fewer?** Five feels
   slightly high for a systems-paper introduction. Consolidating
   #3 (catalog) and #4 (single-seed lesson) into a single
   "benchmark-comparison methodology" contribution would tighten
   the list to four. I lean five because each is independently
   useful and can be cited separately.
2. **Where does the Polars analogy live?** Currently it is in §1.1
   (architectural motivation) and implied in §1.3 (architectural
   split). I considered building §1 around the Polars analogy
   explicitly but decided against it because the paper should be
   comprehensible to a reader who has never used Polars.
3. **Should §1.4 go in the introduction or merge into §2 Related
   Work?** Current version has a short per-framework comparison in
   §1.4 and saves the in-depth discussion for §2. Could instead
   delete §1.4 entirely and start §2 immediately after §1.3. The
   risk of deletion is that a reviewer skimming only §1 would not
   see how rlox fits the landscape.
4. **Venue target.** If we are targeting a general ML venue
   (NeurIPS, ICML) the introduction is about right. If we are
   targeting a systems venue (OSDI, MLSys) the architectural
   discussion in §1.3 should be expanded and the algorithm
   convergence story in §4.1 should be compressed. The current
   draft is balanced for a general-ML venue.
