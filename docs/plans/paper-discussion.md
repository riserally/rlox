# Discussion — rlox paper draft

**Draft status**: v1, 2026-04-08. Section 5 of the paper.

**Pairs with**: §4 Results (this section reflects on what those
numbers mean) and §1 Introduction (this section revisits the
contributions claim made there).

**Target length**: 2–3 pages. The Discussion is where reviewers
look for honesty about limitations; don't pad it, but don't
short-change it either.

---

## 5. Discussion

The multi-seed benchmark results in §4 establish that rlox matches
Stable-Baselines3's convergence quality at materially higher data-
plane throughput under a matched-harness protocol. We now step back
from the specific numbers and discuss (§5.1) what the results imply
for RL framework design, (§5.2) what they deliberately do not claim,
(§5.3) three methodological lessons we internalized during this
work, (§5.4) ongoing and future directions, and (§5.5) a candid
accounting of when rlox is *not* the right choice.

### 5.1 What the results imply

#### 5.1.1 The Python data-plane tax is real, specific, and addressable

The speedups reported in §4.2 are not spread evenly across the
framework. GAE computation is 2–3 orders of magnitude faster in
the Rust implementation; replay buffer writes are an order of
magnitude faster; end-to-end rollouts are closer to 4× faster. The
non-uniformity is the interesting signal: **the overhead that the
Polars pattern eliminates is concentrated in the small-kernel,
high-frequency operations that Python interprets most inefficiently**,
and it is diluted in operations (like the policy forward pass)
that are already dominated by C++ numerical work.

This has a practical implication for framework designers: the
return on moving a given operation into Rust is not constant. It is
high for the operations that currently live in Python `for` loops
with numpy indexing — GAE, buffer pushes, running-stat updates,
per-step normalization — and low for operations that are already
one-shot tensor calls (`torch.nn.functional.*` operations, the
backward pass). The rlox design consciously draws the Rust↔Python
boundary to capture the high-return operations and leave the rest
in PyTorch; we expect future frameworks exploiting this pattern to
make similar boundary choices.

#### 5.1.2 Convergence parity is achievable without algorithmic alignment

One of the surprising findings in §4 is that rlox's default PPO
formulation (CleanRL-style, with inner `0.5` factor and
max-of-clipped value loss) produces statistically equivalent results
to SB3's default (plain MSE, no clipping) on most environments —
despite a 2× difference in effective value-loss gradient magnitude
at matched `vf_coef`. On the one environment where the formulations
genuinely diverge (Hopper-v4 at 1M steps), the divergence is large
(§4.3.1) and systematic: one formulation is reliably better.

The lesson is not "always use the CleanRL formulation". The lesson
is that **the choice of value-loss formulation matters but the
matter is environment-specific and training-length-specific**, and
framework papers should report framework-native results on matched
hyperparameters rather than forcing algorithmic alignment that
might privilege one framework's native recipe over another's. Our
§4.1 table is a measurement of "best native recipe at matched
hyperparameters"; the §4.3.1 ablation isolates the loss-formulation
contribution for the one cell where it dominates.

#### 5.1.3 Same-harness evaluation is more informative than cross-harness

The clearest methodological finding from this work is the value of
evaluating both frameworks in the same harness (§3.7). Every
published SB3-vs-something comparison we could find in the
literature either (a) used a different eval protocol for each
framework, (b) compared framework A's numbers in framework A's
eval harness against framework B's *published* numbers from
framework B's harness, or (c) used a single seed. Our
`multi_seed_runner_sb3.py` takes SB3's model into the rlox eval
harness, producing numbers that are comparable to the rlox numbers
in the same statistical protocol. This isolates framework effects
from protocol effects — the ablation in §4.3.2 shows that protocol
alone accounts for a substantial fraction of typical cross-framework
discrepancies.

We argue that **any framework benchmark paper that wants to claim
convergence parity with an external framework should report both
frameworks in the same harness**. This is not technically difficult
(our SB3 harness is 320 lines of Python), but it is rare in the
literature because the easier "cite published numbers" path is
accepted by reviewers. We encourage reviewers of future framework
papers to ask for the same-harness protocol as a default.

### 5.2 What the results do not claim

Because reviewers and readers will plausibly misread the results,
we are explicit about what they *don't* establish:

#### 5.2.1 rlox is not claiming to be the fastest RL framework

Brax [@brax] on TPU hardware can step ~10⁹ environment steps per
second on JIT-compatible environments. rlox on an M-series
MacBook Pro cannot, and we make no claim otherwise. rlox claims
to be the fastest framework **in the specific quadrant of
single-machine, PyTorch-based, Gymnasium-environment workflows**
(§1.5), and even that claim is conditional: if a user has a
JIT-compatible environment and is willing to move to JAX, Brax is
a better choice.

The speedup numbers in §4.2 are comparisons against Stable-Baselines3
and TorchRL under matched conditions (same CPU, same environments,
same problem sizes). They are not comparisons against JIT-compiled
frameworks. A reader who reads "3–50× faster on the data plane" and
concludes "rlox is the fastest RL framework" has misread us.

#### 5.2.2 rlox is not claiming implementation correctness beyond convergence parity

Our multi-seed tables (§4.1) establish that rlox and SB3 produce
statistically indistinguishable returns on matched hyperparameters.
They do **not** establish that the two implementations are
bit-identical, produce the same gradient magnitudes at any given
step, or agree on any intermediate quantity. In fact §3.8 enumerates
several axes where we know they don't agree. Convergence parity is
a weaker but more useful claim than bit-identical: it says that for
practical purposes a user can substitute rlox for SB3 without
changing the final policy quality, which is the question most
users actually ask. The stronger claim of bit-identical
implementation would require forcing algorithmic alignment on every
axis in §3.8, which would privilege one framework's recipe over the
other's and is explicitly what §5.1.2 argues against.

#### 5.2.3 The Rust data plane is not claiming to be safer than Python

Rust's memory safety guarantees are a side benefit of the language
choice but are not the primary motivation for rlox. The primary
motivation is performance; Rust was chosen because it delivers the
performance without requiring a separate C++ build system (which
would compromise the Polars pattern's "zero build pain" promise).
Users writing custom environments should still write them in
Python for the ergonomics reasons we articulate in §1.5 — the
small set of users who need to write custom data-plane operations
will benefit from Rust's safety, but the much larger set of users
writing custom policies and custom training loops will not touch
Rust code at all and will not benefit from its safety properties.

If safety in the data plane becomes a concern — for example, in a
production context where corrupt observations could propagate to
incorrect actions — the Rust implementation provides stronger
guarantees than an equivalent Python implementation would. This is a
nice property, but it is not the argument for choosing rlox over SB3.

#### 5.2.4 The results are not a claim about PyTorch's efficiency

PyTorch is a well-designed numerical library, and the remaining
bottlenecks in rlox's data path are not PyTorch's fault. Our
microbenchmarks do not involve PyTorch on the data plane at all —
`compute_gae_batched` operates on numpy arrays, as does the replay
buffer, as does VecNormalize. The speedup claims in §4.2 are
measurements of *Python-plus-numpy* versus *Rust*, not of PyTorch
versus Rust. A version of SB3 that moved its GAE implementation into
C++ or into a `torch.compile`-JIT'd function would close most of the
gap; the fact that SB3 has not done so reflects a design priority
(simplicity and approachability) rather than a technical limitation
of PyTorch.

### 5.3 Three methodological lessons

During the development of rlox we learned three things that are
not captured by the quantitative results but are, we think, useful
to share with other framework authors.

#### 5.3.1 Single-seed bisection is systematically misleading

The single most expensive mistake we made during development was to
motivate a PPO loss-formulation change with a single-seed bisection
at 200k and 500k training steps on Hopper-v4. The bisection showed
that the SB3-aligned formulation (plain MSE, no inner `0.5`)
outperformed the CleanRL formulation at both training lengths. We
flipped the default, pushed the change, and discovered on the next
multi-seed run that the SB3-aligned formulation regressed the 1M-step
Hopper number by 57% — the A/B in §4.3.1.

The reason the bisection misled us is that both formulations are
still in early-training regime at 500k steps on Hopper: the policy
has not yet started generating the high-magnitude advantages that
the value function must track, so the formulation difference has
not yet mattered. By 1M the difference is decisive. Looking only at
200k and 500k without running to convergence lost us most of the
signal.

The generalization is: **hyperparameter and loss-formulation
choices whose effect compounds over training length cannot be
validated from early checkpoints**. A 500k-step bisection on Hopper
is informative about whether an algorithm is in the right regime
at 500k steps; it is not informative about whether the algorithm
will converge correctly at 1M or 5M steps. Framework developers
should resist the temptation to validate defaults on truncated
training runs and should budget for full-length ablations even
when the compute cost is significant.

#### 5.3.2 The "same hyperparameters, different framework" comparison has hidden asymmetries

The inconsistencies catalog in §3.8 documents at least a dozen
axes on which rlox and SB3 default differently while nominally
accepting the same hyperparameter names. When we started building
`multi_seed_runner_sb3.py` we expected the translation layer to
handle two or three renamings; by the time we were done we had
found at least twelve asymmetries and fixed the ones that most
affected the measurement. The remaining ones are preserved and
disclosed.

The generalization is: **a "same hyperparameters" comparison
between two frameworks is almost certainly not actually
comparing the same thing** unless the authors have done the work
to build an explicit translation layer and document every
asymmetry. A list of every "same hyperparameter" comparison in
published RL benchmark papers would probably contain dozens of
unintentional asymmetries, and reviewers are collectively unable
to catch them because catching them requires reading two
frameworks' source code in parallel. We encourage future
framework comparisons to publish their translation layer as a
first-class artifact rather than describing "matched
hyperparameters" in prose.

#### 5.3.3 Evaluation protocol dominates small effect sizes

The eval protocol ablation in §4.3.2 shows that the difference
between SB3-style eval (10 episodes, single reset seed, deterministic
policy) and rlox-harness eval (30 episodes, per-episode reset seed,
deterministic policy) is larger than the typical algorithm-design
difference we try to detect in a benchmark. In practice this means
that a reader comparing a published rlox number against a published
SB3-zoo number will see a delta that has nothing to do with the
frameworks and everything to do with the protocol.

The generalization — well-known in the literature [@henderson2018matters;
@colas2018how] but not always internalized in practice — is that
**evaluation protocol is a first-class experimental variable and
must be matched between compared systems**. The rlox paper's
methodological contribution is not the 30-episode protocol
specifically (any consistent protocol would do); it is the
insistence that whatever protocol is used, **the same protocol is
used for both frameworks being compared**.

### 5.4 Future work

Several directions are immediate priorities for the rlox project
and the broader pattern:

#### 5.4.1 Expanding the comparison to CleanRL, TorchRL, and Brax

The §4.1 comparison covers SB3 only. CleanRL and TorchRL are
natural next targets using the same harness approach; the
translation layer work is non-trivial but finite. Brax is more
complex because it requires cross-backend comparisons (CPU vs TPU)
and is better handled as a separate paper focused specifically on
the JIT-compiled vs Rust-accelerated trade-off.

#### 5.4.2 Pre-built wheels

The current installation path requires `maturin develop --release`,
which requires a Rust toolchain. This is a non-trivial barrier for
users who want to try rlox without committing to a systems setup.
Pre-built wheels for common platforms (Linux x86-64, macOS ARM,
Linux ARM, Windows x86-64) would bring rlox's installation
ergonomics in line with SB3's `pip install stable-baselines3`.
The maturin CI integration makes this straightforward; the
blocking concern is the testing budget for platform-specific
correctness.

#### 5.4.3 Custom environments in Rust

rlox currently provides native Rust implementations of CartPole-v1
and Pendulum-v1 for maximum throughput on small-observation
benchmarks. Extending the native Rust environment set — MountainCar,
LunarLander, Acrobot — would make the throughput advantages
accessible on a wider range of classic-control tasks without
requiring users to write their own Rust environments. This is a
bounded amount of work (each environment is 100–300 lines of
Rust) and would primarily benefit the classic-control benchmark
story.

#### 5.4.4 Ablation-paper companion

The methodological lessons in §5.3 and the inconsistencies
catalog in §3.8 are independently interesting as a methods
contribution to the RL literature. A companion paper focused
exclusively on cross-framework benchmark methodology — rather
than on rlox as a framework — would let us go deeper on the
reliability story than the current paper's space budget allows.
We are explicit about this as a potential direction so that
reviewers do not ask why §3.8 is not the entire paper.

#### 5.4.5 Production deployment story

rlox currently provides checkpoint save/load, ONNX export for
policies, and a minimal gRPC server for model inference. A
complete "deploy this checkpoint to production" story would
require additional work on serving, monitoring, safety envelopes,
and integration with standard MLOps tools. We consider this
out of scope for the current paper but acknowledge that the
"single-machine research speedup" story does not directly
address production users.

### 5.5 When not to use rlox

A discussion of a framework that does not include a "when to use
something else" section is incomplete. We list five scenarios where
rlox is not the right choice:

1. **You have a JIT-compatible environment and are comfortable with
   JAX.** Brax or purejaxrl will give you better throughput than
   rlox can, because the JIT graph eliminates Python overhead
   entirely rather than just moving it across a boundary.

2. **You need to distribute training across more than one machine.**
   Ray RLlib's distributed scheduler is production-tested at a scale
   rlox is not designed for. rlox provides gRPC-based distributed
   actor/learner code in `crates/rlox-grpc` but the single-machine
   case remains the primary design target.

3. **You are using rlox exclusively to learn how RL algorithms work
   and are going to read the source code.** CleanRL is a better
   choice for pedagogical purposes because every algorithm is a
   single self-contained file with no shared dependencies.

4. **You are building a production workflow on top of
   Stable-Baselines3 and the ecosystem around it (SB3-contrib,
   rl-baselines3-zoo training scripts, the SB3 callback contrib,
   the Hugging Face SB3 model hub).** Migrating a production
   workflow from SB3 to rlox is not free; the switching cost
   probably exceeds the speedup for short-to-medium training runs,
   and only becomes favorable for large-scale repeated training.

5. **You cannot install a Rust toolchain in your environment.**
   Until pre-built wheels ship (§5.4.2), rlox's `maturin develop
   --release` installation step requires `rustup`, which is a
   non-starter in some institutional computing environments.

The headline claim of rlox is that for researchers and applied ML
engineers who have existing PyTorch workflows, who train on a
single machine, who use Gymnasium-compatible environments, and who
care about wall-clock throughput, rlox provides the speedup they
need without the workflow disruption they want to avoid. That
constituency is large, but it is not everyone, and we would rather
users self-select out of rlox than adopt it and be disappointed.

---

## Writing-time notes for author(s)

*These notes are for the author(s) and should be deleted before
compilation.*

### Discussion is where reviewers decide if you are honest

Reviewer calibration on framework papers typically assigns the
Discussion section 40% of the "is this paper honest" signal. A
Discussion that admits limitations, credits prior work, and names
the cases where the framework is wrong for the user gets the
paper across the accept/reject line at the major venues. A
Discussion that reads as marketing does the opposite. Re-read this
section during the copy-edit pass with an explicitly skeptical eye:
anywhere it sounds like a blog post ("rlox is a great choice for...")
rewrite as a neutral scientific claim ("rlox provides X for users
whose workflow matches Y").

### Citation density in §5.3

The three lessons in §5.3 cite Henderson 2018, Colas 2018,
Agarwal 2021, and Andrychowicz 2021. Double-check these citations
actually exist in `references.bib` and are the correct versions
(conference vs arXiv, 2018 vs 2019). Reviewers at top venues will
catch citation errors and they reflect badly on care.

### §5.1 vs §5.2 balance

§5.1 (implications) has three subsections saying "our results
show X". §5.2 (not claiming) has four subsections saying "our
results do NOT show Y". This 3:4 ratio is deliberate and I want
to preserve it: the "what we're not claiming" section should be
longer than the "what we are claiming" section in a framework
paper, because claim-inflation is the most common reviewer
complaint in this subfield.

### §5.3.1 tone check

The lesson in §5.3.1 is based on a real mistake we made during
rlox development (the Hopper loss-formulation revert, documented
in commit `cd8cbb9` and in `/tmp/hopper_ab.log`). Reviewers will
read the mea culpa as a positive signal of intellectual honesty
**if** the prose is matter-of-fact about it. Avoid both
self-flagellation ("we really messed this up") and minimization
("a small issue we noticed"). The current draft is tuned
appropriately but re-read it after a cooling-off period.

### Things I cut

- A long paragraph on **why Rust over Zig or C++**. Cut because
  the rationale is partly esthetic and partly ecosystem-based
  (PyO3 + ndarray + Polars existing in Rust but not in Zig) and
  belongs in §1 or §2 if anywhere.
- A paragraph on **the developer experience of maintaining a
  Rust+Python codebase**. Interesting but off-topic for an ML
  venue.
- A discussion of **how rlox's inconsistencies catalog could
  evolve into a community resource** tracking framework-to-framework
  differences. This idea is worth pursuing but reads as advocacy
  in the paper's Discussion.
- A comparison of **rlox's approach to determinism** against
  PyTorch's cuDNN-deterministic mode. Too technical for §5 and
  not a load-bearing claim.

### Open questions

1. **Is §5.3 in the right section?** The three lessons are
   methodological rather than specifically about rlox's
   contributions, and could belong in a "lessons learned" section
   in the appendix or in a separate methodology paper. They are
   here because (a) reviewers expect a "things we learned" section
   in a systems paper and (b) lesson §5.3.2 directly supports the
   inconsistencies catalog contribution in §1.2. Revisit if the
   section feels crowded.
2. **Should §5.4 (future work) be a sub-section of Conclusion
   instead?** Some venues expect future work at the end; others
   expect it in the Discussion. Current placement is §5.4 in
   Discussion; if the venue template has a separate Conclusion
   section, move there.
3. **How specific should §5.5 (when not to use rlox) be?** The
   current five scenarios are deliberately broad. Could be
   narrower (cite specific framework version numbers) or
   broader (cite research programs). The breadth-level tuning
   depends on whether reviewers flag it as too vague or too
   self-critical.
