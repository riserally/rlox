"""Type stubs for the rlox Rust extension module (_rlox_core).

This module is compiled from Rust via PyO3 and provides the high-performance
data-plane primitives: environments, experience buffers, GAE computation,
and LLM post-training operations.

All array inputs/outputs use NumPy. The Rust implementations use Rayon for
parallelism and ChaCha8 RNG for reproducible, low-latency sampling.
"""

from typing import Optional
import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Phase 1: Environment Engine
# ---------------------------------------------------------------------------

class CartPole:
    """Rust-native CartPole-v1 environment with exact Gymnasium physics.

    Provides ~2x single-step speedup over Gymnasium by eliminating Python
    interpreter overhead. Identical dynamics, reward, and termination
    conditions to ``gymnasium.make('CartPole-v1')``.

    Parameters
    ----------
    seed : int, optional
        RNG seed for reproducibility. If None, uses entropy from the OS.

    Example
    -------
    >>> env = CartPole(seed=42)
    >>> obs = env.reset()
    >>> result = env.step(1)  # push right
    >>> obs, reward = result["obs"], result["reward"]
    """
    def __init__(self, seed: Optional[int] = None) -> None: ...
    def step(self, action: int) -> dict[str, object]:
        """Take one step. Action must be 0 (left) or 1 (right).

        Returns dict with keys: ``obs`` (f32 array[4]), ``reward`` (f64),
        ``terminated`` (bool), ``truncated`` (bool).
        """
        ...
    def reset(self, seed: Optional[int] = None) -> npt.NDArray[np.float32]:
        """Reset to initial state. Returns observation array of shape (4,)."""
        ...
    def render(self) -> Optional[str]:
        """Return ASCII art rendering of the current state, or None."""
        ...

class VecEnv:
    """Vectorized CartPole environments with Rayon work-stealing parallelism.

    Steps all ``n`` environments in parallel using Rust threads. At high env
    counts (>=64), delivers 3-6x speedup over Gymnasium's SyncVectorEnv.

    .. note::
        Currently only supports CartPole-v1. The ``env_id`` parameter in
        ``__init__`` is accepted but ignored — all envs are CartPole.

    Parameters
    ----------
    n : int
        Number of parallel environments.
    seed : int, optional
        Base seed. Each env gets ``seed + i`` for reproducibility.

    Example
    -------
    >>> vec = VecEnv(n=16, seed=0)
    >>> obs = vec.reset_all()          # shape: (16, 4)
    >>> result = vec.step_all([1]*16)  # step all envs with action=1
    >>> next_obs = result["obs"]       # shape: (16, 4)
    """
    def __init__(self, n: int, seed: Optional[int] = None) -> None: ...
    def num_envs(self) -> int:
        """Return the number of parallel environments."""
        ...
    def step_all(self, actions: list[int]) -> dict[str, object]:
        """Step all environments in parallel.

        Parameters
        ----------
        actions : list[int]
            One action per environment. Length must equal ``num_envs()``.

        Returns
        -------
        dict with keys:
            - ``obs``: f32 array of shape (n_envs, 4)
            - ``rewards``: f64 array of shape (n_envs,)
            - ``terminated``: bool array of shape (n_envs,)
            - ``truncated``: bool array of shape (n_envs,)

        Terminated environments are auto-reset; the returned ``obs`` is
        the *post-reset* observation for those envs.
        """
        ...
    def reset_all(self, seed: Optional[int] = None) -> npt.NDArray[np.float32]:
        """Reset all environments. Returns obs array of shape (n_envs, 4)."""
        ...

class GymEnv:
    """Lightweight wrapper around a Gymnasium environment.

    Normalises the step/reset interface for use with rlox utilities.
    The actual stepping happens in Python (no Rust speedup).

    Parameters
    ----------
    env : gymnasium.Env
        An instantiated Gymnasium environment.
    """
    def __init__(self, env: object) -> None: ...
    def step(self, action: object) -> dict[str, object]:
        """Step the wrapped environment. Returns dict with standard RL keys."""
        ...
    def reset(self, seed: Optional[int] = None) -> object:
        """Reset the wrapped environment. Returns initial observation."""
        ...

# ---------------------------------------------------------------------------
# Phase 2: Experience Storage
# ---------------------------------------------------------------------------

class ExperienceTable:
    """Append-only columnar store for on-policy RL transitions.

    Stores observations, actions, rewards, and done flags in contiguous
    column arrays. Optimised for sequential writes and bulk reads —
    ideal for on-policy algorithms that consume and discard each rollout.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of observation vectors.
    act_dim : int
        Dimensionality of action vectors.

    Example
    -------
    >>> table = ExperienceTable(obs_dim=4, act_dim=1)
    >>> table.push(obs=np.zeros(4, dtype=np.float32), action=0.0,
    ...           reward=1.0, terminated=False, truncated=False)
    >>> len(table)
    1
    >>> table.clear()
    """
    def __init__(self, obs_dim: int, act_dim: int) -> None: ...
    def __len__(self) -> int:
        """Return the number of stored transitions."""
        ...
    def push(
        self,
        obs: npt.NDArray[np.float32],
        action: float,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Append a single transition."""
        ...
    def observations(self) -> npt.NDArray[np.float32]:
        """Return all observations as a contiguous (N, obs_dim) array."""
        ...
    def rewards(self) -> npt.NDArray[np.float32]:
        """Return all rewards as a contiguous (N,) array."""
        ...
    def clear(self) -> None:
        """Remove all stored transitions (retains allocated capacity)."""
        ...

class ReplayBuffer:
    """Fixed-capacity ring buffer with uniform random sampling.

    Uses a pre-allocated ring buffer for O(1) insertion and ChaCha8 RNG
    for fast, reproducible uniform sampling. When full, oldest transitions
    are overwritten.

    Delivers 8-13x faster sampling than TorchRL and SB3, with p99 latency
    under 15us even for batch_size=1024.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    obs_dim : int
        Dimensionality of observation vectors.
    act_dim : int
        Dimensionality of action vectors.

    Example
    -------
    >>> buf = ReplayBuffer(capacity=10_000, obs_dim=4, act_dim=1)
    >>> buf.push(obs, action=0.5, reward=1.0, terminated=False, truncated=False)
    >>> batch = buf.sample(batch_size=32, seed=0)
    >>> batch["obs"].shape  # (32, 4)
    """
    def __init__(self, capacity: int, obs_dim: int, act_dim: int) -> None: ...
    def __len__(self) -> int:
        """Return current number of stored transitions."""
        ...
    def push(
        self,
        obs: npt.NDArray[np.float32],
        action: float,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Insert a transition, overwriting the oldest if at capacity."""
        ...
    def sample(self, batch_size: int, seed: int) -> dict[str, object]:
        """Sample a random batch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.
        seed : int
            RNG seed for this sample (use step counter for reproducibility).

        Returns
        -------
        dict with keys: ``obs``, ``actions``, ``rewards``, ``terminated``,
        ``truncated``. Each value is a NumPy array.
        """
        ...

class PrioritizedReplayBuffer:
    """Replay buffer with proportional prioritisation (Schaul et al., 2016).

    Uses a sum-tree for O(log N) sampling proportional to priority.
    Supports importance-sampling weight correction via beta annealing.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions.
    obs_dim : int
        Observation dimensionality.
    act_dim : int
        Action dimensionality.
    alpha : float
        Priority exponent. 0 = uniform, 1 = fully prioritised.
    beta : float
        Initial importance-sampling exponent. Anneal toward 1.0.
    """
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        alpha: float = 0.6,
        beta: float = 0.4,
    ) -> None: ...
    def __len__(self) -> int: ...
    def push(
        self,
        obs: npt.NDArray[np.float32],
        action: float,
        reward: float,
        terminated: bool,
        truncated: bool,
        priority: float = 1.0,
    ) -> None:
        """Insert a transition with the given priority."""
        ...
    def sample(self, batch_size: int, seed: int) -> dict[str, object]:
        """Sample proportional to priorities.

        Returns dict with additional keys ``weights`` (importance-sampling
        weights) and ``indices`` (for priority updates).
        """
        ...
    def update_priorities(
        self,
        indices: npt.NDArray[np.uint64],
        priorities: npt.NDArray[np.float64],
    ) -> None:
        """Update priorities for previously sampled transitions."""
        ...
    def set_beta(self, beta: float) -> None:
        """Set the importance-sampling exponent (anneal toward 1.0)."""
        ...

class VarLenStore:
    """Packed variable-length sequence storage using offsets + flat data.

    Efficiently stores sequences of different lengths in a single contiguous
    buffer. Used for LLM token sequences in GRPO/DPO pipelines.

    Example
    -------
    >>> store = VarLenStore()
    >>> store.push(np.array([1, 2, 3], dtype=np.uint32))
    >>> store.push(np.array([4, 5], dtype=np.uint32))
    >>> store.num_sequences()  # 2
    >>> store.get(0)           # array([1, 2, 3])
    """
    def __init__(self) -> None: ...
    def push(self, seq: npt.NDArray[np.uint32]) -> None:
        """Append a variable-length sequence."""
        ...
    def num_sequences(self) -> int:
        """Return the number of stored sequences."""
        ...
    def total_elements(self) -> int:
        """Return the total number of elements across all sequences."""
        ...
    def get(self, index: int) -> npt.NDArray[np.uint32]:
        """Retrieve the sequence at the given index."""
        ...

# ---------------------------------------------------------------------------
# Phase 3: Training Core
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    dones: npt.NDArray[np.float64],
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute Generalized Advantage Estimation (Schulman et al., 2016).

    Performs the backward GAE scan in Rust — 140x faster than a Python loop,
    1700x faster than TorchRL's TensorDict-based implementation.

    Parameters
    ----------
    rewards : array of shape (T,)
        Per-step rewards.
    values : array of shape (T,)
        Value function estimates V(s_t).
    dones : array of shape (T,)
        Episode termination flags (1.0 = done).
    last_value : float
        Bootstrap value V(s_{T+1}) for the state after the last step.
    gamma : float
        Discount factor, typically 0.99.
    lam : float
        GAE lambda for bias-variance tradeoff, typically 0.95.

    Returns
    -------
    (advantages, returns) : tuple of f64 arrays, each of shape (T,)
        ``returns = advantages + values``.

    Example
    -------
    >>> adv, ret = compute_gae(rewards, values, dones, last_value=0.0,
    ...                        gamma=0.99, lam=0.95)
    """
    ...

def compute_gae_batched(
    rewards: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    dones: npt.NDArray[np.float64],
    last_values: npt.NDArray[np.float64],
    n_steps: int,
    gamma: float,
    lam: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Batched GAE: compute GAE for multiple environments in a single call.

    All inputs are flat 1-D arrays of length ``n_envs * n_steps``, laid out as
    ``[env0_step0, env0_step1, ..., env1_step0, ...]``.

    Parameters
    ----------
    rewards : array of shape (n_envs * n_steps,)
    values : array of shape (n_envs * n_steps,)
    dones : array of shape (n_envs * n_steps,)
    last_values : array of shape (n_envs,)
    n_steps : int
    gamma : float
    lam : float

    Returns
    -------
    (advantages, returns) : tuple of f64 arrays, each of shape (n_envs * n_steps,)
    """
    ...

def compute_gae_batched_f32(
    rewards: npt.NDArray[np.float32],
    values: npt.NDArray[np.float32],
    dones: npt.NDArray[np.float32],
    last_values: npt.NDArray[np.float32],
    n_steps: int,
    gamma: float,
    lam: float,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Batched GAE in f32 — avoids f64 conversion overhead."""
    ...

def compute_vtrace(
    log_rhos: npt.NDArray[np.float32],
    rewards: npt.NDArray[np.float32],
    values: npt.NDArray[np.float32],
    dones: npt.NDArray[np.float32],
    bootstrap_value: float,
    gamma: float,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Compute V-trace targets and policy gradient advantages (Espeholt et al., 2018).

    Used by IMPALA for off-policy correction. Clips importance weights
    with rho_bar and c_bar thresholds.

    Parameters
    ----------
    log_rhos : array of shape (T,)
        Log importance ratios: log(pi(a|s) / mu(a|s)).
    rewards : array of shape (T,)
        Per-step rewards.
    values : array of shape (T,)
        Value function estimates V(s_t).
    dones : array of shape (T,)
        Episode termination flags (1.0 = done, 0.0 = not done).
    bootstrap_value : float
        V(s_{T+1}) for the state after the final step.
    gamma : float
        Discount factor.
    rho_bar : float
        Truncation threshold for importance weights in value update.
    c_bar : float
        Truncation threshold for trace-cutting coefficients.

    Returns
    -------
    (vs, pg_advantages) : tuple of f32 arrays, each of shape (T,)
    """
    ...

class RunningStats:
    """Welford's online algorithm for computing running mean and variance.

    Useful for reward/observation normalisation during training. Updates
    are O(1) per sample with numerically stable variance computation.

    Example
    -------
    >>> stats = RunningStats()
    >>> stats.batch_update(np.array([1.0, 2.0, 3.0]))
    >>> stats.mean()   # 2.0
    >>> stats.std()    # ~0.816
    """
    def __init__(self) -> None: ...
    def update(self, value: float) -> None:
        """Update with a single scalar value."""
        ...
    def batch_update(self, values: npt.NDArray[np.float64]) -> None:
        """Update with a batch of values (more efficient than repeated update)."""
        ...
    def mean(self) -> float:
        """Return the current running mean."""
        ...
    def std(self) -> float:
        """Return the current running standard deviation."""
        ...
    def count(self) -> int:
        """Return the total number of values seen."""
        ...

class RunningStatsVec:
    """Per-dimension Welford's online algorithm for running mean and variance.

    Tracks independent statistics for each dimension of a vector-valued input,
    enabling proper per-feature observation normalization (matching SB3's
    ``RunningMeanStd``).

    Parameters
    ----------
    dim : int
        Number of dimensions (features) to track.

    Example
    -------
    >>> stats = RunningStatsVec(dim=11)  # e.g. Hopper obs_dim
    >>> stats.batch_update(obs_flat, batch_size=2048)
    >>> normalized = stats.normalize(single_obs)
    """
    def __init__(self, dim: int) -> None: ...
    def update(self, values: npt.NDArray[np.float64]) -> None:
        """Update with a single sample (1-D array of length ``dim``)."""
        ...
    def batch_update(
        self, data: npt.NDArray[np.float64], batch_size: int
    ) -> None:
        """Update with a flat batch of shape ``(batch_size * dim,)``."""
        ...
    def mean(self) -> npt.NDArray[np.float64]:
        """Return per-dimension running mean as array of shape ``(dim,)``."""
        ...
    def var(self) -> npt.NDArray[np.float64]:
        """Return per-dimension population variance as array of shape ``(dim,)``."""
        ...
    def std(self) -> npt.NDArray[np.float64]:
        """Return per-dimension standard deviation as array of shape ``(dim,)``."""
        ...
    def normalize(
        self, values: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Normalize a single sample: ``(values - mean) / max(std, 1e-8)``."""
        ...
    def normalize_batch(
        self, data: npt.NDArray[np.float64], batch_size: int
    ) -> npt.NDArray[np.float64]:
        """Normalize a flat batch of shape ``(batch_size * dim,)``."""
        ...
    def count(self) -> int:
        """Return the total number of samples seen."""
        ...
    def dim(self) -> int:
        """Return the dimensionality."""
        ...
    def reset(self) -> None:
        """Reset all statistics, keeping the dimensionality."""
        ...

# ---------------------------------------------------------------------------
# Phase 4: LLM Post-Training
# ---------------------------------------------------------------------------

def compute_group_advantages(
    rewards: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """GRPO group-relative advantage normalisation.

    Computes ``(reward - mean(rewards)) / std(rewards)`` for a group of
    completions. Returns zeros if std < 1e-8 (all rewards identical).

    Used in Group Relative Policy Optimization (DeepSeek-R1 style) where
    advantages are computed relative to the group rather than a baseline.

    Parameters
    ----------
    rewards : array of shape (G,)
        Reward scores for G completions of the same prompt.

    Returns
    -------
    advantages : f64 array of shape (G,)
    """
    ...

def compute_batch_group_advantages(
    rewards: npt.NDArray[np.float64],
    group_size: int,
) -> npt.NDArray[np.float64]:
    """Batched GRPO group advantages for all groups in a single call.

    Processes ``n_prompts`` groups of ``group_size`` rewards each, applying
    z-score normalisation within each group. Equivalent to calling
    :func:`compute_group_advantages` for each group but with a single
    PyO3 boundary crossing.

    Parameters
    ----------
    rewards : array of shape (n_prompts * group_size,)
        Flat array of all reward scores.
    group_size : int
        Number of completions per prompt (K).

    Returns
    -------
    advantages : f64 array of shape (n_prompts * group_size,)
    """
    ...

def compute_token_kl(
    log_probs_policy: npt.NDArray[np.float64],
    log_probs_ref: npt.NDArray[np.float64],
) -> float:
    """Token-level KL divergence between policy and reference model.

    Computes ``sum(exp(log_p) * (log_p - log_q))`` — the forward KL
    divergence at the token level. Used as a regularisation penalty in
    RLHF/GRPO to prevent the policy from drifting too far from the
    reference model.

    Parameters
    ----------
    log_probs_policy : array of shape (T,)
        Log probabilities under the current policy.
    log_probs_ref : array of shape (T,)
        Log probabilities under the reference model.

    Returns
    -------
    float
        Scalar KL divergence value.
    """
    ...

def compute_token_kl_schulman(
    log_probs_policy: npt.NDArray[np.float64],
    log_probs_ref: npt.NDArray[np.float64],
) -> float:
    """Token-level KL divergence using the Schulman (2020) estimator.

    Computes ``sum(exp(log_p - log_q) - (log_p - log_q) - 1)`` — the
    unbiased KL estimator used by TRL (HuggingFace). Numerically more
    stable than the exact ``exp(log_p) * (log_p - log_q)`` form.

    Parameters
    ----------
    log_probs_policy : array of shape (T,)
        Log probabilities under the current policy.
    log_probs_ref : array of shape (T,)
        Log probabilities under the reference model.

    Returns
    -------
    float
        Scalar KL divergence estimate.
    """
    ...

def compute_batch_token_kl(
    log_probs_policy: npt.NDArray[np.float64],
    log_probs_ref: npt.NDArray[np.float64],
    seq_len: int,
) -> npt.NDArray[np.float64]:
    """Batched token-level KL divergence: process all sequences in a single call.

    Parameters
    ----------
    log_probs_policy : array of shape (batch * seq_len,)
        Flat array of log probabilities under the current policy.
    log_probs_ref : array of shape (batch * seq_len,)
        Flat array of log probabilities under the reference model.
    seq_len : int
        Sequence length (tokens per sequence).

    Returns
    -------
    kl : f64 array of shape (batch,)
        Per-sequence KL divergence values.
    """
    ...

def compute_batch_token_kl_schulman(
    log_probs_policy: npt.NDArray[np.float64],
    log_probs_ref: npt.NDArray[np.float64],
    seq_len: int,
) -> npt.NDArray[np.float64]:
    """Batched token-level KL divergence using the Schulman (2020) estimator.

    Parameters
    ----------
    log_probs_policy : array of shape (batch * seq_len,)
        Flat array of log probabilities under the current policy.
    log_probs_ref : array of shape (batch * seq_len,)
        Flat array of log probabilities under the reference model.
    seq_len : int
        Sequence length (tokens per sequence).

    Returns
    -------
    kl : f64 array of shape (batch,)
        Per-sequence KL divergence estimates.
    """
    ...

def compute_token_kl_f32(
    log_probs_policy: npt.NDArray[np.float32],
    log_probs_ref: npt.NDArray[np.float32],
) -> float:
    """Token-level KL divergence (f32): sum(exp(log_p) * (log_p - log_q))."""
    ...

def compute_token_kl_schulman_f32(
    log_probs_policy: npt.NDArray[np.float32],
    log_probs_ref: npt.NDArray[np.float32],
) -> float:
    """Token-level KL divergence using the Schulman estimator (f32)."""
    ...

def compute_batch_token_kl_f32(
    log_probs_policy: npt.NDArray[np.float32],
    log_probs_ref: npt.NDArray[np.float32],
    seq_len: int,
) -> npt.NDArray[np.float32]:
    """Batched token-level KL divergence (f32)."""
    ...

def compute_batch_token_kl_schulman_f32(
    log_probs_policy: npt.NDArray[np.float32],
    log_probs_ref: npt.NDArray[np.float32],
    seq_len: int,
) -> npt.NDArray[np.float32]:
    """Batched token-level KL divergence using the Schulman estimator (f32)."""
    ...

class DPOPair:
    """Direct Preference Optimization (Rafailov et al., 2023) preference pair.

    Stores a prompt with chosen and rejected completions as token ID arrays.
    Used with ``VarLenStore`` for efficient batch construction.

    Parameters
    ----------
    prompt_tokens : uint32 array
        Token IDs for the shared prompt.
    chosen_tokens : uint32 array
        Token IDs for the preferred completion.
    rejected_tokens : uint32 array
        Token IDs for the dispreferred completion.
    """
    def __init__(
        self,
        prompt_tokens: npt.NDArray[np.uint32],
        chosen_tokens: npt.NDArray[np.uint32],
        rejected_tokens: npt.NDArray[np.uint32],
    ) -> None: ...
    def chosen_len(self) -> int:
        """Return the number of tokens in the chosen completion."""
        ...
    def rejected_len(self) -> int:
        """Return the number of tokens in the rejected completion."""
        ...

def pack_sequences(
    sequences: list[npt.NDArray[np.uint32]],
    max_length: int,
) -> list[dict[str, object]]:
    """Pack variable-length token sequences into fixed-length batches.

    Concatenates sequences with attention masking and position IDs for
    efficient transformer training. Sequences that exceed ``max_length``
    are truncated.

    Parameters
    ----------
    sequences : list of uint32 arrays
        Token ID sequences to pack.
    max_length : int
        Maximum packed sequence length.

    Returns
    -------
    list of dicts, each with keys:
        - ``input_ids``: uint32 array of shape (max_length,)
        - ``attention_mask``: uint32 array of shape (max_length,)
        - ``position_ids``: uint32 array of shape (max_length,)
        - ``sequence_starts``: list of start indices within the packed sequence
    """
    ...

# ---------------------------------------------------------------------------
# Phase 7: Neural Network Backends (Burn / Candle)
# ---------------------------------------------------------------------------

class ActorCritic:
    """Discrete actor-critic policy backed by a pure-Rust NN (Burn or Candle).

    Provides PPO training without PyTorch — the entire forward/backward/step
    pipeline runs in Rust. Useful for CPU-only deployments and low-latency
    inference.

    Parameters
    ----------
    backend : str
        ``"burn"`` (NdArray + autodiff) or ``"candle"`` (candle-core + candle-nn).
    obs_dim : int
        Observation dimension.
    n_actions : int
        Number of discrete actions.
    hidden : int
        Hidden layer width (2 layers). Default 64.
    lr : float
        Learning rate. Default 2.5e-4.
    seed : int
        RNG seed. Default 42.

    Example
    -------
    >>> ac = ActorCritic("candle", obs_dim=4, n_actions=2)
    >>> actions, log_probs = ac.act(obs.ravel())
    >>> values = ac.value(obs.ravel())
    >>> metrics = ac.ppo_step(obs_flat, actions, log_probs, adv, ret, old_v)
    """
    def __init__(
        self,
        backend: str,
        obs_dim: int,
        n_actions: int,
        hidden: int = 64,
        lr: float = 2.5e-4,
        seed: int = 42,
    ) -> None: ...
    def act(
        self,
        obs: npt.NDArray[np.float32],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Sample actions from the policy (no gradient tracking).

        Parameters
        ----------
        obs : flat f32 array of length ``batch * obs_dim``

        Returns
        -------
        (actions, log_probs) : tuple of 1-D f32 arrays of length ``batch``
        """
        ...

    def value(
        self,
        obs: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Compute state values (no gradient tracking).

        Parameters
        ----------
        obs : flat f32 array of length ``batch * obs_dim``

        Returns
        -------
        values : 1-D f32 array of length ``batch``
        """
        ...

    def ppo_step(
        self,
        obs: npt.NDArray[np.float32],
        actions: npt.NDArray[np.float32],
        old_log_probs: npt.NDArray[np.float32],
        advantages: npt.NDArray[np.float32],
        returns: npt.NDArray[np.float32],
        old_values: npt.NDArray[np.float32],
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        clip_vloss: bool = True,
    ) -> dict[str, float]:
        """Perform one PPO gradient step (forward + backward + optimizer step in Rust).

        Parameters
        ----------
        obs : flat f32 array of length ``batch * obs_dim``
        actions, old_log_probs, advantages, returns, old_values :
            flat f32 arrays of length ``batch``

        Returns
        -------
        dict with keys: ``policy_loss``, ``value_loss``, ``entropy``,
        ``approx_kl``, ``clip_fraction``
        """
        ...

    @property
    def learning_rate(self) -> float:
        """Current learning rate."""
        ...

    @learning_rate.setter
    def learning_rate(self, lr: float) -> None: ...
