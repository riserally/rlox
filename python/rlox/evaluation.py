"""Statistical evaluation utilities (Agarwal et al., 2021 style).

Implements robust evaluation metrics for comparing RL algorithms:
- Interquartile Mean (IQM) — more robust than median or mean
- Performance profiles — fraction of runs above threshold
- Stratified bootstrap confidence intervals
- Aggregate metrics dashboard
- Probability of improvement between algorithms
"""

from __future__ import annotations

import numpy as np


def interquartile_mean(scores: list[float] | np.ndarray) -> float:
    """Compute the interquartile mean (IQM) of a list of scores.

    Discards the bottom 25% and top 25%, then takes the mean of the
    remaining middle 50%. More robust than mean (outlier-resistant)
    and more statistically efficient than median.

    Parameters
    ----------
    scores : list or array of scores

    Returns
    -------
    The IQM as a float.
    """
    arr = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(arr)
    q1_idx = int(np.ceil(n * 0.25))
    q3_idx = int(np.floor(n * 0.75))
    if q1_idx >= q3_idx:
        return float(np.mean(arr))
    return float(np.mean(arr[q1_idx:q3_idx]))


def performance_profiles(
    scores_dict: dict[str, list[float]], thresholds: list[float]
) -> dict[str, list[float]]:
    """Compute performance profiles: fraction of runs above each threshold.

    Parameters
    ----------
    scores_dict : dict mapping algorithm name to list of scores
    thresholds : list of threshold values

    Returns
    -------
    dict mapping algorithm name to list of fractions (one per threshold)
    """
    result: dict[str, list[float]] = {}
    for name, scores in scores_dict.items():
        arr = np.asarray(scores, dtype=np.float64)
        fractions = [float(np.mean(arr >= t)) for t in thresholds]
        result[name] = fractions
    return result


def stratified_bootstrap_ci(
    scores: list[float] | np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the IQM.

    Uses percentile bootstrap with a fixed seed for reproducibility.

    Parameters
    ----------
    scores : list or array of scores
    n_bootstrap : number of bootstrap resamples
    ci : confidence level (default 0.95 for 95% CI)

    Returns
    -------
    (lower, upper) bounds of the confidence interval
    """
    arr = np.asarray(scores, dtype=np.float64)
    n = len(arr)
    rng = np.random.default_rng(42)
    boot_iqms = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_iqms[i] = interquartile_mean(sample)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_iqms, 100 * alpha))
    upper = float(np.percentile(boot_iqms, 100 * (1.0 - alpha)))
    return lower, upper


def aggregate_metrics(
    scores_dict: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    """Compute aggregate metrics for each algorithm.

    For each algorithm, computes:
    - iqm: interquartile mean
    - median: median score
    - mean: arithmetic mean
    - optimality_gap: 1 - IQM (assumes scores normalized to [0, 1];
      still computed for unnormalized scores as a convenience)

    Parameters
    ----------
    scores_dict : dict mapping algorithm name to list of scores

    Returns
    -------
    dict mapping algorithm name to dict of metric values
    """
    result: dict[str, dict[str, float]] = {}
    for name, scores in scores_dict.items():
        arr = np.asarray(scores, dtype=np.float64)
        iqm = interquartile_mean(arr)
        result[name] = {
            "iqm": iqm,
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "optimality_gap": 1.0 - iqm,
        }
    return result


# ---------------------------------------------------------------------------
# Score normalization baselines (random / expert)
# ---------------------------------------------------------------------------

# Baseline scores: (random_score, expert_score) for standard environments.
# Random scores are the mean return of a uniform-random policy over 100 episodes.
# Expert scores are from known well-tuned implementations (SB3 rl-zoo3 defaults).
SCORE_BASELINES: dict[str, tuple[float, float]] = {
    # Classic control
    "CartPole-v1": (22.0, 500.0),
    "Acrobot-v1": (-500.0, -72.0),
    "MountainCar-v0": (-200.0, -104.0),
    "Pendulum-v1": (-1600.0, -150.0),
    "LunarLander-v3": (-200.0, 260.0),
    # MuJoCo (from rliable / SB3 zoo defaults, 1M steps)
    "HalfCheetah-v4": (-282.0, 5500.0),
    "Walker2d-v4": (1.6, 4200.0),
    "Hopper-v4": (18.0, 2500.0),
    "Ant-v4": (-50.0, 4000.0),
    "Humanoid-v4": (120.0, 5500.0),
    "Swimmer-v4": (0.3, 360.0),
    "Reacher-v4": (-44.0, -3.5),
    "InvertedPendulum-v4": (0.0, 1000.0),
    "InvertedDoublePendulum-v4": (70.0, 9300.0),
}


def normalize_score(
    score: float,
    env_id: str,
    random_score: float | None = None,
    expert_score: float | None = None,
) -> float:
    """Normalize a raw score to [0, 1] using random and expert baselines.

    Uses the formula: ``(score - random) / (expert - random)``.
    A normalized score of 0 means random-level performance, 1 means
    expert-level. Values can exceed [0, 1].

    Parameters
    ----------
    score : raw environment return
    env_id : Gymnasium environment ID (used to look up baselines)
    random_score : override random baseline (default: looked up from table)
    expert_score : override expert baseline (default: looked up from table)

    Returns
    -------
    Normalized score as a float.

    Raises
    ------
    ValueError
        If env_id not found in baselines and no overrides provided.
    """
    if random_score is None or expert_score is None:
        if env_id not in SCORE_BASELINES:
            raise ValueError(
                f"No baseline scores for {env_id!r}. "
                f"Provide random_score and expert_score, or add to SCORE_BASELINES. "
                f"Known envs: {sorted(SCORE_BASELINES)}"
            )
        default_random, default_expert = SCORE_BASELINES[env_id]
        random_score = random_score if random_score is not None else default_random
        expert_score = expert_score if expert_score is not None else default_expert

    denom = expert_score - random_score
    if abs(denom) < 1e-10:
        return 0.0
    return (score - random_score) / denom


def normalize_scores(
    scores: list[float] | np.ndarray,
    env_id: str,
    random_score: float | None = None,
    expert_score: float | None = None,
) -> np.ndarray:
    """Normalize an array of scores to [0, 1] using baselines.

    Parameters
    ----------
    scores : list or array of raw scores
    env_id : Gymnasium environment ID
    random_score : override random baseline
    expert_score : override expert baseline

    Returns
    -------
    Normalized scores as a numpy array.
    """
    return np.array(
        [normalize_score(s, env_id, random_score, expert_score) for s in scores],
        dtype=np.float64,
    )


def probability_of_improvement(
    scores_a: list[float] | np.ndarray,
    scores_b: list[float] | np.ndarray,
    n_bootstrap: int = 10000,
) -> float:
    """Estimate P(algorithm A > algorithm B) via paired bootstrap.

    Resamples both score sets and computes the fraction of bootstrap
    iterations where IQM(A) > IQM(B).

    Parameters
    ----------
    scores_a : scores from algorithm A
    scores_b : scores from algorithm B
    n_bootstrap : number of bootstrap resamples

    Returns
    -------
    Estimated probability that A is better than B (float in [0, 1]).
    """
    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)
    n_a, n_b = len(arr_a), len(arr_b)
    rng = np.random.default_rng(42)

    a_wins = 0
    for _ in range(n_bootstrap):
        sample_a = rng.choice(arr_a, size=n_a, replace=True)
        sample_b = rng.choice(arr_b, size=n_b, replace=True)
        if interquartile_mean(sample_a) > interquartile_mean(sample_b):
            a_wins += 1

    return a_wins / n_bootstrap


# ---------------------------------------------------------------------------
# Non-stationary RL evaluation metrics
# ---------------------------------------------------------------------------


def dynamic_regret(
    agent_rewards: list[float] | np.ndarray,
    optimal_rewards: list[float] | np.ndarray,
) -> float:
    """Compute dynamic regret: sum of per-step optimality gaps.

    Dynamic regret measures how much worse the agent performs compared to
    the optimal policy *at each time step* (which may change in non-stationary
    settings). Lower is better; 0 means the agent perfectly tracks the optimum.

    DR = sum_{t=1}^T [V*(s_t; M_t) - V^pi(s_t; M_t)]

    Parameters
    ----------
    agent_rewards : per-step rewards achieved by the agent
    optimal_rewards : per-step rewards of the (time-varying) optimal policy

    Returns
    -------
    Total dynamic regret (non-negative float).
    """
    agent = np.asarray(agent_rewards, dtype=np.float64)
    optimal = np.asarray(optimal_rewards, dtype=np.float64)
    assert len(agent) == len(optimal), (
        f"Length mismatch: agent={len(agent)}, optimal={len(optimal)}"
    )
    return float(np.sum(np.maximum(optimal - agent, 0.0)))


def adaptation_latency(
    rewards: list[float] | np.ndarray,
    change_point: int,
    pre_change_mean: float,
    recovery_fraction: float = 0.9,
) -> int | None:
    """Measure steps to recover after a change-point.

    Counts how many steps after the change-point until the agent's
    rolling mean reward reaches `recovery_fraction` of its pre-change
    performance level.

    Parameters
    ----------
    rewards : per-step rewards (full trajectory including before/after change)
    change_point : index at which the environment changed
    pre_change_mean : mean reward before the change (target to recover to)
    recovery_fraction : fraction of pre-change performance to reach (default 0.9)

    Returns
    -------
    Number of steps after change_point to reach recovery, or None if never recovered.
    """
    arr = np.asarray(rewards, dtype=np.float64)
    target = pre_change_mean * recovery_fraction
    post_change = arr[change_point:]
    if len(post_change) == 0:
        return None

    # Use a rolling window of 10 steps for smoothing
    window = min(10, len(post_change))
    for i in range(window - 1, len(post_change)):
        rolling_mean = float(np.mean(post_change[max(0, i - window + 1):i + 1]))
        if rolling_mean >= target:
            return i + 1
    return None


def forgetting_ratio(
    reward_on_task_before: float,
    reward_on_task_after: float,
) -> float:
    """Measure catastrophic forgetting after training on a new task.

    Computes the ratio of performance retained on an old task after
    the agent has been trained on a new task/regime.

    Parameters
    ----------
    reward_on_task_before : performance on task A before training on task B
    reward_on_task_after : performance on task A after training on task B

    Returns
    -------
    Forgetting ratio in [0, 1]. 1.0 = no forgetting, 0.0 = total forgetting.
    Values > 1 indicate positive transfer (improved on task A).
    """
    if abs(reward_on_task_before) < 1e-10:
        return 1.0 if abs(reward_on_task_after) < 1e-10 else 0.0
    return reward_on_task_after / reward_on_task_before
