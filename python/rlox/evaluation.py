"""Statistical evaluation utilities (Agarwal et al., 2021 style)."""

from __future__ import annotations

import numpy as np


def interquartile_mean(scores: list[float] | np.ndarray) -> float:
    """Compute the interquartile mean (IQM) of a list of scores.

    Discards the bottom 25% and top 25%, then takes the mean.
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
    dict mapping algorithm name to list of fractions
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
    """Compute bootstrap confidence interval for the mean.

    Returns
    -------
    (lower, upper) bounds of the confidence interval
    """
    arr = np.asarray(scores, dtype=np.float64)
    n = len(arr)
    rng = np.random.default_rng(42)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        means[i] = np.mean(sample)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(means, 100 * alpha))
    upper = float(np.percentile(means, 100 * (1.0 - alpha)))
    return lower, upper
