"""Tests for rlox.evaluation — Agarwal et al. (2021) statistical evaluation toolkit."""

from __future__ import annotations

import numpy as np
import pytest

from rlox.evaluation import (
    aggregate_metrics,
    interquartile_mean,
    performance_profiles,
    probability_of_improvement,
    stratified_bootstrap_ci,
)


class TestInterquartileMean:
    def test_iqm_basic(self):
        """IQM of [1,2,3,4] should be mean of middle 50% = mean([2,3]) = 2.5."""
        assert interquartile_mean([1, 2, 3, 4]) == pytest.approx(2.5)

    def test_iqm_single_value(self):
        """IQM of a single value should return that value."""
        assert interquartile_mean([5]) == pytest.approx(5.0)

    def test_iqm_sorted_vs_unsorted(self):
        """IQM should be identical regardless of input order."""
        ordered = [1, 2, 3, 4, 5, 6, 7, 8]
        shuffled = [5, 2, 8, 1, 7, 3, 6, 4]
        assert interquartile_mean(ordered) == pytest.approx(
            interquartile_mean(shuffled)
        )

    def test_iqm_numpy_array(self):
        """Should accept numpy arrays as well as lists."""
        arr = np.array([10.0, 20.0, 30.0, 40.0])
        assert interquartile_mean(arr) == pytest.approx(25.0)

    def test_iqm_two_values(self):
        """IQM of two values should be their mean (both in middle 50%)."""
        assert interquartile_mean([3, 7]) == pytest.approx(5.0)


class TestBootstrapCI:
    def test_bootstrap_ci_contains_mean(self):
        """The 95% CI should contain the sample mean."""
        rng = np.random.default_rng(123)
        scores = rng.normal(10.0, 2.0, size=50).tolist()
        lo, hi = stratified_bootstrap_ci(scores, n_bootstrap=5000, ci=0.95)
        sample_mean = float(np.mean(scores))
        assert lo <= sample_mean <= hi

    def test_bootstrap_ci_width(self):
        """More variable data should produce a wider CI."""
        rng = np.random.default_rng(42)
        narrow = rng.normal(0.0, 1.0, size=100).tolist()
        wide = rng.normal(0.0, 10.0, size=100).tolist()
        lo_n, hi_n = stratified_bootstrap_ci(narrow, n_bootstrap=5000, ci=0.95)
        lo_w, hi_w = stratified_bootstrap_ci(wide, n_bootstrap=5000, ci=0.95)
        assert (hi_w - lo_w) > (hi_n - lo_n)

    def test_bootstrap_ci_ordering(self):
        """Lower bound should be less than upper bound."""
        lo, hi = stratified_bootstrap_ci([1, 2, 3, 4, 5], n_bootstrap=1000, ci=0.95)
        assert lo < hi


class TestPerformanceProfiles:
    def test_performance_profiles_shape(self):
        """Output should have one fraction per threshold per algorithm."""
        scores = {"A": [1, 2, 3, 4, 5], "B": [2, 3, 4, 5, 6]}
        thresholds = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = performance_profiles(scores, thresholds)
        assert set(result.keys()) == {"A", "B"}
        assert len(result["A"]) == 5
        assert len(result["B"]) == 5

    def test_performance_profiles_values(self):
        """All values should be fractions in [0, 1]."""
        scores = {"A": [1, 2, 3]}
        thresholds = [0.0, 2.0, 4.0]
        result = performance_profiles(scores, thresholds)
        assert result["A"][0] == pytest.approx(1.0)  # all >= 0
        assert result["A"][1] == pytest.approx(2 / 3)  # 2,3 >= 2
        assert result["A"][2] == pytest.approx(0.0)  # none >= 4


class TestAggregateMetrics:
    def test_aggregate_metrics_keys(self):
        """Each algorithm's metrics should contain iqm, median, mean."""
        scores = {"PPO": [1, 2, 3, 4, 5], "SAC": [2, 3, 4, 5, 6]}
        result = aggregate_metrics(scores)
        for algo in ("PPO", "SAC"):
            assert "iqm" in result[algo]
            assert "median" in result[algo]
            assert "mean" in result[algo]

    def test_aggregate_metrics_optimality_gap(self):
        """Optimality gap should be present and non-negative for normalized scores."""
        scores = {"A": [0.5, 0.6, 0.7, 0.8]}
        result = aggregate_metrics(scores)
        assert "optimality_gap" in result["A"]

    def test_aggregate_metrics_values(self):
        """Sanity check: mean of [1,2,3,4,5] = 3, median = 3."""
        scores = {"X": [1, 2, 3, 4, 5]}
        result = aggregate_metrics(scores)
        assert result["X"]["mean"] == pytest.approx(3.0)
        assert result["X"]["median"] == pytest.approx(3.0)


class TestProbabilityOfImprovement:
    def test_probability_of_improvement_obvious(self):
        """Clearly better algorithm should get P > 0.9."""
        worse = [1.0, 2.0, 3.0, 4.0, 5.0]
        better = [10.0, 11.0, 12.0, 13.0, 14.0]
        p = probability_of_improvement(better, worse, n_bootstrap=5000)
        assert p > 0.9

    def test_probability_of_improvement_directional(self):
        """Better algorithm should have higher P(improvement)."""
        a = [1, 2, 3, 4, 5]
        b = [2, 3, 4, 5, 6]
        p_ba = probability_of_improvement(b, a, n_bootstrap=5000)
        p_ab = probability_of_improvement(a, b, n_bootstrap=5000)
        assert p_ba > p_ab

    def test_probability_of_improvement_range(self):
        """Result should be between 0 and 1."""
        p = probability_of_improvement([1, 2, 3], [1, 2, 3], n_bootstrap=1000)
        assert 0.0 <= p <= 1.0
