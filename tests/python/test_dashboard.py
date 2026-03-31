"""Tests for rlox.dashboard — metrics collection, terminal display, HTML reports."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from rlox.callbacks import Callback
from rlox.dashboard import HTMLReport, MetricsCollector, TerminalDashboard


class TestMetricsCollector:
    """Tests for MetricsCollector callback."""

    def test_is_callback(self):
        mc = MetricsCollector()
        assert isinstance(mc, Callback)

    def test_stores_step_metrics(self):
        mc = MetricsCollector()
        mc.on_step(
            step=1,
            episode_reward=10.0,
            policy_loss=0.5,
            value_loss=0.3,
            entropy=1.2,
            approx_kl=0.01,
            learning_rate=3e-4,
            sps=500.0,
            explained_variance=0.8,
            clip_fraction=0.1,
        )
        data = mc.get_dataframe()
        assert data["step"] == [1]
        assert data["episode_reward"] == [10.0]
        assert data["policy_loss"] == [0.5]
        assert data["value_loss"] == [0.3]
        assert data["entropy"] == [1.2]
        assert data["approx_kl"] == [0.01]
        assert data["learning_rate"] == [3e-4]
        assert data["sps"] == [500.0]
        assert data["explained_variance"] == [0.8]
        assert data["clip_fraction"] == [0.1]

    def test_stores_multiple_steps(self):
        mc = MetricsCollector()
        for i in range(5):
            mc.on_step(step=i, episode_reward=float(i * 10))
        data = mc.get_dataframe()
        assert len(data["step"]) == 5
        assert data["step"] == [0, 1, 2, 3, 4]
        assert data["episode_reward"] == [0.0, 10.0, 20.0, 30.0, 40.0]

    def test_missing_kwargs_stored_as_none(self):
        mc = MetricsCollector()
        mc.on_step(step=1, episode_reward=10.0)
        data = mc.get_dataframe()
        assert data["policy_loss"] == [None]
        assert data["entropy"] == [None]

    def test_on_step_returns_true(self):
        mc = MetricsCollector()
        assert mc.on_step(step=1) is True

    def test_on_training_start_records_start(self):
        mc = MetricsCollector()
        mc.on_training_start(total_timesteps=100_000)
        assert mc._total_timesteps == 100_000
        assert mc._start_time is not None

    def test_on_training_end_records_end(self):
        mc = MetricsCollector()
        mc.on_training_start(total_timesteps=50_000)
        mc.on_training_end()
        assert mc._end_time is not None
        assert mc._end_time >= mc._start_time

    def test_summary_returns_expected_keys(self):
        mc = MetricsCollector()
        mc.on_training_start(total_timesteps=100)
        for i in range(10):
            mc.on_step(
                step=i,
                episode_reward=float(i),
                policy_loss=0.5 - i * 0.01,
                value_loss=0.3,
                entropy=1.0 - i * 0.05,
                sps=500.0,
            )
        mc.on_training_end()

        summary = mc.summary()
        expected_keys = {
            "total_steps",
            "total_timesteps",
            "training_time",
            "peak_reward",
            "final_reward",
            "final_policy_loss",
            "final_value_loss",
            "final_entropy",
            "mean_sps",
        }
        assert expected_keys.issubset(set(summary.keys()))

    def test_summary_peak_reward(self):
        mc = MetricsCollector()
        mc.on_training_start(total_timesteps=5)
        mc.on_step(step=0, episode_reward=5.0)
        mc.on_step(step=1, episode_reward=100.0)
        mc.on_step(step=2, episode_reward=50.0)
        mc.on_training_end()

        summary = mc.summary()
        assert summary["peak_reward"] == 100.0
        assert summary["final_reward"] == 50.0

    def test_summary_empty_collector(self):
        mc = MetricsCollector()
        summary = mc.summary()
        assert summary["total_steps"] == 0

    def test_get_dataframe_returns_copy(self):
        mc = MetricsCollector()
        mc.on_step(step=1, episode_reward=10.0)
        df1 = mc.get_dataframe()
        df1["step"].append(999)
        df2 = mc.get_dataframe()
        assert len(df2["step"]) == 1


class TestTerminalDashboard:
    """Tests for TerminalDashboard callback."""

    def test_is_callback(self):
        td = TerminalDashboard()
        assert isinstance(td, Callback)

    def test_default_update_freq(self):
        td = TerminalDashboard()
        assert td._update_freq > 0

    def test_custom_update_freq(self):
        td = TerminalDashboard(update_freq=50)
        assert td._update_freq == 50

    def test_on_step_returns_true(self):
        td = TerminalDashboard()
        assert td.on_step(step=1) is True

    def test_no_crash_without_rich(self):
        """TerminalDashboard should not crash if rich is not installed."""
        with patch.dict("sys.modules", {"rich": None, "rich.live": None, "rich.table": None, "rich.panel": None, "rich.text": None, "rich.layout": None, "rich.console": None}):
            td = TerminalDashboard()
            td.on_training_start(total_timesteps=1000)
            td.on_step(step=1, episode_reward=10.0)
            td.on_training_end()

    def test_lifecycle_without_rich(self):
        """Full training lifecycle should work without rich."""
        td = TerminalDashboard()
        td._rich_available = False
        td.on_training_start(total_timesteps=100)
        for i in range(10):
            td.on_step(step=i, episode_reward=float(i))
        td.on_training_end()


class TestHTMLReport:
    """Tests for HTMLReport generation."""

    @pytest.fixture()
    def populated_collector(self) -> MetricsCollector:
        mc = MetricsCollector()
        mc.on_training_start(total_timesteps=100)
        for i in range(20):
            mc.on_step(
                step=i,
                episode_reward=float(i * 5),
                policy_loss=0.5 - i * 0.02,
                value_loss=0.3 - i * 0.01,
                entropy=1.5 - i * 0.05,
                sps=400.0 + i * 10,
                approx_kl=0.01 + i * 0.001,
                clip_fraction=0.1 + i * 0.005,
            )
        mc.on_training_end()
        return mc

    def test_generates_html_file(self, populated_collector: MetricsCollector):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.html")
            report = HTMLReport(populated_collector, output_path=path)
            report.generate()
            assert Path(path).exists()
            content = Path(path).read_text()
            assert len(content) > 0

    def test_html_contains_expected_sections(self, populated_collector: MetricsCollector):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.html")
            report = HTMLReport(populated_collector, output_path=path)
            report.generate()
            content = Path(path).read_text()

            assert "<html" in content
            assert "Reward" in content or "reward" in content
            assert "Loss" in content or "loss" in content
            assert "Entropy" in content or "entropy" in content
            assert "<svg" in content

    def test_html_contains_summary_table(self, populated_collector: MetricsCollector):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.html")
            report = HTMLReport(populated_collector, output_path=path)
            report.generate()
            content = Path(path).read_text()
            assert "<table" in content

    def test_html_is_self_contained(self, populated_collector: MetricsCollector):
        """No external resource references (CDN, scripts, stylesheets)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.html")
            report = HTMLReport(populated_collector, output_path=path)
            report.generate()
            content = Path(path).read_text()
            # Filter out xmlns namespace declarations (these are not external resources)
            filtered = content.replace("http://www.w3.org/2000/svg", "")
            assert "http://" not in filtered
            assert "https://" not in filtered

    def test_svg_charts_present(self, populated_collector: MetricsCollector):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.html")
            report = HTMLReport(populated_collector, output_path=path)
            report.generate()
            content = Path(path).read_text()
            # Should have multiple SVG charts
            assert content.count("<svg") >= 3

    def test_empty_collector_still_generates(self):
        mc = MetricsCollector()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.html")
            report = HTMLReport(mc, output_path=path)
            report.generate()
            assert Path(path).exists()

    def test_default_output_path(self):
        mc = MetricsCollector()
        report = HTMLReport(mc)
        assert report._output_path == "training_report.html"
