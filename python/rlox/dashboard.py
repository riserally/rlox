"""Diagnostics dashboard: metrics collection, terminal display, HTML reports.

Provides three components for training observability:

- :class:`MetricsCollector` — callback that accumulates time-series metrics
- :class:`TerminalDashboard` — rich-based live terminal display (optional dep)
- :class:`HTMLReport` — generates a self-contained HTML report with SVG charts
"""

from __future__ import annotations

import html
import time
from typing import Any

from rlox.callbacks import Callback

# Metric field names tracked by MetricsCollector
_METRIC_FIELDS: tuple[str, ...] = (
    "step",
    "episode_reward",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "learning_rate",
    "sps",
    "explained_variance",
    "clip_fraction",
)


class MetricsCollector(Callback):
    """Collects time-series training metrics for visualization.

    Stores per-step values for common RL training metrics and provides
    summary statistics at the end of training.

    Parameters
    ----------
    None

    Examples
    --------
    >>> mc = MetricsCollector()
    >>> mc.on_training_start(total_timesteps=50_000)
    >>> mc.on_step(step=0, episode_reward=10.0, policy_loss=0.5)
    True
    >>> mc.summary()["total_steps"]
    1
    """

    def __init__(self) -> None:
        super().__init__()
        self._data: dict[str, list[Any]] = {field: [] for field in _METRIC_FIELDS}
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._total_timesteps: int = 0

    def on_training_start(self, **kwargs: Any) -> None:
        self._start_time = time.monotonic()
        self._total_timesteps = kwargs.get("total_timesteps", 0)

    def on_step(self, **kwargs: Any) -> bool:
        for field in _METRIC_FIELDS:
            self._data[field].append(kwargs.get(field))
        return True

    def on_training_end(self, **kwargs: Any) -> None:
        self._end_time = time.monotonic()

    def get_dataframe(self) -> dict[str, list[Any]]:
        """Return metrics as a dict of lists (independent copy)."""
        return {k: list(v) for k, v in self._data.items()}

    def summary(self) -> dict[str, Any]:
        """Return a summary dict with final values, peak reward, training time."""
        steps = self._data["step"]
        rewards = [r for r in self._data["episode_reward"] if r is not None]
        sps_values = [s for s in self._data["sps"] if s is not None]

        training_time = 0.0
        if self._start_time is not None and self._end_time is not None:
            training_time = self._end_time - self._start_time

        def _last_non_none(field: str) -> Any:
            for val in reversed(self._data[field]):
                if val is not None:
                    return val
            return None

        return {
            "total_steps": len(steps),
            "total_timesteps": self._total_timesteps,
            "training_time": training_time,
            "peak_reward": max(rewards) if rewards else None,
            "final_reward": rewards[-1] if rewards else None,
            "final_policy_loss": _last_non_none("policy_loss"),
            "final_value_loss": _last_non_none("value_loss"),
            "final_entropy": _last_non_none("entropy"),
            "mean_sps": sum(sps_values) / len(sps_values) if sps_values else None,
        }


class TerminalDashboard(Callback):
    """Rich-based live terminal display of training progress.

    Uses the ``rich`` library to render a live panel showing current
    training metrics. Falls back gracefully to a no-op if rich is not
    installed.

    Parameters
    ----------
    update_freq : int
        Update the display every ``update_freq`` steps (default 100).
    """

    def __init__(self, update_freq: int = 100) -> None:
        super().__init__()
        self._update_freq = update_freq
        self._live: Any = None
        self._total_timesteps = 0
        self._start_time: float | None = None
        self._latest: dict[str, Any] = {}
        self._peak_reward = float("-inf")

        self._rich_available = False
        try:
            import rich  # noqa: F401
            self._rich_available = True
        except ImportError:
            pass

    def on_training_start(self, **kwargs: Any) -> None:
        self._total_timesteps = kwargs.get("total_timesteps", 0)
        self._start_time = time.monotonic()

        if not self._rich_available:
            return

        try:
            from rich.console import Console
            from rich.live import Live

            self._console = Console()
            self._live = Live(console=self._console, refresh_per_second=4)
            self._live.start()
        except Exception:
            self._live = None

    def on_step(self, **kwargs: Any) -> bool:
        step = kwargs.get("step", 0)
        reward = kwargs.get("episode_reward")
        if reward is not None and reward > self._peak_reward:
            self._peak_reward = reward

        self._latest = {
            "step": step,
            "reward": reward,
            "peak_reward": self._peak_reward if self._peak_reward > float("-inf") else None,
            "policy_loss": kwargs.get("policy_loss"),
            "value_loss": kwargs.get("value_loss"),
            "entropy": kwargs.get("entropy"),
            "sps": kwargs.get("sps"),
        }

        if self._live is not None and step % self._update_freq == 0:
            self._render()

        return True

    def on_training_end(self, **kwargs: Any) -> None:
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

    def _render(self) -> None:
        """Render the dashboard panel."""
        if self._live is None:
            return

        try:
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text

            elapsed = time.monotonic() - self._start_time if self._start_time else 0.0

            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Metric", style="bold cyan")
            table.add_column("Value", style="white")

            step = self._latest.get("step", 0)
            total = self._total_timesteps
            progress = f"{step:,} / {total:,}" if total else f"{step:,}"
            table.add_row("Step", progress)

            reward = self._latest.get("reward")
            table.add_row("Reward", f"{reward:.2f}" if reward is not None else "N/A")

            peak = self._latest.get("peak_reward")
            table.add_row("Peak Reward", f"{peak:.2f}" if peak is not None else "N/A")

            for label, key in [
                ("Policy Loss", "policy_loss"),
                ("Value Loss", "value_loss"),
                ("Entropy", "entropy"),
                ("SPS", "sps"),
            ]:
                val = self._latest.get(key)
                table.add_row(label, f"{val:.4f}" if val is not None else "N/A")

            minutes, seconds = divmod(int(elapsed), 60)
            hours, minutes = divmod(minutes, 60)
            table.add_row("Elapsed", f"{hours:02d}:{minutes:02d}:{seconds:02d}")

            panel = Panel(table, title="[bold]rlox Training Dashboard[/bold]", border_style="blue")
            self._live.update(panel)
        except Exception:
            pass


class HTMLReport:
    """Generate a standalone HTML report with embedded SVG charts.

    Parameters
    ----------
    metrics : MetricsCollector
        Populated metrics collector from a training run.
    output_path : str
        Path for the output HTML file (default "training_report.html").
    """

    def __init__(self, metrics: MetricsCollector, output_path: str = "training_report.html") -> None:
        self._metrics = metrics
        self._output_path = output_path

    def generate(self) -> None:
        """Write a self-contained HTML report to disk."""
        data = self._metrics.get_dataframe()
        summary = self._metrics.summary()

        sections: list[str] = []
        sections.append(self._summary_table(summary))

        # Reward curve
        rewards = data.get("episode_reward", [])
        steps = data.get("step", [])
        sections.append(
            self._chart_section(
                "Episode Reward",
                steps,
                rewards,
                color="#2196F3",
            )
        )

        # Loss curves (overlay policy and value)
        sections.append(
            self._dual_chart_section(
                "Training Losses",
                steps,
                data.get("policy_loss", []),
                "Policy Loss",
                "#F44336",
                data.get("value_loss", []),
                "Value Loss",
                "#FF9800",
            )
        )

        # Entropy
        sections.append(
            self._chart_section(
                "Entropy",
                steps,
                data.get("entropy", []),
                color="#4CAF50",
            )
        )

        # SPS
        sections.append(
            self._chart_section(
                "Steps Per Second",
                steps,
                data.get("sps", []),
                color="#9C27B0",
            )
        )

        body = "\n".join(sections)

        page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>rlox Training Report</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 900px;
    margin: 40px auto;
    padding: 0 20px;
    background: #fafafa;
    color: #333;
  }}
  h1 {{ color: #1a237e; border-bottom: 2px solid #1a237e; padding-bottom: 8px; }}
  h2 {{ color: #37474f; margin-top: 32px; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 16px 0;
  }}
  th, td {{
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: left;
  }}
  th {{ background: #e8eaf6; font-weight: 600; }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
  .chart-container {{
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 16px;
    margin: 16px 0;
  }}
  .legend {{
    display: flex;
    gap: 16px;
    margin-top: 8px;
    font-size: 14px;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 4px;
  }}
  .legend-swatch {{
    width: 14px;
    height: 14px;
    border-radius: 2px;
    display: inline-block;
  }}
</style>
</head>
<body>
<h1>rlox Training Report</h1>
{body}
</body>
</html>"""

        with open(self._output_path, "w") as f:
            f.write(page)

    # -- Internal helpers --------------------------------------------------------

    def _summary_table(self, summary: dict[str, Any]) -> str:
        rows: list[str] = []
        labels = {
            "total_steps": "Total Steps",
            "total_timesteps": "Total Timesteps",
            "training_time": "Training Time (s)",
            "peak_reward": "Peak Reward",
            "final_reward": "Final Reward",
            "final_policy_loss": "Final Policy Loss",
            "final_value_loss": "Final Value Loss",
            "final_entropy": "Final Entropy",
            "mean_sps": "Mean SPS",
        }
        for key, label in labels.items():
            val = summary.get(key)
            if val is None:
                formatted = "N/A"
            elif isinstance(val, float):
                formatted = f"{val:.4f}"
            else:
                formatted = str(val)
            rows.append(f"  <tr><td>{html.escape(label)}</td><td>{html.escape(formatted)}</td></tr>")

        return f"""<h2>Summary</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
{"".join(rows)}
</table>"""

    def _chart_section(
        self,
        title: str,
        x_data: list[Any],
        y_data: list[Any],
        color: str,
        width: int = 700,
        height: int = 220,
    ) -> str:
        svg = _svg_line_chart(x_data, y_data, width=width, height=height, color=color, title=title)
        return f"""<h2>{html.escape(title)}</h2>
<div class="chart-container">
{svg}
</div>"""

    def _dual_chart_section(
        self,
        title: str,
        x_data: list[Any],
        y1_data: list[Any],
        y1_label: str,
        y1_color: str,
        y2_data: list[Any],
        y2_label: str,
        y2_color: str,
        width: int = 700,
        height: int = 220,
    ) -> str:
        svg = _svg_dual_line_chart(
            x_data,
            y1_data,
            y2_data,
            width=width,
            height=height,
            color1=y1_color,
            color2=y2_color,
            title=title,
        )
        legend = (
            f'<div class="legend">'
            f'<span class="legend-item"><span class="legend-swatch" style="background:{y1_color}"></span> {html.escape(y1_label)}</span>'
            f'<span class="legend-item"><span class="legend-swatch" style="background:{y2_color}"></span> {html.escape(y2_label)}</span>'
            f"</div>"
        )
        return f"""<h2>{html.escape(title)}</h2>
<div class="chart-container">
{svg}
{legend}
</div>"""


# -- SVG chart helpers (pure Python, no external deps) -------------------------

_PADDING_LEFT = 60
_PADDING_RIGHT = 20
_PADDING_TOP = 10
_PADDING_BOTTOM = 30


def _filter_pairs(xs: list[Any], ys: list[Any]) -> tuple[list[float], list[float]]:
    """Return only pairs where both x and y are not None."""
    fxs: list[float] = []
    fys: list[float] = []
    for x, y in zip(xs, ys):
        if x is not None and y is not None:
            fxs.append(float(x))
            fys.append(float(y))
    return fxs, fys


def _svg_line_chart(
    x_data: list[Any],
    y_data: list[Any],
    width: int = 700,
    height: int = 220,
    color: str = "#2196F3",
    title: str = "",
) -> str:
    """Generate an inline SVG line chart."""
    xs, ys = _filter_pairs(x_data, y_data)

    if len(xs) < 2:
        return (
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
            f'fill="#999" font-size="14">No data</text></svg>'
        )

    plot_w = width - _PADDING_LEFT - _PADDING_RIGHT
    plot_h = height - _PADDING_TOP - _PADDING_BOTTOM

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Avoid zero-range
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1

    def tx(v: float) -> float:
        return _PADDING_LEFT + (v - x_min) / (x_max - x_min) * plot_w

    def ty(v: float) -> float:
        return _PADDING_TOP + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    # Build polyline points
    points = " ".join(f"{tx(x):.1f},{ty(y):.1f}" for x, y in zip(xs, ys))

    # Y-axis labels
    y_labels = ""
    n_ticks = 5
    for i in range(n_ticks + 1):
        val = y_min + (y_max - y_min) * i / n_ticks
        py = ty(val)
        label = f"{val:.2g}"
        y_labels += (
            f'<text x="{_PADDING_LEFT - 6}" y="{py + 4}" '
            f'text-anchor="end" fill="#666" font-size="10">{label}</text>'
            f'<line x1="{_PADDING_LEFT}" y1="{py}" x2="{width - _PADDING_RIGHT}" '
            f'y2="{py}" stroke="#eee" stroke-width="1"/>'
        )

    # X-axis labels
    x_labels = ""
    for i in range(min(6, len(xs))):
        idx = int(i * (len(xs) - 1) / max(5, 1))
        px = tx(xs[idx])
        label = f"{xs[idx]:.0f}"
        x_labels += (
            f'<text x="{px}" y="{height - 4}" '
            f'text-anchor="middle" fill="#666" font-size="10">{label}</text>'
        )

    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'{y_labels}{x_labels}'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round" points="{points}"/>'
        f"</svg>"
    )


def _svg_dual_line_chart(
    x_data: list[Any],
    y1_data: list[Any],
    y2_data: list[Any],
    width: int = 700,
    height: int = 220,
    color1: str = "#F44336",
    color2: str = "#FF9800",
    title: str = "",
) -> str:
    """Generate an inline SVG with two overlaid line series."""
    xs1, ys1 = _filter_pairs(x_data, y1_data)
    xs2, ys2 = _filter_pairs(x_data, y2_data)

    all_xs = xs1 + xs2
    all_ys = ys1 + ys2

    if len(all_xs) < 2 or len(all_ys) < 2:
        return (
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
            f'fill="#999" font-size="14">No data</text></svg>'
        )

    plot_w = width - _PADDING_LEFT - _PADDING_RIGHT
    plot_h = height - _PADDING_TOP - _PADDING_BOTTOM

    x_min, x_max = min(all_xs), max(all_xs)
    y_min, y_max = min(all_ys), max(all_ys)

    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1

    def tx(v: float) -> float:
        return _PADDING_LEFT + (v - x_min) / (x_max - x_min) * plot_w

    def ty(v: float) -> float:
        return _PADDING_TOP + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    points1 = " ".join(f"{tx(x):.1f},{ty(y):.1f}" for x, y in zip(xs1, ys1))
    points2 = " ".join(f"{tx(x):.1f},{ty(y):.1f}" for x, y in zip(xs2, ys2))

    # Y-axis labels
    y_labels = ""
    n_ticks = 5
    for i in range(n_ticks + 1):
        val = y_min + (y_max - y_min) * i / n_ticks
        py = ty(val)
        label = f"{val:.2g}"
        y_labels += (
            f'<text x="{_PADDING_LEFT - 6}" y="{py + 4}" '
            f'text-anchor="end" fill="#666" font-size="10">{label}</text>'
            f'<line x1="{_PADDING_LEFT}" y1="{py}" x2="{width - _PADDING_RIGHT}" '
            f'y2="{py}" stroke="#eee" stroke-width="1"/>'
        )

    lines = ""
    if points1:
        lines += (
            f'<polyline fill="none" stroke="{color1}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round" points="{points1}"/>'
        )
    if points2:
        lines += (
            f'<polyline fill="none" stroke="{color2}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round" points="{points2}"/>'
        )

    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'{y_labels}{lines}'
        f"</svg>"
    )
