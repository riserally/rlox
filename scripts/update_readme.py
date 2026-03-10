#!/usr/bin/env python3
"""Read benchmark_results/*.json and patch the README.md benchmark table."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmark_results"
README = ROOT / "README.md"

MARKER_START = "<!-- BENCH:START -->"
MARKER_END = "<!-- BENCH:END -->"


def load_latest_results() -> dict | None:
    """Find and load the most recent benchmark JSON."""
    jsons = sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not jsons:
        return None
    with open(jsons[-1]) as f:
        return json.load(f)


def fmt_ns(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.0f} ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.1f} us"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    return f"{ns / 1_000_000_000:.2f} s"


def build_table(data: dict) -> str:
    lines = []
    lines.append("| Benchmark | rlox | Baseline | Speedup | Significant |")
    lines.append("|-----------|------|----------|---------|-------------|")

    comparisons = data.get("comparisons", [])
    for c in comparisons:
        name = c["benchmark"]
        rlox_ns = c["rlox_median_ns"]
        base_ns = c["baseline_median_ns"]
        speedup = c["speedup"]
        sig = c.get("significant", False)
        sig_str = "Yes" if sig else "No"
        baseline_name = c.get("baseline_framework", "baseline")
        # Prettify framework names
        pretty = baseline_name.replace("_", " ").title()
        lines.append(
            f"| {name} | {fmt_ns(rlox_ns)} | {fmt_ns(base_ns)} ({pretty}) | **{speedup:.1f}x** | {sig_str} |"
        )

    return "\n".join(lines)


def update_readme(table: str):
    text = README.read_text()
    pattern = re.compile(
        rf"{re.escape(MARKER_START)}.*?{re.escape(MARKER_END)}",
        re.DOTALL,
    )
    replacement = f"{MARKER_START}\n{table}\n{MARKER_END}"

    if pattern.search(text):
        text = pattern.sub(replacement, text)
    else:
        print("Warning: benchmark markers not found in README — skipping update")
        return

    README.write_text(text)
    print(f"Updated {README}")


def main():
    data = load_latest_results()
    if data is None:
        print("No benchmark results found — run benchmarks first.")
        return
    table = build_table(data)
    update_readme(table)


if __name__ == "__main__":
    main()
