#!/usr/bin/env python3
"""
Master benchmark runner for rlox.

Runs all available benchmarks and produces a consolidated report.
Benchmarks that depend on unimplemented features are skipped gracefully.

Usage:
    python benchmarks/run_all.py [--output-dir benchmark_results] [--quick]
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from conftest import system_info, write_report


def run_env_stepping(output_dir: str, quick: bool = False):
    """Run environment stepping benchmarks (3.1)."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3.1: Environment Stepping")
    print("=" * 70)
    from bench_env_stepping import run_all
    run_all(output_dir)


def check_phase_available(phase: str) -> bool:
    """Check if a phase's rlox components are importable."""
    checks = {
        "phase2": lambda: __import__("rlox").ExperienceTable,
        "phase3": lambda: __import__("rlox").compute_gae,
        "phase4": lambda: __import__("rlox").compute_group_advantages,
    }
    try:
        checks[phase]()
        return True
    except (ImportError, AttributeError):
        return False


def main():
    parser = argparse.ArgumentParser(description="rlox benchmark suite")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--quick", action="store_true",
                       help="Run reduced iterations for faster feedback")
    parser.add_argument("--category", choices=[
        "all", "env_stepping", "buffer_ops", "gae", "llm",
    ], default="all")
    args = parser.parse_args()

    print("=" * 70)
    print("rlox Benchmark Suite")
    print("=" * 70)

    info = system_info()
    print(f"Platform: {info['platform']}")
    print(f"Python:   {info['python_version'].split()[0]}")
    print(f"NumPy:    {info['numpy_version']}")
    print(f"CPU:      {info.get('cpu', 'unknown')} ({info['cpu_count']} cores)")
    if info.get("torch_version"):
        print(f"PyTorch:  {info['torch_version']}")
    if info.get("cuda_available"):
        print(f"GPU:      {info.get('gpu', 'unknown')}")
    print(f"rlox:     {'available' if info.get('rlox_available') else 'NOT FOUND'}")

    # Phase availability
    print("\nPhase availability:")
    for phase, desc in [
        ("phase2", "Experience Storage (buffer, VarLenStore)"),
        ("phase3", "Training Orchestrator (GAE, batch assembly)"),
        ("phase4", "LLM Post-Training (GRPO, DPO, token KL)"),
    ]:
        available = check_phase_available(phase)
        status = "READY" if available else "not yet implemented"
        print(f"  {desc}: {status}")

    # Run benchmarks
    if args.category in ("all", "env_stepping"):
        try:
            run_env_stepping(args.output_dir, args.quick)
        except Exception as e:
            print(f"\nENV STEPPING BENCHMARK FAILED: {e}")

    if args.category in ("all", "buffer_ops"):
        if check_phase_available("phase2"):
            print("\n[TODO] Buffer benchmarks — run when Phase 2 is implemented")
        else:
            print("\n[SKIP] Buffer benchmarks — Phase 2 not implemented")

    if args.category in ("all", "gae"):
        if check_phase_available("phase3"):
            print("\n[TODO] GAE benchmarks — run when Phase 3 is implemented")
        else:
            print("\n[SKIP] GAE benchmarks — Phase 3 not implemented")

    if args.category in ("all", "llm"):
        if check_phase_available("phase4"):
            print("\n[TODO] LLM benchmarks — run when Phase 4 is implemented")
        else:
            print("\n[SKIP] LLM benchmarks — Phase 4 not implemented")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
