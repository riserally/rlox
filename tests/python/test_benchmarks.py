"""Import-level tests for benchmark modules.

These verify that each benchmark file is importable and exposes
the expected public functions, without actually running the
(expensive) benchmarks.
"""

import importlib
import sys
import os

import pytest

# Ensure benchmarks directory is on sys.path
BENCH_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmarks",
)
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)


# ---------------------------------------------------------------------------
# bench_distributed
# ---------------------------------------------------------------------------

class TestBenchDistributedImports:
    """bench_distributed.py is importable and declares expected functions."""

    def test_module_importable(self):
        mod = importlib.import_module("bench_distributed")
        assert mod is not None

    @pytest.mark.parametrize("fn_name", [
        "bench_remote_env_pool_mock",
        "bench_pipeline_throughput",
        "bench_multi_env_scaling",
        "run_all",
    ])
    def test_function_exists(self, fn_name: str):
        mod = importlib.import_module("bench_distributed")
        assert hasattr(mod, fn_name), f"Missing function: {fn_name}"
        assert callable(getattr(mod, fn_name))


# ---------------------------------------------------------------------------
# bench_algorithms
# ---------------------------------------------------------------------------

class TestBenchAlgorithmsImports:
    """bench_algorithms.py is importable and declares expected functions."""

    def test_module_importable(self):
        mod = importlib.import_module("bench_algorithms")
        assert mod is not None

    @pytest.mark.parametrize("fn_name", [
        "bench_ppo_cartpole",
        "bench_a2c_cartpole",
        "bench_dqn_cartpole",
        "bench_impala_cartpole",
        "run_all",
    ])
    def test_function_exists(self, fn_name: str):
        mod = importlib.import_module("bench_algorithms")
        assert hasattr(mod, fn_name), f"Missing function: {fn_name}"
        assert callable(getattr(mod, fn_name))


# ---------------------------------------------------------------------------
# bench_mmap_buffer
# ---------------------------------------------------------------------------

class TestBenchMmapBufferImports:
    """bench_mmap_buffer.py is importable and declares expected functions."""

    def test_module_importable(self):
        mod = importlib.import_module("bench_mmap_buffer")
        assert mod is not None

    @pytest.mark.parametrize("fn_name", [
        "bench_push_throughput",
        "bench_sample_throughput",
        "bench_memory_usage",
        "run_all",
    ])
    def test_function_exists(self, fn_name: str):
        mod = importlib.import_module("bench_mmap_buffer")
        assert hasattr(mod, fn_name), f"Missing function: {fn_name}"
        assert callable(getattr(mod, fn_name))
