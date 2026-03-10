# Benchmark Setup & Methodology

## Hardware

| Component | Specification |
|-----------|--------------|
| CPU | Apple M4 (arm64), 14 cores |
| OS | macOS 26.2 (Darwin 25.2.0) |
| RAM | Unified memory architecture |

## Software Versions

| Package | Version |
|---------|---------|
| Python | 3.12.7 |
| NumPy | 2.4.2 |
| PyTorch | 2.10.0 (CPU only, no CUDA) |
| TorchRL | 0.11.1 |
| Stable-Baselines3 | 2.7.1 |
| tensordict | 0.11.0 |
| Gymnasium | 1.1.1 |
| rlox | 0.1.0 (Rust via PyO3 0.23.5, maturin --release) |

## Statistical Methodology

### Timing
- **Clock**: `time.perf_counter_ns()` — monotonic, nanosecond resolution
- **No GC manipulation**: We do not disable garbage collection. GC pauses are part of the real performance profile.
- **CPU**: All benchmarks run on CPU. No GPU benchmarks.

### Warmup & Repetitions

| Benchmark Type | Warmup | Measurement Reps | Rationale |
|---------------|--------|-------------------|-----------|
| Single-step latency | 200 | 1000 | Amortize PyO3/JIT init |
| Vectorized env stepping | 5 rounds | 50 (20 for subprocess) | Amortize Rayon/process warmup |
| Buffer push | 2 rounds | 10 | Amortize initial allocations |
| Buffer sample | 10 | 100 | Amortize cache priming |
| GAE computation | 10 | 100 | Amortize numpy/torch alloc |
| End-to-end rollout | 1 full rollout | 10 | Amortize everything |

### Summary Statistics

- **Primary metric**: **Median** (robust to GC pauses and context switches)
- **Dispersion**: IQR (p25–p75), p99 (tail latency)
- **Comparison**: Bootstrap 95% confidence interval on the speedup ratio
  - 10,000 bootstrap resamples
  - Resample medians of both distributions, compute ratio
  - Report [p2.5, p97.5] of the ratio distribution
- **Significance**: CI lower bound > 1.0 means rlox is statistically faster

### Fairness Constraints

1. **Same env dynamics**: rlox has a native CartPole; TorchRL and SB3 use Gymnasium's CartPole-v1. Both implement identical physics (validated via correctness tests).
2. **Same observation shapes**: CartPole = `(4,)` float32 across all frameworks.
3. **Same batch sizes**: Buffer sample sizes and GAE trajectory lengths are identical.
4. **Deterministic seeding**: `seed=42` everywhere. Buffer sampling uses deterministic ChaCha8 RNG.
5. **CPU only**: TorchRL can offload to GPU; we use `device="cpu"` for fair comparison.
6. **Exclude one-time setup**: Env creation and buffer allocation are not included in timing.
7. **Idiomatic API**: Each framework is used via its recommended API (TorchRL with TensorDict, SB3 with its ReplayBuffer API).

### GIL Considerations

| Framework | GIL Behavior |
|-----------|-------------|
| rlox | Releases GIL during all Rust computation (Rayon threads run freely) |
| TorchRL | Releases GIL in C++ PyTorch kernels; TensorDict operations hold GIL |
| SB3 | Holds GIL throughout (pure Python + NumPy) |

## Reproducing

```bash
# Install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy gymnasium stable-baselines3 torchrl

# Build rlox
maturin develop --release

# Run full suite
python benchmarks/run_all.py

# Or individual benchmarks
python benchmarks/bench_buffer_ops.py
python benchmarks/bench_gae.py
python benchmarks/bench_llm_ops.py
python benchmarks/bench_e2e_rollout.py
python benchmarks/bench_env_stepping.py
```

Raw JSON results are written to `benchmark_results/` (gitignored). Each file contains full timing arrays, system info, and comparison data.
