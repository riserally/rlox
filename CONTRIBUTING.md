# Contributing to rlox

Thank you for your interest in contributing to rlox!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/riserally/rlox.git
cd rlox

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install maturin numpy gymnasium torch pytest ruff

# Build the Rust extension (always use --release)
maturin develop --release

# Verify
python -c "import rlox; print('rlox ready')"
```

## Running Tests

```bash
# Rust tests (all crates)
cargo test --workspace

# Python tests
python -m pytest tests/ -x

# Quick smoke test
python -m pytest tests/python/test_algorithm_smoke.py -v

# Specific test file
python -m pytest tests/python/test_offline_rl.py -v
```

## Code Style

```bash
# Rust
cargo fmt --all
cargo clippy --workspace

# Python
ruff check python/
ruff format python/
```

## Project Structure

```
crates/
  rlox-core/     # Rust data plane: buffers, envs, GAE, KL, pipeline
  rlox-nn/       # Backend-agnostic NN traits
  rlox-candle/   # Candle backend (inference + hybrid collection)
  rlox-burn/     # Burn backend (alternative)
  rlox-python/   # PyO3 bindings
python/rlox/
  algorithms/    # PPO, SAC, DQN, TD3, A2C, MAPPO, DreamerV3, IMPALA, offline RL, LLM
  offline/       # Offline RL base class + protocols
  exploration/   # Noise strategies + intrinsic rewards
  wrappers/      # VecNormalize and other env wrappers
  callbacks.py   # Training callbacks
  policies.py    # Neural network policies
  trainers.py    # High-level trainers for all algorithms
  runner.py      # Config-driven training (train_from_config)
  dashboard.py   # MetricsCollector, TerminalDashboard, HTMLReport
tests/python/    # Python test suite
docs/            # MkDocs documentation
```

## Pull Request Process

1. Create a feature branch from `main`
2. Write tests first (TDD when possible)
3. Ensure all Rust and Python tests pass
4. Run `cargo fmt` and `ruff format`
5. Update documentation if adding new features
6. Keep PRs focused: one feature or fix per PR

## Adding a New Algorithm

1. Create `python/rlox/algorithms/your_algo.py`
2. Implement using existing primitives (buffers, GAE, etc.)
3. Add tests in `tests/python/test_your_algo.py`
4. Add to `docs/examples.md` and `docs/python-guide.md`
5. Update `docs/index.md` algorithm list

## Adding a New Rust Primitive

1. Implement in the appropriate `crates/rlox-core/src/` module
2. Add unit tests in the same file
3. Create PyO3 bindings in `crates/rlox-python/src/`
4. Register in `crates/rlox-python/src/lib.rs`
5. Export in `python/rlox/__init__.py`
6. Add Python integration tests
