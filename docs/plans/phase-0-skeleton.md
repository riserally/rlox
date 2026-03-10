# Phase 0 — Project Skeleton

**Status: COMPLETE**
**Duration: Week 1**
**PRD Features: Foundation (no direct PRD mapping — infrastructure prerequisite)**

---

## Objective

Validate the Rust + Python toolchain end-to-end. A Rust function must be callable from Python via PyO3/maturin. This is the "hello world" that proves the architecture works before investing in domain logic.

## Reasoning

The Polars architecture pattern (Rust core + Python bindings) requires three pieces to work together: a pure Rust library, a PyO3 extension module, and a Python package that re-exports it. If any link in this chain fails (maturin config, PyO3 version compatibility, numpy interop), we discover it immediately — not after writing 5000 lines of RL logic.

The **two-crate split** (`rlox-core` vs `rlox-python`) is a deliberate architectural decision:
- `rlox-core` compiles and tests WITHOUT Python installed (fast CI, future WASM target)
- `rlox-python` is a thin translation layer (PyO3 `#[pyclass]` wrappers)
- This mirrors Polars (`polars-core` / `py-polars`) and Pydantic v2 (`pydantic-core`)

## TDD Test Specifications

### Rust tests (write first, should fail until implementation)

```rust
// crates/rlox-core/src/lib.rs
#[test]
fn core_crate_compiles() {
    // Simply importing from rlox_core should work
    assert!(true);
}
```

### Python tests (write first, should fail until maturin develop)

```python
# tests/python/test_skeleton.py
def test_import_rlox():
    """The rlox package must be importable."""
    import rlox
    assert hasattr(rlox, '__name__')

def test_import_core_module():
    """The compiled Rust extension must be importable."""
    from rlox import _rlox_core
    assert _rlox_core is not None
```

### Build verification (must pass)

```bash
cargo build --workspace         # Rust compilation
cargo test --workspace          # Rust unit tests
maturin develop                 # Build Python extension
python -c "import rlox"         # Python import works
```

## Implementation Steps

1. Create `Cargo.toml` workspace root with members: `rlox-core`, `rlox-python`, `rlox-bench`
2. Create `crates/rlox-core/` with minimal `lib.rs` (module declarations only)
3. Create `crates/rlox-python/` with `#[pymodule]` entry point
4. Create `crates/rlox-bench/` with criterion setup
5. Create `pyproject.toml` with maturin build backend
6. Create `python/rlox/__init__.py` with re-exports from `_rlox_core`
7. Create `python/rlox/_rlox_core.pyi` type stubs

## Files Created

| File | Purpose |
|------|---------|
| `Cargo.toml` | Workspace root, 3 member crates |
| `pyproject.toml` | maturin build backend, Python >=3.9, numpy >=1.21 |
| `python/rlox/__init__.py` | Re-exports CartPole, VecEnv, GymEnv |
| `python/rlox/_rlox_core.pyi` | Type stubs for IDE support |
| `crates/rlox-core/Cargo.toml` | rayon, rand, rand_chacha, thiserror, serde |
| `crates/rlox-core/src/lib.rs` | Module declarations: env, error, seed |
| `crates/rlox-python/Cargo.toml` | pyo3 0.23, numpy 0.23, depends on rlox-core |
| `crates/rlox-python/src/lib.rs` | `#[pymodule]` registering 3 classes |
| `crates/rlox-bench/Cargo.toml` | criterion 0.5 |
| `crates/rlox-bench/src/lib.rs` | Empty marker |

## Acceptance Criteria

- [x] `cargo build --workspace` compiles without errors
- [x] `cargo test --workspace` passes all tests
- [x] `maturin develop` builds Python extension
- [x] `python -c "from rlox import CartPole"` works
- [x] `cargo bench --package rlox-bench` runs
- [x] `rlox-core` has zero PyO3/Python dependencies
- [x] Type stubs provide IDE autocompletion
