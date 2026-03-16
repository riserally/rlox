"""API 1.0 stability tests.

Verify that all public symbols are importable, that the version is a string,
and that key interfaces (configs, trainers, callbacks) expose the expected
methods.
"""

from __future__ import annotations

import importlib

import pytest

import rlox


# =========================================================================
# Public API surface
# =========================================================================


class TestAllPublicSymbolsInAll:
    """Every symbol listed in rlox.__all__ must actually be importable."""

    def test_all_public_symbols_in_all(self):
        missing = []
        for name in rlox.__all__:
            obj = getattr(rlox, name, None)
            if obj is None:
                missing.append(name)
        assert missing == [], f"Symbols in __all__ but not importable: {missing}"

    def test_no_duplicates_in_all(self):
        seen: set[str] = set()
        dupes: list[str] = []
        for name in rlox.__all__:
            if name in seen:
                dupes.append(name)
            seen.add(name)
        assert dupes == [], f"Duplicate entries in __all__: {dupes}"


class TestVersionIsString:
    def test_version_is_string(self):
        assert isinstance(rlox.__version__, str)

    def test_version_is_semver_like(self):
        parts = rlox.__version__.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()


# =========================================================================
# Rust primitives
# =========================================================================


class TestRustPrimitivesImportable:
    """All Rust bindings should import correctly from rlox."""

    RUST_SYMBOLS = [
        "CartPole",
        "VecEnv",
        "GymEnv",
        "ExperienceTable",
        "ReplayBuffer",
        "PrioritizedReplayBuffer",
        "VarLenStore",
        "compute_gae",
        "compute_vtrace",
        "compute_group_advantages",
        "compute_batch_group_advantages",
        "compute_token_kl",
        "compute_token_kl_schulman",
        "DPOPair",
        "RunningStats",
        "pack_sequences",
        "ActorCritic",
    ]

    @pytest.mark.parametrize("symbol", RUST_SYMBOLS)
    def test_rust_primitives_importable(self, symbol: str):
        obj = getattr(rlox, symbol, None)
        assert obj is not None, f"Rust primitive '{symbol}' not importable from rlox"


# =========================================================================
# Python layer
# =========================================================================


class TestPythonLayerImportable:
    """All Python layer classes should import correctly from rlox."""

    PYTHON_SYMBOLS = [
        "RolloutBatch",
        "RolloutCollector",
        "GymVecEnv",
        "PPOLoss",
        "ContinuousPolicy",
        "DiscretePolicy",
        "PPOConfig",
        "SACConfig",
        "DQNConfig",
        "Callback",
        "CallbackList",
        "EvalCallback",
        "EarlyStoppingCallback",
        "CheckpointCallback",
        "LoggerCallback",
        "WandbLogger",
        "TensorBoardLogger",
        "interquartile_mean",
        "performance_profiles",
        "stratified_bootstrap_ci",
        "aggregate_metrics",
        "probability_of_improvement",
        "TrainingDiagnostics",
        "Checkpoint",
    ]

    @pytest.mark.parametrize("symbol", PYTHON_SYMBOLS)
    def test_python_layer_importable(self, symbol: str):
        obj = getattr(rlox, symbol, None)
        assert obj is not None, f"Python symbol '{symbol}' not importable from rlox"


# =========================================================================
# Algorithms
# =========================================================================


class TestAlgorithmsImportable:
    """All algorithm classes should import from rlox.algorithms."""

    ALGORITHM_NAMES = [
        "PPO",
        "A2C",
        "SAC",
        "TD3",
        "DQN",
        "GRPO",
        "DPO",
        "OnlineDPO",
        "BestOfN",
        "MAPPO",
        "DreamerV3",
        "IMPALA",
    ]

    @pytest.mark.parametrize("name", ALGORITHM_NAMES)
    def test_algorithms_importable(self, name: str):
        mod = importlib.import_module("rlox.algorithms")
        obj = getattr(mod, name, None)
        assert obj is not None, f"Algorithm '{name}' not importable from rlox.algorithms"


# =========================================================================
# Configs
# =========================================================================


class TestConfigsHaveFromDictAndToDict:
    """PPOConfig, SACConfig, DQNConfig must have from_dict, to_dict, merge,
    from_yaml, and to_yaml."""

    CONFIG_CLASSES = [
        rlox.PPOConfig,
        rlox.SACConfig,
        rlox.DQNConfig,
    ]

    REQUIRED_METHODS = ["from_dict", "to_dict", "merge", "from_yaml", "to_yaml"]

    @pytest.mark.parametrize("cls", CONFIG_CLASSES, ids=lambda c: c.__name__)
    @pytest.mark.parametrize("method", REQUIRED_METHODS)
    def test_configs_have_from_dict_and_to_dict(self, cls: type, method: str):
        assert hasattr(cls, method), f"{cls.__name__} missing method: {method}"
        assert callable(getattr(cls, method))

    @pytest.mark.parametrize("cls", CONFIG_CLASSES, ids=lambda c: c.__name__)
    def test_config_roundtrip(self, cls: type):
        """from_dict(to_dict()) should produce an equivalent config."""
        original = cls()
        d = original.to_dict()
        restored = cls.from_dict(d)
        assert original.to_dict() == restored.to_dict()


# =========================================================================
# Trainers
# =========================================================================


class TestTrainersHaveTrainMethod:
    """PPOTrainer, SACTrainer, DQNTrainer must all have train()."""

    def _get_trainer_class(self, name: str) -> type:
        mod = importlib.import_module("rlox.trainers")
        return getattr(mod, name)

    TRAINER_NAMES = ["PPOTrainer", "SACTrainer", "DQNTrainer"]

    @pytest.mark.parametrize("name", TRAINER_NAMES)
    def test_trainers_have_train_method(self, name: str):
        cls = self._get_trainer_class(name)
        assert hasattr(cls, "train"), f"{name} missing train() method"
        assert callable(getattr(cls, "train"))

    @pytest.mark.parametrize("name", TRAINER_NAMES)
    def test_trainers_importable(self, name: str):
        cls = self._get_trainer_class(name)
        assert cls is not None


# =========================================================================
# Callback protocol
# =========================================================================


class TestCallbackProtocol:
    """All callback classes must have on_training_start, on_step,
    on_training_end."""

    CALLBACK_CLASSES = [
        rlox.Callback,
        rlox.EvalCallback,
        rlox.EarlyStoppingCallback,
        rlox.CheckpointCallback,
        rlox.TrainingDiagnostics,
    ]

    REQUIRED_METHODS = ["on_training_start", "on_step", "on_training_end"]

    @pytest.mark.parametrize(
        "cls", CALLBACK_CLASSES, ids=lambda c: c.__name__
    )
    @pytest.mark.parametrize("method", REQUIRED_METHODS)
    def test_callback_protocol(self, cls: type, method: str):
        assert hasattr(cls, method), f"{cls.__name__} missing method: {method}"
        assert callable(getattr(cls, method))

    @pytest.mark.parametrize(
        "cls", CALLBACK_CLASSES, ids=lambda c: c.__name__
    )
    def test_callback_on_step_returns_bool(self, cls: type):
        """on_step() should return a bool (True = continue)."""
        instance = cls()
        result = instance.on_step()
        assert isinstance(result, bool)


# =========================================================================
# RemoteEnvPool
# =========================================================================


class TestRemoteEnvPool:
    def test_importable(self):
        from rlox.distributed import RemoteEnvPool
        assert RemoteEnvPool is not None

    def test_num_envs(self):
        from rlox.distributed import RemoteEnvPool
        pool = RemoteEnvPool(
            workers=["node-1:50051", "node-2:50051"],
            envs_per_worker=64,
        )
        assert pool.num_envs() == 128

    def test_step_raises_connection_error(self):
        from rlox.distributed import RemoteEnvPool
        import numpy as np
        pool = RemoteEnvPool(workers=["localhost:50051"], envs_per_worker=4)
        with pytest.raises(ConnectionError, match="gRPC server not running"):
            pool.step_all(np.zeros(4, dtype=np.int32))

    def test_reset_raises_connection_error(self):
        from rlox.distributed import RemoteEnvPool
        pool = RemoteEnvPool(workers=["localhost:50051"], envs_per_worker=4)
        with pytest.raises(ConnectionError, match="gRPC server not running"):
            pool.reset_all()

    def test_empty_workers_raises(self):
        from rlox.distributed import RemoteEnvPool
        with pytest.raises(ValueError, match="must not be empty"):
            RemoteEnvPool(workers=[])

    def test_repr(self):
        from rlox.distributed import RemoteEnvPool
        pool = RemoteEnvPool(workers=["a:1", "b:2"], envs_per_worker=32)
        r = repr(pool)
        assert "workers=2" in r
        assert "total_envs=64" in r

    def test_worker_addresses(self):
        from rlox.distributed import RemoteEnvPool
        pool = RemoteEnvPool(workers=["a:1", "b:2"], envs_per_worker=8)
        assert pool.worker_addresses == ["a:1", "b:2"]

    def test_spaces_with_metadata(self):
        from rlox.distributed import RemoteEnvPool
        pool = RemoteEnvPool(
            workers=["a:1"],
            envs_per_worker=4,
            obs_shape=(4,),
            n_actions=2,
        )
        assert pool.observation_space == {"shape": (4,)}
        assert pool.action_space == {"type": "discrete", "n": 2}

    def test_spaces_without_metadata(self):
        from rlox.distributed import RemoteEnvPool
        pool = RemoteEnvPool(workers=["a:1"], envs_per_worker=4)
        assert pool.observation_space is None
        assert pool.action_space is None
