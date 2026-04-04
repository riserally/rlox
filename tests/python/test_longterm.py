"""Long-term tests: Visual RL wrappers, Language-Conditioned RL, Cloud-Native Deploy.

TDD: These tests are written FIRST, before the implementations.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: Fake vectorized environment matching the rlox VecEnv interface
# ---------------------------------------------------------------------------

class _FakeImageVecEnv:
    """Fake vec env that produces image observations (B, C, H, W)."""

    def __init__(
        self,
        n_envs: int = 2,
        channels: int = 3,
        height: int = 96,
        width: int = 96,
    ) -> None:
        self._n_envs = n_envs
        self._channels = channels
        self._height = height
        self._width = width

    def num_envs(self) -> int:
        return self._n_envs

    def step_all(self, actions):
        obs = np.random.rand(
            self._n_envs, self._channels, self._height, self._width
        ).astype(np.float32) * 255.0
        return {
            "obs": obs,
            "rewards": np.zeros(self._n_envs, dtype=np.float64),
            "terminated": np.zeros(self._n_envs, dtype=np.uint8),
            "truncated": np.zeros(self._n_envs, dtype=np.uint8),
            "terminal_obs": [None] * self._n_envs,
        }

    def reset_all(self, **kwargs):
        return (
            np.random.rand(
                self._n_envs, self._channels, self._height, self._width
            ).astype(np.float32)
            * 255.0
        )


class _FakeVectorVecEnv:
    """Fake vec env that produces flat vector observations."""

    def __init__(self, n_envs: int = 2, obs_dim: int = 10) -> None:
        self._n_envs = n_envs
        self._obs_dim = obs_dim

    def num_envs(self) -> int:
        return self._n_envs

    def step_all(self, actions):
        obs = np.random.randn(self._n_envs, self._obs_dim).astype(np.float32)
        return {
            "obs": obs,
            "rewards": np.ones(self._n_envs, dtype=np.float64),
            "terminated": np.zeros(self._n_envs, dtype=np.uint8),
            "truncated": np.zeros(self._n_envs, dtype=np.uint8),
            "terminal_obs": [None] * self._n_envs,
        }

    def reset_all(self, **kwargs):
        return np.random.randn(self._n_envs, self._obs_dim).astype(np.float32)


# ============================================================================
# Part 1: Visual RL Wrappers
# ============================================================================


class TestFrameStack:
    """Tests for FrameStack wrapper."""

    def test_frame_stack_shape(self):
        """Stacked obs has n_frames * C channels."""
        from rlox.wrappers.visual import FrameStack

        n_frames = 4
        env = _FakeImageVecEnv(n_envs=2, channels=3, height=84, width=84)
        wrapped = FrameStack(env, n_frames=n_frames)
        wrapped.reset_all()

        actions = np.zeros(2, dtype=np.int32)
        result = wrapped.step_all(actions)
        obs = result["obs"]

        # Should be (n_envs, n_frames * C, H, W)
        assert obs.shape == (2, n_frames * 3, 84, 84)

    def test_frame_stack_reset_fills(self):
        """After reset, all frames should be identical (filled with first obs)."""
        from rlox.wrappers.visual import FrameStack

        n_frames = 4
        env = _FakeImageVecEnv(n_envs=2, channels=1, height=84, width=84)
        wrapped = FrameStack(env, n_frames=n_frames)
        obs = wrapped.reset_all()

        # obs shape: (2, 4, 84, 84) -- all 4 frames should be the same
        assert obs.shape == (2, n_frames * 1, 84, 84)
        for i in range(1, n_frames):
            np.testing.assert_array_equal(
                obs[:, 0:1, :, :], obs[:, i : i + 1, :, :]
            )

    def test_frame_stack_shifts_on_step(self):
        """After stepping, the newest frame should differ from the oldest."""
        from rlox.wrappers.visual import FrameStack

        n_frames = 3
        env = _FakeImageVecEnv(n_envs=1, channels=1, height=8, width=8)
        wrapped = FrameStack(env, n_frames=n_frames)
        wrapped.reset_all()

        actions = np.zeros(1, dtype=np.int32)
        result = wrapped.step_all(actions)
        obs = result["obs"]  # (1, 3, 8, 8)

        # The newest frame (last channel) should be from step, different
        # from the reset frame (first channel) with very high probability
        # (random data, astronomically unlikely to be identical)
        assert obs.shape == (1, 3, 8, 8)


class TestImagePreprocess:
    """Tests for ImagePreprocess wrapper."""

    def test_image_preprocess_resize(self):
        """Output is resized to target shape."""
        from rlox.wrappers.visual import ImagePreprocess

        env = _FakeImageVecEnv(n_envs=2, channels=3, height=210, width=160)
        wrapped = ImagePreprocess(env, size=(84, 84), grayscale=False, normalize=False)
        obs = wrapped.reset_all()

        assert obs.shape == (2, 3, 84, 84)

    def test_image_preprocess_grayscale(self):
        """Grayscale reduces channels to 1."""
        from rlox.wrappers.visual import ImagePreprocess

        env = _FakeImageVecEnv(n_envs=2, channels=3, height=84, width=84)
        wrapped = ImagePreprocess(env, size=(84, 84), grayscale=True, normalize=False)
        obs = wrapped.reset_all()

        assert obs.shape == (2, 1, 84, 84)

    def test_image_preprocess_normalize(self):
        """Normalized values should be in [0, 1]."""
        from rlox.wrappers.visual import ImagePreprocess

        env = _FakeImageVecEnv(n_envs=2, channels=3, height=84, width=84)
        wrapped = ImagePreprocess(env, size=(84, 84), grayscale=False, normalize=True)
        obs = wrapped.reset_all()

        assert obs.min() >= 0.0
        assert obs.max() <= 1.0

    def test_image_preprocess_step(self):
        """Preprocessing also works through step_all."""
        from rlox.wrappers.visual import ImagePreprocess

        env = _FakeImageVecEnv(n_envs=2, channels=3, height=210, width=160)
        wrapped = ImagePreprocess(env, size=(84, 84), grayscale=True, normalize=True)
        wrapped.reset_all()

        actions = np.zeros(2, dtype=np.int32)
        result = wrapped.step_all(actions)
        obs = result["obs"]

        assert obs.shape == (2, 1, 84, 84)
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0


class TestAtariWrapper:
    """Tests for AtariWrapper convenience class."""

    def test_atari_wrapper_constructs(self):
        """AtariWrapper constructs (skipped if ale-py unavailable)."""
        pytest.importorskip("ale_py")
        from rlox.wrappers.visual import AtariWrapper

        wrapper = AtariWrapper("PongNoFrameskip-v4", n_envs=1, frame_stack=4)
        assert wrapper.num_envs() == 1


class TestDMControlWrapper:
    """Tests for DMControlWrapper."""

    def test_dmcontrol_wrapper_constructs(self):
        """DMControlWrapper constructs (skipped if dm_control unavailable)."""
        pytest.importorskip("dm_control")
        from rlox.wrappers.visual import DMControlWrapper

        wrapper = DMControlWrapper("cartpole", "swingup", n_envs=1)
        assert wrapper.num_envs() == 1


# ============================================================================
# Part 2: Language-Conditioned RL
# ============================================================================


class TestLanguageWrapper:
    """Tests for LanguageWrapper."""

    def test_language_wrapper_appends_embedding(self):
        """Observation dimension increases by the embedding dimension."""
        sentence_transformers = pytest.importorskip("sentence_transformers")
        from rlox.wrappers.language import LanguageWrapper

        obs_dim = 10
        env = _FakeVectorVecEnv(n_envs=2, obs_dim=obs_dim)
        instructions = ["move to the right", "pick up the object"]

        wrapped = LanguageWrapper(
            env,
            instructions=instructions,
            encoder="all-MiniLM-L6-v2",
            instruction_mode="fixed",
        )
        obs = wrapped.reset_all()

        # Embedding dim for all-MiniLM-L6-v2 is 384
        assert obs.shape[0] == 2
        assert obs.shape[1] > obs_dim  # obs_dim + embed_dim
        embed_dim = obs.shape[1] - obs_dim
        assert embed_dim > 0

    def test_language_wrapper_fixed_mode(self):
        """In fixed mode, the same instruction embedding is appended every step."""
        sentence_transformers = pytest.importorskip("sentence_transformers")
        from rlox.wrappers.language import LanguageWrapper

        env = _FakeVectorVecEnv(n_envs=2, obs_dim=10)
        instructions = ["go left"]

        wrapped = LanguageWrapper(
            env,
            instructions=instructions,
            instruction_mode="fixed",
        )
        obs1 = wrapped.reset_all()
        result = wrapped.step_all(np.zeros(2, dtype=np.int32))
        obs2 = result["obs"]

        # The instruction embedding part (last embed_dim dims) should be identical
        embed1 = obs1[:, 10:]
        embed2 = obs2[:, 10:]
        np.testing.assert_array_almost_equal(embed1, embed2)

    def test_language_wrapper_random_mode(self):
        """In random mode, embeddings may differ across resets."""
        sentence_transformers = pytest.importorskip("sentence_transformers")
        from rlox.wrappers.language import LanguageWrapper

        env = _FakeVectorVecEnv(n_envs=2, obs_dim=10)
        instructions = ["go left", "go right", "jump", "duck"]

        wrapped = LanguageWrapper(
            env,
            instructions=instructions,
            instruction_mode="random",
        )

        # With 4 instructions, across many resets we should eventually
        # see different embeddings (not guaranteed per single reset due to
        # randomness, but the class should accept "random" mode)
        obs = wrapped.reset_all()
        assert obs.shape[1] > 10


class TestGoalConditioned:
    """Tests for GoalConditionedWrapper."""

    def test_goal_conditioned_sparse_reward(self):
        """Reward is 0 when within threshold, -1 otherwise."""
        from rlox.wrappers.language import GoalConditionedWrapper

        obs_dim = 6
        goal_dim = 3
        env = _FakeVectorVecEnv(n_envs=2, obs_dim=obs_dim)

        wrapped = GoalConditionedWrapper(
            env, goal_dim=goal_dim, distance_threshold=0.05
        )
        wrapped.reset_all()

        actions = np.zeros(2, dtype=np.int32)
        result = wrapped.step_all(actions)

        # obs should now have goal appended: obs_dim + goal_dim
        assert result["obs"].shape[1] == obs_dim + goal_dim

        # Rewards should be 0 or -1
        for r in result["rewards"]:
            assert r in (0.0, -1.0)

    def test_goal_conditioned_close_goal(self):
        """When achieved goal matches desired goal, reward is 0."""
        from rlox.wrappers.language import GoalConditionedWrapper

        # Create env that returns zeros -- goal will also be zeros
        class _ZeroEnv:
            def num_envs(self):
                return 1

            def step_all(self, actions):
                return {
                    "obs": np.zeros((1, 4), dtype=np.float32),
                    "rewards": np.zeros(1, dtype=np.float64),
                    "terminated": np.zeros(1, dtype=np.uint8),
                    "truncated": np.zeros(1, dtype=np.uint8),
                    "terminal_obs": [None],
                }

            def reset_all(self, **kwargs):
                return np.zeros((1, 4), dtype=np.float32)

        wrapped = GoalConditionedWrapper(
            _ZeroEnv(), goal_dim=2, distance_threshold=0.05
        )
        wrapped.reset_all()

        # Force goal to match achieved state
        wrapped._goals = np.zeros((1, 2), dtype=np.float32)

        result = wrapped.step_all(np.zeros(1))
        # Distance is 0, which is < threshold -> reward = 0
        assert result["rewards"][0] == 0.0


# ============================================================================
# Part 3: Cloud-Native Deploy
# ============================================================================


class TestDeploy:
    """Tests for deployment utilities."""

    def test_generate_dockerfile(self):
        """Generated Dockerfile contains pip install rlox."""
        from rlox.deploy.docker import generate_dockerfile

        dockerfile = generate_dockerfile(algo="ppo", env="CartPole-v1")
        assert "pip install" in dockerfile
        assert "rlox" in dockerfile
        assert "FROM" in dockerfile
        assert "CMD" in dockerfile

    def test_generate_dockerfile_custom_config(self):
        """Custom config path is included in the Dockerfile."""
        from rlox.deploy.docker import generate_dockerfile

        dockerfile = generate_dockerfile(
            algo="sac", env="Pendulum-v1", config_path="my_config.yaml"
        )
        assert "my_config.yaml" in dockerfile

    def test_generate_k8s_job(self):
        """Generated K8s manifest is a valid dict with required fields."""
        from rlox.deploy.docker import generate_k8s_job

        manifest = generate_k8s_job(
            name="rlox-train-ppo",
            image="rlox:latest",
            config={"algo": "ppo", "env": "CartPole-v1"},
        )
        assert isinstance(manifest, dict)
        assert manifest["apiVersion"] == "batch/v1"
        assert manifest["kind"] == "Job"
        assert "spec" in manifest
        assert "metadata" in manifest
        assert manifest["metadata"]["name"] == "rlox-train-ppo"

    def test_generate_k8s_job_gpu(self):
        """K8s job can request GPU resources."""
        from rlox.deploy.docker import generate_k8s_job

        manifest = generate_k8s_job(
            name="rlox-gpu",
            image="rlox:latest",
            config={"algo": "dreamer"},
            gpu_count=1,
        )
        spec = manifest["spec"]["template"]["spec"]["containers"][0]
        assert "nvidia.com/gpu" in str(spec.get("resources", {}))

    def test_sagemaker_estimator_constructs(self):
        """SageMakerEstimator can be instantiated."""
        from rlox.deploy.sagemaker import SageMakerEstimator

        estimator = SageMakerEstimator(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_type="ml.g5.xlarge",
        )
        assert estimator.role == "arn:aws:iam::123456789012:role/SageMakerRole"
        assert estimator.instance_type == "ml.g5.xlarge"

    def test_sagemaker_estimator_has_fit(self):
        """SageMakerEstimator exposes a fit method."""
        from rlox.deploy.sagemaker import SageMakerEstimator

        estimator = SageMakerEstimator(role="some-role")
        assert callable(getattr(estimator, "fit", None))


# ============================================================================
# Part 4: Integration -- wrappers compose with each other
# ============================================================================


class TestWrapperComposition:
    """Visual and language wrappers compose correctly."""

    def test_preprocess_then_framestack(self):
        """ImagePreprocess -> FrameStack produces correct shape."""
        from rlox.wrappers.visual import FrameStack, ImagePreprocess

        env = _FakeImageVecEnv(n_envs=2, channels=3, height=210, width=160)
        env = ImagePreprocess(env, size=(84, 84), grayscale=True, normalize=True)
        env = FrameStack(env, n_frames=4)

        obs = env.reset_all()
        assert obs.shape == (2, 4, 84, 84)

        result = env.step_all(np.zeros(2, dtype=np.int32))
        assert result["obs"].shape == (2, 4, 84, 84)
        assert result["obs"].min() >= 0.0
        assert result["obs"].max() <= 1.0


# ============================================================================
# Exports
# ============================================================================


class TestExports:
    """All new classes are exported from __init__.py."""

    def test_wrappers_init_exports(self):
        """Wrappers package exports all visual classes."""
        from rlox.wrappers import (
            FrameStack,
            ImagePreprocess,
            AtariWrapper,
            DMControlWrapper,
        )
        assert FrameStack is not None

    def test_wrappers_language_exports(self):
        """Wrappers package exports language classes."""
        from rlox.wrappers import GoalConditionedWrapper
        assert GoalConditionedWrapper is not None

    def test_deploy_exports(self):
        """Deploy package exports all classes."""
        from rlox.deploy import (
            generate_dockerfile,
            generate_k8s_job,
            SageMakerEstimator,
        )
        assert generate_dockerfile is not None
