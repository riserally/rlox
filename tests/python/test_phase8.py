"""Phase 8 tests: PrioritizedReplayBuffer, off-policy algorithms, trainers,
config, callbacks, diagnostics, evaluation, experiment metadata, LLM pipeline."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn

import rlox


# =========================================================================
# Group 0: PrioritizedReplayBuffer Python bindings
# =========================================================================


class TestPrioritizedReplayBuffer:
    def test_create_and_push(self):
        buf = rlox.PrioritizedReplayBuffer(100, 4, 1)
        assert len(buf) == 0
        buf.push(
            np.zeros(4, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            1.0,
            False,
            False,
            priority=2.0,
        )
        assert len(buf) == 1

    def test_sample_returns_dict(self):
        buf = rlox.PrioritizedReplayBuffer(100, 4, 1, alpha=0.6, beta=0.4)
        for i in range(20):
            buf.push(
                np.ones(4, dtype=np.float32) * i,
                np.array([float(i)], dtype=np.float32),
                float(i),
                False,
                False,
                priority=float(i + 1),
            )
        batch = buf.sample(8, seed=42)
        assert "obs" in batch
        assert "actions" in batch
        assert "rewards" in batch
        assert "weights" in batch
        assert "indices" in batch
        obs = np.asarray(batch["obs"])
        assert obs.shape == (8, 4)
        weights = np.asarray(batch["weights"])
        assert weights.shape == (8,)
        assert np.all(weights > 0)
        assert np.all(weights <= 1.0 + 1e-6)

    def test_update_priorities(self):
        buf = rlox.PrioritizedReplayBuffer(100, 4, 1, alpha=1.0, beta=0.4)
        for _ in range(10):
            buf.push(
                np.zeros(4, dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                1.0,
                False,
                False,
            )
        batch = buf.sample(5, seed=0)
        indices = np.asarray(batch["indices"])
        new_prios = np.ones(5, dtype=np.float64) * 10.0
        buf.update_priorities(indices, new_prios)
        # Should not raise

    def test_negative_priority_raises(self):
        buf = rlox.PrioritizedReplayBuffer(100, 4, 1)
        with pytest.raises(RuntimeError, match="non-negative"):
            buf.push(
                np.zeros(4, dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                1.0,
                False,
                False,
                priority=-1.0,
            )

    def test_sample_empty_raises(self):
        buf = rlox.PrioritizedReplayBuffer(100, 4, 1)
        with pytest.raises(RuntimeError):
            buf.sample(1, seed=42)

    def test_set_beta(self):
        buf = rlox.PrioritizedReplayBuffer(100, 4, 1)
        buf.set_beta(1.0)


# =========================================================================
# Group 1: SAC, TD3, DQN smoke tests
# =========================================================================


class TestSAC:
    def test_import(self):
        from rlox.algorithms.sac import SAC
        assert SAC is not None

    @pytest.mark.slow
    def test_smoke_500_steps(self):
        from rlox.algorithms.sac import SAC
        agent = SAC("Pendulum-v1", buffer_size=1000, learning_starts=100, batch_size=64)
        metrics = agent.train(total_timesteps=500)
        assert "critic_loss" in metrics or "mean_reward" in metrics


class TestTD3:
    def test_import(self):
        from rlox.algorithms.td3 import TD3
        assert TD3 is not None

    @pytest.mark.slow
    def test_smoke_500_steps(self):
        from rlox.algorithms.td3 import TD3
        agent = TD3("Pendulum-v1", buffer_size=1000, learning_starts=100, batch_size=64)
        metrics = agent.train(total_timesteps=500)
        assert "critic_loss" in metrics or "mean_reward" in metrics


class TestDQN:
    def test_import(self):
        from rlox.algorithms.dqn import DQN
        assert DQN is not None

    @pytest.mark.slow
    def test_smoke_500_steps(self):
        from rlox.algorithms.dqn import DQN
        agent = DQN("CartPole-v1", buffer_size=1000, learning_starts=100, batch_size=64)
        metrics = agent.train(total_timesteps=500)
        assert "loss" in metrics or "mean_reward" in metrics


# =========================================================================
# Group 2: One-liner trainer API, config system
# =========================================================================


class TestTrainerAPI:
    def test_ppo_trainer_creates(self):
        from rlox.trainers import PPOTrainer
        trainer = PPOTrainer("CartPole-v1")
        assert trainer is not None

    @pytest.mark.slow
    def test_ppo_trainer_train(self):
        from rlox.trainers import PPOTrainer
        trainer = PPOTrainer("CartPole-v1")
        metrics = trainer.train(total_timesteps=256)
        assert "mean_reward" in metrics

    def test_sac_trainer_creates(self):
        from rlox.trainers import SACTrainer
        trainer = SACTrainer("Pendulum-v1")
        assert trainer is not None

    def test_dqn_trainer_creates(self):
        from rlox.trainers import DQNTrainer
        trainer = DQNTrainer("CartPole-v1")
        assert trainer is not None


class TestConfig:
    def test_ppo_config_validation(self):
        from rlox.config import PPOConfig
        with pytest.raises(ValueError):
            PPOConfig(learning_rate=-1.0)
        with pytest.raises(ValueError):
            PPOConfig(n_envs=0)

    def test_sac_config(self):
        from rlox.config import SACConfig
        cfg = SACConfig(tau=0.005, buffer_size=100000)
        assert cfg.tau == 0.005

    def test_dqn_config(self):
        from rlox.config import DQNConfig
        cfg = DQNConfig(double_dqn=True, dueling=True, n_step=3)
        assert cfg.double_dqn is True
        assert cfg.n_step == 3

    def test_config_from_dict(self):
        from rlox.config import PPOConfig
        cfg = PPOConfig.from_dict({"learning_rate": 1e-3, "n_envs": 4})
        assert cfg.learning_rate == 1e-3
        assert cfg.n_envs == 4

    def test_config_merge(self):
        from rlox.config import PPOConfig
        base = PPOConfig()
        merged = base.merge({"learning_rate": 1e-3})
        assert merged.learning_rate == 1e-3
        assert merged.n_envs == base.n_envs


# =========================================================================
# Group 3: Callback system
# =========================================================================


class TestCallbacks:
    def test_callback_on_training_start_fires(self):
        from rlox.callbacks import Callback

        class MyCallback(Callback):
            def __init__(self):
                super().__init__()
                self.started = False

            def on_training_start(self, **kwargs):
                self.started = True

        cb = MyCallback()
        cb.on_training_start()
        assert cb.started

    def test_callback_on_step_fires(self):
        from rlox.callbacks import Callback

        class StepCounter(Callback):
            def __init__(self):
                super().__init__()
                self.count = 0

            def on_step(self, **kwargs):
                self.count += 1
                return True

        cb = StepCounter()
        cb.on_step()
        cb.on_step()
        assert cb.count == 2

    def test_early_stopping_callback(self):
        from rlox.callbacks import EarlyStoppingCallback

        cb = EarlyStoppingCallback(patience=3, min_delta=0.1)
        # Simulate rewards that plateau
        for _ in range(3):
            result = cb.on_step(reward=1.0)
        # After patience steps with no improvement, should return False
        result = cb.on_step(reward=1.0)
        assert result is False

    def test_eval_callback(self):
        from rlox.callbacks import EvalCallback
        cb = EvalCallback(eval_env=None, eval_freq=10, n_eval_episodes=5)
        assert cb.eval_freq == 10

    def test_checkpoint_callback(self):
        from rlox.callbacks import CheckpointCallback
        cb = CheckpointCallback(save_freq=100, save_path="/tmp/rlox_test")
        assert cb.save_freq == 100


# =========================================================================
# Group 4: Diagnostics, evaluation, experiment metadata
# =========================================================================


class TestDiagnostics:
    def test_entropy_collapse_warning(self):
        from rlox.diagnostics import TrainingDiagnostics

        diag = TrainingDiagnostics()
        diag.on_step(entropy=1.0)  # initial
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            diag.on_step(entropy=0.05)  # < 10% of initial
            entropy_warnings = [x for x in w if "entropy" in str(x.message).lower()]
            assert len(entropy_warnings) > 0

    def test_kl_spike_warning(self):
        from rlox.diagnostics import TrainingDiagnostics

        diag = TrainingDiagnostics(target_kl=0.01)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            diag.on_step(approx_kl=0.2)  # 20x target
            kl_warnings = [x for x in w if "kl" in str(x.message).lower()]
            assert len(kl_warnings) > 0

    def test_gradient_explosion_warning(self):
        from rlox.diagnostics import TrainingDiagnostics

        diag = TrainingDiagnostics()
        diag.on_step(grad_norm=1.0)  # baseline
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            diag.on_step(grad_norm=200.0)  # > 100x
            grad_warnings = [x for x in w if "gradient" in str(x.message).lower()]
            assert len(grad_warnings) > 0

    def test_value_divergence_warning(self):
        from rlox.diagnostics import TrainingDiagnostics

        diag = TrainingDiagnostics()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            diag.on_step(explained_var=-2.0)
            val_warnings = [x for x in w if "value" in str(x.message).lower()]
            assert len(val_warnings) > 0


class TestEvaluation:
    def test_interquartile_mean(self):
        from rlox.evaluation import interquartile_mean
        scores = list(range(100))
        iqm = interquartile_mean(scores)
        # IQM of 0..99 with Q1=25, Q3=75 => mean of 25..74 = 49.5
        assert abs(iqm - 49.5) < 1.0

    def test_performance_profiles(self):
        from rlox.evaluation import performance_profiles
        scores_dict = {"algo_a": [1.0, 2.0, 3.0], "algo_b": [0.5, 1.5, 2.5]}
        thresholds = [0.0, 1.0, 2.0, 3.0]
        result = performance_profiles(scores_dict, thresholds)
        assert "algo_a" in result
        assert "algo_b" in result
        # All scores >= 0
        assert result["algo_a"][0] == 1.0

    def test_bootstrap_ci(self):
        from rlox.evaluation import stratified_bootstrap_ci
        np.random.seed(42)
        scores = list(np.random.randn(50))
        lo, hi = stratified_bootstrap_ci(scores, n_bootstrap=1000, ci=0.95)
        assert lo < hi
        mean = np.mean(scores)
        assert lo < mean < hi


class TestExperimentMetadata:
    def test_capture_metadata(self):
        from rlox.experiment import capture_experiment_metadata
        meta = capture_experiment_metadata(config={"lr": 1e-3}, seed=42)
        assert "timestamp" in meta
        assert "platform" in meta
        assert "seed" in meta
        assert meta["seed"] == 42
        assert "config" in meta


# =========================================================================
# Group 5: Network utilities
# =========================================================================


class TestNetworks:
    def test_polyak_update(self):
        from rlox.networks import polyak_update

        source = nn.Linear(4, 2)
        target = nn.Linear(4, 2)
        # Make them different
        with torch.no_grad():
            target.weight.fill_(0.0)
            target.bias.fill_(0.0)
        polyak_update(source, target, tau=1.0)
        # tau=1.0 means full copy
        assert torch.allclose(source.weight, target.weight)

    def test_q_network(self):
        from rlox.networks import QNetwork
        q = QNetwork(obs_dim=4, act_dim=2, hidden=64)
        obs = torch.randn(8, 4)
        act = torch.randn(8, 2)
        out = q(obs, act)
        assert out.shape == (8, 1)

    def test_squashed_gaussian_policy(self):
        from rlox.networks import SquashedGaussianPolicy
        pol = SquashedGaussianPolicy(obs_dim=4, act_dim=2, hidden=64)
        obs = torch.randn(8, 4)
        action, log_prob = pol.sample(obs)
        assert action.shape == (8, 2)
        assert log_prob.shape == (8,)
        # Actions should be in [-1, 1] due to tanh
        assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


# =========================================================================
# Group 6: OnlineDPO, BestOfN smoke tests
# =========================================================================


class TestOnlineDPO:
    def test_import(self):
        from rlox.algorithms.online_dpo import OnlineDPO
        assert OnlineDPO is not None

    def test_smoke(self):
        from rlox.algorithms.online_dpo import OnlineDPO

        # Tiny LM stub
        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(10, 16)
                self.fc = nn.Linear(16, 10)

            def forward(self, x):
                return self.fc(self.embed(x))

            def generate(self, prompt, max_new_tokens=4):
                tokens = prompt.clone()
                for _ in range(max_new_tokens):
                    logits = self.forward(tokens)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    tokens = torch.cat([tokens, next_token], dim=1)
                return tokens

        model = TinyLM()
        ref_model = TinyLM()

        def pref_fn(pairs):
            """Return indices of preferred completion (always first)."""
            return [0] * len(pairs)

        odpo = OnlineDPO(model=model, ref_model=ref_model, preference_fn=pref_fn)
        prompts = torch.randint(0, 10, (2, 3))
        metrics = odpo.train_step(prompts)
        assert "loss" in metrics


class TestBestOfN:
    def test_import(self):
        from rlox.algorithms.best_of_n import BestOfN
        assert BestOfN is not None

    def test_smoke(self):
        from rlox.algorithms.best_of_n import BestOfN

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(10, 16)
                self.fc = nn.Linear(16, 10)

            def forward(self, x):
                return self.fc(self.embed(x))

            def generate(self, prompt, max_new_tokens=4):
                tokens = prompt.clone()
                for _ in range(max_new_tokens):
                    logits = self.forward(tokens)
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    tokens = torch.cat([tokens, next_token], dim=1)
                return tokens

        model = TinyLM()

        def reward_fn(completions, prompts):
            return [float(c.sum()) for c in completions]

        bon = BestOfN(model=model, reward_fn=reward_fn, n=4)
        prompts = torch.randint(0, 10, (2, 3))
        best = bon.generate(prompts)
        assert best.shape[0] == 2  # one best per prompt
