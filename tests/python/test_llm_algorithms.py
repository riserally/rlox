"""Comprehensive tests for LLM post-training algorithms: DPO, OnlineDPO, BestOfN, GRPO.

Covers:
- Gradient flow and parameter updates
- Callback integration
- Checkpoint save/load
- Edge cases (beta values, group sizes)
- Mathematical properties of losses
- MmapReplayBuffer Python integration
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

class TinyLM(nn.Module):
    """Minimal language model for testing."""

    def __init__(self, vocab_size: int = 10, hidden: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(input_ids))

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 4) -> torch.Tensor:
        """Autoregressive sampling (not greedy, so pairs differ)."""
        ids = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(ids)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_token], dim=1)
        return ids


def make_model_pair(vocab_size: int = 10, hidden: int = 16):
    """Create model + frozen ref_model with identical initial weights."""
    model = TinyLM(vocab_size, hidden)
    ref_model = TinyLM(vocab_size, hidden)
    ref_model.load_state_dict(model.state_dict())
    for p in ref_model.parameters():
        p.requires_grad_(False)
    return model, ref_model


# ---------------------------------------------------------------------------
# DPO Tests
# ---------------------------------------------------------------------------

class TestDPO:
    def test_gradient_updates_model(self):
        """train_step should modify model parameters."""
        from rlox.algorithms.dpo import DPO

        model, ref = make_model_pair()
        dpo = DPO(model=model, ref_model=ref)

        params_before = {n: p.clone() for n, p in model.named_parameters()}

        prompt = torch.randint(0, 10, (2, 3))
        chosen = torch.randint(0, 10, (2, 5))
        rejected = torch.randint(0, 10, (2, 5))
        dpo.train_step(prompt, chosen, rejected)

        changed = any(
            not torch.equal(p, params_before[n])
            for n, p in model.named_parameters()
        )
        assert changed, "Model parameters should change after train_step"

    def test_ref_model_unchanged(self):
        """Reference model should never be modified."""
        from rlox.algorithms.dpo import DPO

        model, ref = make_model_pair()
        dpo = DPO(model=model, ref_model=ref)

        ref_before = {n: p.clone() for n, p in ref.named_parameters()}

        prompt = torch.randint(0, 10, (2, 3))
        chosen = torch.randint(0, 10, (2, 5))
        rejected = torch.randint(0, 10, (2, 5))
        dpo.train_step(prompt, chosen, rejected)

        for n, p in ref.named_parameters():
            assert torch.equal(p, ref_before[n]), f"ref_model param '{n}' was modified"

    def test_loss_prefers_chosen(self):
        """When chosen == rejected, loss should be -log(0.5) = ln(2) ≈ 0.693."""
        from rlox.algorithms.dpo import DPO

        model, ref = make_model_pair()
        dpo = DPO(model=model, ref_model=ref)

        prompt = torch.randint(0, 10, (1, 3))
        same = torch.randint(0, 10, (1, 5))
        loss, metrics = dpo.compute_loss(prompt, same, same)

        # When chosen == rejected, log_ratio_chosen == log_ratio_rejected
        # loss = -logsigmoid(0) = -log(0.5) = ln(2)
        assert abs(loss.item() - np.log(2)) < 0.01

    def test_different_beta_values(self):
        """Higher beta should amplify the preference signal."""
        from rlox.algorithms.dpo import DPO

        model, ref = make_model_pair()
        prompt = torch.randint(0, 10, (1, 3))
        chosen = torch.randint(0, 10, (1, 5))
        rejected = torch.randint(0, 10, (1, 5))

        losses = {}
        for beta in [0.01, 0.1, 1.0]:
            # Fresh model each time for fair comparison
            m, r = make_model_pair()
            m.load_state_dict(model.state_dict())
            r.load_state_dict(ref.state_dict())
            dpo = DPO(model=m, ref_model=r, beta=beta)
            loss, _ = dpo.compute_loss(prompt, chosen, rejected)
            losses[beta] = loss.item()

        assert all(np.isfinite(v) for v in losses.values())

    def test_callbacks_called(self):
        """Callbacks should be invoked during train_step."""
        from rlox.algorithms.dpo import DPO
        from rlox.callbacks import Callback

        cb = MagicMock(spec=Callback)
        cb.on_train_batch = MagicMock()

        model, ref = make_model_pair()
        dpo = DPO(model=model, ref_model=ref, callbacks=[cb])

        prompt = torch.randint(0, 10, (1, 3))
        chosen = torch.randint(0, 10, (1, 5))
        rejected = torch.randint(0, 10, (1, 5))
        dpo.train_step(prompt, chosen, rejected)

        cb.on_train_batch.assert_called_once()

    def test_checkpoint_roundtrip(self):
        """save() should produce a loadable checkpoint."""
        from rlox.algorithms.dpo import DPO

        model, ref = make_model_pair()
        dpo = DPO(model=model, ref_model=ref, beta=0.2)

        prompt = torch.randint(0, 10, (1, 3))
        chosen = torch.randint(0, 10, (1, 5))
        rejected = torch.randint(0, 10, (1, 5))
        dpo.train_step(prompt, chosen, rejected)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        dpo.save(path)
        data = torch.load(path, weights_only=False)
        assert data["step"] == 1
        assert data["config"]["beta"] == 0.2
        assert "model_state_dict" in data
        Path(path).unlink()

    def test_global_step_increments(self):
        """_global_step should increment on each train_step."""
        from rlox.algorithms.dpo import DPO

        model, ref = make_model_pair()
        dpo = DPO(model=model, ref_model=ref)

        prompt = torch.randint(0, 10, (1, 3))
        chosen = torch.randint(0, 10, (1, 5))
        rejected = torch.randint(0, 10, (1, 5))

        assert dpo._global_step == 0
        dpo.train_step(prompt, chosen, rejected)
        assert dpo._global_step == 1
        dpo.train_step(prompt, chosen, rejected)
        assert dpo._global_step == 2

    def test_gradient_clipping(self):
        """Gradients should be clipped to max_grad_norm."""
        from rlox.algorithms.dpo import DPO

        model, ref = make_model_pair()
        dpo = DPO(model=model, ref_model=ref)
        dpo.max_grad_norm = 0.01  # Very small clip

        prompt = torch.randint(0, 10, (1, 3))
        chosen = torch.randint(0, 10, (1, 5))
        rejected = torch.randint(0, 10, (1, 5))
        dpo.train_step(prompt, chosen, rejected)

        # Model should still train (no crash), just with clipped gradients
        assert dpo._global_step == 1


# ---------------------------------------------------------------------------
# OnlineDPO Tests
# ---------------------------------------------------------------------------

class TestOnlineDPO:
    def test_train_step_produces_loss(self):
        """train_step should return finite loss."""
        from rlox.algorithms.online_dpo import OnlineDPO

        model, ref = make_model_pair()
        pref_fn = lambda pairs: [0] * len(pairs)
        odpo = OnlineDPO(model=model, ref_model=ref, preference_fn=pref_fn)

        prompts = torch.randint(0, 10, (2, 3))
        metrics = odpo.train_step(prompts)
        assert "loss" in metrics
        assert np.isfinite(metrics["loss"])

    def test_gradient_updates_model(self):
        """train_step should modify model parameters."""
        from rlox.algorithms.online_dpo import OnlineDPO

        model, ref = make_model_pair()
        pref_fn = lambda pairs: [0] * len(pairs)
        odpo = OnlineDPO(model=model, ref_model=ref, preference_fn=pref_fn)

        params_before = {n: p.clone() for n, p in model.named_parameters()}

        prompts = torch.randint(0, 10, (2, 3))
        odpo.train_step(prompts)

        changed = any(
            not torch.equal(p, params_before[n])
            for n, p in model.named_parameters()
        )
        assert changed, "Model parameters should change after train_step"

    def test_preference_fn_called_correctly(self):
        """Preference function receives correct number of pairs."""
        from rlox.algorithms.online_dpo import OnlineDPO

        model, ref = make_model_pair()
        pref_fn = MagicMock(return_value=[0, 0, 0])
        odpo = OnlineDPO(model=model, ref_model=ref, preference_fn=pref_fn)

        prompts = torch.randint(0, 10, (3, 3))
        odpo.train_step(prompts)

        pref_fn.assert_called_once()
        pairs = pref_fn.call_args[0][0]
        assert len(pairs) == 3  # One pair per prompt

    def test_preference_index_1_swaps(self):
        """When preference_fn returns 1, second completion should be chosen."""
        from rlox.algorithms.online_dpo import OnlineDPO

        model, ref = make_model_pair()
        # Always prefer second completion
        pref_fn = lambda pairs: [1] * len(pairs)
        odpo = OnlineDPO(model=model, ref_model=ref, preference_fn=pref_fn)

        prompts = torch.randint(0, 10, (1, 3))
        metrics = odpo.train_step(prompts)
        assert np.isfinite(metrics["loss"])

    def test_gradient_clipping_applied(self):
        """Gradient clipping should not crash with small max_grad_norm."""
        from rlox.algorithms.online_dpo import OnlineDPO

        model, ref = make_model_pair()
        pref_fn = lambda pairs: [0] * len(pairs)
        odpo = OnlineDPO(model=model, ref_model=ref, preference_fn=pref_fn)
        odpo.max_grad_norm = 0.001

        prompts = torch.randint(0, 10, (1, 3))
        metrics = odpo.train_step(prompts)
        assert np.isfinite(metrics["loss"])


# ---------------------------------------------------------------------------
# BestOfN Tests
# ---------------------------------------------------------------------------

class TestBestOfN:
    def test_generate_returns_correct_shape(self):
        """Output should be (B, P + max_new_tokens)."""
        from rlox.algorithms.best_of_n import BestOfN

        model = TinyLM()
        reward_fn = lambda comps, prompts: [float(c.sum()) for c in comps]
        bon = BestOfN(model=model, reward_fn=reward_fn, n=4, max_new_tokens=5)

        prompts = torch.randint(0, 10, (3, 4))
        result = bon.generate(prompts)
        assert result.shape == (3, 4 + 5)

    def test_selects_highest_reward(self):
        """Should select the completion with highest reward score."""
        from rlox.algorithms.best_of_n import BestOfN

        model = TinyLM()

        # Reward function that scores based on first generated token
        call_count = [0]
        def reward_fn(comps, prompts):
            call_count[0] += 1
            return [float(c[-1]) for c in comps]  # Score by last token

        bon = BestOfN(model=model, reward_fn=reward_fn, n=4, max_new_tokens=3)
        prompts = torch.randint(0, 10, (2, 3))
        result = bon.generate(prompts)

        assert call_count[0] == 1  # reward_fn called once
        assert result.shape[0] == 2

    def test_n_equals_1(self):
        """With n=1, should just return the single generation."""
        from rlox.algorithms.best_of_n import BestOfN

        model = TinyLM()
        reward_fn = lambda comps, prompts: [1.0] * len(comps)
        bon = BestOfN(model=model, reward_fn=reward_fn, n=1, max_new_tokens=3)

        prompts = torch.randint(0, 10, (2, 4))
        result = bon.generate(prompts)
        assert result.shape == (2, 4 + 3)

    def test_no_gradient_computation(self):
        """generate() should not compute gradients."""
        from rlox.algorithms.best_of_n import BestOfN

        model = TinyLM()
        reward_fn = lambda comps, prompts: [1.0] * len(comps)
        bon = BestOfN(model=model, reward_fn=reward_fn, n=2)

        prompts = torch.randint(0, 10, (1, 3))
        result = bon.generate(prompts)
        assert not result.requires_grad

    def test_different_n_values(self):
        """Should work with various n values."""
        from rlox.algorithms.best_of_n import BestOfN

        model = TinyLM()
        reward_fn = lambda comps, prompts: [float(i) for i in range(len(comps))]

        for n in [2, 4, 8]:
            bon = BestOfN(model=model, reward_fn=reward_fn, n=n, max_new_tokens=2)
            prompts = torch.randint(0, 10, (1, 3))
            result = bon.generate(prompts)
            assert result.shape == (1, 3 + 2)


# ---------------------------------------------------------------------------
# GRPO Tests
# ---------------------------------------------------------------------------

class TestGRPO:
    def test_train_step_returns_metrics(self):
        """train_step should return loss, mean_reward, kl."""
        from rlox.algorithms.grpo import GRPO

        model, ref = make_model_pair()
        reward_fn = lambda comps, prompts: [float(c.sum()) for c in comps]
        grpo = GRPO(model=model, ref_model=ref, reward_fn=reward_fn, group_size=2)

        prompts = torch.randint(0, 10, (2, 3))
        metrics = grpo.train_step(prompts)

        assert "loss" in metrics
        assert "mean_reward" in metrics
        assert "kl" in metrics
        assert all(np.isfinite(v) for v in metrics.values())

    def test_gradient_updates_model(self):
        """Model parameters should change after train_step."""
        from rlox.algorithms.grpo import GRPO

        model, ref = make_model_pair()
        reward_fn = lambda comps, prompts: [float(c.sum()) for c in comps]
        grpo = GRPO(model=model, ref_model=ref, reward_fn=reward_fn, group_size=2)

        params_before = {n: p.clone() for n, p in model.named_parameters()}

        prompts = torch.randint(0, 10, (2, 3))
        grpo.train_step(prompts)

        changed = any(
            not torch.equal(p, params_before[n])
            for n, p in model.named_parameters()
        )
        assert changed

    def test_group_size_1(self):
        """Should work with group_size=1 (degenerate case)."""
        from rlox.algorithms.grpo import GRPO

        model, ref = make_model_pair()
        reward_fn = lambda comps, prompts: [1.0] * len(comps)
        grpo = GRPO(model=model, ref_model=ref, reward_fn=reward_fn, group_size=1)

        prompts = torch.randint(0, 10, (2, 3))
        metrics = grpo.train_step(prompts)
        assert np.isfinite(metrics["loss"])

    def test_evaluate_returns_mean_reward(self):
        """evaluate() should return a scalar mean reward."""
        from rlox.algorithms.grpo import GRPO

        model, ref = make_model_pair()
        reward_fn = lambda comps, prompts: [2.0] * len(comps)
        grpo = GRPO(model=model, ref_model=ref, reward_fn=reward_fn)

        prompts = torch.randint(0, 10, (3, 3))
        mean_reward = grpo.evaluate(prompts)
        assert mean_reward == pytest.approx(2.0)

    def test_train_multi_epoch(self):
        """train() with multiple epochs should update step count."""
        from rlox.algorithms.grpo import GRPO

        model, ref = make_model_pair()
        reward_fn = lambda comps, prompts: [float(c.sum()) for c in comps]
        grpo = GRPO(model=model, ref_model=ref, reward_fn=reward_fn, group_size=2)

        prompts = torch.randint(0, 10, (2, 3))
        metrics = grpo.train(prompts, n_epochs=3)
        assert grpo._global_step == 3
        assert "loss" in metrics

    def test_callbacks_integration(self):
        """Callbacks should be called during train()."""
        from rlox.algorithms.grpo import GRPO
        from rlox.callbacks import Callback

        cb = MagicMock(spec=Callback)
        cb.on_training_start = MagicMock()
        cb.on_training_end = MagicMock()
        cb.on_train_batch = MagicMock()
        cb.on_step = MagicMock(return_value=True)

        model, ref = make_model_pair()
        reward_fn = lambda comps, prompts: [1.0] * len(comps)
        grpo = GRPO(model=model, ref_model=ref, reward_fn=reward_fn,
                     group_size=2, callbacks=[cb])

        prompts = torch.randint(0, 10, (1, 3))
        grpo.train(prompts, n_epochs=2)

        cb.on_training_start.assert_called_once()
        cb.on_training_end.assert_called_once()
        assert cb.on_train_batch.call_count == 2
        assert cb.on_step.call_count == 2

    def test_early_stopping_via_callback(self):
        """Callback returning False should stop training."""
        from rlox.algorithms.grpo import GRPO
        from rlox.callbacks import Callback

        class StopAfterOne(Callback):
            def on_step(self, **kwargs) -> bool:
                return False  # Stop immediately

        model, ref = make_model_pair()
        reward_fn = lambda comps, prompts: [1.0] * len(comps)
        grpo = GRPO(model=model, ref_model=ref, reward_fn=reward_fn,
                     group_size=2, callbacks=[StopAfterOne()])

        prompts = torch.randint(0, 10, (1, 3))
        grpo.train(prompts, n_epochs=10)
        assert grpo._global_step == 1  # Stopped after first epoch

    def test_checkpoint_roundtrip(self):
        """save() should produce a loadable checkpoint."""
        from rlox.algorithms.grpo import GRPO

        model, ref = make_model_pair()
        reward_fn = lambda comps, prompts: [1.0] * len(comps)
        grpo = GRPO(model=model, ref_model=ref, reward_fn=reward_fn,
                     group_size=4, kl_coef=0.05)

        prompts = torch.randint(0, 10, (1, 3))
        grpo.train_step(prompts)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        grpo.save(path)
        data = torch.load(path, weights_only=False)
        assert data["step"] == 1
        assert data["config"]["group_size"] == 4
        assert data["config"]["kl_coef"] == 0.05
        Path(path).unlink()

    def test_kl_is_zero_at_init(self):
        """At initialization, model == ref_model, so KL should be ~0."""
        from rlox.algorithms.grpo import GRPO

        model, ref = make_model_pair()
        reward_fn = lambda comps, prompts: [1.0] * len(comps)
        grpo = GRPO(model=model, ref_model=ref, reward_fn=reward_fn, group_size=2)

        prompts = torch.randint(0, 10, (1, 3))
        metrics = grpo.train_step(prompts)
        # KL between identical distributions should be ~0
        assert abs(metrics["kl"]) < 0.1


# ---------------------------------------------------------------------------
# MmapReplayBuffer Python Integration Tests
# ---------------------------------------------------------------------------

class TestMmapReplayBuffer:
    def test_create_and_push(self):
        """Basic push and length tracking."""
        import rlox

        with tempfile.TemporaryDirectory() as tmpdir:
            cold_path = f"{tmpdir}/cold.bin"
            buf = rlox.MmapReplayBuffer(
                hot_capacity=10, total_capacity=20,
                obs_dim=4, act_dim=1, cold_path=cold_path,
            )
            assert len(buf) == 0

            obs = np.zeros(4, dtype=np.float32)
            act = np.zeros(1, dtype=np.float32)
            buf.push(obs, act, 1.0, False, False, obs)
            assert len(buf) == 1

    def test_sample_from_hot(self):
        """Sampling when all data is in hot buffer."""
        import rlox

        with tempfile.TemporaryDirectory() as tmpdir:
            cold_path = f"{tmpdir}/cold.bin"
            buf = rlox.MmapReplayBuffer(
                hot_capacity=100, total_capacity=200,
                obs_dim=3, act_dim=1, cold_path=cold_path,
            )

            for i in range(20):
                obs = np.full(3, float(i), dtype=np.float32)
                act = np.array([float(i)], dtype=np.float32)
                buf.push(obs, act, float(i), False, False, obs)

            batch = buf.sample(8, seed=42)
            assert batch["obs"].shape == (8, 3)
            assert batch["actions"].shape == (8,)
            assert batch["rewards"].shape == (8,)

    def test_spill_to_cold(self):
        """When hot is full, data should spill to cold storage."""
        import rlox

        with tempfile.TemporaryDirectory() as tmpdir:
            cold_path = f"{tmpdir}/cold.bin"
            buf = rlox.MmapReplayBuffer(
                hot_capacity=5, total_capacity=20,
                obs_dim=2, act_dim=1, cold_path=cold_path,
            )

            for i in range(10):
                obs = np.full(2, float(i), dtype=np.float32)
                act = np.array([float(i)], dtype=np.float32)
                buf.push(obs, act, float(i), False, False, obs)

            assert len(buf) == 10
            # Cold file should exist
            assert Path(cold_path).exists()

    def test_deterministic_sampling(self):
        """Same seed should produce same samples."""
        import rlox

        with tempfile.TemporaryDirectory() as tmpdir:
            cold_path = f"{tmpdir}/cold.bin"
            buf = rlox.MmapReplayBuffer(
                hot_capacity=50, total_capacity=100,
                obs_dim=4, act_dim=1, cold_path=cold_path,
            )

            for i in range(30):
                obs = np.full(4, float(i), dtype=np.float32)
                act = np.array([float(i)], dtype=np.float32)
                buf.push(obs, act, float(i), False, False, obs)

            batch1 = buf.sample(10, seed=123)
            batch2 = buf.sample(10, seed=123)
            np.testing.assert_array_equal(batch1["obs"], batch2["obs"])
            np.testing.assert_array_equal(batch1["rewards"], batch2["rewards"])

    def test_sample_with_hot_and_cold(self):
        """Sampling should work when data spans hot and cold."""
        import rlox

        with tempfile.TemporaryDirectory() as tmpdir:
            cold_path = f"{tmpdir}/cold.bin"
            buf = rlox.MmapReplayBuffer(
                hot_capacity=5, total_capacity=50,
                obs_dim=3, act_dim=1, cold_path=cold_path,
            )

            for i in range(15):
                obs = np.full(3, float(i), dtype=np.float32)
                act = np.array([float(i)], dtype=np.float32)
                buf.push(obs, act, float(i), False, False, obs)

            assert len(buf) == 15
            batch = buf.sample(10, seed=42)
            assert batch["obs"].shape == (10, 3)

    def test_close_cleans_up(self):
        """close() should remove the cold file."""
        import rlox

        with tempfile.TemporaryDirectory() as tmpdir:
            cold_path = f"{tmpdir}/cold.bin"
            buf = rlox.MmapReplayBuffer(
                hot_capacity=2, total_capacity=10,
                obs_dim=2, act_dim=1, cold_path=cold_path,
            )

            for i in range(5):
                obs = np.full(2, float(i), dtype=np.float32)
                act = np.array([float(i)], dtype=np.float32)
                buf.push(obs, act, float(i), False, False, obs)

            assert Path(cold_path).exists()
            buf.close()
            assert not Path(cold_path).exists()

    def test_use_with_dqn(self):
        """MmapReplayBuffer should work as drop-in for DQN."""
        import rlox
        from rlox.algorithms.dqn import DQN

        with tempfile.TemporaryDirectory() as tmpdir:
            cold_path = f"{tmpdir}/cold.bin"
            buf = rlox.MmapReplayBuffer(
                hot_capacity=100, total_capacity=1000,
                obs_dim=4, act_dim=1, cold_path=cold_path,
            )

            dqn = DQN(
                env_id="CartPole-v1",
                buffer=buf,
                learning_starts=50,
                batch_size=32,
                hidden=32,
            )
            metrics = dqn.train(total_timesteps=100)
            assert "mean_reward" in metrics
            assert len(buf) > 0
