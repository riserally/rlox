"""Tests for rlox.diagnostics — training failure auto-detection."""

from __future__ import annotations

from rlox.callbacks import Callback
from rlox.diagnostics import TrainingDiagnostics


class TestTrainingDiagnostics:
    def test_diagnostics_is_callback(self):
        """TrainingDiagnostics must be a Callback subclass."""
        diag = TrainingDiagnostics()
        assert isinstance(diag, Callback)

    def test_entropy_collapse_detected(self):
        """Entropy dropping below 10% of initial should trigger a warning."""
        diag = TrainingDiagnostics()
        diag.on_train_batch(entropy=1.0)  # sets initial
        diag.on_train_batch(entropy=0.05)  # 5% of initial -> collapse
        assert any("entropy" in w.lower() for w in diag.warnings)

    def test_kl_spike_detected(self):
        """KL exceeding 10x target should trigger a warning."""
        diag = TrainingDiagnostics(target_kl=0.01)
        diag.on_train_batch(approx_kl=0.2)  # 20x target
        assert any("kl" in w.lower() for w in diag.warnings)

    def test_gradient_explosion_detected(self):
        """Grad norm exceeding 100x max_grad_norm should trigger a warning."""
        diag = TrainingDiagnostics(max_grad_norm=0.5)
        diag.on_train_batch(grad_norm=60.0)  # 120x max_grad_norm
        assert any("gradient" in w.lower() for w in diag.warnings)

    def test_value_divergence_detected(self):
        """Explained variance below -1 should trigger a warning."""
        diag = TrainingDiagnostics()
        diag.on_train_batch(explained_var=-2.0)
        assert any("value" in w.lower() for w in diag.warnings)

    def test_no_false_positives(self):
        """Normal training values should not trigger any warnings."""
        diag = TrainingDiagnostics(target_kl=0.01, max_grad_norm=0.5)
        diag.on_train_batch(entropy=1.0)  # initial
        diag.on_train_batch(
            entropy=0.8,
            approx_kl=0.005,
            grad_norm=0.3,
            explained_var=0.5,
        )
        assert len(diag.warnings) == 0

    def test_on_train_batch_returns_none(self):
        """on_train_batch should return None (not a bool like on_step)."""
        diag = TrainingDiagnostics()
        result = diag.on_train_batch(entropy=1.0)
        assert result is None

    def test_multiple_warnings_accumulate(self):
        """Multiple issues in one batch should produce multiple warnings."""
        diag = TrainingDiagnostics(target_kl=0.01, max_grad_norm=0.5)
        diag.on_train_batch(entropy=1.0)  # initial
        diag.on_train_batch(
            entropy=0.01,  # collapse
            approx_kl=0.5,  # spike
            grad_norm=100.0,  # explosion
            explained_var=-3.0,  # divergence
        )
        assert len(diag.warnings) == 4
