"""Tests verifying convergence bug fixes in off-policy algorithms (SAC, TD3, DQN).

These are structural tests that verify the fixed code paths without requiring
a full Rust rebuild -- they inspect source code and test buffer integration
with a mock buffer where necessary.
"""

from __future__ import annotations

import inspect
import textwrap

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _source(obj: object) -> str:
    """Return dedented source of *obj*."""
    return textwrap.dedent(inspect.getsource(obj))


# ---------------------------------------------------------------------------
# SAC
# ---------------------------------------------------------------------------


class TestSACFixes:
    """Verify SAC convergence fixes (Bugs 1-3)."""

    def test_sac_push_includes_next_obs(self) -> None:
        """buffer.push() call in SAC.train() must pass next_obs."""
        from rlox.algorithms.sac import SAC

        src = _source(SAC.train)
        # The push call should contain next_obs as an argument
        assert "next_obs" in src, "SAC.train() must pass next_obs to buffer.push()"
        # Specifically, it should be np.asarray(next_obs, ...)
        assert "np.asarray(next_obs" in src, (
            "SAC.train() should convert next_obs via np.asarray before pushing"
        )

    def test_sac_target_uses_next_obs(self) -> None:
        """SAC._update() must use next_obs (not obs) for target Q computation."""
        from rlox.algorithms.sac import SAC

        src = _source(SAC._update)
        # Must extract next_obs from batch
        assert 'batch["next_obs"]' in src, (
            "SAC._update() must extract next_obs from the sampled batch"
        )
        # actor.sample and critic targets must be called on next_obs
        assert "self.actor.sample(next_obs)" in src, (
            "SAC._update() must sample next_actions from next_obs, not obs"
        )
        assert "self.critic1_target(next_obs" in src
        assert "self.critic2_target(next_obs" in src

    def test_sac_action_scaling_in_train(self) -> None:
        """Actions from SquashedGaussianPolicy ([-1,1]) must be scaled by act_high."""
        from rlox.algorithms.sac import SAC

        src = _source(SAC.train)
        assert "self.act_high" in src, (
            "SAC.train() must scale actions by act_high"
        )
        # Should multiply, not just clip
        assert "action * self.act_high" in src or "action *self.act_high" in src, (
            "SAC.train() should scale action = action * act_high"
        )

    def test_sac_actor_loss_scales_new_actions(self) -> None:
        """new_actions in actor loss must be scaled before passing to critic."""
        from rlox.algorithms.sac import SAC

        src = _source(SAC._update)
        assert "new_actions * self.act_high" in src or "new_actions *self.act_high" in src, (
            "SAC._update() must scale new_actions by act_high in actor loss"
        )


# ---------------------------------------------------------------------------
# TD3
# ---------------------------------------------------------------------------


class TestTD3Fixes:
    """Verify TD3 convergence fixes (Bugs 4-6)."""

    def test_td3_push_includes_next_obs(self) -> None:
        """buffer.push() call in TD3.train() must pass next_obs."""
        from rlox.algorithms.td3 import TD3

        src = _source(TD3.train)
        assert "np.asarray(next_obs" in src, (
            "TD3.train() must pass next_obs to buffer.push()"
        )

    def test_td3_target_uses_next_obs(self) -> None:
        """TD3._update() must use next_obs for target policy and Q computation."""
        from rlox.algorithms.td3 import TD3

        src = _source(TD3._update)
        assert 'batch["next_obs"]' in src
        assert "self.actor_target(next_obs" in src, (
            "TD3._update() must evaluate actor_target on next_obs"
        )
        assert "self.critic1_target(next_obs" in src
        assert "self.critic2_target(next_obs" in src

    def test_td3_critic_target_updates_every_step(self) -> None:
        """Critic polyak updates must happen every step, not only on delayed steps.

        The actor target update should remain inside the policy_delay gate,
        but critic target updates must be outside it.
        """
        from rlox.algorithms.td3 import TD3

        src = _source(TD3._update)
        lines = src.splitlines()

        # Find the lines with polyak_update calls and the policy_delay check
        delay_line_idx = None
        critic1_polyak_idx = None
        critic2_polyak_idx = None
        actor_polyak_idx = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if "self.policy_delay" in stripped and "%" in stripped:
                delay_line_idx = i
            if "polyak_update(self.critic1," in stripped and critic1_polyak_idx is None:
                critic1_polyak_idx = i
            if "polyak_update(self.critic2," in stripped and critic2_polyak_idx is None:
                critic2_polyak_idx = i
            if "polyak_update(self.actor," in stripped:
                actor_polyak_idx = i

        assert delay_line_idx is not None, "Could not find policy_delay check"
        assert critic1_polyak_idx is not None, "Could not find critic1 polyak_update"
        assert critic2_polyak_idx is not None, "Could not find critic2 polyak_update"
        assert actor_polyak_idx is not None, "Could not find actor polyak_update"

        # Critic polyak updates must come BEFORE the delay gate
        assert critic1_polyak_idx < delay_line_idx, (
            "critic1 polyak_update must be outside (before) the policy_delay block"
        )
        assert critic2_polyak_idx < delay_line_idx, (
            "critic2 polyak_update must be outside (before) the policy_delay block"
        )
        # Actor polyak update must come AFTER the delay gate
        assert actor_polyak_idx > delay_line_idx, (
            "actor polyak_update must remain inside the policy_delay block"
        )

    def test_td3_deterministic_policy_scales(self) -> None:
        """DeterministicPolicy.forward() must multiply output by max_action."""
        from rlox.networks import DeterministicPolicy

        src = _source(DeterministicPolicy.forward)
        assert "self.max_action" in src, (
            "DeterministicPolicy.forward() must scale by max_action"
        )


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------


class TestDQNFixes:
    """Verify DQN convergence fixes (Bugs 7-9)."""

    def test_dqn_push_includes_next_obs(self) -> None:
        """_store_transition must accept and forward next_obs to buffer.push()."""
        from rlox.algorithms.dqn import DQN

        sig = inspect.signature(DQN._store_transition)
        param_names = list(sig.parameters.keys())
        assert "next_obs" in param_names, (
            "DQN._store_transition() must accept a next_obs parameter"
        )

        src = _source(DQN._store_transition)
        assert "np.asarray(last_next_obs" in src, (
            "_store_transition must pass next_obs to buffer.push()"
        )

    def test_dqn_train_passes_next_obs_to_store(self) -> None:
        """DQN.train() must pass next_obs to _store_transition."""
        from rlox.algorithms.dqn import DQN

        src = _source(DQN.train)
        assert "_store_transition(obs, action, reward, next_obs" in src, (
            "DQN.train() must pass next_obs to _store_transition"
        )

    def test_dqn_target_uses_next_obs(self) -> None:
        """DQN._update() must use next_obs for target Q computation."""
        from rlox.algorithms.dqn import DQN

        src = _source(DQN._update)
        assert 'batch["next_obs"]' in src
        assert "self.q_network(next_obs)" in src, (
            "Double DQN action selection must use next_obs"
        )
        assert "self.target_network(next_obs)" in src, (
            "Target network evaluation must use next_obs"
        )

    def test_dqn_nstep_flush_uses_actual_flags(self) -> None:
        """N-step flush must use actual done/truncated from the last transition,
        not hardcoded True/False.
        """
        from rlox.algorithms.dqn import DQN

        src = _source(DQN.train)
        # Find the flush section (after "Flush n-step buffer" comment)
        flush_start = src.find("Flush n-step buffer")
        assert flush_start != -1, "Could not find n-step flush section"
        flush_section = src[flush_start:]

        # Should NOT have hardcoded True, False as done/trunc args to push
        # Instead should reference last_done_b / last_trunc_b or similar
        assert "last_done_b" in flush_section or "last_done" in flush_section, (
            "N-step flush must use actual done flag from last transition"
        )
        assert "last_trunc_b" in flush_section or "last_trunc" in flush_section, (
            "N-step flush must use actual truncated flag from last transition"
        )
