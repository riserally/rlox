"""Tests for DreamerV3: RSSM, symlog/symexp, sequence replay, continuous actions."""

from __future__ import annotations

import pytest
import torch
import numpy as np


# ---------------------------------------------------------------------------
# symlog / symexp
# ---------------------------------------------------------------------------


class TestSymlogSymexp:
    def test_symlog_zero(self):
        from rlox.algorithms.dreamer import symlog

        x = torch.tensor([0.0])
        assert torch.allclose(symlog(x), torch.tensor([0.0]))

    def test_symlog_positive(self):
        from rlox.algorithms.dreamer import symlog

        x = torch.tensor([1.0, 10.0, 100.0])
        result = symlog(x)
        expected = torch.sign(x) * torch.log1p(torch.abs(x))
        assert torch.allclose(result, expected)

    def test_symlog_negative(self):
        from rlox.algorithms.dreamer import symlog

        x = torch.tensor([-5.0])
        result = symlog(x)
        assert result.item() < 0.0

    def test_symexp_inverts_symlog(self):
        from rlox.algorithms.dreamer import symlog, symexp

        x = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        roundtrip = symexp(symlog(x))
        assert torch.allclose(roundtrip, x, atol=1e-5)

    def test_symexp_zero(self):
        from rlox.algorithms.dreamer import symexp

        x = torch.tensor([0.0])
        assert torch.allclose(symexp(x), torch.tensor([0.0]))


# ---------------------------------------------------------------------------
# RSSM
# ---------------------------------------------------------------------------


class TestRSSM:
    def test_rssm_init(self):
        from rlox.algorithms.dreamer import RSSM

        rssm = RSSM(obs_dim=4, act_dim=2, deter_dim=64, stoch_dim=8, classes=8)
        assert rssm.deter_dim == 64
        assert rssm.stoch_dim == 8
        assert rssm.classes == 8

    def test_rssm_initial_state(self):
        from rlox.algorithms.dreamer import RSSM

        rssm = RSSM(obs_dim=4, act_dim=2, deter_dim=64, stoch_dim=8, classes=8)
        batch = 3
        h, z = rssm.initial_state(batch)
        assert h.shape == (batch, 64)
        assert z.shape == (batch, 8 * 8)

    def test_rssm_observe_step(self):
        from rlox.algorithms.dreamer import RSSM

        rssm = RSSM(obs_dim=4, act_dim=2, deter_dim=64, stoch_dim=8, classes=8)
        batch = 5
        h, z = rssm.initial_state(batch)
        obs = torch.randn(batch, 4)
        action = torch.randn(batch, 2)

        h_new, z_new, z_prior = rssm.observe_step(h, z, action, obs)
        assert h_new.shape == (batch, 64)
        assert z_new.shape == (batch, 8 * 8)
        assert z_prior.shape == (batch, 8 * 8)

    def test_rssm_imagine_step(self):
        from rlox.algorithms.dreamer import RSSM

        rssm = RSSM(obs_dim=4, act_dim=2, deter_dim=64, stoch_dim=8, classes=8)
        batch = 5
        h, z = rssm.initial_state(batch)
        action = torch.randn(batch, 2)

        h_new, z_prior = rssm.imagine_step(h, z, action)
        assert h_new.shape == (batch, 64)
        assert z_prior.shape == (batch, 8 * 8)

    def test_rssm_get_feature(self):
        from rlox.algorithms.dreamer import RSSM

        rssm = RSSM(obs_dim=4, act_dim=2, deter_dim=64, stoch_dim=8, classes=8)
        batch = 3
        h = torch.randn(batch, 64)
        z = torch.randn(batch, 64)  # 8*8=64
        feat = rssm.get_feature(h, z)
        assert feat.shape == (batch, 64 + 64)


# ---------------------------------------------------------------------------
# DreamerV3 single step
# ---------------------------------------------------------------------------


class TestDreamerV3:
    def test_instantiation(self):
        from rlox.algorithms.dreamer import DreamerV3

        agent = DreamerV3(env_id="CartPole-v1", seed=42)
        assert agent is not None

    def test_world_model_is_rssm(self):
        from rlox.algorithms.dreamer import DreamerV3, RSSM

        agent = DreamerV3(env_id="CartPole-v1", seed=42)
        assert isinstance(agent.world_model, RSSM)

    @pytest.mark.slow
    def test_trains_short_run(self):
        from rlox.algorithms.dreamer import DreamerV3

        agent = DreamerV3(env_id="CartPole-v1", seed=42)
        metrics = agent.train(total_timesteps=200)
        assert isinstance(metrics, dict)
        assert "mean_reward" in metrics

    def test_sequence_replay(self):
        """Buffer supports sample_sequences method."""
        from rlox.algorithms.dreamer import DreamerV3

        agent = DreamerV3(env_id="CartPole-v1", seed=42, buffer_size=500)
        # Collect some experience first
        agent._collect_experience(50)

        seqs = agent._sample_sequences(batch_size=2, seq_len=5)
        if seqs is not None:  # may be None if not enough data
            assert "obs" in seqs
            assert seqs["obs"].shape[1] == 5  # seq_len


# ---------------------------------------------------------------------------
# DreamerV3Config
# ---------------------------------------------------------------------------


class TestDreamerV3Config:
    def test_defaults(self):
        from rlox.config import DreamerV3Config

        cfg = DreamerV3Config()
        assert cfg.learning_rate == 1e-4
        assert cfg.deter_dim == 512
        assert cfg.stoch_dim == 32
        assert cfg.stoch_classes == 32
        assert cfg.imagination_horizon == 15
        assert cfg.seq_len == 50

    def test_validation_rejects_bad_values(self):
        from rlox.config import DreamerV3Config

        with pytest.raises(ValueError):
            DreamerV3Config(learning_rate=-1.0)
        with pytest.raises(ValueError):
            DreamerV3Config(batch_size=0)

    def test_from_dict_ignores_unknown(self):
        from rlox.config import DreamerV3Config

        cfg = DreamerV3Config.from_dict({"deter_dim": 256, "foo": "bar"})
        assert cfg.deter_dim == 256

    def test_roundtrip_dict(self):
        from rlox.config import DreamerV3Config

        cfg = DreamerV3Config(deter_dim=256, seq_len=30)
        d = cfg.to_dict()
        loaded = DreamerV3Config.from_dict(d)
        assert loaded.to_dict() == d

    def test_yaml_roundtrip(self, tmp_path):
        from rlox.config import DreamerV3Config

        cfg = DreamerV3Config(stoch_dim=16, stoch_classes=16)
        path = tmp_path / "dreamer.yaml"
        cfg.to_yaml(path)
        loaded = DreamerV3Config.from_yaml(path)
        assert loaded.stoch_dim == 16
        assert loaded.stoch_classes == 16

    def test_toml_roundtrip(self, tmp_path):
        from rlox.config import DreamerV3Config

        cfg = DreamerV3Config(imagination_horizon=20)
        path = tmp_path / "dreamer.toml"
        cfg.to_toml(path)
        loaded = DreamerV3Config.from_toml(path)
        assert loaded.imagination_horizon == 20


# ---------------------------------------------------------------------------
# CNN Encoder / Decoder
# ---------------------------------------------------------------------------


class TestCNNEncoder:
    """Test convolutional encoder for image observations."""

    def test_output_shape_64x64_rgb(self):
        from rlox.algorithms.dreamer import CNNEncoder

        encoder = CNNEncoder(in_channels=3, depth=48)
        x = torch.randn(2, 3, 64, 64)
        out = encoder(x)
        assert out.dim() == 2
        assert out.shape[0] == 2
        # 4 layers of stride-2 => 64 / 16 = 4, depth*8 = 384, so 384*4*4 = 6144
        assert out.shape[1] == 48 * 8 * 4 * 4

    def test_output_shape_grayscale(self):
        from rlox.algorithms.dreamer import CNNEncoder

        encoder = CNNEncoder(in_channels=1, depth=48)
        x = torch.randn(4, 1, 64, 64)
        out = encoder(x)
        assert out.shape == (4, 48 * 8 * 4 * 4)

    def test_nhwc_input_auto_permuted(self):
        from rlox.algorithms.dreamer import CNNEncoder

        encoder = CNNEncoder(in_channels=3, depth=48)
        x = torch.randn(2, 64, 64, 3)  # NHWC format
        out = encoder(x)
        assert out.dim() == 2
        assert out.shape[0] == 2

    def test_custom_depth(self):
        from rlox.algorithms.dreamer import CNNEncoder

        encoder = CNNEncoder(in_channels=3, depth=32)
        x = torch.randn(1, 3, 64, 64)
        out = encoder(x)
        assert out.shape[1] == 32 * 8 * 4 * 4

    def test_non_square_input(self):
        from rlox.algorithms.dreamer import CNNEncoder

        encoder = CNNEncoder(in_channels=3, depth=48)
        x = torch.randn(1, 3, 96, 96)
        out = encoder(x)
        # 96 / 16 = 6, so 384 * 6 * 6 = 13824
        assert out.shape == (1, 48 * 8 * 6 * 6)


class TestCNNDecoder:
    """Test transposed convolutional decoder for image reconstruction."""

    def test_output_shape_matches_input_64(self):
        from rlox.algorithms.dreamer import CNNDecoder

        feat_dim = 128
        decoder = CNNDecoder(feat_dim=feat_dim, out_channels=3, depth=48, img_size=64)
        features = torch.randn(2, feat_dim)
        out = decoder(features)
        assert out.shape == (2, 3, 64, 64)

    def test_output_shape_grayscale(self):
        from rlox.algorithms.dreamer import CNNDecoder

        feat_dim = 128
        decoder = CNNDecoder(feat_dim=feat_dim, out_channels=1, depth=48, img_size=64)
        features = torch.randn(3, feat_dim)
        out = decoder(features)
        assert out.shape == (3, 1, 64, 64)

    def test_output_shape_96(self):
        from rlox.algorithms.dreamer import CNNDecoder

        feat_dim = 256
        decoder = CNNDecoder(feat_dim=feat_dim, out_channels=3, depth=48, img_size=96)
        features = torch.randn(1, feat_dim)
        out = decoder(features)
        assert out.shape == (1, 3, 96, 96)

    def test_encoder_decoder_roundtrip_shape(self):
        """Encoder output fed to decoder should reconstruct original spatial dims."""
        from rlox.algorithms.dreamer import CNNEncoder, CNNDecoder

        encoder = CNNEncoder(in_channels=3, depth=48)
        dummy = torch.randn(1, 3, 64, 64)
        embed_dim = encoder(dummy).shape[-1]

        feat_dim = 128 + embed_dim  # simulating deter + stoch + embed
        decoder = CNNDecoder(feat_dim=feat_dim, out_channels=3, depth=48, img_size=64)
        features = torch.randn(2, feat_dim)
        out = decoder(features)
        assert out.shape == (2, 3, 64, 64)


class TestPreprocessObs:
    """Test image observation preprocessing."""

    def test_normalize_pixel_values(self):
        from rlox.algorithms.dreamer import preprocess_obs

        obs = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8).float()
        result = preprocess_obs(obs, use_cnn=True)
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_adds_batch_dim_if_missing(self):
        from rlox.algorithms.dreamer import preprocess_obs

        obs = torch.randn(3, 64, 64)
        result = preprocess_obs(obs, use_cnn=True)
        assert result.dim() == 4
        assert result.shape[0] == 1

    def test_flat_obs_passthrough(self):
        from rlox.algorithms.dreamer import preprocess_obs

        obs = torch.randn(4, 8)
        result = preprocess_obs(obs, use_cnn=False)
        assert torch.equal(result, obs)


def _has_box2d() -> bool:
    """Check if Box2D is available for CarRacing env."""
    try:
        import Box2D  # noqa: F401
        return True
    except ImportError:
        return False


_skip_no_box2d = pytest.mark.skipif(
    not _has_box2d(), reason="Box2D not installed (pip install 'gymnasium[box2d]')"
)


class TestDreamerV3CNN:
    """Test DreamerV3 instantiation and forward pass with image observations."""

    @_skip_no_box2d
    def test_instantiation_with_image_env(self):
        """DreamerV3 should detect image obs and create CNN encoder/decoder."""
        from rlox.algorithms.dreamer import DreamerV3, CNNEncoder, CNNDecoder

        agent = DreamerV3(env_id="CarRacing-v3", seed=42)
        assert agent._use_cnn is True
        assert isinstance(agent.encoder, CNNEncoder)
        assert isinstance(agent.decoder, CNNDecoder)

    def test_mlp_path_unchanged(self):
        """Flat observation envs should still use MLP path."""
        from rlox.algorithms.dreamer import DreamerV3

        agent = DreamerV3(env_id="CartPole-v1", seed=42)
        assert agent._use_cnn is False
        assert not hasattr(agent, "encoder")

    @_skip_no_box2d
    def test_encoder_embed_dim_computed(self):
        """The embed_dim should be computed from a dummy forward pass."""
        from rlox.algorithms.dreamer import DreamerV3

        agent = DreamerV3(env_id="CarRacing-v3", seed=42)
        # embed_dim should be positive and match encoder output
        assert agent._embed_dim > 0
