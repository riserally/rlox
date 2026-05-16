# Config API

Configuration dataclasses for all algorithms. Every config class provides
`from_dict()`, `merge()`, and `to_dict()` helpers, plus TOML/YAML loading
via `TrainingConfig`.

## Base

::: rlox.config.ConfigMixin

## Unified

::: rlox.config.TrainingConfig

## On-Policy

::: rlox.config.PPOConfig

::: rlox.config.A2CConfig

::: rlox.config.VPGConfig

::: rlox.config.TRPOConfig

## Off-Policy

::: rlox.config.SACConfig

::: rlox.config.TD3Config

::: rlox.config.DQNConfig

::: rlox.config.MPOConfig

## Multi-Agent

::: rlox.config.MAPPOConfig

::: rlox.config.QMIXConfig

## Model-Based

::: rlox.config.DreamerV3Config

## Distributed

::: rlox.config.IMPALAConfig

## Offline RL

::: rlox.config.DecisionTransformerConfig

::: rlox.config.CalQLConfig

## Exploration & Meta

::: rlox.config.DiffusionPolicyConfig

::: rlox.config.DTPConfig

::: rlox.config.SelfPlayConfig

::: rlox.config.GoExploreConfig

::: rlox.config.PBTConfig
