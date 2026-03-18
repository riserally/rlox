# Core Rust Primitives (PyO3)

High-performance data-plane primitives compiled from Rust via PyO3: environments, experience buffers, GAE computation, and LLM post-training operations.

## Environments

::: rlox.CartPole

::: rlox.VecEnv

::: rlox.GymEnv

## Experience Storage

::: rlox.ExperienceTable

::: rlox.ReplayBuffer

::: rlox.PrioritizedReplayBuffer

::: rlox.VarLenStore

## Advantage Estimation

::: rlox.compute_gae

::: rlox.compute_vtrace

## Running Statistics

::: rlox.RunningStats

## LLM Post-Training Operations

::: rlox.compute_group_advantages

::: rlox.compute_batch_group_advantages

::: rlox.compute_token_kl

::: rlox.compute_token_kl_schulman

::: rlox.DPOPair

::: rlox.pack_sequences

## Neural Network Backend

::: rlox.ActorCritic

## Python Layer

::: rlox.batch.RolloutBatch

::: rlox.collectors.RolloutCollector

::: rlox.gym_vec_env.GymVecEnv

::: rlox.losses.PPOLoss

::: rlox.diagnostics.TrainingDiagnostics

::: rlox.checkpoint.Checkpoint

::: rlox.hub.push_to_hub

::: rlox.hub.load_from_hub

::: rlox.compile.compile_policy
