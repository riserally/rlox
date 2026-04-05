# Convergence Benchmark v6 — Results Summary

**Date:** 2026-04-05
**Code:** rlox v1.0.0+ (post convergence fixes)
**Hardware:** GCP e2-standard-8 (8 vCPU, 32GB RAM), CPU-only
**Status:** 21/32 completed, running rlox SAC Walker2d

---

## v6 Results (post-fix) vs v5 (pre-fix) vs SB3

| Algorithm | Environment | Steps | v5 rlox (pre-fix) | v6 rlox (post-fix) | SB3 | Change | Winner |
|-----------|-------------|-------|--------------------|--------------------|-----|--------|--------|
| **PPO** | CartPole-v1 | 100K | 454 | 410 | 421 | -10% | ~Tied |
| **PPO** | Acrobot-v1 | 500K | -89 | **-87** | -118 | +2% | **rlox** |
| **PPO** | HalfCheetah-v4 | 1M | 4,226 | 1,365* | 3,143 | -68%* | SB3 |
| **PPO** | **Hopper-v4** | 1M | **628** | **3,342** | 3,578 | **+432%** | ~SB3 |
| **PPO** | Walker2d-v4 | 2M | 5,007 | 4,055 | 4,384 | -19% | ~Tied |
| **A2C** | CartPole-v1 | 100K | 54 | **500** | 500 | **+827%** | **Tied** |
| **DQN** | CartPole-v1 | 100K | 165 | 249 | 196 | +51% | **rlox** |
| **DQN** | MountainCar-v0 | 500K | -179 | -196 | -110 | -10% | SB3 |
| **SAC** | Pendulum-v1 | 50K | -169 | -167 | -167 | +1% | Tied |
| **SAC** | HalfCheetah-v4 | 1M | 11,468 | 10,711 | 10,563 | -7% | **rlox** |
| **SAC** | Hopper-v4 | 1M | 3,291 | 3,301 | 3,170 | 0% | **rlox** |
| **TD3** | Pendulum-v1 | 50K | -163 | -167 | -169 | -2% | Tied |
| **TD3** | HalfCheetah-v4 | 1M | 10,400 | 9,555 | 9,899 | -8% | ~Tied |

*PPO HalfCheetah: v6 result (1,365) is significantly lower than v5 (4,226) and SB3 (3,143).
This may be a regression from the VecNormalize changes or a seed-dependent result. Needs investigation.

## Key Findings

### Fix Validated
- **PPO Hopper: 628 → 3,342** (+432%) — matches SB3's 3,578. Truncation bootstrap + VecNormalize working.
- **A2C CartPole: 54 → 500** — advantage normalization fix validated.
- **DQN CartPole: 165 → 249** — improved.

### Regression Found
- **PPO HalfCheetah: 4,226 → 1,365** — significant drop. Needs investigation.
  Possible causes: VecNormalize interaction with HalfCheetah, or seed variance.

### Consistent
- SAC and TD3 results similar across v5/v6 (expected — fixes targeted on-policy collector).
- Pendulum results unchanged (no normalization used).

## SPS Comparison (v6)

| Algorithm | Environment | rlox SPS | SB3 SPS | Speedup |
|-----------|-------------|----------|---------|---------|
| PPO | CartPole | 1,897 | 750 | **2.5x** |
| PPO | Acrobot | 3,206 | 1,538 | **2.1x** |
| PPO | HalfCheetah | 951 | 516 | **1.8x** |
| PPO | Hopper | 862 | 660 | **1.3x** |
| PPO | Walker2d | 1,119 | 730 | **1.5x** |
| A2C | CartPole | 1,678 | 716 | **2.3x** |
| SAC | HalfCheetah | 63 | 53 | **1.2x** |
| SAC | Hopper | 53 | 53 | 1.0x |
| TD3 | HalfCheetah | 86 | 72 | **1.2x** |

## Still Running

- rlox SAC Walker2d (step 270K/2M) — return 1,894, climbing
- Remaining: ~8 experiments (SAC/TD3 Walker2d, PPO/SAC Ant, SAC Humanoid)

## Issues for Paper

1. **PPO HalfCheetah regression needs investigation** before paper submission
2. **rlox PPO results missing from JSON** — `.close()` bug caused crash during eval
   (fixed in latest code, but v6 ran old code). Log data is complete.
3. **Single seed** — need 5+ seeds for IQM/bootstrap CI
4. **No x86 baseline** — all results on GCP e2 (x86 Xeon), which is good for paper
