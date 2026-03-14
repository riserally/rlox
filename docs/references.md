# References

This document lists the academic papers that rlox implements or builds upon. References are numbered to match citations in the [Mathematical Reference](math-reference.md).

---

## Core Algorithms

**[1]** J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," *arXiv preprint arXiv:1707.06347*, 2017.
- Clipped surrogate objective (PPO-Clip)
- Implemented in: `rlox.algorithms.ppo.PPO`, `rlox.losses.PPOLoss`

**[2]** J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation," in *Proc. ICLR*, 2016.
- GAE: exponentially-weighted average of multi-step TD errors
- Implemented in: `rlox_core::training::gae::compute_gae`, `rlox.compute_gae`

**[3]** T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor," in *Proc. ICML*, 2018, pp. 1861--1870.
- Entropy-regularised RL, twin Q-networks, squashed Gaussian policy
- Implemented in: `rlox.algorithms.sac.SAC`

**[4]** S. Fujimoto, H. van Hoof, and D. Meger, "Addressing Function Approximation Error in Actor-Critic Methods," in *Proc. ICML*, 2018, pp. 1587--1596.
- TD3: twin critics, delayed policy updates, target policy smoothing
- Implemented in: `rlox.algorithms.td3.TD3`

**[5]** V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis, "Human-level control through deep reinforcement learning," *Nature*, vol. 518, no. 7540, pp. 529--533, 2015.
- DQN: deep Q-learning with experience replay and target networks
- Implemented in: `rlox.algorithms.dqn.DQN`

---

## DQN Extensions (Rainbow Components)

**[6]** H. van Hasselt, A. Guez, and D. Silver, "Deep Reinforcement Learning with Double Q-learning," in *Proc. AAAI*, 2016, pp. 2094--2100.
- Double DQN: decoupled action selection and evaluation
- Implemented in: `rlox.algorithms.dqn.DQN` (`double_dqn=True`)

**[7]** Z. Wang, T. Schaul, M. Hessel, H. van Hasselt, M. Lanctot, and N. de Freitas, "Dueling Network Architectures for Deep Reinforcement Learning," in *Proc. ICML*, 2016, pp. 1995--2003.
- Dueling architecture: separate value and advantage streams
- Implemented in: `rlox.networks.DuelingQNetwork`, `rlox.algorithms.dqn.DQN` (`dueling=True`)

**[8]** T. Schaul, J. Quan, I. Antonoglou, and D. Silver, "Prioritized Experience Replay," in *Proc. ICLR*, 2016.
- Sum-tree based proportional prioritisation with importance-sampling correction
- Implemented in: `rlox_core::buffer::priority::PrioritizedReplayBuffer`, `rlox.PrioritizedReplayBuffer`

---

## Distributed and Off-Policy Correction

**[9]** L. Espeholt, H. Soyer, R. Munos, K. Simonyan, V. Mnih, T. Ward, Y. Doron, V. Firoiu, T. Harley, I. Dunning, S. Legg, and K. Kavukcuoglu, "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures," in *Proc. ICML*, 2018, pp. 1407--1416.
- V-trace: clipped importance weight off-policy correction
- Implemented in: `rlox_core::training::vtrace::compute_vtrace`, `rlox.compute_vtrace`

---

## LLM Post-Training

**[10]** R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," in *Proc. NeurIPS*, 2023.
- DPO: bypasses reward modelling by directly optimising from preferences
- Implemented in: `rlox.algorithms.dpo.DPO`, `rlox_core::llm::ops::DPOPair`

**[11]** Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, M. Zhang, Y. K. Li, Y. Wu, and D. Guo, "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," *arXiv preprint arXiv:2402.03300*, 2024.
- GRPO: group-relative policy optimization, eliminates learned value baseline
- Implemented in: `rlox_core::llm::ops::compute_group_advantages`, `rlox.algorithms.grpo.GRPO`

---

## Evaluation Methodology

**[12]** R. Agarwal, M. Schwarzer, P. S. Castro, A. Courville, and M. G. Bellemare, "Deep Reinforcement Learning at the Edge of the Statistical Precipice," in *Proc. NeurIPS*, 2021.
- IQM, performance profiles, stratified bootstrap CIs for RL evaluation
- Implemented in: `rlox.evaluation`

---

## Implementation References

**[13]** V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. Harley, T. P. Lillicrap, D. Silver, and K. Kavukcuoglu, "Asynchronous Methods for Deep Reinforcement Learning," in *Proc. ICML*, 2016, pp. 1928--1937.
- A3C / A2C: synchronous advantage actor-critic
- Implemented in: `rlox.algorithms.a2c.A2C`

**[14]** M. Andrychowicz, A. Raichuk, P. Stanczyk, M. Orsini, S. Girgin, R. Marinier, L. Hussenot, M. Geist, O. Pietquin, M. Michalski, S. Gelly, and O. Bachem, "What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study," *arXiv preprint arXiv:2006.05990*, 2021.
- Empirical analysis of on-policy RL implementation details (orthogonal init, advantage normalisation, etc.)
- Used for: initialisation and training practice choices in `rlox.policies`

---

## Architecture Inspiration

**[15]** Polars contributors, "Polars: Blazingly fast DataFrames," https://pola.rs, 2024.
- Architecture pattern: Rust data plane + Python control plane via PyO3
- rlox applies this pattern to RL: Rust handles environments, buffers, and numerical computation; Python handles training logic and neural networks

**[16]** H. Huang, S. Dossa, C. Ye, J. Braga, D. Chakraborty, K. Mehta, and J. G. Araujo, "CleanRL: High-quality Single-file Implementations of DeepRL Algorithms," *Journal of Machine Learning Research*, vol. 23, no. 274, pp. 1--18, 2022.
- Reference implementations for PPO, A2C, SAC, TD3, DQN hyperparameters and training practices
- rlox's default hyperparameters match CleanRL where applicable
