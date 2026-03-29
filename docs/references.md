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

## Multi-Agent RL

**[17]** C. Yu, A. Velu, E. Vinitsky, J. Gao, Y. Wang, A. Baez, B. Awbi, et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games," in *Proc. NeurIPS*, 2022.
- MAPPO: PPO with centralized critic for multi-agent cooperation
- Implemented in: `rlox.algorithms.mappo.MAPPO`
- https://arxiv.org/abs/2103.01955

---

## Model-Based RL

**[18]** D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap, "Mastering Diverse Domains through World Models," *arXiv preprint arXiv:2301.04104*, 2023.
- DreamerV3: world model with actor-critic in latent space
- Implemented in: `rlox.algorithms.dreamer.DreamerV3`
- https://arxiv.org/abs/2301.04104

---

## Offline RL

**[19]** S. Fujimoto and S. S. Gu, "A Minimalist Approach to Offline Reinforcement Learning," in *Proc. NeurIPS*, 2021.
- TD3+BC: TD3 with behavioral cloning regularization
- Implemented in: `rlox.algorithms.td3_bc.TD3BC`
- https://arxiv.org/abs/2106.06860

**[20]** I. Kostrikov, A. Nair, and S. Levine, "Offline Reinforcement Learning with Implicit Q-Learning," in *Proc. ICLR*, 2022.
- IQL: avoids OOD actions via expectile regression on the value function
- Implemented in: `rlox.algorithms.iql.IQL`
- https://arxiv.org/abs/2110.06169

**[21]** A. Kumar, A. Zhou, G. Tucker, and S. Levine, "Conservative Q-Learning for Offline Reinforcement Learning," in *Proc. NeurIPS*, 2020.
- CQL: penalizes Q-values for out-of-distribution actions
- Implemented in: `rlox.algorithms.cql.CQL`
- https://arxiv.org/abs/2006.04779

---

## Imitation Learning

**[22]** M. Bain and C. Sammut, "A Framework for Behavioural Cloning," in *Machine Intelligence 15*, pp. 103--129, 1995.
- Behavioral cloning: supervised learning on expert demonstrations
- Implemented in: `rlox.algorithms.bc.BC`

---

## LLM Alignment (Additional)

**[23]** Z. Guo, A. Rashid, B. Suber, S. Sharma, D. Sui, et al., "Direct Language Model Alignment from Online AI Feedback," *arXiv preprint arXiv:2402.04792*, 2024.
- Online DPO: extends DPO to the online generation setting
- Implemented in: `rlox.algorithms.online_dpo.OnlineDPO`
- https://arxiv.org/abs/2402.04792

**[24]** Y. Nakano, J. Hilton, S. Balaji, J. Wu, et al., "WebGPT: Browser-assisted question-answering with human feedback," *arXiv preprint arXiv:2112.09332*, 2021.
- Best-of-N sampling as an alignment baseline
- Implemented in: `rlox.algorithms.best_of_n.BestOfN`
- https://arxiv.org/abs/2112.09332

**[25]** A. Gao, J. Schulman, and J. Hilton, "Scaling Laws for Reward Model Overoptimization," in *Proc. ICML*, 2023.
- Analysis of best-of-N vs RL fine-tuning overoptimization
- https://arxiv.org/abs/2210.10760

---

## Evaluation Methodology

**[26]** R. Agarwal, M. Schwarzer, P. S. Castro, A. Courville, and M. G. Bellemare, "Deep Reinforcement Learning at the Edge of the Statistical Precipice," in *Proc. NeurIPS*, 2021.
- IQM, performance profiles, stratified bootstrap CIs for RL evaluation
- Implemented in: `rlox.evaluation`

---

## Implementation References

**[27]** V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. Harley, T. P. Lillicrap, D. Silver, and K. Kavukcuoglu, "Asynchronous Methods for Deep Reinforcement Learning," in *Proc. ICML*, 2016, pp. 1928--1937.
- A3C / A2C: synchronous advantage actor-critic
- Implemented in: `rlox.algorithms.a2c.A2C`

**[28]** M. Andrychowicz, A. Raichuk, P. Stanczyk, M. Orsini, S. Girgin, R. Marinier, L. Hussenot, M. Geist, O. Pietquin, M. Michalski, S. Gelly, and O. Bachem, "What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study," *arXiv preprint arXiv:2006.05990*, 2021.
- Empirical analysis of on-policy RL implementation details (orthogonal init, advantage normalisation, etc.)
- Used for: initialisation and training practice choices in `rlox.policies`

---

## Architecture Inspiration

**[29]** Polars contributors, "Polars: Blazingly fast DataFrames," https://pola.rs, 2024.
- Architecture pattern: Rust data plane + Python control plane via PyO3
- rlox applies this pattern to RL: Rust handles environments, buffers, and numerical computation; Python handles training logic and neural networks

**[30]** H. Huang, S. Dossa, C. Ye, J. Braga, D. Chakraborty, K. Mehta, and J. G. Araujo, "CleanRL: High-quality Single-file Implementations of DeepRL Algorithms," *Journal of Machine Learning Research*, vol. 23, no. 274, pp. 1--18, 2022.
- Reference implementations for PPO, A2C, SAC, TD3, DQN hyperparameters and training practices
- rlox's default hyperparameters match CleanRL where applicable
