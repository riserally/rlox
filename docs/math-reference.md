# rlox Mathematical Reference

This document provides the mathematical formulations for every algorithm and computation implemented in rlox. Equations are numbered for cross-referencing.

## Notation

| Symbol | Meaning |
|--------|---------|
| $s_t, a_t$ | State and action at time $t$ |
| $r_t$ | Reward at time $t$ |
| $\gamma \in [0, 1)$ | Discount factor |
| $\pi_\theta(a \mid s)$ | Policy parameterised by $\theta$ |
| $V^\pi(s)$ | State value function under policy $\pi$ |
| $Q^\pi(s, a)$ | State-action value function under policy $\pi$ |
| $A^\pi(s, a)$ | Advantage function: $Q^\pi(s,a) - V^\pi(s)$ |
| $\hat{A}_t$ | Estimated advantage at time $t$ |
| $\mathcal{H}[\pi]$ | Entropy of the policy: $-\mathbb{E}[\log \pi]$ |
| $D_\text{KL}$ | Kullback-Leibler divergence |
| $\tau$ | Polyak averaging coefficient |

---

## 1. Generalized Advantage Estimation (GAE)

**Reference**: Schulman et al. (2016) [2]

**Implementation**: `rlox_core::training::gae::compute_gae` (Rust), `rlox.compute_gae` (Python)

### Derivation

The TD residual at time $t$ is:

$$\delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t) \tag{1}$$

where $d_t \in \{0, 1\}$ is the episode termination flag. GAE defines the advantage estimator as an exponentially-weighted average of $k$-step TD errors:

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \delta_{t+l} \prod_{k=0}^{l-1}(1 - d_{t+k}) \tag{2}$$

This is computed via the backward recursion:

$$\hat{A}_t = \delta_t + \gamma \lambda (1 - d_t) \hat{A}_{t+1} \tag{3}$$

with $\hat{A}_T = 0$. The return target is:

$$\hat{R}_t = \hat{A}_t + V(s_t) \tag{4}$$

### Special Cases

- $\lambda = 0$: $\hat{A}_t = \delta_t$ (one-step TD error, low variance, high bias)
- $\lambda = 1$: $\hat{A}_t = \sum_{l=0}^{T-t-1} \gamma^l r_{t+l} - V(s_t)$ (Monte Carlo return minus baseline, high variance, low bias)
- $\lambda \in (0,1)$: Interpolates between the two

### Complexity

- Time: $O(T)$ per trajectory (single backward pass)
- Space: $O(T)$ for the advantages vector

---

## 2. Proximal Policy Optimization (PPO)

**Reference**: Schulman et al. (2017) [1]

**Implementation**: `rlox.losses.PPOLoss` (Python), `rlox.algorithms.ppo.PPO`

### Clipped Surrogate Objective

Let $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$ be the probability ratio. The clipped objective is:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right] \tag{5}$$

where $\epsilon$ is the clip range (default 0.2).

### Value Loss

With optional clipping (matching CleanRL):

$$L^V(\theta) = \frac{1}{2}\mathbb{E}_t\left[\max\left((V_\theta(s_t) - \hat{R}_t)^2, \; (V_\text{clip}(s_t) - \hat{R}_t)^2\right)\right] \tag{6}$$

where $V_\text{clip}(s_t) = V_{\theta_\text{old}}(s_t) + \text{clip}(V_\theta(s_t) - V_{\theta_\text{old}}(s_t), -\epsilon, \epsilon)$.

Without clipping: $L^V(\theta) = \frac{1}{2}\mathbb{E}_t\left[(V_\theta(s_t) - \hat{R}_t)^2\right]$.

### Entropy Bonus

$$L^H(\theta) = \mathbb{E}_t\left[\mathcal{H}[\pi_\theta(\cdot \mid s_t)]\right] \tag{7}$$

### Total Loss

$$L(\theta) = -L^{\text{CLIP}}(\theta) + c_v L^V(\theta) - c_h L^H(\theta) \tag{8}$$

Default coefficients: $c_v = 0.5$, $c_h = 0.01$.

### Diagnostics

Approximate KL divergence (from ratio):

$$\hat{D}_\text{KL} = \mathbb{E}_t\left[(r_t(\theta) - 1) - \log r_t(\theta)\right] \tag{9}$$

Clip fraction: $\mathbb{E}_t\left[\mathbf{1}[|r_t(\theta) - 1| > \epsilon]\right]$.

### Training Procedure

For each update:
1. Collect $n_\text{envs} \times n_\text{steps}$ transitions using $\pi_{\theta_\text{old}}$
2. Compute GAE advantages using Eq. (3)
3. For $K$ epochs (default 4):
   - Shuffle and split into minibatches
   - Normalise advantages per minibatch: $\hat{A}_t \leftarrow (\hat{A}_t - \bar{A}) / (\sigma_A + 10^{-8})$
   - Compute loss (Eq. 8) and update $\theta$ via Adam
   - Clip gradients: $\|\nabla\|_2 \leq 0.5$
4. Linearly anneal learning rate (optional)

---

## 3. Advantage Actor-Critic (A2C)

**Reference**: Mnih et al. (2016) [13]

**Implementation**: `rlox.algorithms.a2c.A2C`

A2C is the synchronous variant of A3C. It uses the same advantage estimation as PPO but without ratio clipping or multiple epochs.

### Policy Gradient

$$\nabla_\theta J(\theta) = \mathbb{E}_t\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \hat{A}_t\right] \tag{10}$$

### Total Loss

$$L(\theta) = -\mathbb{E}_t\left[\log \pi_\theta(a_t \mid s_t) \hat{A}_t\right] + c_v \cdot \frac{1}{2}\mathbb{E}_t\left[(V_\theta(s_t) - \hat{R}_t)^2\right] - c_h \cdot \mathcal{H}[\pi_\theta] \tag{11}$$

Default: $c_v = 0.5$, $c_h = 0.01$, optimised with RMSprop.

### Key Differences from PPO

| | PPO | A2C |
|---|-----|-----|
| Clipping | Yes (Eq. 5) | No |
| Epochs per rollout | $K$ (typically 4) | 1 |
| GAE lambda | 0.95 | 1.0 (Monte Carlo) |
| Optimizer | Adam | RMSprop |
| n_steps | 128 | 5 |

---

## 4. Soft Actor-Critic (SAC)

**Reference**: Haarnoja et al. (2018) [3]

**Implementation**: `rlox.algorithms.sac.SAC`

### Entropy-Regularised Objective

SAC maximises the maximum-entropy objective:

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi}\left[r(s_t, a_t) + \alpha \mathcal{H}[\pi(\cdot \mid s_t)]\right] \tag{12}$$

where $\alpha$ is the temperature parameter controlling the entropy-reward tradeoff.

### Soft Bellman Equation

The soft Q-function satisfies:

$$Q^\pi(s, a) = r(s, a) + \gamma \mathbb{E}_{s'}\left[V^\pi(s')\right] \tag{13}$$

$$V^\pi(s) = \mathbb{E}_{a \sim \pi}\left[Q^\pi(s, a) - \alpha \log \pi(a \mid s)\right] \tag{14}$$

### Critic Loss (Twin Q-Networks)

Two independent Q-networks are trained with:

$$L_Q(\phi_i) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{B}}\left[\left(Q_{\phi_i}(s,a) - y\right)^2\right], \quad i \in \{1, 2\} \tag{15}$$

$$y = r + \gamma (1 - d) \left(\min_{i=1,2} Q_{\bar{\phi}_i}(s', \tilde{a}') - \alpha \log \pi_\theta(\tilde{a}' \mid s')\right) \tag{16}$$

where $\tilde{a}' \sim \pi_\theta(\cdot \mid s')$ and $\bar{\phi}_i$ are target network parameters.

### Actor Loss

$$L_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{B}}\left[\alpha \log \pi_\theta(\tilde{a} \mid s) - \min_{i=1,2} Q_{\phi_i}(s, \tilde{a})\right] \tag{17}$$

where $\tilde{a} \sim \pi_\theta(\cdot \mid s)$ via the reparameterisation trick.

### Squashed Gaussian Policy

Actions are sampled as $a = \tanh(\mu(s) + \sigma(s) \odot \xi)$, $\xi \sim \mathcal{N}(0, I)$. The log-probability with the Jacobian correction is:

$$\log \pi(a \mid s) = \log \mathcal{N}(u; \mu, \sigma^2) - \sum_{i=1}^{d} \log(1 - a_i^2 + \epsilon) \tag{18}$$

### Automatic Entropy Tuning

The temperature $\alpha$ is optimised to satisfy a target entropy $\bar{\mathcal{H}}$:

$$L(\alpha) = -\alpha \mathbb{E}_{a \sim \pi}\left[\log \pi(a \mid s) + \bar{\mathcal{H}}\right] \tag{19}$$

Default target entropy: $\bar{\mathcal{H}} = -\dim(\mathcal{A})$.

### Soft Target Update

$$\bar{\phi} \leftarrow \tau \phi + (1 - \tau) \bar{\phi}, \quad \tau = 0.005 \tag{20}$$

---

## 5. Twin Delayed DDPG (TD3)

**Reference**: Fujimoto et al. (2018) [4]

**Implementation**: `rlox.algorithms.td3.TD3`

TD3 addresses overestimation bias in DDPG with three techniques:

### Twin Critics

Same as SAC (Eq. 15-16), but with a deterministic target policy:

$$y = r + \gamma (1 - d) \min_{i=1,2} Q_{\bar{\phi}_i}(s', \pi_{\bar{\theta}}(s') + \epsilon) \tag{21}$$

### Target Policy Smoothing

Adds clipped noise to the target action:

$$\epsilon \sim \text{clip}(\mathcal{N}(0, \sigma^2), -c, c) \tag{22}$$

Default: $\sigma = 0.2$, $c = 0.5$.

### Delayed Policy Updates

The actor and target networks are updated every $d$ critic updates (default $d = 2$):

$$L_\pi(\theta) = -\mathbb{E}_{s \sim \mathcal{B}}\left[Q_{\phi_1}(s, \pi_\theta(s))\right] \tag{23}$$

---

## 6. Deep Q-Network (DQN)

**Reference**: Mnih et al. (2015) [5]

**Implementation**: `rlox.algorithms.dqn.DQN`

### Bellman Equation

The Q-function satisfies:

$$Q^*(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right] \tag{24}$$

### DQN Loss

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{B}}\left[\left(Q_\theta(s,a) - y\right)^2\right] \tag{25}$$

$$y = r + \gamma^n (1 - d) \max_{a'} Q_{\bar{\theta}}(s', a') \tag{26}$$

where $n$ is the N-step return horizon and $\bar{\theta}$ are target network parameters updated every $K$ steps (hard copy).

### Double DQN

**Reference**: van Hasselt et al. (2016) [6]

Decouples action selection from evaluation to reduce overestimation:

$$y = r + \gamma^n (1 - d) \; Q_{\bar{\theta}}\!\left(s', \underset{a'}{\arg\max}\; Q_\theta(s', a')\right) \tag{27}$$

### Dueling Architecture

**Reference**: Wang et al. (2016) [7]

Decomposes Q into value and advantage streams:

$$Q_\theta(s, a) = V_\theta(s) + A_\theta(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A_\theta(s, a') \tag{28}$$

### N-Step Returns

Instead of single-step bootstrapping, uses the $n$-step return:

$$R_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n (1 - d_{t+n-1}) \max_{a'} Q_{\bar{\theta}}(s_{t+n}, a') \tag{29}$$

### Prioritized Experience Replay (PER)

**Reference**: Schaul et al. (2016) [8]

Samples transitions proportional to their TD error magnitude:

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \quad p_i = |\delta_i| + \epsilon \tag{30}$$

Importance-sampling weights correct the bias:

$$w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta \bigg/ \max_j w_j \tag{31}$$

where $\beta$ is annealed from $\beta_0$ (default 0.4) to 1.0 over training.

**Implementation**: `rlox_core::buffer::priority` uses a sum-tree for $O(\log N)$ sampling and priority updates.

### Epsilon-Greedy Exploration

$$a = \begin{cases} \arg\max_{a'} Q_\theta(s, a') & \text{with probability } 1 - \epsilon \\ \text{Uniform}(\mathcal{A}) & \text{with probability } \epsilon \end{cases} \tag{32}$$

$\epsilon$ is linearly decayed from $\epsilon_0$ (default 1.0) to $\epsilon_f$ (default 0.05) over the first fraction of training.

---

## 7. V-trace

**Reference**: Espeholt et al. (2018) [9]

**Implementation**: `rlox_core::training::vtrace::compute_vtrace` (Rust), `rlox.compute_vtrace` (Python)

V-trace provides off-policy correction for the IMPALA architecture.

### Importance Weights

Let $\rho_t = \frac{\pi(a_t \mid s_t)}{\mu(a_t \mid s_t)}$ be the importance ratio, where $\mu$ is the behaviour policy. Define clipped weights:

$$\bar{\rho}_t = \min(\bar{\rho}, \rho_t), \quad \bar{c}_t = \min(\bar{c}, \rho_t) \tag{33}$$

### V-trace Target

$$v_s = V(s) + \sum_{t=s}^{s+n-1} \gamma^{t-s} \left(\prod_{i=s}^{t-1} \bar{c}_i\right) \delta_t V \tag{34}$$

where the temporal difference is:

$$\delta_t V = \bar{\rho}_t (r_t + \gamma V(s_{t+1}) - V(s_t)) \tag{35}$$

### Backward Recursion

The implementation computes V-trace via backward iteration:

$$v_t = V(s_t) + \delta_t + \gamma \bar{c}_t (v_{t+1} - V(s_{t+1})) \tag{36}$$

Policy gradient advantages:

$$\hat{A}_t^{\text{pg}} = \bar{\rho}_t \left(r_t + \gamma v_{t+1} - V(s_t)\right) \tag{37}$$

### Properties

- $\bar{\rho} = \bar{c} = \infty$: on-policy, equivalent to GAE($\lambda=1$)
- $\bar{\rho} = \bar{c} = 1$: default, limits variance while allowing some off-policy correction
- $\bar{\rho}$ controls the bias of the value function fixed point
- $\bar{c}$ controls the speed of convergence (trace cutting)

---

## 8. Group Relative Policy Optimization (GRPO)

**Reference**: Shao et al. (2024) [11]

**Implementation**: `rlox_core::llm::ops::compute_group_advantages` (Rust), `rlox.algorithms.grpo.GRPO` (Python)

### Group-Relative Advantages

For a prompt $x$, generate $G$ completions $\{y_1, \ldots, y_G\}$ and compute rewards $\{r_1, \ldots, r_G\}$. The advantage for completion $i$ is:

$$\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r + \epsilon} \tag{38}$$

where $\bar{r} = \frac{1}{G}\sum_{j=1}^G r_j$ and $\sigma_r = \sqrt{\frac{1}{G}\sum_{j=1}^G (r_j - \bar{r})^2}$.

If $\sigma_r < 10^{-8}$ (constant rewards), all advantages are set to zero.

### GRPO Loss

$$L(\theta) = -\frac{1}{G}\sum_{i=1}^G \hat{A}_i \sum_{t} \log \pi_\theta(y_{i,t} \mid x, y_{i,<t}) + \beta \cdot D_\text{KL}[\pi_\theta \| \pi_\text{ref}] \tag{39}$$

The KL penalty prevents the policy from drifting too far from the reference model.

### Key Difference from PPO

GRPO eliminates the need for a learned value function. Instead of $V(s)$ as a baseline, it uses the group mean reward. This is particularly suited for LLM post-training where:
- Episodes are single-turn (generate once, score once)
- The reward function is an external model (e.g. reward model, verifier)
- Training a value head for language models is expensive

---

## 9. Direct Preference Optimization (DPO)

**Reference**: Rafailov et al. (2023) [10]

**Implementation**: `rlox.algorithms.dpo.DPO`

### Bradley-Terry Preference Model

Given chosen completion $y_w$ and rejected completion $y_l$ for prompt $x$, the preference probability under the Bradley-Terry model is:

$$p^*(y_w \succ y_l \mid x) = \sigma(r^*(x, y_w) - r^*(x, y_l)) \tag{40}$$

where $\sigma$ is the sigmoid function and $r^*$ is the ground-truth reward.

### From RLHF to DPO

The optimal policy under KL-constrained reward maximisation has the form:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_\text{ref}(y \mid x) \exp\left(\frac{1}{\beta} r^*(x, y)\right) \tag{41}$$

Solving for the reward and substituting into the Bradley-Terry model yields:

$$p^*(y_w \succ y_l \mid x) = \sigma\left(\beta \log \frac{\pi^*(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}\right) \tag{42}$$

### DPO Loss

$$L_\text{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\left(\beta \left(\log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}\right)\right)\right] \tag{43}$$

where $\beta$ is the temperature parameter (default 0.1).

### Implicit Reward

DPO implicitly defines a reward:

$$\hat{r}(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_\text{ref}(y \mid x)} + \beta \log Z(x) \tag{44}$$

The diagnostic metrics track `chosen_reward` and `rejected_reward` as $\beta \cdot \mathbb{E}[\log(\pi_\theta / \pi_\text{ref})]$ for chosen and rejected completions respectively. A well-trained model should have `chosen_reward > rejected_reward`.

### Sequence Log-Probabilities

For a sequence $y = (y_1, \ldots, y_T)$:

$$\log \pi_\theta(y \mid x) = \sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{<t}) \tag{45}$$

---

## 10. Token-Level KL Divergence

**Implementation**: `rlox_core::llm::ops::compute_token_kl` (Rust), `rlox.compute_token_kl` (Python)

The forward KL divergence at the token level:

$$D_\text{KL}(p \| q) = \sum_{t=1}^{T} p(y_t) \left(\log p(y_t) - \log q(y_t)\right) = \sum_{t=1}^{T} \exp(\log p_t) \cdot (\log p_t - \log q_t) \tag{46}$$

where $\log p_t$ and $\log q_t$ are the per-token log-probabilities under the policy and reference model respectively.

**Properties**:
- $D_\text{KL}(p \| p) = 0$ (identical distributions)
- $D_\text{KL}(p \| q) \geq 0$ (Gibbs' inequality)
- Not symmetric: $D_\text{KL}(p \| q) \neq D_\text{KL}(q \| p)$

Used as a regularisation penalty in GRPO (Eq. 39) and general RLHF training to prevent policy collapse.

---

## 11. Bootstrap Confidence Intervals and IQM

**Reference**: Agarwal et al. (2021) [12]

**Implementation**: `rlox.evaluation`

### Interquartile Mean (IQM)

A robust measure of central tendency that discards the bottom and top 25% of scores:

$$\text{IQM}(X) = \frac{1}{\lfloor 0.5n \rfloor} \sum_{i=\lceil 0.25n \rceil + 1}^{\lfloor 0.75n \rfloor} X_{(i)} \tag{47}$$

where $X_{(i)}$ are the order statistics. IQM is more robust to outliers than the mean while being more statistically efficient than the median.

### Stratified Bootstrap Confidence Interval

For a set of scores $\{x_1, \ldots, x_n\}$:

1. Draw $B$ bootstrap resamples $\{x_1^*, \ldots, x_n^*\}$ with replacement
2. Compute $\bar{x}^*_b = \frac{1}{n}\sum_i x_{i,b}^*$ for each resample $b$
3. The $(1 - \alpha)$ confidence interval is $[\bar{x}^*_{(\alpha/2)}, \bar{x}^*_{(1-\alpha/2)}]$

Default: $B = 10{,}000$ resamples, 95% CI.

### Performance Profiles

The performance profile of algorithm $A$ is:

$$F_A(\tau) = \frac{1}{M} \sum_{m=1}^{M} \mathbf{1}\left[\text{score}_{A,m} \geq \tau\right] \tag{48}$$

This gives the fraction of runs where algorithm $A$ achieves at least score $\tau$, aggregating across environments and seeds.

---

## 12. Polyak (Soft) Target Update

Used by SAC, TD3, and other off-policy algorithms:

$$\theta_\text{target} \leftarrow \tau \theta + (1 - \tau) \theta_\text{target} \tag{49}$$

**Implementation**: `rlox.networks.polyak_update`

Default $\tau = 0.005$. This is equivalent to an exponential moving average of the online parameters, providing a slowly-evolving target that stabilises training.

---

## 13. Orthogonal Initialisation

**Reference**: Andrychowicz et al. (2021) [14]

**Implementation**: `rlox.policies._orthogonal_init`

Policy networks use orthogonal weight initialisation with gain-dependent scaling:
- Hidden layers: gain $= \sqrt{2}$ (for ReLU/Tanh activations)
- Policy head: gain $= 0.01$ (encourages near-uniform initial distribution)
- Value head: gain $= 1.0$

---

## Cross-References

- [Rust User Guide](rust-guide.md) -- code-level documentation of each implementation
- [Python User Guide](python-guide.md) -- API usage examples
- [References](references.md) -- full academic citations for all referenced papers
