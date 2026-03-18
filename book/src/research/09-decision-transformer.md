# Decision Transformer / Offline RL

> Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling," NeurIPS, 2021.
> Janner et al., "Offline Reinforcement Learning as One Big Sequence Modeling Problem," NeurIPS, 2021.

## Key Idea

Decision Transformer reframes RL as a sequence modeling problem. Instead of learning value functions or policy gradients, it uses a causal Transformer (GPT-style) to predict actions conditioned on desired returns, past states, and past actions. At test time, conditioning on a high return-to-go elicits high-performing behavior. This leverages Transformer power and sidesteps TD-learning instabilities, though it requires a pre-collected offline dataset.

## Mathematical Formulation

**Input sequence (context window of length K):**

```
τ = (R̂_1, s_1, a_1, R̂_2, s_2, a_2, ..., R̂_T, s_T, a_T)

where R̂_t = Σ_{k=t}^{T} r_k  (return-to-go)
```

**Training objective (autoregressive on actions):**

```
L(θ) = E_{τ~D} [ Σ_t -log π_θ(a_t | R̂_t, s_t, R̂_{t-1}, s_{t-1}, a_{t-1}, ...) ]
```

**At inference:**

```
a_t = π_θ(· | R̂_t^desired, s_t, past context)
R̂_{t+1} = R̂_t^desired - r_t  (update return-to-go)
```

The model applies a GPT-style Transformer with causal masking over interleaved (R̂, s, a) tokens.

## Properties

- Offline (trains on pre-collected dataset)
- Neither value-based nor policy-gradient — sequence modeling
- Model-free (though the Transformer implicitly models dynamics)
- Return-conditioned: desired return acts as a "goal"

## Key Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Context length K | 20 | (R̂,s,a) triples |
| Transformer layers | 3–6 | GPT-2 style |
| Attention heads | 4–8 | |
| Embedding dim | 128–256 | |
| Learning rate | 1e-4 (Adam) | |
| Batch size | 64 | Sequences |
| Dropout | 0.1 | |
| Return-to-go (test) | Max in dataset | Or higher for extrapolation |

## Complexity

- **Training:** Standard Transformer cost O(K² × d_model) per step
- **Inference:** Autoregressive, but only predicts actions — fast per step
- **Memory:** Context window bounded by K — manageable
- **Data:** Requires large offline dataset of varied quality

## Primary Use Cases

- Offline RL benchmarks (D4RL): MuJoCo locomotion, Antmaze
- Multi-task and multi-domain agents (Gato)
- Settings where offline data is abundant but online interaction is expensive
- Language-conditioned control
- Game playing from demonstrations

## Known Limitations

1. **Cannot exceed** best trajectories in dataset (no stitching in base form)
2. **No TD bootstrapping** — cannot combine sub-optimal trajectory segments
3. **Return conditioning** assumes the agent can achieve the conditioned return
4. **Requires high-quality data** with diverse returns
5. **Underperforms TD-learning** methods (CQL, IQL) on many D4RL benchmarks
6. **Not designed for online interaction** (though Online DT variants exist)
7. More conceptually elegant than practically dominant

## Major Variants

| Variant | Reference | Key Change |
|---------|-----------|------------|
| Trajectory Transformer | Janner et al., NeurIPS 2021 | Models states/rewards too, beam search |
| Online DT | Lee et al., ICML 2022 | Fine-tunes with online interaction |
| Elastic DT | Wu et al., NeurIPS 2023 | Variable-length history |
| Gato | Reed et al., 2022 | Multi-modal, multi-task generalist |
| QDT | Yamagata et al., 2022 | Combines TD-learning with seq modeling |

## Relationship to Other Algorithms

- **Fundamentally different paradigm** from PPO/SAC/DQN — casts RL as supervised learning
- Competes with offline RL methods: **CQL**, **IQL**, **BCQ**
- **Gato** connects to the "foundation model for control" vision
- The paradigm informed LLM-as-agent approaches
- **Dreamer** also implicitly models dynamics but trains with imagination, not conditioning

## Industry Deployment

- **DeepMind:** Gato (multi-modal agent)
- Research-stage at most companies; not widely deployed in production
- Influential conceptually — bridged NLP and RL communities
- Informed LLM-as-agent approaches (but those typically use prompting, not DT-style training)
