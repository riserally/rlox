# Parcours d'apprentissage

> **ℹ️ Traduction abrégée.** Cette page présente la carte du parcours et le Niveau 1.
> Le parcours complet (Niveaux 2 à 5 : concepts clés, algorithmes on-policy/off-policy,
> sujets avancés, production) n'est disponible qu'en [version anglaise](../learning-path.md).

Votre guide pour maîtriser l'apprentissage par renforcement avec rlox — de zéro à la production.

```mermaid
flowchart TD
    L1[Niveau 1 : premiers pas]
    L2[Niveau 2 : concepts clés]
    L3A[Niveau 3a : on-policy]
    L3B[Niveau 3b : off-policy]
    L3C[Niveau 3c : model-based et multi-agent]
    L4[Niveau 4 : sujets avancés]
    L5[Niveau 5 : production et mise à l'échelle]

    L1 --> L2
    L2 --> L3A
    L2 --> L3B
    L2 --> L3C
    L3A --> L4
    L3B --> L4
    L3C --> L4
    L4 --> L5

    L3A -.- VPG[VPG] & A2C[A2C] & PPO[PPO] & TRPO[TRPO]
    L3B -.- DQN[DQN] & TD3[TD3] & SAC[SAC] & IMPALA[IMPALA]
    L3C -.- Dreamer[DreamerV3] & MAPPO[MAPPO] & QMIX[QMIX]

    style L1 fill:#e8f5e9,stroke:#388e3c
    style L2 fill:#e3f2fd,stroke:#1976d2
    style L3A fill:#fff3e0,stroke:#f57c00
    style L3B fill:#fff3e0,stroke:#f57c00
    style L3C fill:#fff3e0,stroke:#f57c00
    style L4 fill:#fce4ec,stroke:#c62828
    style L5 fill:#f3e5f5,stroke:#7b1fa2
```

---

## Niveau 1 : premiers pas (30 minutes)

**Objectif :** installer rlox, entraîner votre premier agent et voir les résultats.

### Installer rlox

```bash
pip install rlox
```

### Entraîner votre premier agent

```python
from rlox import Trainer

trainer = Trainer("ppo", env="CartPole-v1", seed=42)
metrics = trainer.train(total_timesteps=100_000)
print(f"Retour final : {metrics['mean_reward']:.1f}")
```

### Comprendre l'API Trainer

`Trainer` est le seul point d'entrée pour tous les algorithmes :

```python
# Créer avec un nom d'algorithme + un environnement
trainer = Trainer("sac", env="Pendulum-v1")

# Entraîner pour N pas de temps
metrics = trainer.train(total_timesteps=50_000)

# Sauvegarder / charger des checkpoints
trainer.save("my_model")
trainer = Trainer.from_checkpoint("my_model", algorithm="sac", env="Pendulum-v1")

# Prédire des actions
action = trainer.predict(obs, deterministic=True)
```

### Pour aller plus loin

- [Premiers pas](getting-started.md) — introduction abrégée en français
- [Parcours d'apprentissage complet (en)](../learning-path.md) — Niveaux 2 à 5
- [Exemples](../examples.md) — extraits de code prêts à l'emploi
