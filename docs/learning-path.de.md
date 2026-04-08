# Lernpfad

> **ℹ️ Gekürzte Übersetzung.** Diese Seite zeigt die Lernpfadkarte und Stufe 1.
> Der vollständige Pfad (Stufen 2–5: Kernkonzepte, On-Policy-/Off-Policy-Algorithmen,
> fortgeschrittene Themen, Produktion) ist nur in der [englischen Version](../learning-path.md) verfügbar.

Ihr Leitfaden zur Beherrschung von Reinforcement Learning mit rlox — von null bis zur Produktion.

```mermaid
flowchart TD
    L1[Stufe 1: Erste Schritte]
    L2[Stufe 2: Kernkonzepte]
    L3A[Stufe 3a: On-Policy]
    L3B[Stufe 3b: Off-Policy]
    L3C[Stufe 3c: Modellbasiert & Multi-Agent]
    L4[Stufe 4: Fortgeschrittene Themen]
    L5[Stufe 5: Produktion & Skalierung]

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

## Stufe 1: Erste Schritte (30 Minuten)

**Ziel:** rlox installieren, ersten Agenten trainieren und Ergebnisse sehen.

### rlox installieren

```bash
pip install rlox
```

### Ersten Agenten trainieren

```python
from rlox import Trainer

trainer = Trainer("ppo", env="CartPole-v1", seed=42)
metrics = trainer.train(total_timesteps=100_000)
print(f"Finale Rückgabe: {metrics['mean_reward']:.1f}")
```

### Die Trainer-API verstehen

`Trainer` ist der einzige Einstiegspunkt für alle Algorithmen:

```python
# Mit Algorithmus-Name und Umgebung erstellen
trainer = Trainer("sac", env="Pendulum-v1")

# Für N Zeitschritte trainieren
metrics = trainer.train(total_timesteps=50_000)

# Checkpoints speichern / laden
trainer.save("my_model")
trainer = Trainer.from_checkpoint("my_model", algorithm="sac", env="Pendulum-v1")

# Aktionen vorhersagen
action = trainer.predict(obs, deterministic=True)
```

### Weiterlesen

- [Erste Schritte](getting-started.md) — gekürzte Einführung auf Deutsch
- [Vollständiger Lernpfad (en)](../learning-path.md) — Stufen 2–5
- [Beispiele](../examples.md) — gebrauchsfertige Code-Schnipsel
