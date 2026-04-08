# Ścieżka nauki

> **ℹ️ Skrócone tłumaczenie.** Ta strona pokazuje mapę ścieżki nauki i Poziom 1.
> Pełna ścieżka (Poziomy 2–5: koncepcje podstawowe, algorytmy on-policy/off-policy,
> tematy zaawansowane, produkcja) jest dostępna wyłącznie w [wersji angielskiej](../learning-path.md).

Twój przewodnik po opanowaniu uczenia ze wzmocnieniem z rlox — od zera do produkcji.

```mermaid
flowchart TD
    L1[Poziom 1: pierwsze kroki]
    L2[Poziom 2: koncepcje podstawowe]
    L3A[Poziom 3a: on-policy]
    L3B[Poziom 3b: off-policy]
    L3C[Poziom 3c: model-based i multi-agent]
    L4[Poziom 4: tematy zaawansowane]
    L5[Poziom 5: produkcja i skala]

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

## Poziom 1: pierwsze kroki (30 minut)

**Cel:** zainstalować rlox, wytrenować pierwszego agenta i zobaczyć wyniki.

### Instalacja rlox

```bash
pip install rlox
```

### Wytrenuj pierwszego agenta

```python
from rlox import Trainer

trainer = Trainer("ppo", env="CartPole-v1", seed=42)
metrics = trainer.train(total_timesteps=100_000)
print(f"Końcowy zwrot: {metrics['mean_reward']:.1f}")
```

### Poznaj API Trainera

`Trainer` jest jedynym punktem wejścia dla wszystkich algorytmów:

```python
# Utwórz z nazwą algorytmu i środowiskiem
trainer = Trainer("sac", env="Pendulum-v1")

# Trenuj przez N kroków
metrics = trainer.train(total_timesteps=50_000)

# Zapisz / wczytaj checkpointy
trainer.save("my_model")
trainer = Trainer.from_checkpoint("my_model", algorithm="sac", env="Pendulum-v1")

# Przewiduj akcje
action = trainer.predict(obs, deterministic=True)
```

### Dalsza lektura

- [Przewodnik pierwszych kroków](getting-started.md) — skrócone wprowadzenie po polsku
- [Pełna ścieżka nauki (en)](../learning-path.md) — Poziomy 2–5
- [Przykłady](../examples.md) — gotowe fragmenty kodu
