# Erste Schritte mit rlox

> **ℹ️ Gekürzte Übersetzung.** Diese Seite behandelt nur Installation und den ersten Agenten.
> Die vollständige Anleitung (Low-Level-API, Rust-Komponenten, Checkpoints, eigene
> Collectors) ist nur in der [englischen Version](../getting-started.md) verfügbar.
> Möchten Sie bei der Erweiterung dieser Übersetzung helfen? Siehe den
> [Übersetzungsleitfaden](CONTRIBUTING-translations.md).

Dieses Tutorial führt Sie durch die Installation von rlox und das Training Ihres ersten Agenten.

## Voraussetzungen

- Python 3.10+ (getestet auf 3.10, 3.11, 3.12, 3.13)
- Rust-Toolchain (`rustup` — Installation von https://rustup.rs)
- macOS (ARM/Intel), Linux oder Windows

## Installation

```bash
# Repository klonen
git clone https://github.com/riserally/rlox.git
cd rlox

# Virtuelle Python-Umgebung erstellen
python3.12 -m venv .venv
source .venv/bin/activate

# Build-Abhängigkeiten installieren
pip install maturin numpy gymnasium torch

# Rust-Erweiterung bauen und im Entwicklungsmodus installieren
maturin develop --release

# Installation überprüfen
python -c "from rlox import CartPole; print('rlox bereit')"
```

> **Tipp**: Verwenden Sie immer `--release` mit maturin. Debug-Builds sind 10–50× langsamer
> und machen Benchmarks bedeutungslos.

## Schritt 1: Ihr erster CartPole-Agent

Der schnellste Weg, einen RL-Agenten mit rlox zu trainieren:

```python
from rlox import Trainer

# PPO auf CartPole-v1 mit Standard-Hyperparametern trainieren
trainer = Trainer("ppo", env="CartPole-v1", seed=42)
metrics = trainer.train(total_timesteps=50_000)

print(f"Durchschnittliche Belohnung: {metrics['mean_reward']:.1f}")
# Erwartet: ~400–500 (CartPole-Maximum ist 500)
```

Das war's — 3 Zeilen bis zum trainierten Agenten. Unter der Haube verwendet rlox:

- Rust-`VecEnv` für paralleles Stepping von Umgebungen (standardmäßig 8 Umgebungen)
- Rust-`compute_gae` für Advantage-Schätzung (140× schneller als Python)
- PyTorch für Policy-Netzwerk und SGD-Updates

## Schritt 2: Die Architektur verstehen

rlox folgt dem **Polars-Muster** — eine Rust-Datenschicht für schwere Berechnungen,
mit Python als Steuerung:

```
┌─────────────────────────────────────────┐
│  Python (Steuerungsschicht)             │
│  Trainingsschleifen, Policies (PyTorch),│
│  Konfiguration, Logging, Callbacks      │
├────────── PyO3-Grenze ──────────────────┤
│  Rust (Datenschicht)                    │
│  Umgebungen, paralleles Stepping,       │
│  Buffer, GAE, GRPO, KL-Divergenz        │
└─────────────────────────────────────────┘
```

**Warum diese Aufteilung?** RL-Training hat zwei Flaschenhälse:

1. **Datenerfassung** — Umgebungen schritten und Übergänge speichern. Das ist
   peinlich parallel und profitiert enorm von Rusts Null-Kosten-Abstraktionen.
2. **Gradientenberechnung** — Vorwärts-/Rückwärtspass des neuronalen Netzes. Das
   handhabt PyTorch/CUDA bereits gut.

rlox beschleunigt (1) und überlässt (2) PyTorch.

## Nächste Schritte

Weitere Kapitel (Low-Level-API, direkte Verwendung von Rust-Primitiven, Checkpoints,
eigene Policies und Collectors) finden Sie in der [vollständigen englischen Version](../getting-started.md).
Siehe auch:

- [Python User Guide](../python-guide.md) — vollständige API-Referenz
- [Beispiele](../examples.md) — Copy-paste-Code für jeden Algorithmus
- [Migration von SB3](../tutorials/migration-sb3.md) — Side-by-side API-Vergleich
