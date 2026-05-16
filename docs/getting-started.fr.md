# Premiers pas avec rlox

> **ℹ️ Traduction abrégée.** Cette page couvre uniquement l'installation et le premier agent.
> Le guide complet (API bas niveau, composants Rust, checkpoints, collecteurs
> personnalisés) n'est disponible qu'en [version anglaise](../getting-started.md).
> Vous souhaitez aider à étendre cette traduction ? Consultez le
> [guide pour les traducteur·rices](CONTRIBUTING-translations.md).

Ce tutoriel vous guide à travers l'installation de rlox et l'entraînement de votre premier agent.

## Prérequis

- Python 3.10+ (testé sur 3.10, 3.11, 3.12, 3.13)
- Toolchain Rust (`rustup` — installation depuis https://rustup.rs)
- macOS (ARM/Intel), Linux ou Windows

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/wojciechkpl/rlox.git
cd rlox

# Créer un environnement virtuel Python
python3.12 -m venv .venv
source .venv/bin/activate

# Installer les dépendances de build
pip install maturin numpy gymnasium torch

# Construire l'extension Rust et installer en mode développement
maturin develop --release

# Vérifier l'installation
python -c "from rlox import CartPole; print('rlox prêt')"
```

> **Astuce** : Utilisez toujours `--release` avec maturin. Les builds debug sont
> 10 à 50× plus lents et rendent les benchmarks sans valeur.

## Étape 1 : votre premier agent CartPole

La façon la plus rapide d'entraîner un agent RL avec rlox :

```python
from rlox import Trainer

# Entraîner PPO sur CartPole-v1 avec les hyperparamètres par défaut
trainer = Trainer("ppo", env="CartPole-v1", seed=42)
metrics = trainer.train(total_timesteps=50_000)

print(f"Récompense moyenne : {metrics['mean_reward']:.1f}")
# Attendu : ~400–500 (le maximum de CartPole est 500)
```

C'est tout — 3 lignes pour un agent entraîné. En coulisses, rlox utilise :

- `VecEnv` en Rust pour le stepping parallèle des environnements (8 par défaut)
- `compute_gae` en Rust pour l'estimation d'advantage (140× plus rapide que Python)
- PyTorch pour le réseau de politique et les mises à jour SGD

## Étape 2 : comprendre l'architecture

rlox suit le **modèle Polars** — une couche de données en Rust pour le calcul intensif,
avec Python aux commandes :

```
┌─────────────────────────────────────────┐
│  Python (couche de contrôle)            │
│  Boucles d'entraînement, politiques,    │
│  config, logging, callbacks             │
├────────── frontière PyO3 ───────────────┤
│  Rust (couche de données)               │
│  Environnements, stepping parallèle,    │
│  buffers, GAE, GRPO, divergence KL      │
└─────────────────────────────────────────┘
```

**Pourquoi ce découpage ?** L'entraînement RL a deux goulets d'étranglement :

1. **Collecte de données** — stepping des environnements et stockage des transitions.
   C'est embarrassingly parallel et bénéficie énormément des abstractions à coût zéro
   de Rust.
2. **Calcul du gradient** — passes avant/arrière du réseau de neurones. PyTorch/CUDA
   gèrent déjà cela très bien.

rlox accélère (1) tout en laissant (2) à PyTorch.

## Étapes suivantes

Les chapitres suivants (API bas niveau, utilisation directe des primitives Rust,
checkpoints, politiques et collecteurs personnalisés) se trouvent dans la
[version anglaise complète](../getting-started.md). Voir aussi :

- [Guide Python](../python-guide.md) — référence complète de l'API
- [Exemples](../examples.md) — code copier-coller pour chaque algorithme
- [Migration depuis SB3](../tutorials/migration-sb3.md) — comparaison d'API côte à côte
