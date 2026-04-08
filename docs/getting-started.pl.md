# Pierwsze kroki z rlox

> **ℹ️ Skrócone tłumaczenie.** Ta strona obejmuje tylko instalację i pierwszego agenta.
> Pełny przewodnik (API niskiego poziomu, komponenty Rust, checkpointy, kolektory
> niestandardowe) jest dostępny wyłącznie w [wersji angielskiej](../getting-started.md).
> Chcesz pomóc rozszerzyć to tłumaczenie? Zobacz [przewodnik dla tłumaczy](CONTRIBUTING-translations.md).

Ten samouczek prowadzi przez instalację rlox i wytrenowanie pierwszego agenta.

## Wymagania wstępne

- Python 3.10+ (testowane na 3.10, 3.11, 3.12, 3.13)
- Toolchain Rusta (`rustup` — instalacja z https://rustup.rs)
- macOS (ARM/Intel), Linux lub Windows

## Instalacja

```bash
# Sklonuj repozytorium
git clone https://github.com/riserally/rlox.git
cd rlox

# Utwórz wirtualne środowisko Pythona
python3.12 -m venv .venv
source .venv/bin/activate

# Zainstaluj zależności
pip install maturin numpy gymnasium torch

# Zbuduj rozszerzenie Rust i zainstaluj w trybie deweloperskim
maturin develop --release

# Zweryfikuj instalację
python -c "from rlox import CartPole; print('rlox gotowe')"
```

> **Wskazówka**: Zawsze używaj `--release` z maturin. Buildy debug są 10–50× wolniejsze
> i sprawiają, że benchmarki tracą sens.

## Krok 1: pierwszy agent CartPole

Najszybszy sposób na wytrenowanie agenta RL w rlox:

```python
from rlox import Trainer

# Trenuj PPO na CartPole-v1 z domyślnymi hiperparametrami
trainer = Trainer("ppo", env="CartPole-v1", seed=42)
metrics = trainer.train(total_timesteps=50_000)

print(f"Średnia nagroda: {metrics['mean_reward']:.1f}")
# Oczekiwane: ~400–500 (maksimum CartPole to 500)
```

To wszystko — 3 linijki do wytrenowanego agenta. Pod maską rlox używa:

- Rustowego `VecEnv` do równoległego krokowania środowisk (domyślnie 8 środowisk)
- Rustowego `compute_gae` do estymacji przewag (140× szybsze niż Python)
- PyTorch do sieci polityki i aktualizacji SGD

## Krok 2: zrozumienie architektury

rlox stosuje **wzorzec Polars** — warstwa danych w Ruście do ciężkich obliczeń, ze
sterowaniem z poziomu Pythona:

```
┌─────────────────────────────────────────┐
│  Python (warstwa sterowania)            │
│  Pętle treningowe, polityki (PyTorch),  │
│  konfiguracja, logowanie, callbacki     │
├────────── granica PyO3 ─────────────────┤
│  Rust (warstwa danych)                  │
│  Środowiska, krokowanie równoległe,     │
│  bufory, GAE, GRPO, dywergencja KL      │
└─────────────────────────────────────────┘
```

**Dlaczego taki podział?** Trening RL ma dwa wąskie gardła:

1. **Zbieranie danych** — krokowanie środowisk i przechowywanie przejść. Jest to
   zadanie wstydliwie równoległe i ogromnie zyskuje na abstrakcjach bez kosztów Rusta.
2. **Obliczanie gradientu** — propagacja w przód/wstecz sieci neuronowej. PyTorch/CUDA
   już to dobrze obsługują.

rlox przyspiesza (1), pozostawiając (2) PyTorchowi.

## Następne kroki

Dalsze rozdziały (API niskiego poziomu, używanie prymitywów Rust bezpośrednio,
checkpointy, własne polityki i kolektory) są w [pełnej wersji angielskiej](../getting-started.md).
Zobacz też:

- [Python User Guide](../python-guide.md) — pełna referencja API
- [Przykłady](../examples.md) — gotowy kod dla każdego algorytmu
- [Migracja z SB3](../tutorials/migration-sb3.md) — porównanie API obok siebie
