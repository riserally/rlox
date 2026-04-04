"""rlox.wrappers -- Visual RL and Language-Conditioned RL wrappers.

Provides environment wrappers for:
- Frame stacking (temporal information)
- Image preprocessing (resize, grayscale, normalize)
- Atari and DMControl environment wrapping
- Language-conditioned observations
- Goal-conditioned sparse reward
"""

from rlox.wrappers.visual import (
    FrameStack,
    ImagePreprocess,
    AtariWrapper,
    DMControlWrapper,
)
from rlox.wrappers.language import (
    LanguageWrapper,
    GoalConditionedWrapper,
)

__all__ = [
    "FrameStack",
    "ImagePreprocess",
    "AtariWrapper",
    "DMControlWrapper",
    "LanguageWrapper",
    "GoalConditionedWrapper",
]
