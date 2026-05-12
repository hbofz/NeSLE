"""Reward profiles grouped by Mario level/state.

Add a new ``level_*.py`` file when a map needs its own reward tuning, then
register the Stable-Retro state name in ``LEVEL_CONFIGS`` below.
"""

from __future__ import annotations

from mario_rl.reward_profiles.future_levels import FUTURE_LEVEL_CONFIGS
from mario_rl.reward_profiles.level_1_1 import LEVEL1_1, LEVEL1_1_99_LIVES
from mario_rl.reward_profiles.level_7_1 import LEVEL7_1


LEVEL_CONFIGS: dict[str, dict] = {
    "Level1-1": LEVEL1_1,
    "Level1-1-99lives": LEVEL1_1_99_LIVES,
    "Level7-1": LEVEL7_1,
    **FUTURE_LEVEL_CONFIGS,
}


__all__ = ["LEVEL_CONFIGS"]
