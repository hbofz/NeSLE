"""Level-specific reward profile registry.

Each Stable-Retro state maps to keyword overrides for
:class:`~mario_rl.rewards.SmartMarioReward`.  The actual numbers live in
``mario_rl.reward_profiles`` so each map can grow its own tuning file instead
of crowding one central reward module.
"""

from __future__ import annotations

from mario_rl.reward_profiles import LEVEL_CONFIGS


def get_level_reward_kwargs(state: str) -> dict:
    """Return SmartMarioReward kwargs for *state*, falling back to defaults."""
    return dict(LEVEL_CONFIGS.get(state, {}))
