"""World 1-1 reward profile.

This profile keeps the agent strongly motivated to finish the level while
making optional human-like play more profitable: coins, enemy stomps, and
score events are meant to compete with pure rightward speed.
"""

from __future__ import annotations


LEVEL1_1: dict[str, float | int] = {
    # Geometry: approximate x-scroll positions for one-time end-zone bonuses.
    "finish_zone_x": 3100.0,
    "flag_zone_x": 3300.0,
    # Movement: still reward forward play, but do not let "hold right forever"
    # drown out coins, stomps, and score events.
    "progress_scale": 0.18,
    "backtrack_scale": 0.03,
    "checkpoint_bonus": 3.0,
    "checkpoint_width": 128,
    # Human-ish events.  Stable-Retro exposes total score/coins, not exact enemy
    # identities, so kills are inferred from score jumps that are not coins.
    "score_scale": 0.04,
    "coin_bonus": 3.0,
    "kill_bonus": 8.0,
    # Finish bonuses preserve the main objective: beat the map.
    "finish_zone_bonus": 100.0,
    "flag_bonus": 100.0,
    # Single-life training already terminates on death.  Keep the death signal
    # strong, but avoid stacking a separate life penalty on the same event.
    "death_penalty": 75.0,
    "life_loss_penalty": 0.0,
    "level_bonus": 0.0,
    # Give Mario room to line up jumps/stomps/coins without rewarding idling.
    "time_penalty": 0.005,
    "stall_penalty": 0.02,
    "stall_window": 40,
    "max_stall_penalty": 0.5,
    # Jumping is necessary for human-like play, so punish spam lightly instead
    # of making every jump feel expensive.
    "jump_penalty": 0.005,
    "neutral_jump_penalty": 0.02,
    "repeated_jump_penalty": 0.02,
    "repeated_jump_window": 6,
    "max_repeated_jump_penalty": 0.25,
    "left_penalty": 0.01,
    "bad_button_penalty": 0.25,
}


LEVEL1_1_99_LIVES: dict[str, float | int] = dict(LEVEL1_1)
