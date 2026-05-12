"""Starter geometry profiles for future Mario levels.

These are intentionally rough.  When a level becomes a real training target,
move it into its own ``level_*.py`` file and tune coins/kills/progress there.
"""

from __future__ import annotations


FUTURE_LEVEL_CONFIGS: dict[str, dict[str, float]] = {
    "Level2-1": {
        "finish_zone_x": 3400.0,
        "flag_zone_x": 3580.0,
    },
    "Level3-1": {
        "finish_zone_x": 2500.0,
        "flag_zone_x": 2700.0,
    },
    "Level4-1": {
        "finish_zone_x": 2700.0,
        "flag_zone_x": 2900.0,
    },
    "Level5-1": {
        "finish_zone_x": 3000.0,
        "flag_zone_x": 3200.0,
    },
    "Level6-1": {
        "finish_zone_x": 2700.0,
        "flag_zone_x": 2900.0,
    },
    "Level8-1": {
        "finish_zone_x": 3000.0,
        "flag_zone_x": 3200.0,
    },
}
