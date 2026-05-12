"""World 7-1 reward profile.

World 7-1 is a cannon and Hammer Bros. gauntlet.  Bullet Bills can be farmed
forever, so this profile rewards real progress and gates the big payout on the
actual flag (detected via flag_get / level transition), not just an x guess.
"""

from __future__ import annotations


LEVEL7_1: dict[str, object] = {
    # Geometry: SMB1 stages are ~200-212 tiles (~3200-3400px).  The flag sits
    # after the spring-pit + Buzzy staircase — push these later than the old
    # values so the giant bonus does NOT fire before the hardest section.
    "finish_zone_x": 3160.0,
    "flag_zone_x": 3270.0,
    # Movement: forward progress is the dominant lesson on 7-1.  Make it
    # stronger than 1-1, not weaker.
    "progress_scale": 0.22,
    "backtrack_scale": 0.06,
    "checkpoint_bonus": 2.0,
    "checkpoint_width": 128,
    # Section bonuses are smaller and more spread out so partial progress can
    # not out-earn dying.  Last bonus is placed just past the spring-pit so
    # clearing the hardest mechanical moment is specifically rewarded.
    "zone_bonuses": (
        (520.0, 6.0),
        (1040.0, 9.0),
        (1500.0, 11.0),
        (2020.0, 12.0),
        (2460.0, 9.0),
        (2900.0, 30.0),  # post spring-pit landing
    ),
    # Events: cap farmable score/kill reward hard.  Bullet Bills spawn forever
    # from cannons, so kills should be incidental income, not a strategy.
    "score_scale": 0.02,
    "coin_bonus": 1.5,
    "kill_bonus": 2.0,
    "max_score_reward": 60.0,
    "max_kill_reward": 24.0,
    # Finish: most of the payout is gated on the real flag, not the x guess.
    "finish_zone_bonus": 80.0,
    "flag_bonus": 400.0,
    # Survival: dying must clearly cost more than two flag attempts' worth of
    # partial progress.
    "death_penalty": 200.0,
    "life_loss_penalty": 0.0,
    "level_bonus": 0.0,
    # Pacing: 7-1 requires waiting out Hammer Bro patterns and Bullet Bill
    # volleys — patience is correct play, not stalling.  Widen the window and
    # cap the penalty so brief waits do not dominate the reward signal.
    "time_penalty": 0.012,
    "stall_penalty": 0.02,
    "stall_window": 48,
    "max_stall_penalty": 0.5,
    # Action quality: 7-1 needs long held-A jumps over Hammer Bros, so widen
    # the repeated-jump window and shrink the per-jump cost.
    "jump_penalty": 0.004,
    "neutral_jump_penalty": 0.02,
    "repeated_jump_penalty": 0.01,
    "repeated_jump_window": 10,
    "max_repeated_jump_penalty": 0.25,
    "left_penalty": 0.015,
    "bad_button_penalty": 0.25,
}
