# Reward Design

The Stable-Retro Mario integration has a very small default reward:

```json
{
  "reward": {
    "variables": {
      "xscrollLo": {
        "reward": 1
      }
    }
  }
}
```

That teaches "move right", but it does not directly value safer play, coins, score, or new territory beyond the low scroll byte.

This project therefore trains with `--reward-mode smart` by default.

## Per-Level Profiles

The reward wrapper lives in `src/mario_rl/rewards.py`, but the numbers are
split by map under `src/mario_rl/reward_profiles/`.

Current layout:

```text
src/mario_rl/reward_profiles/
|-- level_1_1.py       # tuned World 1-1 profile
|-- level_7_1.py       # tuned World 7-1 cannon/Hammer Bros profile
|-- future_levels.py   # rough geometry-only starter profiles
`-- __init__.py        # Stable-Retro state -> profile registry
```

To tune a map, create a new `level_*.py` file with the reward kwargs for
`SmartMarioReward`, then register the Stable-Retro state name in
`reward_profiles/__init__.py`.

World 1-1 is currently tuned toward more human-like play: beating the map is
still the main goal, but coins, enemy kills, and score events are stronger so
the policy has a reason to do more than sprint right.

World 7-1 is tuned differently because Bullet Bills can be farmed forever.  It
adds one-time section-clear bonuses, stronger death pressure, and caps on
score/kill reward so the agent learns to clear the cannon and Hammer Bros.
gauntlet instead of standing still for infinite enemy points.

## Smart Reward Components

Each agent decision is repeated for 4 emulator frames by default, then the smart reward is computed from Stable-Retro `info` variables.

| Component | Purpose |
|---|---|
| `progress` | Rewards forward movement using `xscrollLo + 256 * xscrollHi`; lightly penalizes moving backward. |
| `checkpoint` | Gives a bonus when Mario reaches each new 128-pixel progress band. |
| `zone` | Optional one-time profile-defined bonuses for clearing major map sections. |
| `score` | Rewards score increases from enemies, blocks, powerups, and flag scoring. |
| `coin` | Rewards coin collection, including coin counter wraparound. |
| `kill` | Adds a bonus when score jumps indicate an enemy kill that was not just a coin. |
| `level` | Rewards a level variable change when the integration exposes one. |
| `finish` | Gives large one-time bonuses near the end of the level and the approximate flag zone. |
| `life` | Penalizes losing lives. |
| `death` | Adds a larger penalty when an episode terminates from death or single-life loss. |
| `time` | Adds a small per-decision time cost so standing still is not free. |
| `stall` | Adds increasing pressure when progress has been flat/backward for several decisions. |

The wrapper also adds an `info["smart_reward"]` dictionary so we can inspect component-level rewards during debugging.
`mario-eval` also prints `smart_reward_total`, which is the episode-level sum
of these components. Prefer that over the final `info["smart_reward"]`, because
the final `info` only describes the last environment step.

Training uses single-life episodes by default. When Mario loses a life, the
episode ends immediately, which gives PPO cleaner feedback than waiting until
every life is gone.

The recommended `mario` action mode also adds a small action-quality term. It
lightly penalizes jump spam, neutral jumps, left movement, and impossible/bad
buttons. Forward progress is still worth much more than the action penalty, so
useful jumps remain profitable.

## Reward Profile Parameters

Profiles are dictionaries of keyword arguments for `SmartMarioReward`.  Common
fields:

| Field | Meaning |
|---|---|
| `progress_scale` | Reward multiplier for forward x movement. |
| `backtrack_scale` | Penalty multiplier for moving backward. |
| `checkpoint_bonus` | One-time reward for each progress band. |
| `zone_bonuses` | Optional one-time `(x_position, bonus)` section rewards. |
| `score_scale` | Reward multiplier for score increases. |
| `coin_bonus` | Flat reward per coin. |
| `kill_bonus` | Flat inferred enemy-kill reward. |
| `max_score_reward` | Optional cap on episode score reward. |
| `max_kill_reward` | Optional cap on episode kill reward. |
| `finish_zone_x` | Approximate x-position for near-finish bonus. |
| `flag_zone_x` | Approximate x-position for flag-zone bonus. |
| `death_penalty` | Penalty when the episode ends from death. |
| `time_penalty` | Small per-decision cost. |
| `stall_penalty` | Increasing cost when x progress stops. |

Caps are especially important for levels with infinitely spawning enemies, such
as World 7-1's Bullet Bills.

## Modes

```bash
--reward-mode base
```

Use only the Stable-Retro scenario reward.

```bash
--reward-mode shaped
```

Use the Stable-Retro reward plus light full-x progress shaping.

```bash
--reward-mode smart
```

Use the project reward above. This is the default for training.
