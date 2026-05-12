# Architecture

This document explains how the project is wired together from ROM import to
training, evaluation, videos, and reward-profile customization.

## High-Level Flow

```text
Stable-Retro emulator
    -> MarioActionSpace
    -> observation wrapper (RAM or pixel)
    -> ActionRepeat
    -> optional FrameStack for pixels
    -> episode-boundary wrappers
    -> reward wrapper
    -> Stable-Baselines3 PPO/RecurrentPPO
```

The central entrypoint is `make_mario_env()` in `src/mario_rl/env.py`.  Every
CLI command uses this function so training, evaluation, smoke tests, and videos
share the same environment construction path.

## Environment Factory

`make_mario_env()` accepts the main experiment choices:

| Argument | Purpose |
|---|---|
| `state` | Stable-Retro state, such as `Level1-1` or `Level7-1`. |
| `obs_mode` | `ram` for a RAM vector or `pixel` for processed screen frames. |
| `action_mode` | `mario` for the curated action set, or raw Retro modes. |
| `reward_mode` | `base`, `shaped`, or `smart`. |
| `single_life` | End the episode when Mario loses one life. |
| `flag_lock` | End the episode when the game transitions to the next level. |

The wrapper order matters.  Actions are mapped before they reach the emulator,
observations are preprocessed before PPO sees them, and rewards are shaped after
episode-boundary wrappers add helpful flags such as `single_life_done`.

`mario-train` enables `flag_lock` by default.  This keeps training focused on
the selected Stable-Retro state: `--state Level7-1` starts each episode on
Level 7-1, and once Mario clears the flag, the episode terminates and the next
episode resets back to Level 7-1.  Passing `--no-flag-lock` disables that
behavior.

## Observation Modes

### RAM

`obs_mode="ram"` uses Stable-Retro RAM observations.  `RamFloat32` converts the
raw byte vector into a float32 vector in `[0, 1]`.  In this Stable-Retro Mario
integration the RAM observation prints as `(10240,)`.  The default policy is
`MlpPolicy` with separate 256x256 policy/value networks.

This mode is fast and useful as a baseline because the agent sees memory
directly.

### Pixel

`obs_mode="pixel"` uses RGB screen observations.  The pipeline is:

```text
RGB screen
    -> grayscale
    -> resize to 84x84
    -> stack 4 frames
    -> CnnPolicy / NatureCNN
```

This mode is slower but closer to how a human plays because the policy learns
from visual frames.

## Action Space

The recommended action mode is `--action-mode mario`.  It maps the controller
to 11 useful discrete actions: right movement, running, jumping, left movement,
down, and right+down.  It removes noisy actions like START/SELECT and impossible
LEFT+RIGHT combinations.

See `docs/ACTIONS.md` for the full table.

## Reward Modes

| Mode | Description |
|---|---|
| `base` | Stable-Retro's original scenario reward. |
| `shaped` | Lightweight x-progress shaping. |
| `smart` | Dense reward with progress, score, coins, kills, time, stall, death, and per-level profile tuning. |

The smart reward lives in `src/mario_rl/rewards.py`.  Per-map numbers live under
`src/mario_rl/reward_profiles/`.

## Per-Level Profiles

`src/mario_rl/levels.py` maps the requested `--state` to the correct reward
profile.  For example:

```bash
mario-train --state Level7-1 --reward-mode smart
```

automatically loads `src/mario_rl/reward_profiles/level_7_1.py`.

Add a new map profile by:

1. Creating a new file in `src/mario_rl/reward_profiles/`.
2. Defining a dictionary of `SmartMarioReward` keyword arguments.
3. Registering the state name in `reward_profiles/__init__.py`.

## CLIs

| Command | File | Purpose |
|---|---|---|
| `mario-doctor` | `scripts/doctor.py` | Checks imports, platform, and ROM registration. |
| `mario-smoke` | `scripts/smoke.py` | Runs random actions to verify environment construction. |
| `mario-train` | `scripts/train_ppo.py` | Trains PPO/RecurrentPPO and saves checkpoints. |
| `mario-eval` | `scripts/evaluate.py` | Loads a model, runs episodes, prints metrics, and optionally records videos. |

## Training Outputs

Training writes generated artifacts to:

```text
models/<run-name>/     # checkpoints and final_model.zip
runs/<run-name>/       # TensorBoard logs
videos/<run-name>/     # evaluation videos
```

These directories are intentionally ignored by Git.

## Evaluation Metrics

`mario-eval` prints:

- episode reward
- max x-position reached
- score
- coins
- time
- lives
- `smart_reward_total`
- final Stable-Retro `info`

`smart_reward_total` is especially useful because the final `info["smart_reward"]`
only describes the last environment step, while `smart_reward_total` summarizes
the whole rollout.
