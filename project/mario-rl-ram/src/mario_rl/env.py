from __future__ import annotations

from typing import Literal

import gymnasium as gym
import stable_retro as retro
from stable_baselines3.common.monitor import Monitor

from mario_rl.levels import get_level_reward_kwargs
from mario_rl.rewards import InfoRewardShaping, SmartMarioReward
from mario_rl.wrappers import (
    ActionRepeat,
    FlagLockEpisode,
    FrameStack,
    MarioActionSpace,
    PixelPreprocess,
    RamFloat32,
    SingleLifeEpisode,
)


ActionMode = Literal["all", "discrete", "multidiscrete", "mario"]
RewardMode = Literal["base", "shaped", "smart"]
ObsMode = Literal["ram", "pixel"]


def make_mario_env(
    game: str = "SuperMarioBros-Nes-v0",
    state: str = "Level1-1",
    obs_mode: ObsMode = "ram",
    action_mode: ActionMode = "mario",
    reward_mode: RewardMode = "smart",
    action_repeat: int = 4,
    render_mode: str | None = None,
    record: str | bool = False,
    monitor: bool = True,
    single_life: bool = True,
    flag_lock: bool = True,
    level_reward_kwargs: dict | None = None,
) -> gym.Env:
    """Create a Mario environment backed by RAM or pixel observations.

    Parameters
    ----------
    game:
        Stable-Retro game ID.
    state:
        Level state to load, e.g. ``"Level1-1"``.  Must match a ``.state``
        file shipped with the integration.
    obs_mode:
        ``"ram"``   – Stable-Retro RAM vector, float32, MLP policy.
        ``"pixel"`` – grayscale 84×84×4 frame stack, uint8, CNN policy.
    action_mode:
        Button mapping mode (``"mario"`` = curated 11-action discrete set).
    reward_mode:
        Reward shaping: ``"base"`` | ``"shaped"`` | ``"smart"``.
    action_repeat:
        Number of emulator frames to repeat each action.
    render_mode:
        Passed to ``retro.make()``; use ``"rgb_array"`` for video recording.
    record:
        Passed to ``retro.make()`` for BK2 recording.
    monitor:
        Wrap with SB3 ``Monitor`` for episode stats.
    single_life:
        End episode on first life lost.
    flag_lock:
        End episode when Mario completes the level (level variables change).
    level_reward_kwargs:
        Extra kwargs forwarded to :class:`~mario_rl.wrappers.SmartMarioReward`,
        overriding per-level defaults from :mod:`mario_rl.levels`.
    """
    # ------------------------------------------------------------------ base
    obs_type = retro.Observations.RAM if obs_mode == "ram" else retro.Observations.IMAGE
    env = retro.make(
        game=game,
        state=state,
        obs_type=obs_type,
        use_restricted_actions=_retro_action_mode(action_mode),
        render_mode=render_mode,
        record=record,
    )

    # --------------------------------------------------------- action mapping
    if action_mode == "mario":
        env = MarioActionSpace(env)

    # ------------------------------------------------------- observation pre
    if obs_mode == "ram":
        env = RamFloat32(env)
    else:  # pixel
        env = PixelPreprocess(env, width=84, height=84)

    # ------------------------------------------------------- frame repeat
    env = ActionRepeat(env, repeat=action_repeat)

    # ------------------------------------------------------ pixel frame stack
    if obs_mode == "pixel":
        env = FrameStack(env, n_stack=4)

    # --------------------------------------------------------- episode bounds
    if single_life:
        env = SingleLifeEpisode(env)

    if flag_lock:
        env = FlagLockEpisode(env)

    # --------------------------------------------------------- reward shaping
    if reward_mode == "shaped":
        env = InfoRewardShaping(env)
    elif reward_mode == "smart":
        # Merge per-level defaults with any caller overrides
        kwargs: dict = get_level_reward_kwargs(state)
        if level_reward_kwargs:
            kwargs.update(level_reward_kwargs)
        env = SmartMarioReward(env, **kwargs)

    if monitor:
        env = Monitor(env)

    return env


def _retro_action_mode(action_mode: ActionMode):
    if action_mode in ("mario", "all"):
        return retro.Actions.ALL
    if action_mode == "discrete":
        return retro.Actions.DISCRETE
    if action_mode == "multidiscrete":
        return retro.Actions.MULTI_DISCRETE
    raise ValueError(f"Unknown action mode: {action_mode}")
