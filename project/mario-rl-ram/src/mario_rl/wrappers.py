from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Reward wrappers live in rewards.py; re-exported here for backward compatibility.
from mario_rl.rewards import InfoRewardShaping, SmartMarioReward

__all__ = [
    "InfoRewardShaping",
    "SmartMarioReward",
    "MarioActionSpace",
    "RamFloat32",
    "ActionRepeat",
    "SingleLifeEpisode",
    "FlagLockEpisode",
    "PixelPreprocess",
    "FrameStack",
]


MARIO_ACTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("NOOP", ()),
    ("RIGHT", ("RIGHT",)),
    ("RIGHT_RUN", ("RIGHT", "B")),
    ("RIGHT_JUMP", ("RIGHT", "A")),
    ("RIGHT_RUN_JUMP", ("RIGHT", "B", "A")),
    ("JUMP", ("A",)),
    ("RUN_JUMP", ("B", "A")),
    ("LEFT", ("LEFT",)),
    ("LEFT_JUMP", ("LEFT", "A")),
    ("DOWN", ("DOWN",)),
    ("RIGHT_DOWN", ("RIGHT", "DOWN")),
)


class MarioActionSpace(gym.Wrapper):
    """Curated discrete Mario actions over the full NES button vector."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        buttons = getattr(env.unwrapped, "buttons", None)
        if not buttons:
            raise ValueError("MarioActionSpace requires an environment with NES button metadata.")
        self.button_names = list(buttons)
        self._button_index = {button: index for index, button in enumerate(self.button_names) if button}
        self._actions = tuple((name, self._button_vector(buttons)) for name, buttons in MARIO_ACTIONS)
        self.action_space = spaces.Discrete(len(self._actions))

    @property
    def action_meanings(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self._actions)

    def step(self, action: Any):
        action_index = int(action)
        name, vector = self._actions[action_index]
        obs, reward, terminated, truncated, info = self.env.step(vector)
        info = dict(info)
        info["action_index"] = action_index
        info["action_name"] = name
        info["buttons_pressed"] = tuple(self.button_names[i] for i, pressed in enumerate(vector) if pressed)
        return obs, reward, terminated, truncated, info

    def _button_vector(self, buttons: tuple[str, ...]) -> np.ndarray:
        vector = np.zeros(len(self.button_names), dtype=np.int8)
        for button in buttons:
            vector[self._button_index[button]] = 1
        return vector


class RamFloat32(gym.ObservationWrapper):
    """Scale uint8 RAM observations to float32 in [0, 1]."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32) / 255.0


class ActionRepeat(gym.Wrapper):
    """Repeat a full controller action for a few emulator frames."""

    def __init__(self, env: gym.Env, repeat: int = 4):
        super().__init__(env)
        if repeat < 1:
            raise ValueError("repeat must be >= 1")
        self.repeat = repeat

    def step(self, action: Any):
        total_reward = 0.0
        last_obs = None
        last_info: dict[str, Any] = {}
        terminated = False
        truncated = False

        for _ in range(self.repeat):
            last_obs, reward, terminated, truncated, last_info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return last_obs, total_reward, terminated, truncated, last_info


class SingleLifeEpisode(gym.Wrapper):
    """End an episode immediately when Mario loses a life."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._last_lives: int | None = None

    def reset(self, **kwargs):
        self._last_lives = None
        return self.env.reset(**kwargs)

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "lives" in info:
            lives = int(info["lives"])
            if self._last_lives is not None and lives < self._last_lives:
                terminated = True
                info = dict(info)
                info["single_life_done"] = True
            self._last_lives = lives
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Level-lock wrapper
# ---------------------------------------------------------------------------

class FlagLockEpisode(gym.Wrapper):
    """End an episode when Mario completes the level.

    Stable-Retro's scenario only terminates on ``lives == -1``.  This wrapper
    watches ``levelHi`` / ``levelLo`` from the info dict.  As soon as those
    values differ from what they were at ``reset()`` (i.e. Mario passed the
    flag and the game transitioned to the next level) the episode is marked
    terminated so training stays locked to the starting level.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._start_level: tuple[int, int] | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._start_level = self._get_level(info)
        return obs, info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Only fire if the episode has not already ended (e.g. from SingleLifeEpisode)
        if not terminated and not truncated:
            current_level = self._get_level(info)
            if self._start_level is not None and current_level is not None:
                if current_level != self._start_level:
                    terminated = True
                    info = dict(info)
                    info["flag_lock_done"] = True
        return obs, reward, terminated, truncated, info

    @staticmethod
    def _get_level(info: dict[str, Any]) -> tuple[int, int] | None:
        if "levelHi" in info and "levelLo" in info:
            try:
                return (int(info["levelHi"]), int(info["levelLo"]))
            except (TypeError, ValueError):
                return None
        return None


# ---------------------------------------------------------------------------
# Pixel observation wrappers (for CNN training)
# ---------------------------------------------------------------------------

class PixelPreprocess(gym.ObservationWrapper):
    """Convert an RGB screen to a grayscale uint8 frame of shape (H, W, 1).

    The raw Stable-Retro screen is ``(H, W, 3)`` uint8.  This wrapper:
    1. Converts RGB → grayscale using standard luminance weights.
    2. Resizes to ``(height, width)`` using PIL (bicubic).
    3. Returns shape ``(height, width, 1)`` uint8 so SB3's
       ``VecTransposeImage`` can later convert it to channel-first.
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(height, width, 1),
            dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        from PIL import Image  # PIL is a transitive dep via SB3[extra]
        # obs: (H, W, 3) uint8
        gray = np.dot(obs[..., :3].astype(np.float32), [0.299, 0.587, 0.114])
        img = Image.fromarray(gray.astype(np.uint8))
        img = img.resize((self.width, self.height), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)[..., np.newaxis]  # (H, W, 1)


class FrameStack(gym.Wrapper):
    """Stack the last ``n_stack`` frames along the channel axis.

    Input observation space must be ``(H, W, C)``.
    Output shape is ``(H, W, C * n_stack)``.

    SB3's ``VecTransposeImage`` will later convert ``(H, W, C*n)`` →
    ``(C*n, H, W)`` which is what ``CnnPolicy`` / NatureCNN expects.
    """

    def __init__(self, env: gym.Env, n_stack: int = 4):
        super().__init__(env)
        self.n_stack = n_stack
        low = env.observation_space.low
        high = env.observation_space.high
        self.observation_space = spaces.Box(
            low=np.repeat(low, n_stack, axis=-1),
            high=np.repeat(high, n_stack, axis=-1),
            dtype=env.observation_space.dtype,
        )
        self._frames: deque[np.ndarray] = deque(maxlen=n_stack)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_stack):
            self._frames.append(obs)
        return self._stack(), info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._stack(), reward, terminated, truncated, info

    def _stack(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=-1)
