"""Mario RAM RL package.

Heavy emulator dependencies are imported lazily so utility modules such as the
NeSLE benchmark CLI can be used in setup/test contexts before Stable-Retro is
installed.
"""

from mario_rl.levels import LEVEL_CONFIGS, get_level_reward_kwargs

__all__ = [
    "__version__",
    "make_mario_env",
    "LEVEL_CONFIGS",
    "get_level_reward_kwargs",
    "InfoRewardShaping",
    "SmartMarioReward",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "make_mario_env":
        from mario_rl.env import make_mario_env

        return make_mario_env
    if name in {"InfoRewardShaping", "SmartMarioReward"}:
        from mario_rl import rewards

        return getattr(rewards, name)
    raise AttributeError(name)
