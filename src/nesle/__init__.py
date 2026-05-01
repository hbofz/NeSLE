"""NeSLE Python API."""

from .actions import (
    COMPLEX_MOVEMENT,
    COMPLEX_MOVEMENT_MASKS,
    RIGHT_ONLY,
    RIGHT_ONLY_MASKS,
    SIMPLE_MOVEMENT,
    SIMPLE_MOVEMENT_MASKS,
    SIMPLE_MOVEMENT_WITH_START,
    SIMPLE_MOVEMENT_WITH_START_MASKS,
    Button,
    encode_action,
)
from .rom import INESRom, parse_ines
from .smb import MarioRamState, RewardComponents, compute_reward, read_ram

__all__ = [
    "Button",
    "COMPLEX_MOVEMENT",
    "COMPLEX_MOVEMENT_MASKS",
    "INESRom",
    "MarioRamState",
    "RIGHT_ONLY",
    "RIGHT_ONLY_MASKS",
    "RewardComponents",
    "SIMPLE_MOVEMENT",
    "SIMPLE_MOVEMENT_MASKS",
    "SIMPLE_MOVEMENT_WITH_START",
    "SIMPLE_MOVEMENT_WITH_START_MASKS",
    "compute_reward",
    "encode_action",
    "make",
    "make_vec",
    "parse_ines",
    "read_ram",
]


def make(*args, **kwargs):
    from .env import NesleEnv

    return NesleEnv(*args, **kwargs)


def make_vec(*args, **kwargs):
    from .env import NesleVecEnv

    return NesleVecEnv(*args, **kwargs)
