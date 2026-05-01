from __future__ import annotations

from enum import IntEnum
from typing import Iterable


class Button(IntEnum):
    A = 0
    B = 1
    SELECT = 2
    START = 3
    UP = 4
    DOWN = 5
    LEFT = 6
    RIGHT = 7


_ALIASES = {
    "a": Button.A,
    "b": Button.B,
    "select": Button.SELECT,
    "start": Button.START,
    "up": Button.UP,
    "down": Button.DOWN,
    "left": Button.LEFT,
    "right": Button.RIGHT,
    "noop": None,
    "no-op": None,
    "none": None,
}


def encode_action(buttons: Iterable[str | Button]) -> int:
    mask = 0
    for button in buttons:
        if isinstance(button, Button):
            resolved = button
        else:
            key = str(button).strip().lower()
            if key not in _ALIASES:
                raise ValueError(f"unknown NES controller button: {button!r}")
            resolved = _ALIASES[key]
        if resolved is not None:
            mask |= 1 << int(resolved)
    return mask


RIGHT_ONLY = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
]

SIMPLE_MOVEMENT = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
    ["A"],
    ["left"],
]

SIMPLE_MOVEMENT_WITH_START = [
    ["start"],
    *SIMPLE_MOVEMENT,
]

COMPLEX_MOVEMENT = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
    ["A"],
    ["left"],
    ["left", "A"],
    ["left", "B"],
    ["left", "A", "B"],
    ["down"],
    ["up"],
]

RIGHT_ONLY_MASKS = tuple(encode_action(action) for action in RIGHT_ONLY)
SIMPLE_MOVEMENT_MASKS = tuple(encode_action(action) for action in SIMPLE_MOVEMENT)
SIMPLE_MOVEMENT_WITH_START_MASKS = tuple(encode_action(action) for action in SIMPLE_MOVEMENT_WITH_START)
COMPLEX_MOVEMENT_MASKS = tuple(encode_action(action) for action in COMPLEX_MOVEMENT)
