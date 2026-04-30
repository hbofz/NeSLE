from __future__ import annotations

from dataclasses import dataclass


CPU_RAM_BYTES = 2048

PLAYER_STATE = 0x000E
PLAYER_FLOAT_STATE = 0x001D
ENEMY_TYPE_BASE = 0x0016
X_PAGE = 0x006D
X_SCREEN = 0x0086
Y_VIEWPORT = 0x00B5
Y_PIXEL = 0x03B8
LIVES = 0x075A
STAGE = 0x075C
WORLD = 0x075F
AREA = 0x0760
STATUS = 0x0756
GAME_MODE = 0x0770
COINS_DIGITS = 0x07ED
TIME_DIGITS = 0x07F8
SCORE_DIGITS = 0x07DE


@dataclass(frozen=True)
class MarioRamState:
    x_pos: int
    y_pos: int
    time: int
    coins: int
    score: int
    life: int
    world: int
    stage: int
    area: int
    status: str
    status_code: int
    player_state: int
    flag_get: bool
    is_dying: bool
    is_dead: bool
    is_game_over: bool


@dataclass(frozen=True)
class RewardComponents:
    x: int
    time: int
    death: int
    total: int


def _read_digits(ram: bytes, address: int, length: int) -> int:
    value = 0
    for byte in ram[address : address + length]:
        value = value * 10 + (byte & 0x0F)
    return value


def _status_name(status_code: int) -> str:
    if status_code == 0:
        return "small"
    if status_code == 1:
        return "tall"
    return "fireball"


def _is_stage_over(ram: bytes) -> bool:
    stage_over_enemies = {0x2D, 0x31}
    for offset in range(5):
        if ram[ENEMY_TYPE_BASE + offset] in stage_over_enemies and ram[PLAYER_FLOAT_STATE] == 3:
            return True
    return False


def read_ram(data: bytes | bytearray | memoryview) -> MarioRamState:
    ram = bytes(data)
    if len(ram) < CPU_RAM_BYTES:
        raise ValueError("Super Mario Bros. RAM view must contain 2048 bytes")

    y_viewport = ram[Y_VIEWPORT]
    y_pixel = ram[Y_PIXEL]
    player_state = ram[PLAYER_STATE]
    status_code = ram[STATUS]

    return MarioRamState(
        x_pos=ram[X_PAGE] * 0x100 + ram[X_SCREEN],
        y_pos=255 + (255 - y_pixel) if y_viewport < 1 else 255 - y_pixel,
        time=_read_digits(ram, TIME_DIGITS, 3),
        coins=_read_digits(ram, COINS_DIGITS, 2),
        score=_read_digits(ram, SCORE_DIGITS, 6),
        life=ram[LIVES],
        world=ram[WORLD] + 1,
        stage=ram[STAGE] + 1,
        area=ram[AREA] + 1,
        status=_status_name(status_code),
        status_code=status_code,
        player_state=player_state,
        flag_get=ram[GAME_MODE] == 2 or _is_stage_over(ram),
        is_dying=player_state == 0x0B or y_viewport > 1,
        is_dead=player_state == 0x06,
        is_game_over=ram[LIVES] == 0xFF,
    )


def compute_reward(previous: MarioRamState, current: MarioRamState) -> RewardComponents:
    x_reward = current.x_pos - previous.x_pos
    if x_reward < -5 or x_reward > 5:
        x_reward = 0

    time_reward = current.time - previous.time
    if time_reward > 0:
        time_reward = 0

    death_penalty = -25 if current.is_dying or current.is_dead else 0
    return RewardComponents(
        x=x_reward,
        time=time_reward,
        death=death_penalty,
        total=x_reward + time_reward + death_penalty,
    )
