from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import numpy as np

    if not hasattr(np, "uint8"):
        raise ImportError("numpy is present but incomplete")
except ImportError:  # pragma: no cover - only hit in broken local installs
    np = None  # type: ignore[assignment]

from .actions import (
    COMPLEX_MOVEMENT_MASKS,
    RIGHT_ONLY_MASKS,
    SIMPLE_MOVEMENT_MASKS,
)
from .rom import INESRom, parse_ines
from .smb import CPU_RAM_BYTES, MarioRamState, compute_reward, read_ram


try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised when optional deps are absent
    gym = None
    spaces = None

try:
    from stable_baselines3.common.vec_env import VecEnv as _StableBaselinesVecEnv
except ImportError:  # pragma: no cover - exercised when optional deps are absent
    _StableBaselinesVecEnv = None


FRAME_SHAPE = (240, 256, 3)
FRAME_DTYPE = np.uint8 if np is not None else None


class DiscreteSpace:
    def __init__(self, n: int) -> None:
        self.n = int(n)
        self.shape = ()
        self.dtype = np.int64

    def sample(self) -> int:
        return int(np.random.default_rng().integers(self.n))

    def contains(self, value: Any) -> bool:
        try:
            item = int(value)
        except (TypeError, ValueError):
            return False
        return 0 <= item < self.n


class BoxSpace:
    def __init__(self, low: int, high: int, shape: tuple[int, ...], dtype: Any) -> None:
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = np.dtype(dtype)

    def sample(self) -> np.ndarray:
        rng = np.random.default_rng()
        return rng.integers(self.low, self.high + 1, size=self.shape, dtype=self.dtype)

    def contains(self, value: Any) -> bool:
        array = np.asarray(value)
        return array.shape == self.shape and array.dtype == self.dtype


def _box(low: int, high: int, shape: tuple[int, ...], dtype: Any) -> Any:
    if spaces is not None:
        return spaces.Box(low, high, shape=shape, dtype=dtype)
    return BoxSpace(low, high, shape, dtype)


def _discrete(n: int) -> Any:
    if spaces is not None:
        return spaces.Discrete(n)
    return DiscreteSpace(n)


def _load_rom(rom_path: str | Path) -> tuple[bytes, INESRom]:
    data = Path(rom_path).read_bytes()
    return data, parse_ines(data)


def _action_masks(action_space: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(action_space, str):
        key = action_space.lower().replace("-", "_")
        if key in {"right", "right_only"}:
            return RIGHT_ONLY_MASKS
        if key == "simple":
            return SIMPLE_MOVEMENT_MASKS
        if key == "complex":
            return COMPLEX_MOVEMENT_MASKS
        if key in {"full", "raw"}:
            return tuple(range(256))
        raise ValueError(f"unknown action space: {action_space!r}")
    masks = tuple(int(mask) & 0xFF for mask in action_space)
    if not masks:
        raise ValueError("custom action space must contain at least one action mask")
    return masks


def _require_numpy() -> Any:
    if np is None:
        raise ImportError("Install a complete numpy package to use nesle.env")
    return np


def _default_ram() -> bytearray:
    ram = bytearray(CPU_RAM_BYTES)
    ram[0x006D] = 1
    ram[0x0086] = 2
    ram[0x00B5] = 1
    ram[0x03B8] = 100
    ram[0x075A] = 2
    ram[0x075C] = 0
    ram[0x075F] = 0
    ram[0x0760] = 0
    ram[0x0756] = 1
    ram[0x0770] = 0
    ram[0x07F8] = 4
    ram[0x07F9] = 0
    ram[0x07FA] = 0
    return ram


def _write_time(ram: bytearray, value: int) -> None:
    value = max(0, min(999, int(value)))
    ram[0x07F8] = value // 100
    ram[0x07F9] = (value // 10) % 10
    ram[0x07FA] = value % 10


def _write_x(ram: bytearray, value: int) -> None:
    value = max(0, min(0xFFFF, int(value)))
    ram[0x006D] = (value >> 8) & 0xFF
    ram[0x0086] = value & 0xFF


def _info_from_state(state: MarioRamState, reward_components: Any | None, backend: str) -> dict[str, Any]:
    info = asdict(state)
    info["backend"] = backend
    if reward_components is not None:
        info["reward_components"] = asdict(reward_components)
    return info


@dataclass(frozen=True)
class BackendConfig:
    rom_path: str
    num_envs: int
    frameskip: int = 4
    action_space: str | Sequence[int] = "simple"
    device: str = "auto"
    backend: str = "auto"
    max_episode_steps: int = 0


class _SyntheticBackend:
    name = "synthetic"

    def __init__(self, rom: INESRom, seed: int | None = None, max_episode_steps: int = 0) -> None:
        numpy = _require_numpy()
        self.rom = rom
        self.max_episode_steps = max_episode_steps
        self.rng = numpy.random.default_rng(seed)
        self.ram = _default_ram()
        self.frame = numpy.zeros(FRAME_SHAPE, dtype=FRAME_DTYPE)
        self.step_count = 0
        self.previous_state = read_ram(self.ram)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = _require_numpy().random.default_rng(seed)
        self.ram = _default_ram()
        self.frame.fill(0)
        self.step_count = 0
        self.previous_state = read_ram(self.ram)
        return self.frame.copy(), _info_from_state(self.previous_state, None, self.name)

    def step(self, action_mask: int, frameskip: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        previous = self.previous_state
        x_delta = 0
        if action_mask & 0x80:
            x_delta += 1 + int(bool(action_mask & 0x02))
        if action_mask & 0x40:
            x_delta -= 1
        _write_x(self.ram, previous.x_pos + x_delta)
        if self.step_count % max(1, 24 // max(1, frameskip)) == 0:
            _write_time(self.ram, previous.time - 1)
        if action_mask & 0x01:
            self.ram[0x03B8] = max(0, self.ram[0x03B8] - 1)

        self.step_count += 1
        current = read_ram(self.ram)
        reward_components = compute_reward(previous, current)
        self.previous_state = current
        terminated = current.is_dying or current.is_dead or current.is_game_over or current.flag_get
        truncated = self.max_episode_steps > 0 and self.step_count >= self.max_episode_steps
        done = terminated or truncated
        self.frame[:, :, 0] = (current.x_pos + self.step_count) & 0xFF
        self.frame[:, :, 1] = current.time & 0xFF
        self.frame[:, :, 2] = action_mask & 0xFF
        info = _info_from_state(current, reward_components, self.name)
        info["terminated"] = terminated
        info["truncated"] = truncated
        return self.frame.copy(), float(reward_components.total), done, info

    def render(self) -> np.ndarray:
        return self.frame.copy()


class _NativeBackend:
    name = "native"

    def __init__(self, rom_bytes: bytes, max_episode_steps: int = 0) -> None:
        from . import _core  # type: ignore[attr-defined]

        if not hasattr(_core, "NativeConsole"):
            raise RuntimeError("installed nesle._core does not expose NativeConsole")
        self.console = _core.NativeConsole(rom_bytes)
        self.max_episode_steps = max_episode_steps
        self.step_count = 0
        self.previous_state: MarioRamState | None = None

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        del seed
        self.console.reset()
        self.step_count = 0
        self.previous_state = read_ram(self.console.ram())
        return self.render(), _info_from_state(self.previous_state, None, self.name)

    def step(self, action_mask: int, frameskip: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.previous_state is None:
            self.reset()
        previous = self.previous_state or read_ram(self.console.ram())
        native_info = self.console.step(int(action_mask) & 0xFF, int(frameskip), 200_000)
        self.step_count += 1
        current = read_ram(self.console.ram())
        reward_components = compute_reward(previous, current)
        self.previous_state = current
        terminated = current.is_dying or current.is_dead or current.is_game_over or current.flag_get
        truncated = self.max_episode_steps > 0 and self.step_count >= self.max_episode_steps
        done = terminated or truncated
        info = _info_from_state(current, reward_components, self.name)
        info.update(native_info)
        info["terminated"] = terminated
        info["truncated"] = truncated
        return self.render(), float(reward_components.total), done, info

    def render(self) -> np.ndarray:
        numpy = _require_numpy()
        return numpy.frombuffer(self.console.frame(), dtype=numpy.uint8).reshape(FRAME_SHAPE).copy()


def _make_backend(
    backend: str,
    device: str,
    rom_bytes: bytes,
    rom: INESRom,
    seed: int | None,
    max_episode_steps: int,
) -> _NativeBackend | _SyntheticBackend:
    requested = backend.lower()
    if requested not in {"auto", "native", "synthetic"}:
        raise ValueError(f"unknown backend: {backend!r}")
    if requested in {"auto", "native"} and device.lower() in {"auto", "cpu", "cuda"}:
        try:
            return _NativeBackend(rom_bytes, max_episode_steps=max_episode_steps)
        except Exception:
            if requested == "native":
                raise
    return _SyntheticBackend(rom, seed=seed, max_episode_steps=max_episode_steps)


_VecEnvBase = _StableBaselinesVecEnv if _StableBaselinesVecEnv is not None else object


class NesleVecEnv(_VecEnvBase):
    """SB3-style vector API for NeSLE environments.

    The current Phase 4 surface supports SB3's ``reset``/``step`` vector
    contract and auto-reset behavior. It uses the native C++ console when the
    extension exposes it, otherwise it falls back to a deterministic Python
    compatibility backend.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        rom_path: str,
        num_envs: int,
        frameskip: int = 4,
        action_space: str | Sequence[int] = "simple",
        device: str = "auto",
        backend: str = "auto",
        render_mode: str | None = "rgb_array",
        seed: int | None = None,
        max_episode_steps: int = 0,
    ) -> None:
        numpy = _require_numpy()
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if frameskip <= 0:
            raise ValueError("frameskip must be positive")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        rom_bytes, rom = _load_rom(rom_path)
        self.config = BackendConfig(
            rom_path=str(rom_path),
            num_envs=num_envs,
            frameskip=frameskip,
            action_space=action_space,
            device=device,
            backend=backend,
            max_episode_steps=max_episode_steps,
        )
        self.rom = rom
        self.render_mode = render_mode
        self.num_envs = num_envs
        self.frameskip = frameskip
        self.action_masks = _action_masks(action_space)
        self.action_space = _discrete(len(self.action_masks))
        self.observation_space = _box(0, 255, FRAME_SHAPE, numpy.uint8)
        if _StableBaselinesVecEnv is not None:
            _StableBaselinesVecEnv.__init__(self, num_envs, self.observation_space, self.action_space)
        self.reset_infos: list[dict[str, Any]] = [{} for _ in range(num_envs)]
        self.buf_infos: list[dict[str, Any]] = [{} for _ in range(num_envs)]
        self._pending_actions: np.ndarray | None = None
        self._seeds: list[int | None] = [None for _ in range(num_envs)]
        self._options: list[dict[str, Any]] = [{} for _ in range(num_envs)]
        self._closed = False
        self._backends = [
            _make_backend(
                backend,
                device,
                rom_bytes,
                rom,
                None if seed is None else seed + env,
                max_episode_steps,
            )
            for env in range(num_envs)
        ]

    def reset(self) -> np.ndarray:
        numpy = _require_numpy()
        observations = []
        infos = []
        for env, backend in enumerate(self._backends):
            obs, info = backend.reset(self._seeds[env])
            if self._options[env]:
                info["reset_options"] = dict(self._options[env])
            observations.append(obs)
            infos.append(info)
        self.reset_infos = infos
        self._seeds = [None for _ in range(self.num_envs)]
        self._options = [{} for _ in range(self.num_envs)]
        return numpy.stack(observations, axis=0)

    def step_async(self, actions: Iterable[int]) -> None:
        numpy = _require_numpy()
        action_array = numpy.asarray(list(actions), dtype=numpy.int64)
        if action_array.shape != (self.num_envs,):
            raise ValueError(f"expected actions with shape ({self.num_envs},), got {action_array.shape}")
        self._pending_actions = action_array

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if self._pending_actions is None:
            raise RuntimeError("step_async must be called before step_wait")
        return self.step(self._pending_actions)

    def step(self, actions: Iterable[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        numpy = _require_numpy()
        action_array = numpy.asarray(list(actions), dtype=numpy.int64)
        if action_array.shape != (self.num_envs,):
            raise ValueError(f"expected actions with shape ({self.num_envs},), got {action_array.shape}")

        observations = []
        rewards = numpy.zeros(self.num_envs, dtype=numpy.float32)
        dones = numpy.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = []
        for env, (backend, action_index) in enumerate(zip(self._backends, action_array, strict=True)):
            if action_index < 0 or action_index >= len(self.action_masks):
                raise ValueError(f"action index out of range for env {env}: {action_index}")
            obs, reward, done, info = backend.step(self.action_masks[int(action_index)], self.frameskip)
            rewards[env] = reward
            dones[env] = done
            if done:
                info["terminal_observation"] = obs.copy()
                obs, reset_info = backend.reset()
                self.reset_infos[env] = reset_info
            observations.append(obs)
            infos.append(info)
        self._pending_actions = None
        self.buf_infos = infos
        return numpy.stack(observations, axis=0), rewards, dones, infos

    def render(self, mode: str | None = None) -> np.ndarray | None:
        mode = self.render_mode if mode is None else mode
        if mode is None:
            return None
        if mode != "rgb_array":
            raise ValueError("only rgb_array rendering is supported")
        return _require_numpy().stack([backend.render() for backend in self._backends], axis=0)

    def get_images(self) -> list[np.ndarray]:
        return [backend.render() for backend in self._backends]

    def close(self) -> None:
        self._closed = True

    def seed(self, seed: int | None = None) -> list[int | None]:
        self._seeds = [None if seed is None else seed + env for env in range(self.num_envs)]
        return list(self._seeds)

    def set_options(self, options: dict[str, Any] | Sequence[dict[str, Any]] | None = None) -> None:
        if options is None:
            self._options = [{} for _ in range(self.num_envs)]
            return
        if isinstance(options, dict):
            self._options = [dict(options) for _ in range(self.num_envs)]
            return
        if len(options) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} option dictionaries, got {len(options)}")
        self._options = [dict(item) for item in options]

    def get_attr(self, attr_name: str, indices: Sequence[int] | int | None = None) -> list[Any]:
        return [getattr(self._backends[i], attr_name) for i in self._resolve_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices: Sequence[int] | int | None = None) -> None:
        for i in self._resolve_indices(indices):
            setattr(self._backends[i], attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: Sequence[int] | int | None = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        return [
            getattr(self._backends[i], method_name)(*method_args, **method_kwargs)
            for i in self._resolve_indices(indices)
        ]

    def env_is_wrapped(self, wrapper_class: type[Any], indices: Sequence[int] | int | None = None) -> list[bool]:
        del wrapper_class
        return [False for _ in self._resolve_indices(indices)]

    def _resolve_indices(self, indices: Sequence[int] | int | None) -> list[int]:
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return [int(index) for index in indices]


if gym is not None:
    _EnvBase = gym.Env
else:
    _EnvBase = object


class NesleEnv(_EnvBase):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        rom_path: str,
        frameskip: int = 4,
        action_space: str | Sequence[int] = "simple",
        device: str = "auto",
        backend: str = "auto",
        render_mode: str | None = "rgb_array",
        max_episode_steps: int = 0,
    ) -> None:
        if gym is not None:
            super().__init__()
        self.vector_env = NesleVecEnv(
            rom_path=rom_path,
            num_envs=1,
            frameskip=frameskip,
            action_space=action_space,
            device=device,
            backend=backend,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )
        self.render_mode = render_mode
        self.observation_space = self.vector_env.observation_space
        self.action_space = self.vector_env.action_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if gym is not None:
            super().reset(seed=seed)
        if seed is not None:
            self.vector_env.seed(seed)
        if options is not None:
            self.vector_env.set_options(options)
        observations = self.vector_env.reset()
        return observations[0], self.vector_env.reset_infos[0]

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observations, rewards, dones, infos = self.vector_env.step([action])
        info = infos[0]
        return (
            observations[0],
            float(rewards[0]),
            bool(info.get("terminated", dones[0])),
            bool(info.get("truncated", False)),
            info,
        )

    def render(self) -> np.ndarray | None:
        rendered = self.vector_env.render()
        return None if rendered is None else rendered[0]

    def close(self) -> None:
        self.vector_env.close()
