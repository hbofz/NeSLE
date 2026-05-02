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
    SIMPLE_MOVEMENT_WITH_START_MASKS,
    encode_action,
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
RAM_SHAPE = (CPU_RAM_BYTES,)
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
        if key in {"simple_with_start", "simple_start"}:
            return SIMPLE_MOVEMENT_WITH_START_MASKS
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


def _normalize_observation_mode(observation_mode: str) -> str:
    key = observation_mode.lower().replace("-", "_")
    if key in {"rgb", "rgb_array", "frame"}:
        return "rgb_array"
    if key in {"ram", "cpu_ram"}:
        return "ram"
    raise ValueError("observation_mode must be 'rgb_array' or 'ram'")


@dataclass(frozen=True)
class BackendConfig:
    rom_path: str
    num_envs: int
    frameskip: int = 4
    action_space: str | Sequence[int] = "simple"
    device: str = "auto"
    backend: str = "auto"
    observation_mode: str = "rgb_array"
    observation_cadence: int = 1
    max_episode_steps: int = 0
    start_on_reset: bool = False
    reset_wait_steps: int = 10
    reset_start_steps: int = 2
    reset_post_start_steps: int = 60


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

    def ram_observation(self) -> np.ndarray:
        numpy = _require_numpy()
        return numpy.frombuffer(bytes(self.ram), dtype=numpy.uint8).copy()


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

    def ram_observation(self) -> np.ndarray:
        numpy = _require_numpy()
        return numpy.frombuffer(self.console.ram(), dtype=numpy.uint8).copy()


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


def _make_cuda_batch(num_envs: int, frameskip: int, rom_bytes: bytes | None = None) -> Any:
    from . import _cuda_core  # type: ignore[attr-defined]

    if not hasattr(_cuda_core, "CudaBatch"):
        raise RuntimeError("installed nesle._cuda_core does not expose CudaBatch")
    if rom_bytes is not None:
        return _cuda_core.CudaBatch(num_envs, frameskip, rom_bytes)
    return _cuda_core.CudaBatch(num_envs, frameskip)


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
        observation_mode: str = "rgb_array",
        seed: int | None = None,
        observation_cadence: int = 1,
        max_episode_steps: int = 0,
        start_on_reset: bool = False,
        reset_wait_steps: int = 10,
        reset_start_steps: int = 2,
        reset_post_start_steps: int = 60,
    ) -> None:
        numpy = _require_numpy()
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if frameskip <= 0:
            raise ValueError("frameskip must be positive")
        if observation_cadence <= 0:
            raise ValueError("observation_cadence must be positive")
        if reset_wait_steps < 0 or reset_start_steps < 0 or reset_post_start_steps < 0:
            raise ValueError("reset warmup step counts must be non-negative")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        observation_mode = _normalize_observation_mode(observation_mode)
        rom_bytes, rom = _load_rom(rom_path)
        self.config = BackendConfig(
            rom_path=str(rom_path),
            num_envs=num_envs,
            frameskip=frameskip,
            action_space=action_space,
            device=device,
            backend=backend,
            observation_mode=observation_mode,
            observation_cadence=observation_cadence,
            max_episode_steps=max_episode_steps,
            start_on_reset=start_on_reset,
            reset_wait_steps=reset_wait_steps,
            reset_start_steps=reset_start_steps,
            reset_post_start_steps=reset_post_start_steps,
        )
        self.rom = rom
        self.render_mode = render_mode
        self.observation_mode = observation_mode
        self.num_envs = num_envs
        self.frameskip = frameskip
        self.observation_cadence = observation_cadence
        self.start_on_reset = bool(start_on_reset)
        self.reset_wait_steps = int(reset_wait_steps)
        self.reset_start_steps = int(reset_start_steps)
        self.reset_post_start_steps = int(reset_post_start_steps)
        self.action_masks = _action_masks(action_space)
        self.action_space = _discrete(len(self.action_masks))
        observation_shape = RAM_SHAPE if observation_mode == "ram" else FRAME_SHAPE
        self.observation_space = _box(0, 255, observation_shape, numpy.uint8)
        if _StableBaselinesVecEnv is not None:
            _StableBaselinesVecEnv.__init__(self, num_envs, self.observation_space, self.action_space)
        self.reset_infos: list[dict[str, Any]] = [{} for _ in range(num_envs)]
        self.buf_infos: list[dict[str, Any]] = [{} for _ in range(num_envs)]
        self._pending_actions: np.ndarray | None = None
        self._seeds: list[int | None] = [None for _ in range(num_envs)]
        self._options: list[dict[str, Any]] = [{} for _ in range(num_envs)]
        self._closed = False
        self._cuda_observations: np.ndarray | None = None
        self._cuda_step_count = 0
        self._cuda_env_step_counts: np.ndarray | None = None
        cuda_requested = backend.lower() == "cuda" or (
            backend.lower() == "auto" and device.lower() == "cuda"
        )
        self._cuda_batch = None
        if cuda_requested:
            try:
                self._cuda_batch = _make_cuda_batch(num_envs, frameskip, rom_bytes)
            except Exception:
                if backend.lower() == "cuda":
                    raise
        if self._cuda_batch is not None:
            self._backends = []
            self._cuda_env_step_counts = numpy.zeros(num_envs, dtype=numpy.int64)
        else:
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
        if self._cuda_batch is not None:
            reset_frame = numpy.asarray(self._cuda_batch.reset(), dtype=numpy.uint8)
            if self.observation_mode == "ram":
                observations = numpy.asarray(self._cuda_batch.ram(), dtype=numpy.uint8)
                self._cuda_observations = reset_frame
            else:
                observations = reset_frame
                self._cuda_observations = observations
            self._cuda_step_count = 0
            if self._cuda_env_step_counts is not None:
                self._cuda_env_step_counts[:] = 0
            backend_name = str(self._cuda_batch.name)
            infos = [
                {
                    "backend": backend_name,
                    "observation_mode": self.observation_mode,
                }
                for _ in range(self.num_envs)
            ]
            for env in range(self.num_envs):
                if self._options[env]:
                    infos[env]["reset_options"] = dict(self._options[env])
            self.reset_infos = infos
            self._seeds = [None for _ in range(self.num_envs)]
            self._options = [{} for _ in range(self.num_envs)]
            if self.start_on_reset:
                observations = self._run_cuda_start_sequence()
                for info in self.reset_infos:
                    info["start_on_reset"] = True
            return observations

        observations = []
        infos = []
        for env, backend in enumerate(self._backends):
            obs, info = backend.reset(self._seeds[env])
            if self.observation_mode == "ram":
                obs = backend.ram_observation()
                info["observation_mode"] = "ram"
            if self._options[env]:
                info["reset_options"] = dict(self._options[env])
            observations.append(obs)
            infos.append(info)
        self.reset_infos = infos
        self._seeds = [None for _ in range(self.num_envs)]
        self._options = [{} for _ in range(self.num_envs)]
        if self.start_on_reset:
            observations = self._run_native_start_sequence()
            for info in self.reset_infos:
                info["start_on_reset"] = True
            return observations
        return numpy.stack(observations, axis=0)

    def _reset_start_sequence(self) -> tuple[int, int, int]:
        return self.reset_wait_steps, self.reset_start_steps, self.reset_post_start_steps

    def _run_cuda_start_sequence(self) -> np.ndarray:
        numpy = _require_numpy()
        if self._cuda_batch is None:
            raise RuntimeError("CUDA reset sequence requires a CUDA batch")
        noop = numpy.zeros(self.num_envs, dtype=numpy.uint8)
        start = numpy.full(self.num_envs, encode_action(["start"]), dtype=numpy.uint8)
        for mask, steps in zip((noop, start, noop), self._reset_start_sequence(), strict=True):
            for _ in range(steps):
                self._cuda_batch.step(mask, render_frame=False, copy_obs=False)
        if hasattr(self._cuda_batch, "poke_ram"):
            self._cuda_batch.poke_ram(0x0770, 1)
        self._cuda_step_count = 0
        if self._cuda_env_step_counts is not None:
            self._cuda_env_step_counts[:] = 0
        if self.observation_mode == "ram":
            observations = numpy.asarray(self._cuda_batch.ram(), dtype=numpy.uint8)
            self._cuda_observations = observations
            ram_observations = observations
        else:
            observations = numpy.asarray(self._cuda_batch.render(), dtype=numpy.uint8)
            self._cuda_observations = observations
            ram_observations = numpy.asarray(self._cuda_batch.ram(), dtype=numpy.uint8)
        backend_name = str(self._cuda_batch.name)
        self.reset_infos = [
            {
                **_info_from_state(read_ram(ram_observations[env]), None, backend_name),
                "observation_mode": self.observation_mode,
            }
            for env in range(self.num_envs)
        ]
        return observations

    def _run_native_start_sequence(self) -> np.ndarray:
        numpy = _require_numpy()
        noop = 0
        start = encode_action(["start"])
        for backend in self._backends:
            for mask, steps in zip((noop, start, noop), self._reset_start_sequence(), strict=True):
                for _ in range(steps):
                    backend.step(mask, self.frameskip)
            backend.step_count = 0
        observations = []
        infos = []
        for backend in self._backends:
            obs = backend.ram_observation() if self.observation_mode == "ram" else backend.render()
            state = read_ram(backend.ram_observation())
            info = _info_from_state(state, None, backend.name)
            info["observation_mode"] = self.observation_mode
            observations.append(obs)
            infos.append(info)
        self.reset_infos = infos
        return numpy.stack(observations, axis=0)

    def step_async(self, actions: Iterable[int]) -> None:
        numpy = _require_numpy()
        action_array = numpy.asarray(actions, dtype=numpy.int64).ravel()
        if action_array.shape != (self.num_envs,):
            raise ValueError(f"expected actions with shape ({self.num_envs},), got {action_array.shape}")
        self._pending_actions = action_array

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if self._pending_actions is None:
            raise RuntimeError("step_async must be called before step_wait")
        return self.step(self._pending_actions)

    def step(self, actions: Iterable[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        numpy = _require_numpy()
        action_array = numpy.asarray(actions, dtype=numpy.int64).ravel()
        if action_array.shape != (self.num_envs,):
            raise ValueError(f"expected actions with shape ({self.num_envs},), got {action_array.shape}")

        if self._cuda_batch is not None:
            if numpy.any(action_array < 0) or numpy.any(action_array >= len(self.action_masks)):
                raise ValueError("action index out of range")
            action_masks = numpy.asarray(
                [self.action_masks[int(action)] for action in action_array],
                dtype=numpy.uint8,
            )
            self._cuda_step_count += 1
            if self._cuda_env_step_counts is not None:
                self._cuda_env_step_counts += 1
            if self.observation_mode == "ram":
                result = self._cuda_batch.step(action_masks, render_frame=False, copy_obs=False)
                observations = numpy.asarray(self._cuda_batch.ram(), dtype=numpy.uint8)
                observations_copied = True
                observations_stale = False
                rgb_observations_copied = False
            else:
                copy_observations = self.observation_cadence <= 1 or (
                    self._cuda_step_count % self.observation_cadence == 0
                )
                result = self._cuda_batch.step(
                    action_masks,
                    render_frame=copy_observations,
                    copy_obs=copy_observations,
                )
                if copy_observations:
                    observations = numpy.asarray(result["obs"], dtype=numpy.uint8)
                    self._cuda_observations = observations
                elif self._cuda_observations is not None:
                    observations = self._cuda_observations
                else:
                    observations = numpy.asarray(self._cuda_batch.render(), dtype=numpy.uint8)
                    self._cuda_observations = observations
                observations_copied = copy_observations
                observations_stale = not copy_observations
                rgb_observations_copied = copy_observations
            rewards = numpy.asarray(result["rewards"], dtype=numpy.float32)
            dones = numpy.asarray(result["dones"], dtype=bool)

            # Episode truncation for CUDA path.
            max_steps = self.config.max_episode_steps
            if max_steps > 0 and self._cuda_env_step_counts is not None:
                truncated_mask = self._cuda_env_step_counts >= max_steps
                dones = dones | truncated_mask
            else:
                truncated_mask = numpy.zeros(self.num_envs, dtype=bool)

            backend_name = str(self._cuda_batch.name)
            infos = [
                {
                    "backend": backend_name,
                    "observation_mode": self.observation_mode,
                    "observation_cadence": self.observation_cadence,
                    "observations_copied": observations_copied,
                    "observations_stale": observations_stale,
                    "rgb_observations_copied": rgb_observations_copied,
                }
                for _ in range(self.num_envs)
            ]

            # SB3 auto-reset contract: save terminal observations and reset
            # done environments before returning.
            any_done = numpy.any(dones)
            if any_done:
                reset_mask = numpy.asarray(dones, dtype=numpy.uint8)
                for env in range(self.num_envs):
                    if dones[env]:
                        infos[env]["terminal_observation"] = observations[env].copy()
                        infos[env]["terminated"] = bool(dones[env] and not truncated_mask[env])
                        infos[env]["truncated"] = bool(truncated_mask[env])
                        self.reset_infos[env] = {
                            "backend": backend_name,
                            "observation_mode": self.observation_mode,
                        }
                # Reset done envs on device.
                self._cuda_batch.reset_envs(reset_mask)
                # Re-read observations for reset envs so the returned obs
                # reflects the new episode, not the terminal state.
                if self.observation_mode == "ram":
                    fresh_obs = numpy.asarray(self._cuda_batch.ram(), dtype=numpy.uint8)
                else:
                    fresh_obs = numpy.asarray(self._cuda_batch.render(), dtype=numpy.uint8)
                    self._cuda_observations = fresh_obs
                for env in range(self.num_envs):
                    if dones[env]:
                        observations[env] = fresh_obs[env]
                # Reset per-env step counts for done envs.
                if self._cuda_env_step_counts is not None:
                    self._cuda_env_step_counts[dones] = 0

            self._pending_actions = None
            self.buf_infos = infos
            return observations, rewards, dones, infos

        observations = []
        rewards = numpy.zeros(self.num_envs, dtype=numpy.float32)
        dones = numpy.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = []
        for env, (backend, action_index) in enumerate(zip(self._backends, action_array, strict=True)):
            if action_index < 0 or action_index >= len(self.action_masks):
                raise ValueError(f"action index out of range for env {env}: {action_index}")
            obs, reward, done, info = backend.step(self.action_masks[int(action_index)], self.frameskip)
            if self.observation_mode == "ram":
                obs = backend.ram_observation()
                info["observation_mode"] = "ram"
            rewards[env] = reward
            dones[env] = done
            if done:
                info["terminal_observation"] = obs.copy()
                obs, reset_info = backend.reset()
                if self.observation_mode == "ram":
                    obs = backend.ram_observation()
                    reset_info["observation_mode"] = "ram"
                self.reset_infos[env] = reset_info
            observations.append(obs)
            infos.append(info)
        self._pending_actions = None
        self.buf_infos = infos
        return numpy.stack(observations, axis=0), rewards, dones, infos

    def step_reward(self, actions: Iterable[int]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Step the CUDA backend without rendering or copying RGB observations.

        This is intended for high-throughput training loops that consume rewards,
        done flags, and occasional explicit renders instead of full RGB frames on
        every environment step.
        """

        numpy = _require_numpy()
        if self._cuda_batch is None:
            raise RuntimeError("step_reward requires backend='cuda'")
        action_array = numpy.asarray(actions, dtype=numpy.int64).ravel()
        if action_array.shape != (self.num_envs,):
            raise ValueError(f"expected actions with shape ({self.num_envs},), got {action_array.shape}")
        if numpy.any(action_array < 0) or numpy.any(action_array >= len(self.action_masks)):
            raise ValueError("action index out of range")

        action_masks = numpy.asarray(
            [self.action_masks[int(action)] for action in action_array],
            dtype=numpy.uint8,
        )
        if self._cuda_env_step_counts is not None:
            self._cuda_env_step_counts += 1
        result = self._cuda_batch.step(action_masks, render_frame=False, copy_obs=False)
        rewards = numpy.asarray(result["rewards"], dtype=numpy.float32)
        dones = numpy.asarray(result["dones"], dtype=bool)

        max_steps = self.config.max_episode_steps
        if max_steps > 0 and self._cuda_env_step_counts is not None:
            truncated_mask = self._cuda_env_step_counts >= max_steps
            dones = dones | truncated_mask
        else:
            truncated_mask = numpy.zeros(self.num_envs, dtype=bool)

        backend_name = str(self._cuda_batch.name)
        infos = [
            {
                "backend": backend_name,
                "observation_mode": self.observation_mode,
                "observations_copied": False,
            }
            for _ in range(self.num_envs)
        ]

        if numpy.any(dones):
            terminal_observations = (
                numpy.asarray(self._cuda_batch.ram(), dtype=numpy.uint8)
                if self.observation_mode == "ram"
                else numpy.asarray(self._cuda_batch.render(), dtype=numpy.uint8)
            )
            reset_mask = numpy.asarray(dones, dtype=numpy.uint8)
            for env in range(self.num_envs):
                if dones[env]:
                    infos[env]["terminal_observation"] = terminal_observations[env].copy()
                    infos[env]["terminated"] = bool(dones[env] and not truncated_mask[env])
                    infos[env]["truncated"] = bool(truncated_mask[env])
                    self.reset_infos[env] = {
                        "backend": backend_name,
                        "observation_mode": self.observation_mode,
                    }
            self._cuda_batch.reset_envs(reset_mask)
            if self._cuda_env_step_counts is not None:
                self._cuda_env_step_counts[dones] = 0

        self._pending_actions = None
        self.buf_infos = infos
        return rewards, dones, infos

    def render(self, mode: str | None = None) -> np.ndarray | None:
        mode = self.render_mode if mode is None else mode
        if mode is None:
            return None
        if mode != "rgb_array":
            raise ValueError("only rgb_array rendering is supported")
        if self._cuda_batch is not None:
            return _require_numpy().asarray(self._cuda_batch.render(), dtype=FRAME_DTYPE)
        return _require_numpy().stack([backend.render() for backend in self._backends], axis=0)

    def get_images(self) -> list[np.ndarray]:
        if self._cuda_batch is not None:
            rendered = self.render()
            return [rendered[env].copy() for env in range(self.num_envs)] if rendered is not None else []
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
        if attr_name == "render_mode":
            return [self.render_mode for _ in self._resolve_indices(indices)]
        if self._cuda_batch is not None and attr_name == "name":
            return [str(self._cuda_batch.name) for _ in self._resolve_indices(indices)]
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
        if self._cuda_batch is not None and method_name == "render":
            rendered = self.render()
            return [rendered[i].copy() for i in self._resolve_indices(indices)] if rendered is not None else []
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
        observation_mode: str = "rgb_array",
        observation_cadence: int = 1,
        max_episode_steps: int = 0,
        start_on_reset: bool = False,
        reset_wait_steps: int = 10,
        reset_start_steps: int = 2,
        reset_post_start_steps: int = 60,
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
            observation_mode=observation_mode,
            observation_cadence=observation_cadence,
            max_episode_steps=max_episode_steps,
            start_on_reset=start_on_reset,
            reset_wait_steps=reset_wait_steps,
            reset_start_steps=reset_start_steps,
            reset_post_start_steps=reset_post_start_steps,
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
