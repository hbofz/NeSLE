from __future__ import annotations

from dataclasses import dataclass
from typing import Any


try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised only without optional deps
    gym = None
    spaces = None


@dataclass(frozen=True)
class BackendConfig:
    rom_path: str
    num_envs: int
    frameskip: int = 4
    action_space: str = "simple"
    device: str = "cuda"


class NesleVecEnv:
    """SB3-style vector API placeholder for the upcoming CUDA backend."""

    def __init__(
        self,
        rom_path: str,
        num_envs: int,
        frameskip: int = 4,
        action_space: str = "simple",
        device: str = "cuda",
        render_mode: str | None = "rgb_array",
    ) -> None:
        self.config = BackendConfig(
            rom_path=rom_path,
            num_envs=num_envs,
            frameskip=frameskip,
            action_space=action_space,
            device=device,
        )
        self.render_mode = render_mode
        self.num_envs = num_envs
        self.reset_infos: list[dict[str, Any]] = [{} for _ in range(num_envs)]
        raise RuntimeError(
            "NeSLE CUDA backend is not implemented yet. Phase 0 provides the "
            "project scaffold, ROM parser, Mario RAM utilities, and tests."
        )


if gym is not None:

    class NesleEnv(gym.Env):
        metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

        def __init__(self, rom_path: str, render_mode: str | None = "rgb_array") -> None:
            super().__init__()
            self.rom_path = rom_path
            self.render_mode = render_mode
            self.observation_space = spaces.Box(0, 255, shape=(240, 256, 3), dtype="uint8")
            self.action_space = spaces.Discrete(256)
            raise RuntimeError(
                "NeSLE single-env backend is not implemented yet. Use the Phase 0 "
                "utilities while the emulator core is under construction."
            )

else:

    class NesleEnv:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Install gymnasium to use NesleEnv")
