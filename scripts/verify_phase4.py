from __future__ import annotations

import tempfile
from pathlib import Path

import nesle

try:
    import numpy as np

    if not hasattr(np, "uint8"):
        raise ImportError("numpy is present but incomplete")
except ImportError:
    print("complete numpy is not available; skipping Phase 4 smoke.")
    raise SystemExit(0)


PRG_BANK_SIZE = 16 * 1024
CHR_BANK_SIZE = 8 * 1024


def make_rom() -> bytes:
    header = bytearray(b"NES\x1a")
    header.extend([2, 1, 0, 0])
    header.extend(b"\x00" * 8)
    prg = bytearray([0xEA] * (2 * PRG_BANK_SIZE))
    prg[0x7FFC] = 0x00
    prg[0x7FFD] = 0x80
    return bytes(header + prg + bytearray(CHR_BANK_SIZE))


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        rom_path = Path(tmp) / "phase4.nes"
        rom_path.write_bytes(make_rom())

        env = nesle.make_vec(
            str(rom_path),
            num_envs=2,
            backend="synthetic",
            action_space="simple",
            max_episode_steps=2,
        )
        obs = env.reset()
        obs, rewards, dones, infos = env.step([1, 4])
        env.step_async([1, 1])
        obs, rewards, dones, infos = env.step_wait()
        assert obs.shape == (2, 240, 256, 3)
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert "terminal_observation" in infos[0]
        assert env.render().shape == (2, 240, 256, 3)
        env.close()

        single = nesle.make(str(rom_path), backend="synthetic", max_episode_steps=2)
        obs, info = single.reset(seed=123, options={"smoke": True})
        assert obs.shape == (240, 256, 3)
        assert info["reset_options"] == {"smoke": True}
        obs, reward, terminated, truncated, info = single.step(1)
        assert obs.shape == (240, 256, 3)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        single.close()

        try:
            from gymnasium.utils.env_checker import check_env
        except ImportError:
            print("gymnasium is not available; skipping Gymnasium env checker.")
        else:
            check_env(nesle.make(str(rom_path), backend="synthetic"), skip_render_check=True)
            print("gymnasium_check ok")

        try:
            from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
        except ImportError:
            print("stable-baselines3 is not available; skipping SB3 wrapper smoke.")
        else:
            sb3_env = nesle.make_vec(str(rom_path), num_envs=2, backend="synthetic")
            sb3_env = VecFrameStack(VecTransposeImage(sb3_env), n_stack=4)
            obs = sb3_env.reset()
            obs, rewards, dones, infos = sb3_env.step([1, 1])
            assert obs.shape[0] == 2
            assert rewards.shape == (2,)
            sb3_env.close()
            print("sb3_vec_smoke ok")

    print("phase4_smoke ok")


if __name__ == "__main__":
    main()
