from __future__ import annotations

import argparse
from pathlib import Path

import nesle


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke the Phase 4 Python API with a local ROM.")
    parser.add_argument("rom_path")
    parser.add_argument("--backend", default="auto", choices=["auto", "native", "synthetic"])
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--action-space", default="simple")
    args = parser.parse_args()

    rom_path = Path(args.rom_path)
    rom = nesle.parse_ines(rom_path.read_bytes())
    env = nesle.make_vec(
        str(rom_path),
        num_envs=args.num_envs,
        frameskip=args.frameskip,
        action_space=args.action_space,
        backend=args.backend,
        render_mode="rgb_array",
    )
    obs = env.reset()
    action_index = min(4, env.action_space.n - 1)
    total_reward = 0.0
    last_infos = env.reset_infos
    for _ in range(args.steps):
        obs, rewards, dones, last_infos = env.step([action_index] * args.num_envs)
        total_reward += float(rewards.sum())

    rendered = env.render()
    backend_names = sorted({info.get("backend", "unknown") for info in last_infos})
    env.close()

    print(
        "phase4_user_rom"
        f" mapper={rom.mapper}"
        f" nrom={int(rom.is_nrom)}"
        f" mario_target={int(rom.is_supported_mario_target)}"
        f" envs={args.num_envs}"
        f" steps={args.steps}"
        f" backend={','.join(backend_names)}"
        f" obs_shape={tuple(obs.shape)}"
        f" render_shape={tuple(rendered.shape)}"
        f" reward_sum={total_reward:.1f}"
        f" dones={','.join(str(int(done)) for done in dones)}"
    )


if __name__ == "__main__":
    main()
