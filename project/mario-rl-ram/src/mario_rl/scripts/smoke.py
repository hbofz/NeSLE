from __future__ import annotations

import argparse

from mario_rl.env import make_mario_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test a Mario RL environment.")
    parser.add_argument("--game", default="SuperMarioBros-Nes-v0")
    parser.add_argument("--state", default="Level1-1", help="Level state to load, e.g. Level1-1.")
    parser.add_argument("--obs-mode", default="ram", choices=["ram", "pixel"], help="Observation mode: ram or pixel.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--action-mode", default="mario", choices=["all", "discrete", "multidiscrete", "mario"])
    parser.add_argument("--reward-mode", default="base", choices=["base", "shaped", "smart"])
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--single-life", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flag-lock", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        env = make_mario_env(
            game=args.game,
            state=args.state,
            obs_mode=args.obs_mode,
            action_mode=args.action_mode,
            reward_mode=args.reward_mode,
            action_repeat=args.action_repeat,
            render_mode=None,
            single_life=args.single_life,
            flag_lock=args.flag_lock,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\n\nImport your legally obtained ROM first:\n"
            "  python -m stable_retro.import roms/"
        ) from exc

    obs, info = env.reset()
    total_reward = 0.0
    last_info = info

    for _ in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, last_info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            obs, last_info = env.reset()

    print(f"observation_space={env.observation_space}")
    print(f"action_space={env.action_space}")
    print(f"last_obs_shape={getattr(obs, 'shape', None)}")
    print(f"total_reward={total_reward:.2f}")
    print(f"info_keys={sorted(last_info.keys())}")
    env.close()


if __name__ == "__main__":
    main()
