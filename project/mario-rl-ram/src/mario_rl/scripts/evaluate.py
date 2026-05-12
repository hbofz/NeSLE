from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from mario_rl.env import make_mario_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Mario policy.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--algo", default="ppo", choices=["ppo", "recurrent-ppo"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--game", default="SuperMarioBros-Nes-v0")
    parser.add_argument("--state", default="Level1-1", help="Level state used during training, e.g. Level1-1.")
    parser.add_argument("--obs-mode", default="ram", choices=["ram", "pixel"], help="Must match the obs mode used during training.")
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--action-mode", default="mario", choices=["all", "discrete", "multidiscrete", "mario"])
    parser.add_argument("--reward-mode", default="smart", choices=["base", "shaped", "smart"])
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--single-life", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flag-lock", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_mode = "rgb_array" if args.video_dir else None
    env = make_mario_env(
        game=args.game,
        state=args.state,
        obs_mode=args.obs_mode,
        action_mode=args.action_mode,
        reward_mode=args.reward_mode,
        action_repeat=args.action_repeat,
        render_mode=render_mode,
        monitor=False,
        single_life=args.single_life,
        flag_lock=args.flag_lock,
    )

    if args.video_dir:
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            name_prefix=Path(args.model).stem,
            episode_trigger=lambda episode_id: True,
        )

    model_cls = PPO if args.algo == "ppo" else RecurrentPPO
    model = model_cls.load(args.model, env=env, device=args.device)

    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        lstm_states = None
        episode_start = True
        max_x = 0.0
        final_info = {}
        smart_reward_totals: dict[str, float] = {}

        while not done:
            if args.algo == "recurrent-ppo":
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=[episode_start],
                    deterministic=args.deterministic,
                )
            else:
                action, _ = model.predict(obs, deterministic=args.deterministic)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_start = done
            episode_reward += float(reward)
            final_info = info
            max_x = max(max_x, _x_position(info) or 0.0)
            _add_smart_reward_totals(smart_reward_totals, info)

        print(
            f"episode={episode + 1} "
            f"reward={episode_reward:.2f} "
            f"max_x={max_x:.0f} "
            f"score={final_info.get('score')} "
            f"coins={final_info.get('coins')} "
            f"time={final_info.get('time')} "
            f"lives={final_info.get('lives')} "
            f"smart_reward_total={_rounded_totals(smart_reward_totals)} "
            f"info={final_info}"
        )

    env.close()


def _x_position(info: dict) -> float | None:
    if "xscrollLo" in info and "xscrollHi" in info:
        return float(info["xscrollLo"]) + 256.0 * float(info["xscrollHi"])
    return None


def _add_smart_reward_totals(totals: dict[str, float], info: dict) -> None:
    components = info.get("smart_reward")
    if not isinstance(components, dict):
        return
    for name, value in components.items():
        try:
            totals[name] = totals.get(name, 0.0) + float(value)
        except (TypeError, ValueError):
            continue


def _rounded_totals(totals: dict[str, float]) -> dict[str, float]:
    return {name: round(value, 2) for name, value in sorted(totals.items())}


if __name__ == "__main__":
    main()
