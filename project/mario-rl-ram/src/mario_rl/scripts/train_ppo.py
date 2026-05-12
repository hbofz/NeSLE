from __future__ import annotations

import argparse
import re
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from mario_rl.env import make_mario_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Mario RL agent with PPO.")
    parser.add_argument("--algo", default="ppo", choices=["ppo", "recurrent-ppo"])
    parser.add_argument("--game", default="SuperMarioBros-Nes-v0")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="ppo-ram-mario-actions")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--log-dir", default="runs")
    parser.add_argument("--action-mode", default="mario", choices=["all", "discrete", "multidiscrete", "mario"])
    parser.add_argument("--reward-mode", default="smart", choices=["base", "shaped", "smart"])
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--single-life", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-freq", type=int, default=250_000)
    parser.add_argument("--resume-from", default=None, help="Path to a saved SB3 checkpoint to continue training.")
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True, help="Automatically resume from the latest checkpoint in model-dir.")
    # --- new args ---
    parser.add_argument("--state", default="Level1-1", help="Stable-Retro state (level) to train on, e.g. Level1-1.")
    parser.add_argument("--obs-mode", default="ram", choices=["ram", "pixel"], help="ram = MLP policy on RAM; pixel = CNN policy on 84x84 grayscale frames.")
    parser.add_argument("--flag-lock", action=argparse.BooleanOptionalAction, default=True, help="End episode when Mario completes the level (flag locked).")
    return parser.parse_args()


def make_env_factory(args: argparse.Namespace, rank: int):
    def _init():
        env = make_mario_env(
            game=args.game,
            state=args.state,
            obs_mode=args.obs_mode,
            action_mode=args.action_mode,
            reward_mode=args.reward_mode,
            action_repeat=args.action_repeat,
            render_mode=None,
            monitor=False,
            single_life=args.single_life,
            flag_lock=args.flag_lock,
        )
        env.reset(seed=args.seed + rank)
        return env

    return _init


def main() -> None:
    args = parse_args()
    run_model_dir = Path(args.model_dir) / args.run_name
    run_log_dir = Path(args.log_dir) / args.run_name
    run_model_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir.mkdir(parents=True, exist_ok=True)

    if args.auto_resume and not args.resume_from:
        checkpoints = list(run_model_dir.glob("*.zip"))
        if checkpoints:
            def get_step(path: Path) -> int:
                match = re.search(r"(\d+)_steps", path.name)
                return int(match.group(1)) if match else -1
            latest_checkpoint = max(checkpoints, key=get_step)
            if get_step(latest_checkpoint) >= 0:
                args.resume_from = str(latest_checkpoint)
                print(f"Auto-resume found latest checkpoint: {args.resume_from}")

    env = SubprocVecEnv(
        [make_env_factory(args, rank) for rank in range(args.n_envs)],
        start_method="spawn",
    )
    env = VecMonitor(env)

    is_pixel = args.obs_mode == "pixel"
    # CNN policy for pixel mode; MLP policy for RAM mode
    if args.algo == "ppo":
        policy = "CnnPolicy" if is_pixel else "MlpPolicy"
    else:
        policy = "CnnLstmPolicy" if is_pixel else "MlpLstmPolicy"

    # net_arch is meaningful only for MLP policies; CNN uses NatureCNN backbone
    policy_kwargs: dict = {} if is_pixel else dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    common_kwargs = dict(
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        tensorboard_log=str(args.log_dir),
        verbose=1,
        seed=args.seed,
        device=args.device,
        policy_kwargs=policy_kwargs,
    )

    model_cls = PPO if args.algo == "ppo" else RecurrentPPO
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        model = model_cls.load(
            args.resume_from,
            env=env,
            device=args.device,
            tensorboard_log=str(args.log_dir),
            print_system_info=True,
        )
        print(f"Loaded checkpoint with num_timesteps={model.num_timesteps}")
    else:
        model = model_cls(policy, **common_kwargs)

    checkpoint_freq = max(args.checkpoint_freq // args.n_envs, 1)
    checkpoint = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(run_model_dir),
        name_prefix=args.algo,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint,
        tb_log_name=args.run_name,
        reset_num_timesteps=not bool(args.resume_from),
        progress_bar=True,
    )
    model.save(run_model_dir / "final_model")
    env.close()


if __name__ == "__main__":
    main()
