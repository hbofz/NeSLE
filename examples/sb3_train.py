from __future__ import annotations

import argparse

import nesle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an SB3 PPO agent with NeSLE.")
    parser.add_argument("rom_path", help="Path to Super Mario Bros. (World).nes")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--action-space", default="simple", choices=["right_only", "simple", "complex", "raw"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backend", default="auto", choices=["auto", "native", "synthetic", "cuda"])
    parser.add_argument("--observation-mode", default="ram", choices=["ram", "rgb_array"])
    parser.add_argument("--policy", default="auto", choices=["auto", "MlpPolicy", "CnnPolicy"])
    parser.add_argument("--sb3-device", default="auto")
    parser.add_argument("--model-path", default="nesle_ppo")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("Install the 'rl' extra to run this example: pip install -e '.[rl]'") from exc

    if args.observation_mode == "ram" and args.policy == "CnnPolicy":
        raise SystemExit("CnnPolicy requires --observation-mode rgb_array.")
    if args.observation_mode == "rgb_array" and args.policy == "MlpPolicy":
        raise SystemExit("MlpPolicy requires --observation-mode ram.")

    env = nesle.make_vec(
        rom_path=args.rom_path,
        num_envs=args.num_envs,
        frameskip=args.frameskip,
        action_space=args.action_space,
        device=args.device,
        backend=args.backend,
        render_mode="rgb_array",
        observation_mode=args.observation_mode,
    )
    if args.observation_mode == "rgb_array":
        from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

        env = VecTransposeImage(env)
        env = VecFrameStack(env, n_stack=4)

    policy = args.policy
    if policy == "auto":
        policy = "MlpPolicy" if args.observation_mode == "ram" else "CnnPolicy"

    model = PPO(
        policy,
        env,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        device=args.sb3_device,
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(args.model_path)
    env.close()


if __name__ == "__main__":
    main()
