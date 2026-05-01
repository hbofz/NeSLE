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
    parser.add_argument("--action-space", default="simple_with_start", choices=["right_only", "simple", "simple_with_start", "complex", "raw"])
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
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Stable-Baselines3 requires PyTorch; install the 'rl' extra.") from exc

    if args.observation_mode == "ram" and args.policy == "CnnPolicy":
        raise SystemExit("CnnPolicy requires --observation-mode rgb_array.")
    if args.observation_mode == "rgb_array" and args.policy == "MlpPolicy":
        raise SystemExit("MlpPolicy requires --observation-mode ram.")
    if args.sb3_device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "--sb3-device cuda was requested, but PyTorch CUDA is not available. "
            "Install a CUDA-enabled PyTorch build or use --sb3-device cpu."
        )

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
    env_backend = "unknown"
    if getattr(env, "_cuda_batch", None) is not None:
        env_backend = str(env._cuda_batch.name)
    elif hasattr(env, "config"):
        env_backend = str(env.config.backend)
    if args.observation_mode == "rgb_array":
        from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

        env = VecTransposeImage(env)
        env = VecFrameStack(env, n_stack=4)

    policy = args.policy
    if policy == "auto":
        policy = "MlpPolicy" if args.observation_mode == "ram" else "CnnPolicy"

    torch_device = args.sb3_device
    if torch_device == "auto":
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
    print(
        f"nesle_backend={env_backend} observation_mode={args.observation_mode} "
        f"sb3_device={torch_device} torch={torch.__version__} torch_cuda={cuda_name}"
    )

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
