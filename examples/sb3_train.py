from __future__ import annotations

import argparse
import time

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
    parser.add_argument("--start-on-reset", action="store_true", help="Boot each reset into controllable gameplay.")
    parser.add_argument("--reset-wait-steps", type=int, default=10)
    parser.add_argument("--reset-start-steps", type=int, default=2)
    parser.add_argument("--reset-post-start-steps", type=int, default=60)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=1.0)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
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

    class ProgressCallback(BaseCallback):
        def __init__(self, total_timesteps: int, interval_sec: float) -> None:
            super().__init__()
            self.total_timesteps = max(1, int(total_timesteps))
            self.interval_sec = max(0.1, float(interval_sec))
            self.last_update = 0.0
            self.bar = None

        def _on_training_start(self) -> None:
            self.last_update = time.monotonic()
            try:
                from tqdm.auto import tqdm
            except ImportError:
                self._print_progress()
                return
            self.bar = tqdm(total=self.total_timesteps, desc="training", unit="steps")

        def _on_step(self) -> bool:
            current = min(self.num_timesteps, self.total_timesteps)
            if self.bar is not None:
                delta = current - self.bar.n
                if delta > 0:
                    self.bar.update(delta)
                return True
            now = time.monotonic()
            if now - self.last_update >= self.interval_sec or current >= self.total_timesteps:
                self._print_progress()
                self.last_update = now
            return True

        def _on_training_end(self) -> None:
            current = min(self.num_timesteps, self.total_timesteps)
            if self.bar is not None:
                delta = current - self.bar.n
                if delta > 0:
                    self.bar.update(delta)
                self.bar.close()
            else:
                self._print_progress()
                print()

        def _print_progress(self) -> None:
            current = min(self.num_timesteps, self.total_timesteps)
            frac = current / self.total_timesteps
            width = 32
            filled = int(width * frac)
            bar = "#" * filled + "-" * (width - filled)
            print(
                f"\rtraining [{bar}] {frac * 100:6.2f}% "
                f"{current}/{self.total_timesteps}",
                end="",
                flush=True,
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
        start_on_reset=args.start_on_reset,
        reset_wait_steps=args.reset_wait_steps,
        reset_start_steps=args.reset_start_steps,
        reset_post_start_steps=args.reset_post_start_steps,
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
    callback = ProgressCallback(args.timesteps, args.progress_interval) if args.progress_bar else None
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(args.model_path)
    env.close()


if __name__ == "__main__":
    main()
