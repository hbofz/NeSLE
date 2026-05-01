from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import nesle

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/nesle-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/nesle-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"], "fontconfig").mkdir(parents=True, exist_ok=True)

try:
    import numpy as np

    if not hasattr(np, "uint8"):
        raise ImportError("numpy is present but incomplete")
except ImportError as exc:  # pragma: no cover - this is a command-line guard
    raise SystemExit("Phase 5 benchmarks require a complete NumPy install.") from exc


DEFAULT_ENV_COUNTS = (1, 8, 32, 128, 512, 1024, 2048, 4096)
DEFAULT_MODES = ("step", "render", "inference")


@dataclass(frozen=True)
class BenchmarkResult:
    runner: str
    mode: str
    backend: str
    num_envs: int
    steps: int
    frameskip: int
    duration_sec: float
    env_steps_per_sec: float
    training_frames_per_sec: float
    fps_per_env: float
    reset_rate: float
    reward_sum: float
    gpu: dict[str, Any]


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError("all counts must be positive")
    return items


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    items = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected at least one mode")
    unknown = sorted(set(items) - set(DEFAULT_MODES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown benchmark modes: {', '.join(unknown)}")
    return items


def _gpu_snapshot() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"available": False}

    gpus = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        name, util, mem_used, mem_total = parts
        gpus.append(
            {
                "name": name,
                "utilization_percent": int(util),
                "memory_used_mib": int(mem_used),
                "memory_total_mib": int(mem_total),
            }
        )
    return {"available": bool(gpus), "gpus": gpus}


def _make_actions(action_space: Any, rng: np.random.Generator, steps: int, num_envs: int) -> np.ndarray:
    return rng.integers(0, int(action_space.n), size=(steps, num_envs), dtype=np.int64)


def _benchmark_nesle(
    rom_path: str,
    mode: str,
    num_envs: int,
    steps: int,
    warmup_steps: int,
    frameskip: int,
    backend: str,
    action_space: str,
    seed: int,
) -> BenchmarkResult:
    env = nesle.make_vec(
        rom_path,
        num_envs=num_envs,
        frameskip=frameskip,
        backend=backend,
        action_space=action_space,
        render_mode="rgb_array",
    )
    rng = np.random.default_rng(seed)
    env.reset()
    try:
        selected_backend = str(env.get_attr("name", indices=0)[0])
    except Exception:
        selected_backend = backend
    warmup_actions = _make_actions(env.action_space, rng, warmup_steps, num_envs)
    for actions in warmup_actions:
        env.step(actions)
        if mode == "render":
            env.render()

    torch_model = None
    torch_device = None
    if mode == "inference":
        torch_model, torch_device = _make_torch_policy()

    actions_batch = _make_actions(env.action_space, rng, steps, num_envs)
    reward_sum = 0.0
    done_count = 0
    gpu_before = _gpu_snapshot()
    started = time.perf_counter()
    obs = None
    for actions in actions_batch:
        obs, rewards, dones, _infos = env.step(actions)
        reward_sum += float(np.sum(rewards))
        done_count += int(np.sum(dones))
        if mode == "render":
            env.render()
        elif mode == "inference":
            _run_torch_policy(torch_model, torch_device, obs)
    duration = max(time.perf_counter() - started, 1e-12)
    gpu_after = _gpu_snapshot()
    env.close()
    return BenchmarkResult(
        runner="nesle",
        mode=mode,
        backend=selected_backend,
        num_envs=num_envs,
        steps=steps,
        frameskip=frameskip,
        duration_sec=duration,
        env_steps_per_sec=(num_envs * steps) / duration,
        training_frames_per_sec=(num_envs * steps * frameskip) / duration,
        fps_per_env=steps / duration,
        reset_rate=done_count / max(num_envs * steps, 1),
        reward_sum=reward_sum,
        gpu={"before": gpu_before, "after": gpu_after, "torch_device": torch_device},
    )


def _make_torch_policy() -> tuple[Any, str]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise SystemExit("Install the 'rl' extra to run inference benchmarks.") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, kernel_size=8, stride=4),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, kernel_size=4, stride=2),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(32, 7),
    ).to(device)
    model.eval()
    return model, device


def _run_torch_policy(model: Any, device: str, obs: np.ndarray) -> None:
    import torch

    with torch.inference_mode():
        tensor = torch.as_tensor(obs, device=device, dtype=torch.float32)
        tensor = tensor.permute(0, 3, 1, 2).div_(255.0)
        model(tensor)


def _benchmark_legacy(
    env_id: str,
    num_envs: int,
    steps: int,
    warmup_steps: int,
    seed: int,
) -> BenchmarkResult:
    try:
        import gym_super_mario_bros
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        from nes_py.wrappers import JoypadSpace
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise SystemExit("Install the 'legacy-mario' extra to run the legacy benchmark.") from exc

    envs = [JoypadSpace(gym_super_mario_bros.make(env_id), SIMPLE_MOVEMENT) for _ in range(num_envs)]
    rng = np.random.default_rng(seed)
    for index, env in enumerate(envs):
        try:
            env.reset(seed=seed + index)
        except TypeError:
            env.reset()
    for _ in range(warmup_steps):
        for env in envs:
            env.step(int(rng.integers(0, env.action_space.n)))

    reward_sum = 0.0
    done_count = 0
    gpu_before = _gpu_snapshot()
    started = time.perf_counter()
    for _ in range(steps):
        for env in envs:
            result = env.step(int(rng.integers(0, env.action_space.n)))
            if len(result) == 5:
                _obs, reward, terminated, truncated, _info = result
                done = bool(terminated or truncated)
            else:
                _obs, reward, done, _info = result
            reward_sum += float(reward)
            done_count += int(done)
            if done:
                env.reset()
    duration = max(time.perf_counter() - started, 1e-12)
    for env in envs:
        env.close()
    return BenchmarkResult(
        runner="gym-super-mario-bros",
        mode="step",
        backend="legacy",
        num_envs=num_envs,
        steps=steps,
        frameskip=4,
        duration_sec=duration,
        env_steps_per_sec=(num_envs * steps) / duration,
        training_frames_per_sec=(num_envs * steps * 4) / duration,
        fps_per_env=steps / duration,
        reset_rate=done_count / max(num_envs * steps, 1),
        reward_sum=reward_sum,
        gpu={"before": gpu_before, "after": _gpu_snapshot(), "torch_device": None},
    )


def _write_results(results: Iterable[BenchmarkResult], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    if output.suffix.lower() == ".csv":
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                row = dict(row)
                row["gpu"] = json.dumps(row["gpu"], sort_keys=True)
                writer.writerow(row)
        return
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5 NeSLE throughput benchmarks.")
    parser.add_argument("rom_path", help="Path to a local Super Mario Bros. .nes ROM")
    parser.add_argument("--env-counts", type=_parse_csv_ints, default=DEFAULT_ENV_COUNTS)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--backend", choices=("auto", "native", "synthetic"), default="auto")
    parser.add_argument("--action-space", default="simple")
    parser.add_argument("--modes", type=_parse_csv_strings, default=DEFAULT_MODES)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--legacy-env-id", default="SuperMarioBros-v0")
    parser.add_argument("--legacy-max-envs", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/phase5.json"))
    args = parser.parse_args()

    rom_path = Path(args.rom_path)
    if not rom_path.is_file():
        raise SystemExit(f"ROM path does not point to a file: {rom_path}")
    if args.steps <= 0 or args.warmup_steps < 0 or args.frameskip <= 0:
        raise SystemExit("steps and frameskip must be positive; warmup-steps must be non-negative")

    results: list[BenchmarkResult] = []
    for num_envs in args.env_counts:
        for mode in args.modes:
            result = _benchmark_nesle(
                str(rom_path),
                mode,
                num_envs,
                args.steps,
                args.warmup_steps,
                args.frameskip,
                args.backend,
                args.action_space,
                args.seed,
            )
            results.append(result)
            print(
                f"{result.runner} mode={result.mode} envs={result.num_envs} "
                f"backend={result.backend} env_steps/s={result.env_steps_per_sec:.1f} "
                f"frames/s={result.training_frames_per_sec:.1f}"
            )

        if args.include_legacy and num_envs <= args.legacy_max_envs:
            result = _benchmark_legacy(
                args.legacy_env_id,
                num_envs,
                args.steps,
                args.warmup_steps,
                args.seed,
            )
            results.append(result)
            print(
                f"{result.runner} envs={result.num_envs} "
                f"env_steps/s={result.env_steps_per_sec:.1f} "
                f"frames/s={result.training_frames_per_sec:.1f}"
            )

    _write_results(results, args.output)
    print(f"wrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()
