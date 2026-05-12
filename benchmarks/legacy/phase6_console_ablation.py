from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from phase5_benchmark import _GpuSampler, _gpu_snapshot, _parse_csv_ints

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - command-line guard
    raise SystemExit("Phase 6 ablations require NumPy.") from exc


DEFAULT_ENV_COUNTS = (1, 2, 8, 32, 64, 128)
DEFAULT_FRAMESKIPS = (1, 2, 4, 8)
DEFAULT_MODES = ("rgb", "ram_obs", "render_only", "no_copy")


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    items = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected at least one mode")
    unknown = sorted(set(items) - set(DEFAULT_MODES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown modes: {', '.join(unknown)}")
    return items


def _make_actions(action_masks: np.ndarray, rng: np.random.Generator, num_envs: int) -> np.ndarray:
    return action_masks[rng.integers(0, len(action_masks), size=num_envs)]


def _copy_flags(mode: str) -> tuple[bool, bool]:
    if mode == "rgb":
        return True, True
    if mode == "ram_obs":
        return False, False
    if mode == "render_only":
        return True, False
    if mode == "no_copy":
        return False, False
    raise ValueError(f"unknown mode: {mode}")


def _peak_gpu_field(gpu: dict[str, Any], field: str) -> int | None:
    gpus = gpu.get("gpus", [])
    if not gpus:
        return None
    value = gpus[0].get(field)
    return int(value) if value is not None else None


def _run_case(
    rom_bytes: bytes,
    action_masks: np.ndarray,
    num_envs: int,
    frameskip: int,
    mode: str,
    steps: int,
    warmup_steps: int,
    seed: int,
) -> dict[str, Any]:
    from nesle import _cuda_core  # type: ignore[attr-defined]

    render_frame, copy_obs = _copy_flags(mode)
    rng = np.random.default_rng(seed)
    batch = _cuda_core.CudaBatch(num_envs, frameskip, rom_bytes)
    batch.reset()
    for _ in range(warmup_steps):
        batch.step(_make_actions(action_masks, rng, num_envs), render_frame=render_frame, copy_obs=copy_obs)
        if mode == "ram_obs":
            batch.ram()

    reward_sum = 0.0
    done_count = 0
    gpu_before = _gpu_snapshot()
    with _GpuSampler() as gpu_sampler:
        started = time.perf_counter()
        for _ in range(steps):
            result = batch.step(
                _make_actions(action_masks, rng, num_envs),
                render_frame=render_frame,
                copy_obs=copy_obs,
            )
            if mode == "ram_obs":
                batch.ram()
            reward_sum += float(np.asarray(result["rewards"], dtype=np.float32).sum())
            done_count += int(np.asarray(result["dones"], dtype=bool).sum())
        duration = max(time.perf_counter() - started, 1e-12)

    peak = gpu_sampler.peak()
    env_steps_per_sec = (num_envs * steps) / duration
    return {
        "runner": "nesle",
        "backend": "cuda-console",
        "mode": mode,
        "num_envs": num_envs,
        "frameskip": frameskip,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "duration_sec": duration,
        "env_steps_per_sec": env_steps_per_sec,
        "training_frames_per_sec": env_steps_per_sec * frameskip,
        "fps_per_env": steps / duration,
        "reset_rate": done_count / max(num_envs * steps, 1),
        "reward_sum": reward_sum,
        "gpu": {
            "before": gpu_before,
            "after": _gpu_snapshot(),
            "peak": peak,
            "peak_utilization_percent": _peak_gpu_field(peak, "peak_utilization_percent"),
            "peak_memory_used_mib": _peak_gpu_field(peak, "peak_memory_used_mib"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 6 CUDA-console ablations.")
    parser.add_argument("rom_path", help="Path to a local Super Mario Bros. .nes ROM")
    parser.add_argument("--env-counts", type=_parse_csv_ints, default=DEFAULT_ENV_COUNTS)
    parser.add_argument("--frameskips", type=_parse_csv_ints, default=DEFAULT_FRAMESKIPS)
    parser.add_argument("--modes", type=_parse_csv_strings, default=DEFAULT_MODES)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--action-space", default="simple")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/phase6_console_ablation.json"))
    args = parser.parse_args()

    rom_path = Path(args.rom_path)
    if not rom_path.is_file():
        raise SystemExit(f"ROM path does not point to a file: {rom_path}")
    if args.steps <= 0 or args.warmup_steps < 0:
        raise SystemExit("steps must be positive; warmup-steps must be non-negative")

    from nesle.env import _action_masks

    rom_bytes = rom_path.read_bytes()
    action_masks = np.asarray(_action_masks(args.action_space), dtype=np.uint8)
    rows: list[dict[str, Any]] = []
    for frameskip in args.frameskips:
        for num_envs in args.env_counts:
            for mode in args.modes:
                row = _run_case(
                    rom_bytes,
                    action_masks,
                    num_envs,
                    frameskip,
                    mode,
                    args.steps,
                    args.warmup_steps,
                    args.seed,
                )
                rows.append(row)
                print(
                    f"mode={mode} envs={num_envs} frameskip={frameskip} "
                    f"env_steps/s={row['env_steps_per_sec']:.1f} "
                    f"frames/s={row['training_frames_per_sec']:.1f}"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
