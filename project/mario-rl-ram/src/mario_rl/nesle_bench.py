from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


BUTTON_BITS: dict[str, int] = {
    "A": 0,
    "B": 1,
    "SELECT": 2,
    "START": 3,
    "UP": 4,
    "DOWN": 5,
    "LEFT": 6,
    "RIGHT": 7,
}

NESLE_MARIO_ACTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("NOOP", ()),
    ("RIGHT", ("RIGHT",)),
    ("RIGHT_RUN", ("RIGHT", "B")),
    ("RIGHT_JUMP", ("RIGHT", "A")),
    ("RIGHT_RUN_JUMP", ("RIGHT", "B", "A")),
    ("JUMP", ("A",)),
    ("RUN_JUMP", ("B", "A")),
    ("LEFT", ("LEFT",)),
    ("LEFT_JUMP", ("LEFT", "A")),
    ("DOWN", ("DOWN",)),
    ("RIGHT_DOWN", ("RIGHT", "DOWN")),
)

DEFAULT_ENV_COUNTS = (
    1,
    8,
    32,
    128,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    98304,
    131072,
)
DEFAULT_PPO_COUNTS = (1024, 4096, 8192, 16384, 32768, 65536, 98304, 131072)


@dataclass(frozen=True)
class CaseResult:
    suite: str
    mode: str
    status: str
    num_envs: int
    frameskip: int
    warmup_steps: int
    timed_steps: int
    duration_sec: float
    env_steps_per_sec: float
    frame_steps_per_sec: float
    gpu_before: dict[str, Any]
    gpu_after: dict[str, Any]
    error: str = ""
    extra: dict[str, Any] | None = None


def encode_nes_buttons(buttons: Iterable[str]) -> int:
    mask = 0
    for button in buttons:
        key = button.upper()
        if key not in BUTTON_BITS:
            raise ValueError(f"unknown NES button: {button!r}")
        mask |= 1 << BUTTON_BITS[key]
    return mask


def nesle_mario_action_masks() -> tuple[int, ...]:
    return tuple(encode_nes_buttons(buttons) for _, buttons in NESLE_MARIO_ACTIONS)


def parse_csv_ints(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        items = tuple(int(item) for item in value)
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError("all counts must be positive")
    return items


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_rom_path() -> Path:
    return repo_root() / "roms" / "Super Mario Bros. (World).nes"


def find_nesle_root(explicit: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("NESLE_ROOT"):
        candidates.append(Path(os.environ["NESLE_ROOT"]))
    candidates.extend([Path.cwd(), *Path(__file__).resolve().parents])
    for base in candidates:
        base = base.expanduser().resolve()
        for candidate in (base, base.parent, base.parent.parent):
            if (candidate / "src" / "nesle" / "env.py").is_file():
                return candidate
    return None


def default_snapshot_path(nesle_root: Path | None) -> Path:
    if nesle_root is not None:
        return nesle_root / "docs" / "data" / "smb_level1_1.state"
    return Path("docs/data/smb_level1_1.state")


def load_reset_state(path: Path) -> bytes:
    raw = path.read_bytes()
    if len(raw) >= 2 and raw[0] == 0x1F and raw[1] == 0x8B:
        return gzip.decompress(raw)
    return raw


def run_command(cmd: Sequence[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=None if cwd is None else str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def setup_nesle(nesle_root: Path, cuda_arch: str) -> None:
    install = run_command([sys.executable, "-m", "pip", "install", "-e", str(nesle_root)])
    if install.returncode != 0:
        raise RuntimeError(f"NeSLE editable install failed:\n{install.stdout}\n{install.stderr}")
    ensure_nesle_on_path(nesle_root)
    env = dict(os.environ)
    env["NESLE_CUDA_ARCH"] = cuda_arch
    env["PYTHON"] = sys.executable
    build = run_command(["sh", "scripts/build_cuda_extension.sh"], cwd=nesle_root, env=env)
    if build.returncode != 0:
        raise RuntimeError(f"NeSLE CUDA build failed:\n{build.stdout}\n{build.stderr}")


def ensure_nesle_on_path(nesle_root: Path | None) -> None:
    if nesle_root is None:
        return
    src = str(nesle_root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def git_sha(path: Path | None) -> str | None:
    if path is None:
        return None
    proc = run_command(["git", "rev-parse", "HEAD"], cwd=path)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def gpu_snapshot() -> dict[str, Any]:
    query = "name,driver_version,memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu,power.draw"
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"available": False}
    if proc.returncode != 0:
        return {"available": False, "error": proc.stderr.strip()}
    gpus = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue
        name, driver, used, free, total, util, temp, power = parts
        gpus.append(
            {
                "name": name,
                "driver": driver,
                "memory_used_mib": _to_number(used),
                "memory_free_mib": _to_number(free),
                "memory_total_mib": _to_number(total),
                "utilization_percent": _to_number(util),
                "temperature_c": _to_number(temp),
                "power_w": _to_number(power),
            }
        )
    return {"available": bool(gpus), "gpus": gpus}


def _to_number(value: str) -> float | int | str:
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def free_vram_mib(snapshot: dict[str, Any]) -> int | None:
    try:
        return int(snapshot["gpus"][0]["memory_free_mib"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    nesle_root = find_nesle_root(args.nesle_root)
    if args.setup:
        if nesle_root is None:
            raise SystemExit("Cannot --setup because NeSLE root was not found; pass --nesle-root.")
        setup_nesle(nesle_root, args.cuda_arch)
    ensure_nesle_on_path(nesle_root)

    import nesle  # noqa: F401
    import torch
    from nesle import _cuda_core  # type: ignore[attr-defined]

    rom_path = Path(args.rom)
    snapshot_path = Path(args.snapshot)
    batch = _cuda_core.CudaBatch(1, args.frameskip, rom_path.read_bytes(), load_reset_state(snapshot_path))
    obs = batch.reset()
    ram = batch.ram()
    if int(ram[0, 0x0770]) != 1:
        raise RuntimeError("snapshot reset smoke failed: RAM[0x0770] is not gameplay mode 1")

    correctness = None
    if args.run_correctness:
        if nesle_root is None:
            raise SystemExit("--run-correctness requires --nesle-root or NESLE_ROOT.")
        proc = run_command([sys.executable, "benchmarks/verify_correctness.py"], cwd=nesle_root)
        correctness = {
            "returncode": proc.returncode,
            "stdout_tail": tail(proc.stdout),
            "stderr_tail": tail(proc.stderr),
        }
        if proc.returncode != 0:
            raise RuntimeError("benchmark correctness check failed")

    info = {
        "mario_rl_sha": git_sha(repo_root()),
        "nesle_sha": git_sha(nesle_root),
        "nesle_root": None if nesle_root is None else str(nesle_root),
        "python": sys.version,
        "torch": {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "gpu": gpu_snapshot(),
        "cuda_core": str(_cuda_core),
        "snapshot_reset_shape": list(obs.shape),
        "snapshot_world": int(ram[0, 0x075F]) + 1,
        "correctness": correctness,
    }
    return info


def run_env_sweep(args: argparse.Namespace) -> list[CaseResult]:
    from nesle import _cuda_core  # type: ignore[attr-defined]

    rom_bytes = Path(args.rom).read_bytes()
    snapshot_bytes = load_reset_state(Path(args.snapshot))
    action_mask = encode_nes_buttons(args.env_action.split("+") if args.env_action else ())
    counts = list(parse_csv_ints(args.env_counts))
    results: list[CaseResult] = []
    tested: set[int] = set()

    while counts:
        num_envs = counts.pop(0)
        if num_envs in tested:
            continue
        tested.add(num_envs)
        before = gpu_snapshot()
        started = time.perf_counter()
        try:
            batch = _cuda_core.CudaBatch(num_envs, args.frameskip, rom_bytes, snapshot_bytes)
            batch.reset()
            actions = np.full(num_envs, action_mask, dtype=np.uint8)
            for _ in range(args.warmup_steps):
                batch.step(actions, render_frame=False, copy_obs=False)
            timed_start = time.perf_counter()
            for _ in range(args.timed_steps):
                batch.step(actions, render_frame=False, copy_obs=False)
            duration = max(time.perf_counter() - timed_start, 1e-12)
            after = gpu_snapshot()
            env_steps = num_envs * args.timed_steps
            results.append(
                CaseResult(
                    suite="env",
                    mode="cuda-console-no-copy",
                    status="ok",
                    num_envs=num_envs,
                    frameskip=args.frameskip,
                    warmup_steps=args.warmup_steps,
                    timed_steps=args.timed_steps,
                    duration_sec=duration,
                    env_steps_per_sec=env_steps / duration,
                    frame_steps_per_sec=env_steps * args.frameskip / duration,
                    gpu_before=before,
                    gpu_after=after,
                    extra={"action_mask": action_mask},
                )
            )
            free = free_vram_mib(after)
            if (
                args.continue_until_free_gb > 0
                and not counts
                and free is not None
                and free > args.continue_until_free_gb * 1024
                and num_envs * 2 <= args.max_extra_envs
            ):
                counts.append(num_envs * 2)
        except Exception as exc:  # noqa: BLE001 - benchmark must record failure boundaries
            duration = max(time.perf_counter() - started, 1e-12)
            results.append(
                CaseResult(
                    suite="env",
                    mode="cuda-console-no-copy",
                    status="error",
                    num_envs=num_envs,
                    frameskip=args.frameskip,
                    warmup_steps=args.warmup_steps,
                    timed_steps=args.timed_steps,
                    duration_sec=duration,
                    env_steps_per_sec=0.0,
                    frame_steps_per_sec=0.0,
                    gpu_before=before,
                    gpu_after=gpu_snapshot(),
                    error=repr(exc),
                    extra={"action_mask": action_mask},
                )
            )
            break
    return results


def run_ppo_sweep(args: argparse.Namespace) -> list[CaseResult]:
    results = []
    for num_envs in parse_csv_ints(args.ppo_env_counts):
        results.append(run_ppo_case(args, num_envs, args.ppo_updates))
    return results


def run_ppo_case(args: argparse.Namespace, num_envs: int, updates: int) -> CaseResult:
    output_dir = Path(args.output_dir)
    ckpt = output_dir / "checkpoints" / f"native_ppo_envs_{num_envs}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    total_timesteps = num_envs * args.n_steps * updates
    cmd = [
        sys.executable,
        "-m",
        "nesle.native_ppo",
        str(args.rom),
        "--reset-state-path",
        str(args.snapshot),
        "--action-space",
        args.ppo_action_space,
        "--num-envs",
        str(num_envs),
        "--total-timesteps",
        str(total_timesteps),
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--hidden-size",
        str(args.hidden_size),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--checkpoint-path",
        str(ckpt),
        "--log-interval",
        "1",
    ]
    before = gpu_snapshot()
    started = time.perf_counter()
    proc = run_command(cmd)
    duration = max(time.perf_counter() - started, 1e-12)
    after = gpu_snapshot()
    metrics = parse_native_ppo_stdout(proc.stdout)
    status = "ok" if proc.returncode == 0 else "error"
    return CaseResult(
        suite="ppo",
        mode="native-ppo-minimal-reward",
        status=status,
        num_envs=num_envs,
        frameskip=args.frameskip,
        warmup_steps=0,
        timed_steps=total_timesteps,
        duration_sec=duration,
        env_steps_per_sec=total_timesteps / duration if status == "ok" else 0.0,
        frame_steps_per_sec=total_timesteps * args.frameskip / duration if status == "ok" else 0.0,
        gpu_before=before,
        gpu_after=after,
        error="" if status == "ok" else tail(proc.stderr or proc.stdout),
        extra={
            "cmd": cmd,
            "checkpoint_path": str(ckpt),
            "updates": updates,
            "stdout_tail": tail(proc.stdout),
            "stderr_tail": tail(proc.stderr),
            "last_update": metrics,
        },
    )


def parse_native_ppo_stdout(stdout: str) -> dict[str, Any]:
    pattern = re.compile(
        r"update=(?P<update>\d+)/(?P<updates>\d+)\s+"
        r"step=(?P<step>\d+)\s+fps=(?P<fps>\d+)\s+"
        r"loss=(?P<loss>[-+0-9.eE]+)\s+clipfrac=(?P<clipfrac>[-+0-9.eE]+)\s+"
        r"explained_var=(?P<explained_var>[-+0-9.eE]+|nan)\s+"
        r"ep_return_100=(?P<ep_return_100>[-+0-9.eE]+|nan)\s+"
        r"ep_len_100=(?P<ep_len_100>[-+0-9.eE]+|nan)"
    )
    matches = list(pattern.finditer(stdout))
    if not matches:
        return {}
    return {key: _parse_metric(value) for key, value in matches[-1].groupdict().items()}


def _parse_metric(value: str) -> int | float | str:
    if value == "nan":
        return value
    try:
        return int(value)
    except ValueError:
        return float(value)


def tail(text: str, lines: int = 40) -> str:
    return "\n".join(text.splitlines()[-lines:])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [flatten(row) for row in rows]
    if not flat_rows:
        path.write_text("")
        return
    fields = sorted({key for row in flat_rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)


def flatten(row: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten(value, full_key))
        elif isinstance(value, (list, tuple)):
            out[full_key] = json.dumps(value)
        else:
            out[full_key] = value
    return out


def write_report(path: Path, metadata: dict[str, Any], env_rows: Sequence[dict[str, Any]], ppo_rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# NeSLE A100 Limits Report",
        "",
        "## Setup",
        "",
        f"- Mario RL SHA: `{metadata.get('mario_rl_sha')}`",
        f"- NeSLE SHA: `{metadata.get('nesle_sha')}`",
        f"- Python: `{sys.version.split()[0]}`",
        f"- GPU: `{_gpu_name(metadata.get('gpu', {}))}`",
        "",
        "## Env-Only Sweep",
        "",
        "| Status | Envs | Env steps/sec | Frame steps/sec | Free VRAM after | Error |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in env_rows:
        lines.append(_table_row(row))
    lines.extend(
        [
            "",
            "## Native PPO Sweep",
            "",
            "| Status | Envs | Env steps/sec | Frame steps/sec | Free VRAM after | Last PPO metrics |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in ppo_rows:
        extra = row.get("extra") or {}
        metrics = extra.get("last_update") or {}
        lines.append(_table_row(row, json.dumps(metrics, sort_keys=True)))
    lines.extend(
        [
            "",
            "## Commands",
            "",
            "```sh",
            "mario-nesle-bench all --setup --run-correctness",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def _gpu_name(snapshot: dict[str, Any]) -> str:
    try:
        return str(snapshot["gpus"][0]["name"])
    except (KeyError, IndexError, TypeError):
        return "unavailable"


def _table_row(row: dict[str, Any], note: str | None = None) -> str:
    free = free_vram_mib(row.get("gpu_after", {}))
    return (
        f"| {row.get('status')} | {row.get('num_envs')} | "
        f"{float(row.get('env_steps_per_sec') or 0):.1f} | "
        f"{float(row.get('frame_steps_per_sec') or 0):.1f} | "
        f"{'' if free is None else free} | "
        f"{(note if note is not None else row.get('error', ''))} |"
    )


def result_dicts(results: Sequence[CaseResult]) -> list[dict[str, Any]]:
    return [asdict(result) for result in results]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    nesle_root = find_nesle_root(None)
    parser.add_argument("--nesle-root", default=None)
    parser.add_argument("--rom", default=str(default_rom_path()))
    parser.add_argument("--snapshot", default=str(default_snapshot_path(nesle_root)))
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--cuda-arch", default="sm_80")
    parser.add_argument("--setup", action="store_true", help="pip install NeSLE editable and build _cuda_core.")
    parser.add_argument("--run-correctness", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark NeSLE CUDA envs and native PPO on A100.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("preflight", "env-sweep", "ppo-sweep", "stress", "all"):
        sub = subparsers.add_parser(name)
        add_common_args(sub)
        if name in {"env-sweep", "all"}:
            sub.add_argument("--env-counts", default=",".join(str(x) for x in DEFAULT_ENV_COUNTS))
            sub.add_argument("--warmup-steps", type=int, default=30)
            sub.add_argument("--timed-steps", type=int, default=200)
            sub.add_argument("--env-action", default="RIGHT")
            sub.add_argument("--continue-until-free-gb", type=int, default=8)
            sub.add_argument("--max-extra-envs", type=int, default=262144)
        if name in {"ppo-sweep", "stress", "all"}:
            sub.add_argument("--ppo-env-counts", default=",".join(str(x) for x in DEFAULT_PPO_COUNTS))
            sub.add_argument("--ppo-updates", type=int, default=2)
            sub.add_argument("--n-steps", type=int, default=128)
            sub.add_argument("--batch-size", type=int, default=8192)
            sub.add_argument("--hidden-size", type=int, default=256)
            sub.add_argument("--max-episode-steps", type=int, default=512)
            sub.add_argument("--ppo-action-space", default="mario")
        if name == "stress":
            sub.add_argument("--stress-envs", type=int, default=65536)
            sub.add_argument("--stress-timesteps", type=int, default=75_000_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "preflight":
        write_json(output_dir / "nesle_a100_preflight.json", preflight(args))
        return

    metadata = preflight(args)
    env_results: list[CaseResult] = []
    ppo_results: list[CaseResult] = []

    if args.command in {"env-sweep", "all"}:
        env_results = run_env_sweep(args)
        write_json(output_dir / "nesle_a100_limits.json", {"metadata": metadata, "results": result_dicts(env_results)})
        write_csv(output_dir / "nesle_a100_limits.csv", result_dicts(env_results))

    if args.command in {"ppo-sweep", "all"}:
        ppo_results = run_ppo_sweep(args)
        write_json(output_dir / "nesle_a100_native_ppo.json", {"metadata": metadata, "results": result_dicts(ppo_results)})

    if args.command == "stress":
        updates = max(1, math.ceil(args.stress_timesteps / (args.stress_envs * args.n_steps)))
        result = run_ppo_case(args, args.stress_envs, updates)
        ppo_results = [result]
        write_json(output_dir / "nesle_a100_native_ppo_stress.json", {"metadata": metadata, "results": result_dicts(ppo_results)})

    if args.command in {"env-sweep", "ppo-sweep", "stress", "all"}:
        write_report(
            output_dir / "nesle_a100_report.md",
            metadata,
            result_dicts(env_results),
            result_dicts(ppo_results),
        )


if __name__ == "__main__":
    main()
