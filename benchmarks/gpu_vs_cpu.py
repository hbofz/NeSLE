"""Head-to-head throughput: native (CPU, one env) vs cuda-console (GPU, batched).

Both backends:
- Run the same SMB ROM
- Use frameskip=4 (4 emulator frames per step call) — same as RL default
- Use RIGHT for every action (eliminates random-policy noise)
- Are warmed up before timing (no first-step JIT / cuMemcpyAsync penalty)
- Are timed over a fixed step budget

Reports:
- env-steps/sec (one step = one env.step call advancing one env by frameskip frames)
- frame-steps/sec (one step × frameskip = N effective NES frames simulated)
- relative speedup vs the single-env CPU baseline
"""
from __future__ import annotations

import gzip
import time
from pathlib import Path

import numpy as np

import nesle
from nesle._cuda_core import CudaBatch

REPO = Path(__file__).resolve().parent
ROM_BYTES = (REPO / "Super Mario Bros. (World).nes").read_bytes()
STATE_BYTES = gzip.decompress((REPO / "docs/data/smb_level1_1.state").read_bytes())
FRAMESKIP = 4
WARMUP_STEPS = 30
TIMED_STEPS = 200
RIGHT_RAW = 0x80


def bench_cpu_single() -> dict:
    env = nesle.make(
        rom_path=str(REPO / "Super Mario Bros. (World).nes"),
        backend="native",
        observation_mode="ram",
        action_space="raw",
        frameskip=FRAMESKIP,
    )
    env.reset()
    # Warmup
    for _ in range(WARMUP_STEPS):
        env.step(RIGHT_RAW)
    t0 = time.perf_counter()
    for _ in range(TIMED_STEPS):
        env.step(RIGHT_RAW)
    elapsed = time.perf_counter() - t0
    env.close()
    env_steps = TIMED_STEPS  # 1 env × N step calls
    return {
        "label": "native CPU (1 env)",
        "num_envs": 1,
        "env_steps": env_steps,
        "elapsed_s": elapsed,
        "env_steps_per_s": env_steps / elapsed,
        "frame_steps_per_s": env_steps * FRAMESKIP / elapsed,
    }


def bench_gpu_batched(num_envs: int, use_snapshot: bool = True) -> dict:
    snapshot = STATE_BYTES if use_snapshot else None
    if snapshot is not None:
        batch = CudaBatch(num_envs, FRAMESKIP, ROM_BYTES, snapshot)
    else:
        batch = CudaBatch(num_envs, FRAMESKIP, ROM_BYTES)
    batch.reset()
    actions = np.full(num_envs, RIGHT_RAW, dtype=np.uint8)
    # Warmup
    for _ in range(WARMUP_STEPS):
        batch.step(actions, render_frame=False, copy_obs=False)
    t0 = time.perf_counter()
    for _ in range(TIMED_STEPS):
        batch.step(actions, render_frame=False, copy_obs=False)
    elapsed = time.perf_counter() - t0
    env_steps = TIMED_STEPS * num_envs
    return {
        "label": f"cuda-console (snapshot, {num_envs} env{'s' if num_envs != 1 else ''})",
        "num_envs": num_envs,
        "env_steps": env_steps,
        "elapsed_s": elapsed,
        "env_steps_per_s": env_steps / elapsed,
        "frame_steps_per_s": env_steps * FRAMESKIP / elapsed,
    }


def fmt_row(r: dict, baseline_eps: float) -> str:
    speedup = r["env_steps_per_s"] / baseline_eps if baseline_eps > 0 else 0.0
    return (
        f"  {r['label']:40s}  "
        f"{r['env_steps_per_s']:>10,.0f} env-steps/s  "
        f"{r['frame_steps_per_s']:>12,.0f} frame-steps/s  "
        f"{speedup:>6.2f}x"
    )


def main() -> None:
    print(f"NeSLE benchmark — frameskip={FRAMESKIP}, action=RIGHT, "
          f"warmup={WARMUP_STEPS}, timed={TIMED_STEPS} steps per run")
    print(f"ROM:    Super Mario Bros. (World).nes ({len(ROM_BYTES):,} bytes)")
    print(f"Reset:  Stable Retro Level 1-1 snapshot (start of W1-1)")
    print()
    print("Running...")
    print()

    rows = []
    rows.append(bench_cpu_single())
    baseline_eps = rows[0]["env_steps_per_s"]

    # cuda-console at a range of batch sizes the 1050 Ti can plausibly handle.
    for n in [1, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        try:
            rows.append(bench_gpu_batched(n))
        except Exception as exc:
            print(f"  cuda-console {n} envs FAILED: {exc}")
            break

    print(f"{'Backend':42s}  {'Env-steps/s':>10s}  {'Frame-steps/s':>15s}  {'vs CPU':>8s}")
    print("-" * 90)
    for r in rows:
        print(fmt_row(r, baseline_eps))
    print()
    best = max(rows, key=lambda r: r["env_steps_per_s"])
    print(f"Best total throughput:  {best['label']}  "
          f"@ {best['env_steps_per_s']:,.0f} env-steps/s "
          f"({best['env_steps_per_s'] / baseline_eps:.1f}x vs CPU)")
    print(f"Best per-env latency:   "
          f"{min(r['elapsed_s']/(TIMED_STEPS) for r in rows)*1000:.2f} ms/step (smallest batch)")


if __name__ == "__main__":
    main()
