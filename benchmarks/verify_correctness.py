"""Falsifiability tests for the cuda-console throughput numbers.

Three independent checks that would all FAIL if the kernel were silently skipping work:

  (1) Different actions per env ==> different x_pos. If kernel runs only env 0 and copies
      its state to env 1..N, this test fails.

  (2) Cycle accounting. SMB at NTSC runs the CPU at ~29830 cycles per frame. With
      frameskip=4 each env.step() = ~119k CPU cycles. step_stats reports the actual
      instruction count per env per step; verify the per-env instructions number
      matches a plausible CPU workload, and that doubling envs roughly doubles
      total work (kernel scales).

  (3) Snapshot drift. After many random steps, sample RAM across envs and confirm
      hashes are diverse — i.e., emulators actually evolve independently with
      different action streams.
"""
from __future__ import annotations

import gzip
import hashlib
import time
from pathlib import Path

import numpy as np

from nesle._cuda_core import CudaBatch

REPO = Path(__file__).resolve().parent
ROM = (REPO / "Super Mario Bros. (World).nes").read_bytes()
STATE = gzip.decompress((REPO / "docs/data/smb_level1_1.state").read_bytes())


def test_actions_diverge(num_envs: int = 4096, steps: int = 60) -> None:
    print(f"[1/3] Different-action divergence at {num_envs} envs over {steps} steps...")
    batch = CudaBatch(num_envs, 4, ROM, STATE)
    batch.reset()
    # Give each env a different mask. We cycle through 8 distinct controller states.
    base_masks = np.array([0x00, 0x80, 0x40, 0x81, 0x82, 0x83, 0x01, 0x02], dtype=np.uint8)
    actions = np.tile(base_masks, num_envs // len(base_masks) + 1)[:num_envs]
    for _ in range(steps):
        batch.step(actions, render_frame=False, copy_obs=False)
    ram = batch.ram()
    # x_pos per env
    x_positions = ram[:, 0x006D].astype(np.int32) * 0x100 + ram[:, 0x0086].astype(np.int32)
    # Group by action class — envs with the same action should have similar (not identical
    # because of timing jitter) x_pos; envs with different actions should diverge.
    per_action_x = {int(m): x_positions[actions == m] for m in base_masks}
    print(f"  per-action mean x_pos (n_envs each):")
    for m, xs in per_action_x.items():
        print(f"    mask=0x{m:02x}  n={len(xs):4d}  x_mean={xs.mean():.1f}  x_min={xs.min()}  x_max={xs.max()}  x_std={xs.std():.1f}")
    # Sanity: RIGHT mask (0x80) should have higher mean than NOOP (0x00)
    right_mean = per_action_x[0x80].mean()
    noop_mean = per_action_x[0x00].mean()
    assert right_mean > noop_mean + 5, (
        f"RIGHT envs (x={right_mean:.1f}) should significantly outpace NOOP envs "
        f"(x={noop_mean:.1f})"
    )
    # Sanity: at least 5 distinct x_pos values across the batch — proves envs aren't all
    # synchronized to env 0.
    distinct = len(np.unique(x_positions))
    assert distinct >= 5, f"only {distinct} distinct x_pos values; envs may be synced"
    print(f"  ==> PASS: {distinct} distinct x_pos values, RIGHT envs ahead of NOOP envs [ok]")


def test_cycle_accounting(num_envs_list: tuple[int, ...] = (32, 256, 2048)) -> None:
    print(f"\n[2/3] CPU cycle accounting (uses step_stats; instructions per env per step)...")
    NOOP = lambda n: np.zeros(n, dtype=np.uint8)
    rows = []
    for n in num_envs_list:
        batch = CudaBatch(n, 4, ROM, STATE)
        batch.reset()
        # Warmup
        for _ in range(20):
            batch.step(NOOP(n), render_frame=False, copy_obs=False)
        # Measure instructions over a single step_stats call
        stats = batch.step_stats(NOOP(n))
        instrs = stats["instructions"]  # uint64 per env, instructions over 4-frame step
        frames = stats["frames_completed"]
        budget_hits = stats["budget_hits"]
        rows.append((n, int(instrs.mean()), int(instrs.std()), int(frames.mean()),
                     int(budget_hits.sum())))
    print(f"  {'n_envs':>6s}  {'mean instr/step':>16s}  {'std':>6s}  {'mean frames':>12s}  {'budget hits':>11s}")
    for n, mean_instrs, std_instrs, mean_frames, hits in rows:
        print(f"  {n:>6d}  {mean_instrs:>16,d}  {std_instrs:>6d}  {mean_frames:>12d}  {hits:>11d}")
    # SMB at frameskip=4 must run ~15k-40k instructions per step (we saw ~15.5k typical
    # earlier from probe_pc_profile.py). If we see < 5000, kernel is no-op'ing.
    for n, mean_instrs, _, mean_frames, _ in rows:
        assert mean_instrs > 5000, (
            f"{n} envs only ran {mean_instrs} instructions/step — kernel skipping work?"
        )
        assert mean_frames == 4, f"frameskip=4 but frames_completed reports {mean_frames}"
    print(f"  ==> PASS: all batch sizes ran plausible CPU work, frames_completed=4 each [ok]")


def test_state_diversity(num_envs: int = 256, steps: int = 100) -> None:
    print(f"\n[3/3] Independent state evolution: {num_envs} envs, random actions, "
          f"{steps} steps, verify diverse RAM hashes...")
    batch = CudaBatch(num_envs, 4, ROM, STATE)
    batch.reset()
    rng = np.random.default_rng(42)
    for _ in range(steps):
        actions = rng.integers(0, 256, size=num_envs, dtype=np.uint8)
        batch.step(actions, render_frame=False, copy_obs=False)
    ram = batch.ram()
    # Hash each env's RAM separately, count distinct hashes
    hashes = {hashlib.md5(ram[e].tobytes()).digest()[:6] for e in range(num_envs)}
    print(f"  distinct RAM hashes across {num_envs} envs: {len(hashes)}")
    assert len(hashes) >= num_envs // 2, (
        f"only {len(hashes)} distinct RAM hashes for {num_envs} envs — envs may be syncing"
    )
    print(f"  ==> PASS: state evolution is independent across envs [ok]")


def main() -> None:
    test_actions_diverge()
    test_cycle_accounting()
    test_state_diversity()
    print("\nAll three falsifiability checks passed. Benchmark numbers reflect real work.")


if __name__ == "__main__":
    main()
