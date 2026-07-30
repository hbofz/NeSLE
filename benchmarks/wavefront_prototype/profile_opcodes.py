"""Gather the real SMB opcode frequency distribution via CudaBatch.step_profile.

Runs 256 envs from the W1-1 snapshot with a mix of controller inputs so the
population decorrelates, accumulates per-opcode execution counts over many
steps, and writes opcode_hist.json next to this script. Feed the JSON to
make_hist_header.py to regenerate opcode_hist.inc for proto.cu.

Requires the built CUDA extension and the SMB ROM at the repo root.
"""
import gzip
import json
from pathlib import Path

import numpy as np

from nesle._cuda_core import CudaBatch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROM = (REPO / "Super Mario Bros. (World).nes").read_bytes()
STATE = gzip.decompress((REPO / "docs" / "data" / "smb_level1_1.state").read_bytes())

NUM_ENVS = 256
FRAMESKIP = 4
WARMUP_STEPS = 30
PROFILE_STEPS = 120

batch = CudaBatch(NUM_ENVS, FRAMESKIP, ROM, STATE)
batch.reset()

rng = np.random.default_rng(1234)
# Mix of plausible SMB controller masks: right, right+A (jump), right+B (run),
# right+A+B, noop, left. Weighted toward moving right like real training.
masks = np.array([0x80, 0x81, 0x82, 0x83, 0x00, 0x40], dtype=np.uint8)
weights = np.array([0.30, 0.25, 0.15, 0.15, 0.10, 0.05])

for _ in range(WARMUP_STEPS):
    actions = rng.choice(masks, size=NUM_ENVS, p=weights)
    batch.step(actions, render_frame=False, copy_obs=False)

total = np.zeros(256, dtype=np.uint64)
for _ in range(PROFILE_STEPS):
    actions = rng.choice(masks, size=NUM_ENVS, p=weights)
    out = batch.step_profile(actions)
    total += out["opcode_counts"].astype(np.uint64)

grand = int(total.sum())
print(f"profiled {PROFILE_STEPS} steps x {NUM_ENVS} envs (frameskip {FRAMESKIP})")
print(f"total opcode executions: {grand:,}")

order = np.argsort(total)[::-1]
print("\ntop 25 opcodes:")
for rank, op in enumerate(order[:25], 1):
    c = int(total[op])
    print(f"{rank:3d}. 0x{op:02X}  {c:>14,}  {100.0 * c / grand:6.3f}%")
print(f"\ndistinct opcodes executed: {int((total > 0).sum())}")

out_path = HERE / "opcode_hist.json"
out_path.write_text(json.dumps({
    "num_envs": NUM_ENVS,
    "frameskip": FRAMESKIP,
    "profile_steps": PROFILE_STEPS,
    "total_executions": grand,
    "counts": [int(c) for c in total],
}, indent=1))
print(f"wrote {out_path}")
