# NeSLE throughput: GPU vs CPU on a GTX 1050 Ti

Measured 2026-10-25 on the current kernel (after the 2026-07-29 optimization
program: table-driven 6502 decode, register-resident PPU state, lazy PPU
settlement, hybrid render dispatch). The pre-optimization sweep this replaces
(29,670 env-steps/s at 4,096 envs, crossover at 64 envs) is in git history.
Per-change measurements are in [gpu-scaling.md](gpu-scaling.md); the A100
equivalent is `docs/data/verification-2026-09-01-a100.json`.

**Hardware**
- GPU: NVIDIA GeForce GTX 1050 Ti (Pascal, sm_61, 4 GB VRAM, 2017-era)
- Driver: 582.66
- CUDA Toolkit used to build the extension: 12.9 (13.x dropped Pascal)
- CPU: Intel Core i7-8750H (6 cores / 12 threads)
- Host: Ubuntu under WSL2 on Windows 11, Python 3.14, g++ 15

Earlier runs on this card were on native Windows. The 4,096-env figure here
(177,401) is within 2% of the native-Windows 180,437 in the README.

**Workload**
- ROM: Super Mario Bros. (World).nes - NROM mapper 0, 32 KB PRG + 8 KB CHR
- Reset: Stable Retro Level 1-1 FCS snapshot (start of W1-1, fully-initialized gameplay)
- frameskip=4 (one env.step = 4 emulated NES frames)
- Constant action: RIGHT held every step. Every env runs the same instruction
  stream, so this is the best case for warp coherence. See
  [Realistic policy](#realistic-policy-per-env-random-actions) below for
  per-env random actions
- Per run: 30 step warmup, 200 timed steps, mean throughput reported
- `cuda-console` runs with `render_frame=False, copy_obs=False` (RAM-observation path)
- Table is the median of three back-to-back sweeps

**Reproduce**
```
python benchmarks/gpu_vs_cpu.py --max-envs 16384
```

## Results

| Backend | Env-steps/s | Frame-steps/s | vs CPU baseline |
|---|---:|---:|---:|
| native CPU (1 env) | 344 | 1,376 | 1.00x |
| cuda-console, 1 env | 61 | 244 | 0.18x |
| cuda-console, 2 envs | 123 | 492 | 0.36x |
| cuda-console, 4 envs | 244 | 976 | 0.71x |
| cuda-console, 8 envs | 485 | 1,940 | 1.41x |
| cuda-console, 16 envs | 960 | 3,840 | 2.79x |
| cuda-console, 32 envs | 1,908 | 7,632 | 5.55x |
| cuda-console, 64 envs | 3,779 | 15,116 | 10.99x |
| cuda-console, 128 envs | 7,413 | 29,652 | 21.55x |
| cuda-console, 256 envs | 14,521 | 58,084 | 42.21x |
| cuda-console, 512 envs | 29,065 | 116,260 | 84.49x |
| cuda-console, 1,024 envs | 55,459 | 221,836 | 161.22x |
| cuda-console, 2,048 envs | 103,990 | 415,960 | 302.30x |
| cuda-console, 4,096 envs | **177,401** | **709,604** | **515.70x** |
| cuda-console, 8,192 envs | 172,809 | 691,236 | 502.35x |
| cuda-console, 16,384 envs | 174,308 | 697,232 | 506.71x |

Run-to-run spread across the three sweeps was under 1% at every batch size,
CPU baseline included (343-345).

## Observations

- **Crossover is between 4 and 8 envs** (0.71x at 4, 1.41x at 8), the same
  8-env crossover the A100 showed. One env takes about 16 ms per step: a single
  GPU thread emulating four frames serially is far slower than a CPU core.
  Step time stays at 16-18 ms all the way to 1,024 envs, so extra envs are
  nearly free until the card fills.
- **Scaling is linear from 1 to 1,024 envs**: each doubling doubles throughput.
  It softens at 2,048 (1.88x) and 4,096 (1.71x).
- **The card saturates at 4,096 envs.** 8,192 and 16,384 are flat, so the A100's
  linear range to ~8,192 does not carry over to a 6-SM Pascal part. 16,384 envs
  still fit in 4 GB (about 195 KB each).
- **Peak: 710k effective NES frames per second** across 4,096 parallel SMB
  instances, roughly **11,800x real time** across the batch.

## Realistic policy: per-env random actions

Same protocol, but each env draws its own seeded random action every step from
`simple_with_start` (the `native_ppo` default action space), as an untrained
stochastic policy does. Median of three sweeps, spread under 1%:

```
python benchmarks/gpu_vs_cpu.py --max-envs 16384 --policy random
```

| Backend | Env-steps/s | vs CPU baseline | RIGHT-held / random |
|---|---:|---:|---:|
| native CPU (1 env) | 363 | 1.00x | 0.95x |
| cuda-console, 1 env | 73 | 0.20x | 0.84x |
| cuda-console, 8 envs | 181 | 0.50x | 2.68x |
| cuda-console, 32 envs | 379 | 1.04x | 5.03x |
| cuda-console, 256 envs | 2,726 | 7.51x | 5.33x |
| cuda-console, 1,024 envs | 10,124 | 27.89x | 5.48x |
| cuda-console, 2,048 envs | 17,711 | 48.79x | 5.87x |
| cuda-console, 4,096 envs | **28,084** | **77.37x** | 6.32x |
| cuda-console, 8,192 envs | 29,798 | 82.09x | 5.80x |
| cuda-console, 16,384 envs | 31,388 | 86.47x | 5.55x |

- **The gap is warp divergence.** It grows with batch size up to 32 envs, one
  warp, and then stays at 5-6x: once every warp holds 32 independent envs, it
  cannot get more divergent. Giving all envs one shared random action instead
  runs at 214k at 4,096 envs, so it is not the actions themselves.
- **Crossover moves to 32 envs** (from 4-8 with RIGHT held).
- The CPU baseline barely changes, since a single core has no lanes to diverge.

## What this means for RL training

A typical PPO run on SMB takes ~10M timesteps. On a single-env CPU emulator:

- 10M timesteps / 344 env-steps/s ~= **8.1 hours**

On the 1050 Ti at 4,096 envs:

- 10M timesteps / 177,401 env-steps/s ~= **56 seconds** of environment stepping
  with RIGHT held, or 10M / 28,084 ~= **6 minutes** with per-env random actions

(Total wall-clock includes the gradient update step, which is
policy-network-dependent. With a small MLP policy and RAM observations, network
forward/backward isn't the bottleneck; on Windows the WDDM submission ceiling in
`KNOWN_ISSUES.md` is.)

For larger GPUs, use the measured A100 numbers in the README rather than
extrapolating from this card.

## Baseline: nes-py / gym-super-mario-bros (2026-07-29)

The widely-used CPU Mario stack was measured on the same machine with the same
protocol (hold RIGHT+B, warmup then timed steps): **530 frames/s ≈ 132
env-steps/s** at frameskip-4 accounting (`benchmarks/nespy_baseline.py`; needs
its own old-gym venv, see the script header). nes-py was not re-run for this
sweep, so its row is from the earlier date. For scale:

| Stack | Env-steps/s (fs4) | Relative |
|---|---:|---:|
| nes-py (`SuperMarioBros-v0`), 1 env | 132 | 0.4x |
| NeSLE native CPU, 1 env | 344 | 1.0x |
| NeSLE cuda-console, 4,096 envs | 177,401 | 516x (≈1,344x nes-py) |
