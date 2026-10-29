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

## Baseline: nes-py on every core

Measured 2026-10-29. `nes-py` / `gym-super-mario-bros` is the widely-used CPU
Mario stack. It ran on the same machine, OS and Python 3.14 interpreter as the
NeSLE runs above, holding RIGHT like them. `benchmarks/nespy_baseline.py
--workers N` starts N independent processes with one env each. Every worker
warms up for 30 s, then all of them time the same five 10 s windows behind a
barrier, so a window's summed rate is real concurrent throughput, and each run
reports its median window. Each row below is the median of three independent
runs, interleaved:

```
python benchmarks/nespy_baseline.py --workers 12   # in its own venv, see the script header
```

| nes-py processes | Env-steps/s (fs4) | Range, 3 runs | Per process | CPU clock |
|---|---:|---:|---:|---:|
| 1 | 168 | 168-169 | 168 | 4.1 GHz |
| 6 (= core count) | 654 | 630-655 | 109 | 3.4 GHz |
| 12 (every thread) | **767** | 764-768 | 64 | 3.05 GHz |

Twelve processes is the most this CPU gets out of nes-py, so that row is the
fair comparison:

| Against nes-py | RIGHT held (177,401 at 4,096 envs) | Per-env random (28,084) |
|---|---:|---:|
| 1 process (168) | 1,056x | 167x |
| 12 processes (767) | **231x** | **37x** |

At 767 env-steps/s, the 10M-timestep PPO run above takes about 3.6 hours of
environment stepping on all 12 threads.

`benchmarks/verify_claims.py` reproduced this end to end the same day, building
its own nes-py venv: 169 and 769 env-steps/s for 1 and 12 processes, and
177,623 at 4,096 GPU envs. Raw output:
`docs/data/verification-2026-10-29-1050ti.json`. Its native CPU figure (384)
is above the 344 in the table at the top, which was not re-run here.

**How it was measured, and what it leaves out**

- **Process count.** A sweep from 1 to 12 processes (on the slower build
  described below) rose at every step. On the final build, 24 processes on 12
  threads gave the same as 12 (766 vs 767). Filling the second hardware thread
  of each core adds about 17% over one process per core, so `verify_claims.py`
  runs one process per CPU it may use.
- **Pinning.** In the same sweep, pinning each process to its own CPU (physical
  cores first) never beat the default scheduler, and at 6 processes it was 4%
  slower (490 vs 509). Under WSL2 a pin only picks a virtual CPU; Hyper-V still
  decides where that runs. The script does not pin.
- **Thermal and power limits.** This is a laptop i7-8750H. With all 12 threads
  loaded, the clock decays from about 3.5 GHz to a sustained 3.05 GHz over the
  first 15-20 s (read on the Windows side from `% Processor Performance`, since
  WSL2 exposes no cpufreq). Timed from a cold start, 12 processes read 9%
  higher in the first 10 s window than afterwards. The 30 s warmup skips that,
  so the 12-process row is the sustained rate a long training run gets. One
  process keeps its 4.1 GHz single-core turbo, which flatters the 1-process
  row.
- **Spread.** Three runs agreed within 1% at 1 and 12 processes, and within 4%
  at 6, where single 10 s windows ranged from 618 to 676. One window in the
  earlier sweep dipped 30% at 2 processes, from host activity, which is why
  each run reports its median window.
- **Skipped frames are counted.** gym-super-mario-bros fast-forwards deaths
  and area changes inside a single `step()`, which is 11% of emulated frames
  with RIGHT held. The script counts them, as NeSLE's figures do, since they
  are emulation work. An agent never sees them, so per agent-visible step,
  nes-py is about 11% slower than these tables.
- **The nes-py build matters.** nes-py compiles its C++ core when it is
  installed. Built against Ubuntu's Python 3.14, its raw emulator step takes
  1.41 ms per frame; built against uv's standalone Python 3.11 with the same
  g++ 15.2, it takes 1.59 ms, and whole runs are about 11% slower. The cause was
  not isolated (the two interpreters hand extensions different compile flags).
  The tables use the faster build. Where stdlib venv is missing (Colab),
  `verify_claims.py` falls back to uv's Python 3.11, so its nes-py figures there
  may read about 11% low. The earlier 132 env-steps/s (2026-07-29) came from a
  different install and a 2,000-step protocol holding RIGHT+B without counting
  skipped frames, so it is not comparable.
- **Not like-for-like.** nes-py draws the full 256x240 frame every step and
  cannot skip it. The NeSLE GPU figures are the RAM-observation path with
  `render_frame=False`, which does not draw. nes-py's Python wrapper costs under
  0.1 ms of each ~1.7 ms step. So this compares the standard stack as people use
  it, not the two emulator cores.
- **An upper bound for SB3.** Independent processes skip the per-step
  observation transfer of a `SubprocVecEnv`, so an SB3 pipeline over nes-py
  would likely get somewhat less than the 12-process row (not measured).
