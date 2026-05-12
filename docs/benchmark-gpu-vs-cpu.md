# NeSLE throughput: GPU vs CPU on a GTX 1050 Ti

**Hardware**
- GPU: NVIDIA GeForce GTX 1050 Ti (Pascal, sm_61, 4 GB VRAM, 2017-era)
- Driver: 582.28 (CUDA 13.0 runtime)
- CUDA Toolkit used to build the extension: 12.9 (sidecar; 13.x dropped Pascal)
- Host: Windows 11, Python 3.14, MSVC 2022 BuildTools

**Workload**
- ROM: Super Mario Bros. (World).nes - NROM mapper 0, 32 KB PRG + 8 KB CHR
- Reset: Stable Retro Level 1-1 FCS snapshot (start of W1-1, fully-initialized gameplay)
- frameskip=4 (one env.step = 4 emulated NES frames)
- Constant action: RIGHT held every step
- Per run: 30 step warmup, 200 timed steps, mean throughput reported
- `cuda-console` runs with `render_frame=False, copy_obs=False` (RAM-observation path)

**Reproduce**
```
python benchmarks/gpu_vs_cpu.py
```

## Results

| Backend | Env-steps/s | Frame-steps/s | vs CPU baseline |
|---|---:|---:|---:|
| native CPU (1 env) | 290 | 1,162 | 1.00x |
| cuda-console, 1 env | 7 | 30 | 0.03x |
| cuda-console, 8 envs | 58 | 232 | 0.20x |
| cuda-console, 32 envs | 232 | 930 | 0.80x |
| cuda-console, 64 envs | 448 | 1,791 | 1.54x |
| cuda-console, 128 envs | 1,276 | 5,105 | 4.39x |
| cuda-console, 256 envs | 4,043 | 16,171 | 13.92x |
| cuda-console, 512 envs | 8,135 | 32,540 | 28.00x |
| cuda-console, 1024 envs | 14,863 | 59,452 | 51.17x |
| cuda-console, 2048 envs | 26,973 | 107,893 | 92.86x |
| cuda-console, 4096 envs | **29,670** | **118,680** | **102.14x** |

## Observations

- **GPU loses at small batch sizes** (<=32 envs in this run): kernel-launch overhead per step dominates. Per-env latency is ~3 ms - comparable to a single CUDA launch round-trip.
- **GPU wins at 64 envs** (parity crossover in this run). Each parallel env now amortizes its share of the launch cost.
- **Near-linear scaling from 64 to 2048 envs**: doubling batch size ~doubles throughput. The kernel grid is large enough to saturate the GPU's CUDA cores.
- **Plateau between 2048 and 4096 envs**: scaling falls to 1.1x per doubling. We're now bandwidth/occupancy-limited rather than compute-limited.
- **Peak: 119k effective NES frames per second** across 4096 parallel SMB instances. Real NES runs at 60 fps - this is roughly **1978x real-time, across the batch.**

## What this means for RL training

A typical PPO run on SMB takes ~10M timesteps. On a single-env CPU emulator:

- 10M timesteps / 290 env-steps/s ~= **9.6 hours**

On the 1050 Ti at 4096 envs:

- 10M timesteps / 29,670 env-steps/s ~= **5.6 minutes** of environment stepping

(Total wall-clock includes the SB3 gradient update step, which is policy-network-dependent and runs on whichever device `--sb3-device` selects. With a small MlpPolicy and RAM observations, network forward/backward isn't the bottleneck.)

## Extrapolation to A100 / H100

The 1050 Ti has ~2 TFLOPS FP32. A100 has ~19.5 TFLOPS, H100 has ~67 TFLOPS. The cuda-console kernel is CUDA-core-limited (not tensor-core, so the formula scales with FP32). Naively:

- A100 estimate: ~10x this card's throughput -> **~300k env-steps/s** at high batch size
- H100 estimate: ~30x -> **~1M env-steps/s**

Memory-wise: each env needs ~200 KB of state. A100 (40-80 GB) can comfortably hold 200,000+ parallel envs. The local 4 GB cap (and the 2048-to-4096 plateau) is a host-side constraint, not a fundamental algorithmic one.
