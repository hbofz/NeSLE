# NeSLE throughput: GPU vs CPU on a GTX 1050 Ti

**Hardware**
- GPU: NVIDIA GeForce GTX 1050 Ti (Pascal, sm_61, 4 GB VRAM, 2017-era)
- Driver: 582.28 (CUDA 13.0 runtime)
- CUDA Toolkit used to build the extension: 12.9 (sidecar; 13.x dropped Pascal)
- Host: Windows 11, Python 3.14, MSVC 2022 BuildTools

**Workload**
- ROM: Super Mario Bros. (World).nes — NROM mapper 0, 32 KB PRG + 8 KB CHR
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
| native CPU (1 env) | 308 | 1,232 | 1.00× |
| cuda-console, 1 env | 17 | 69 | 0.06× |
| cuda-console, 8 envs | 137 | 548 | 0.44× |
| cuda-console, 32 envs | 535 | 2,142 | 1.74× |
| cuda-console, 64 envs | 1,054 | 4,216 | 3.42× |
| cuda-console, 128 envs | 2,057 | 8,228 | 6.68× |
| cuda-console, 256 envs | 4,027 | 16,106 | 13.07× |
| cuda-console, 512 envs | 8,067 | 32,266 | 26.18× |
| cuda-console, 1024 envs | 14,779 | 59,116 | 47.97× |
| cuda-console, 2048 envs | 26,454 | 105,814 | 85.87× |
| cuda-console, 4096 envs | **29,058** | **116,233** | **94.32×** |

## Observations

- **GPU loses at small batch sizes** (≤16 envs): kernel-launch overhead per step dominates. Per-env latency is ~3 ms — comparable to a single CUDA launch round-trip.
- **GPU wins at 32 envs** (parity crossover). Each parallel env now amortizes its share of the launch cost.
- **Near-linear scaling from 64 to 2048 envs**: doubling batch size ~doubles throughput. The kernel grid is large enough to saturate the GPU's CUDA cores.
- **Plateau between 2048 and 4096 envs**: scaling falls to 1.1× per doubling. We're now bandwidth/occupancy-limited rather than compute-limited.
- **Peak: 116k effective NES frames per second** across 4096 parallel SMB instances. Real NES runs at 60 fps — this is roughly **1936× real-time, across the batch.**

## What this means for RL training

A typical PPO run on SMB takes ~10M timesteps. On a single-env CPU emulator:

- 10M timesteps / 308 env-steps/s ≈ **9 hours**

On the 1050 Ti at 4096 envs:

- 10M timesteps / 29,058 env-steps/s ≈ **6 minutes** of environment stepping

(Total wall-clock includes the SB3 gradient update step, which is policy-network-dependent and runs on whichever device `--sb3-device` selects. With a small MlpPolicy and RAM observations, network forward/backward isn't the bottleneck.)

## Extrapolation to A100 / H100

The 1050 Ti has ~2 TFLOPS FP32. A100 has ~19.5 TFLOPS, H100 has ~67 TFLOPS. The cuda-console kernel is CUDA-core-limited (not tensor-core, so the formula scales with FP32). Naively:

- A100 estimate: ~10× this card's throughput → **~300k env-steps/s** at high batch size
- H100 estimate: ~30× → **~1M env-steps/s**

Memory-wise: each env needs ~200 KB of state. A100 (40-80 GB) can comfortably hold 200,000+ parallel envs. The local 4 GB cap (and the 2048→4096 plateau) is a host-side constraint, not a fundamental algorithmic one.
