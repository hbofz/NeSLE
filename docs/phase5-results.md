# Phase 5 Benchmark Results

Date: 2026-05-01

GPU environment:

- Colab SSH tunnel host: `believes-digital-moss-guest.trycloudflare.com`
- GPU: NVIDIA A100-SXM4-80GB
- Driver: 580.82.07
- Torch: 2.10.0+cu128
- NVCC: `/usr/local/cuda-12.8/bin/nvcc`

## Python API Calibration

Command shape:

```sh
PYTHONPATH=src python3 benchmarks/phase5_benchmark.py \
  "Super Mario Bros. (World).nes" \
  --env-counts 1,8,32,128 \
  --steps 50 \
  --warmup-steps 10 \
  --modes step,render,inference \
  --backend auto
```

Extended rows for `256,512` used `--steps 30 --warmup-steps 5`.

| Mode | Envs | Env steps/sec | Training frames/sec | Peak GPU util | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| step | 1 | 213.4 | 853.5 | 0% | 460 MiB |
| render | 1 | 159.3 | 637.3 | 0% | 460 MiB |
| inference | 1 | 94.4 | 377.6 | 5% | 993 MiB |
| step | 8 | 215.0 | 860.1 | 0% | 993 MiB |
| render | 8 | 160.9 | 643.6 | 0% | 993 MiB |
| inference | 8 | 195.7 | 782.9 | 2% | 1015 MiB |
| step | 32 | 213.5 | 853.8 | 0% | 1015 MiB |
| render | 32 | 160.0 | 639.9 | 0% | 1015 MiB |
| inference | 32 | 201.4 | 805.6 | 2% | 1063 MiB |
| step | 128 | 212.9 | 851.6 | 0% | 1063 MiB |
| render | 128 | 159.5 | 638.0 | 0% | 1063 MiB |
| inference | 128 | 200.5 | 802.0 | 5% | 1275 MiB |
| step | 256 | 213.1 | 852.6 | 0% | 460 MiB |
| render | 256 | 158.7 | 634.8 | 0% | 460 MiB |
| inference | 256 | 199.6 | 798.5 | 13% | 1577 MiB |
| step | 512 | 213.5 | 854.1 | 0% | 1577 MiB |
| render | 512 | 159.7 | 638.7 | 0% | 1577 MiB |
| inference | 512 | 201.4 | 805.6 | 17% | 2299 MiB |

Interpretation: the packaged Python API currently uses the native CPU backend.
Throughput is flat across environment counts, so Phase 5 cannot claim GPU-scale
environment stepping until a packaged CUDA backend is wired into `NesleVecEnv`.
The inference path uses CUDA, but the GPU waits on CPU env stepping.

## Raw CUDA Kernel Benchmark

Command:

```sh
NESLE_CUDA_ARCH=sm_80 sh scripts/benchmark_cuda_kernels.sh \
  --env-counts 1024,4096,8192,16384 \
  --step-iterations 2000 \
  --render-iterations 50 \
  --warmup-iterations 20
```

| Envs | Reward env steps/sec | Render frames/sec |
| ---: | ---: | ---: |
| 1024 | 144,467,000 | 34,350 |
| 4096 | 709,597,000 | 136,468 |
| 8192 | 1,374,450,000 | 130,098 |
| 16384 | 2,633,530,000 | 166,120 |

Interpretation: the lower-level CUDA reward and render kernels scale on A100.
The next implementation step is a packaged CUDA environment backend that owns
device buffers, resets from snapshots, steps batched actions, and exposes NumPy
observations/rewards/dones to the Python vector API.
