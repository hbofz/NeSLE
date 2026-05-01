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

## CUDA Python Backend

Command:

```sh
NESLE_CUDA_ARCH=sm_80 PYTHON=python3 sh scripts/build_cuda_extension.sh
PYTHONPATH=src python3 benchmarks/phase5_benchmark.py \
  "Super Mario Bros. (World).nes" \
  --env-counts 128,512,2048,4096 \
  --steps 100 \
  --warmup-steps 20 \
  --modes step,render,inference \
  --backend cuda
```

| Mode | Envs | Env steps/sec | Training frames/sec | Peak GPU util | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| step | 128 | 3,652.9 | 14,611.7 | 91% | 911 MiB |
| render | 128 | 3,187.0 | 12,747.8 | 83% | 911 MiB |
| inference | 128 | 1,714.7 | 6,858.8 | 59% | 1301 MiB |
| step | 512 | 8,051.8 | 32,207.4 | 61% | 1367 MiB |
| render | 512 | 5,328.4 | 21,313.4 | 49% | 1367 MiB |
| inference | 512 | 2,426.5 | 9,706.1 | 56% | 2207 MiB |
| step | 2048 | 12,487.3 | 49,949.1 | 45% | 2481 MiB |
| render | 2048 | 6,719.8 | 26,879.3 | 36% | 2481 MiB |
| inference | 2048 | 2,790.8 | 11,163.2 | 40% | 5829 MiB |
| step | 4096 | 13,446.5 | 53,786.0 | 36% | 6193 MiB |
| render | 4096 | 7,177.6 | 28,710.6 | 36% | 6193 MiB |
| inference | 4096 | 2,871.5 | 11,486.0 | 46% | 11955 MiB |

Interpretation: `backend="cuda"` now exercises a packaged CUDA vector backend
through the public Python API. This first backend moves batched action
application, reward, and RGB render kernels onto the GPU, while still returning
NumPy observations/rewards/dones each step. It is a large improvement over the
native CPU backend, but it is still below raw kernel capacity because every step
copies full RGB observations back to host memory and does not yet run the full
CUDA CPU/PPU console loop per action.

## CUDA Reward No-Copy Mode

Command:

```sh
PYTHONPATH=src python3 benchmarks/phase5_benchmark.py \
  "Super Mario Bros. (World).nes" \
  --env-counts 128,512,2048,4096,8192,16384 \
  --steps 500 \
  --warmup-steps 50 \
  --modes reward \
  --backend cuda
```

| Mode | Envs | Env steps/sec | Training frames/sec | Peak GPU util | Peak GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| reward | 128 | 1,649,929.4 | 6,599,717.7 | 3% | 911 MiB |
| reward | 512 | 6,110,703.3 | 24,442,813.1 | 40% | 1001 MiB |
| reward | 2048 | 22,600,223.2 | 90,400,892.9 | 41% | 1275 MiB |
| reward | 4096 | 40,810,137.0 | 163,240,548.1 | 22% | 1639 MiB |
| reward | 8192 | 68,550,659.5 | 274,202,637.9 | 19% | 2367 MiB |
| reward | 16384 | 102,991,288.4 | 411,965,153.7 | 13% | 3823 MiB |

Interpretation: the benchmark can now separate training-style reward/done
throughput from full RGB observation throughput. This path skips per-step RGB
render and host observation copy, while still copying rewards and done flags
back to Python for accounting.

## Legacy Comparison

The legacy comparison was run with `gym-super-mario-bros==7.4.0`,
`nes-py==8.2.1`, `gym==0.25.2`, and `numpy==1.26.4`. Newer Colab images ship
NumPy 2.x and Gym 0.26+, which are incompatible with this older stack unless
the legacy dependencies are pinned in an isolated install.

| Runner | Envs | Env steps/sec | Training frames/sec |
| --- | ---: | ---: | ---: |
| gym-super-mario-bros | 1 | 432.9 | 1,731.5 |
| gym-super-mario-bros | 8 | 423.3 | 1,693.3 |

Interpretation: the legacy CPU emulator saturates around 423 env-steps/sec on
the Colab A100 host CPU for this simple sequential comparison. The packaged
CUDA backend is roughly 31.8x faster at 4096 envs when returning RGB
observations, and the no-copy reward mode is roughly 243,300x faster at 16,384
envs for reward/done accounting.

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
The CUDA Python backend now owns device buffers and exposes NumPy
observations/rewards/dones to the Python vector API. The next implementation
step is replacing its synthetic action-application kernel with the full CUDA
CPU/PPU console loop and adding render-cadence/no-copy modes for training.
