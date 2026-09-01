# Verification run, 2026-09-01

Re-measurement of every performance claim in the README, on a clean clone.

**Provenance:** transcribed from the notebook session's console output. The
machine-readable `verification.json` was not retrieved before the runtime was
released. Regenerate it with `benchmarks/verify_claims.py`, which writes the
JSON directly.

## Environment

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB (81,920 MiB, sm_80) |
| Driver | 580.82.07, CUDA 13.0 |
| torch | 2.11.0+cu128 |
| Python | 3.13.15, Linux-6.6.122+-x86_64-with-glibc2.35 |
| Commit | f3140dbc18cac34f86acec2f7ad8bb7b9e1ed79e |
| Started | 2026-09-01T18:18:45Z |

## Protocol

Identical to `benchmarks/gpu_vs_cpu.py`: frameskip 4, RIGHT held every step,
30 warmup and 200 timed steps per point, `render_frame=False, copy_obs=False`,
snapshot reset from `docs/data/smb_level1_1.state`.

## Gates

- `pytest tests/` : **69 passed** in 15.05s
- `benchmarks/verify_correctness.py` : **3/3 passed**
  (5 distinct x_pos values across 8 action masks; 15,356 instr/step at every
  batch size with frames_completed=4; 256 distinct RAM hashes across 256 envs)

## Throughput

Single-environment native CPU baseline on this VM: **222 env-steps/s**.

| Envs | Env-steps/s | vs 1-env CPU |
|---:|---:|---:|
| 1 | 82 | 0.4x |
| 8 | 654 | 2.9x |
| 16 | 1,300 | 5.8x |
| 32 | 2,551 | 11.5x |
| 64 | 5,031 | 22.6x |
| 128 | 9,784 | 44.0x |
| 256 | 19,549 | 87.9x |
| 512 | 39,092 | 175.8x |
| 1,024 | 78,234 | 351.9x |
| 2,048 | 156,339 | 703.2x |
| 4,096 | 311,770 | 1,402.3x |
| 8,192 | 617,990 | 2,779.7x |
| 16,384 | 1,110,450 | 4,994.8x |
| 32,768 | 2,029,814 | 9,130.0x |
| **65,536** | **3,274,290** | **14,727.6x** |
| 131,072 | 3,101,507 | 13,950.5x |

Peak 3,274,290 env-steps/s = 13,097,160 NES frames/s, about 218,286x real time.

## Claims

| Envs | Published | Measured | Ratio | Verdict |
|---:|---:|---:|---:|---|
| 4,096 | 311,844 | 311,770 | 1.000x | PASS |
| 16,384 | 1,050,710 | 1,110,450 | 1.057x | PASS |
| 32,768 | 1,931,460 | 2,029,814 | 1.051x | PASS |
| 65,536 | 3,142,205 | 3,274,290 | 1.042x | PASS |
| 131,072 | 2,816,144 | 3,101,507 | 1.101x | PASS |

Every claim reproduced at or above its published value. The v0.3.0 tables also
assumed a 219 env-steps/s VM CPU baseline; this run measured 222, within 1.4%.

## Measured for the first time

**Crossover: 8 envs.** 1 env runs at 0.4x the single-env CPU (launch overhead
dominates); 8 envs already reach 2.9x. The true crossover lies between 2 and 8,
untested. The README previously said ~32 and `benchmark-gpu-vs-cpu.md` said 64,
both measured before the 2026-07-29 optimization program.

**Scaling stays linear to ~8,192**, not ~2,048 as previously documented: clean
doubling from 8 through 4,096, holding at 8,192, softening after 16,384.

**Device memory: ~195 KB per environment.** At 131,072 envs, sampled with the
batch resident:

| | MiB |
|---|---:|
| before | 430 |
| live | 25,348 |
| after free | 430 |
| **attributable** | **24,918** (24.3 GiB) |
| card total | 81,920 |

Full release back to the 430 MiB baseline confirms the reading. This implies
about 12.2 GiB at 65,536 envs, which corroborates the previously unbacked
"13 GB" figure, and matches the ~200 KB/env estimate in
[benchmark-gpu-vs-cpu.md](../benchmark-gpu-vs-cpu.md).

## Not measured

The multi-core nes-py baseline. Colab's `python3-venv` ships without
`ensurepip`, so the isolated old-gym environment could not be created. It
remains a [roadmap item](../../README.md#roadmap) and is best measured on the
GTX 1050 Ti host, so that GPU and CPU figures come from one machine.
