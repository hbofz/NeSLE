# Contributing

Issues and pull requests are welcome. NeSLE is a small project, so there is no
process to fight; the notes below exist mostly so that changes to the CUDA
kernel stay measurable.

## Setup

See [docs/install.md](docs/install.md). Short version: Python 3.10+, a CUDA GPU,
CUDA Toolkit 12.x, a C++20 toolchain, then

```sh
python -m pip install -e '.[dev,rl]'
python -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
python scripts/build_cuda_extension.py
```

A stale `_cuda_core` extension fails silently and keeps the old behavior.
Rebuild after every change under `cpp/`.

You need to supply your own legally obtained Super Mario Bros. ROM. **Never
commit a ROM**; `*.nes` is gitignored and should stay that way.

## Checks

```sh
ruff check .                              # CI runs this
python -m pytest tests/                   # 69 tests; GPU/ROM tests auto-skip
sh scripts/run_cpp_tests.sh               # C++ unit tests
python benchmarks/verify_correctness.py   # batch is N independent emulators
```

CI runs ruff and pytest on Python 3.10 and 3.12, plus the C++ tests. GPU and
ROM dependent tests skip automatically on the runners, so **a green CI badge
does not mean the CUDA paths were exercised**. Run the suite locally on a GPU
before submitting anything that touches `cpp/`.

## Changes to the emulator core

The 6502 and PPU are behavior-critical: a subtle change can leave every test
passing while quietly altering emulation. Two gates exist for this.

- **Klaus functional tests** for the CPU ([docs/cpu-validation.md](docs/cpu-validation.md)).
- **Differential testing.** The table-driven decoder was validated against the
  previous interpreter across 8M random instructions, comparing results, bus
  traffic, and cycle counts. If you rewrite a decode or addressing path, do
  something equivalent and say so in the PR.

Bit-exactness matters more than it looks. A useful check is that a fixed-seed
training run produces identical losses before and after your change.

## Performance changes

Kernel work is only worth taking if it is measured. Include in the PR:

1. Before and after from `python benchmarks/gpu_vs_cpu.py`, on the same machine,
   same session, with the GPU otherwise idle.
2. The env counts you measured. Small-batch and large-batch behavior diverge;
   a win at 256 envs can be a regression at 65,536.
3. Confirmation that the test suite still passes.

`benchmarks/verify_claims.py` runs the full sweep and prints a
claimed-vs-measured table, and exits non-zero on failure. It also runs on a
rented GPU without a local CUDA setup:

```sh
colab new -s nesle --gpu A100
colab upload -s nesle "Super Mario Bros. (World).nes" /content/rom.nes
colab exec   -s nesle -f benchmarks/verify_claims.py
colab stop -s nesle
```

Two cautions learned the hard way:

- **The CPU baseline is noisy.** It moves 290 to 335 env-steps/s on the same
  machine depending on background load, while GPU numbers are stable within
  ~1%. Report absolute env-steps/s, not just a speedup multiple.
- **Watch for inlining cliffs.** Growing the CUDA module once slowed both render
  kernels by 2.3x, because nvcc's inlining heuristics gave up. Forcing inlining
  on the render helper chain recovered it and unlocked the change's real win.
  If a change makes something unrelated slower, suspect this first.

## Negative results are welcome

[docs/gpu-scaling.md](docs/gpu-scaling.md) records approaches that were tried
and rejected with the measurements that killed them: zero page in shared memory
(1.9x regression from occupancy collapse on sm_61), opcode-binned wavefront
dispatch (3.2x slower than thread-per-env). A PR that establishes something
does *not* work, with numbers, is a real contribution. Add it to that document.

Please read it before proposing a kernel redesign, so we do not re-litigate
settled ground.

## Where to start

The [open issues](https://github.com/hbofz/NeSLE/issues) are the live backlog,
labelled by difficulty. Good entry points:

- Wire the C++ tests in `tests/cpp/` into CI.
- Unify the POSIX and Windows build recipes into `setup.py`.
- Measure a multi-core CPU baseline, so the speedup claim is against a loaded
  CPU rather than one core.

The largest open item is **mapper support beyond NROM** (MMC1, MMC3), which
would take NeSLE from a Mario trainer to a general NES RL platform. Start at
`cpp/include/nesle/cuda/batch_bus.cuh`.

## Pull requests

Small and focused beats large and sweeping. Explain what you changed and why,
and include the measurements if it touches performance. If you are unsure
whether an idea fits, open an issue first and ask.
