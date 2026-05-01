# NeSLE

NeSLE is a GPU-native NES learning environment aimed at running thousands of
Super Mario Bros. instances on NVIDIA GPUs behind a Gymnasium/SB3-compatible
Python API.

The repository is intentionally staged. Phase 0 established the target
architecture, project layout, NROM/iNES parsing, Mario RAM decoding, reward
extraction, action mappings, and tests. Phases 1 and 2 built the portable CPU,
PPU, input, rendering, and OpenEmu reference gates. Phase 3 moved the emulator
correctness contract into CUDA batch execution. Phase 4 added the Gymnasium/SB3
Python API.

## Current Phase

The current path covers the Phase 1/2 CPU emulator, the Phase 3 CUDA batch
contract, and the completed Phase 4 Python API:

- 2A03/6502 state and official-opcode execution core
- Flat 64 KB test bus, NROM memory-map smoke tests, and NES console CPU bus
- RAM, PPU register, APU/input, and PRG ROM mirroring behavior
- Basic NTSC PPU timing, vblank/NMI delivery, OAMDMA stalls, and frame stepping
- Coarse sprite-0-hit behavior for early Super Mario Bros. boot progress
- CPU RGB frame rendering for background and sprite tiles
- Deterministic action traces with Mario RAM, reward, RAM hash, and frame hash
- OpenEmu/Nestopia save-state rendering bridge for reference-frame debugging
- OpenEmu screenshot comparison gate for local Nestopia reference captures
- Headless `.nes` boot runner for NROM smoke tests
- C++ tests for CPU execution, stack calls, branches, arithmetic, and NROM reads
- CUDA smoke for 4096-env reward/done batches
- CUDA device smoke for the shared CPU core, batch console stepping, OAM DMA,
  PPU timing, PPU register-fed RGB rendering, and device-side reset snapshot
  restore
- `NesleEnv` and `NesleVecEnv` Python wrappers with Gymnasium-style single-env
  reset/step and SB3-style vector reset/step/auto-reset semantics
- Native C++ console binding hook plus deterministic Python compatibility
  backend for API development without a packaged native runtime

## Quick Verification

```sh
sh scripts/verify.sh
```

Phase 4 API checks can also be run directly:

```sh
python -m pip install -e '.[dev,rl]'
sh scripts/verify_phase4.sh
sh scripts/verify_native_binding.sh
```

`verify_native_binding.sh` compiles and imports the pybind extension, exercises
`NativeConsole`, and runs the native Python backend when the selected Python has
a complete NumPy install.

With a local Super Mario Bros. `.nes` file, run the optional real-ROM gate:

```sh
NESLE_ROM_PATH="/path/to/Super Mario Bros. (World).nes" sh scripts/smoke_user_rom.sh
NESLE_ROM_PATH="/path/to/Super Mario Bros. (World).nes" sh scripts/smoke_phase2_user_rom.sh
NESLE_ROM_PATH="/path/to/Super Mario Bros. (World).nes" sh scripts/smoke_phase4_user_rom.sh
NESLE_ROM_PATH="/path/to/Super Mario Bros. (World).nes" sh scripts/render_openemu_state.sh
NESLE_ROM_PATH="/path/to/Super Mario Bros. (World).nes" sh scripts/compare_openemu_state.sh
```

On an NVIDIA CUDA machine, run the optional device smoke:

```sh
sh scripts/verify_cuda.sh
```

That smoke compiles the CUDA kernels, launches a 4096-env reward/done batch,
runs a tiny on-device NROM CPU trace through the batch CPU bus, steps an
integrated CPU/PPU console path through OAM DMA, and verifies device-side reset
snapshot restore and byte-for-byte CPU/GPU RGB frame parity for a synthetic
background+sprite scene.

## Target API

The end state is:

```python
import nesle

env = nesle.make_vec(
    rom_path="Super Mario Bros. (World).nes",
    num_envs=4096,
    action_space="simple",
    render_mode="rgb_array",
)

obs = env.reset()
obs, rewards, dones, infos = env.step(actions)
```

The vector wrapper follows SB3's `VecEnv` reset/step shape and auto-reset
contract, including `terminal_observation`. The single environment wrapper uses
Gymnasium's reset/step return convention when Gymnasium is installed.

An SB3 PPO starter is available at [examples/sb3_train.py](examples/sb3_train.py):

```sh
python -m pip install -e '.[rl]'
python examples/sb3_train.py "Super Mario Bros. (World).nes" --num-envs 8
```

Legacy `nes-py` and `gym-super-mario-bros` comparison dependencies are kept in
the `legacy-mario` extra for benchmark work.

## Documents

- [Research notes](docs/research-notes.md)
- [Architecture](docs/architecture.md)
- [Phases](docs/phases.md)
- [CPU validation](docs/cpu-validation.md)
- [Headless runner](docs/headless-runner.md)
