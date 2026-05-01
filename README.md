# NeSLE

NeSLE is a GPU-native NES learning environment aimed at running thousands of
Super Mario Bros. instances on NVIDIA GPUs behind a Gymnasium/SB3-compatible
Python API.

The repository is intentionally staged. Phase 0 established the target
architecture, project layout, NROM/iNES parsing, Mario RAM decoding, reward
extraction, action mappings, and tests. Phase 1 is now building the portable NES
CPU and NROM memory-map core that will later compile into CUDA kernels.

## Current Phase

The current CPU path covers the Phase 1 core and the first Phase 2 Mario
PPU/input gate:

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

## Quick Verification

```sh
sh scripts/verify.sh
```

With a local Super Mario Bros. `.nes` file, run the optional real-ROM gate:

```sh
NESLE_ROM_PATH="/path/to/Super Mario Bros. (World).nes" sh scripts/smoke_user_rom.sh
NESLE_ROM_PATH="/path/to/Super Mario Bros. (World).nes" sh scripts/smoke_phase2_user_rom.sh
NESLE_ROM_PATH="/path/to/Super Mario Bros. (World).nes" sh scripts/render_openemu_state.sh
NESLE_ROM_PATH="/path/to/Super Mario Bros. (World).nes" sh scripts/compare_openemu_state.sh
```

On an NVIDIA CUDA machine, run the optional device smoke:

```sh
sh scripts/verify_cuda.sh
```

That smoke compiles the CUDA kernels, launches a 4096-env reward/done batch,
and runs a tiny on-device NROM CPU trace through the batch CPU bus.

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

SB3 compatibility is planned around SB3's `VecEnv` contract, while the single
environment wrapper will follow the maintained Gymnasium step/reset API.

## Documents

- [Research notes](docs/research-notes.md)
- [Architecture](docs/architecture.md)
- [Phases](docs/phases.md)
- [CPU validation](docs/cpu-validation.md)
- [Headless runner](docs/headless-runner.md)
