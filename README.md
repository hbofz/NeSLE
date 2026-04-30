# NeSLE

NeSLE is a GPU-native NES learning environment aimed at running thousands of
Super Mario Bros. instances on NVIDIA GPUs behind a Gymnasium/SB3-compatible
Python API.

The repository is intentionally staged. The first checked-in slice establishes
the target architecture, project layout, NROM/iNES parsing, Mario RAM decoding,
reward extraction, action mappings, and tests. CUDA emulation kernels are
scaffolded but not yet implemented.

## Current Phase

Phase 0 is the executable project skeleton:

- C++ core metadata parser for `.nes` iNES ROMs
- Super Mario Bros. RAM interpretation and reward components
- Python mirrors for action, ROM, and RAM utilities
- CUDA-facing state layout headers for the GPU emulator
- Research notes, architecture plan, and phase gates
- Python and C++ smoke tests

## Quick Verification

```sh
sh scripts/verify_phase0.sh
```

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
