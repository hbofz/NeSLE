# Research Notes

## CuLE Takeaways

CuLE demonstrates the thesis we want for NES: keep emulator state and rendered
frames resident on the GPU, run thousands of environments in parallel, and avoid
CPU/GPU observation transfer on the inference path. The paper reports up to
155M raw Atari frames per hour on one GPU and explicitly attributes the win to
GPU-side emulation, GPU-side rendering, and batching.

Important design lessons from CuLE:

- Use one logical emulator per GPU thread first. It is not theoretically optimal,
  but it is debuggable and already enough to beat CPU emulation at large env
  counts.
- Split execution kernels when their resource profiles differ. CuLE separates
  CPU/game execution from TIA rendering; NeSLE should separate 2A03 CPU/APU/PPU
  register execution from PPU frame rendering.
- Expect warp divergence after random policies decorrelate environments. Measure
  it and use reset-state caches, env grouping, and batch sizing to reduce harm.
- Render only when needed. RL commonly frame-skips and uses max-pooled or stacked
  frames, so many raw frames do not need full RGB output.

Sources:

- CuLE paper: https://arxiv.org/abs/1907.08467
- CuLE repo: https://github.com/NVlabs/cule

## NES Scope For Mario First

Super Mario Bros. is an NROM game, which is the right first target. NROM has no
bank switching, fixed PRG ROM, fixed CHR ROM, no mapper IRQs, and no cartridge
audio. That lets the first emulator avoid the hardest mapper problems while
still being a real NES emulator.

Core hardware needed for Mario:

- 2A03 CPU: NMOS 6502 core without decimal mode, plus NMI handling.
- CPU memory map: 2 KB internal RAM mirrored through `$1FFF`, PPU registers at
  `$2000-$2007` mirrored through `$3FFF`, APU/input around `$4000-$4017`, and
  cartridge space from `$4020-$FFFF`.
- PPU registers and enough cycle behavior for NMI, vblank, OAMDMA, scrolling,
  sprite 0 hit, background, and sprite rendering.
- Standard controller serial protocol through `$4016/$4017`.
- NROM mapper 0 with 16 KB or 32 KB PRG ROM and 8 KB CHR ROM.

Sources:

- NESdev CPU memory map: https://www.nesdev.org/wiki/CPU_memory_map
- NESdev PPU registers: https://www.nesdev.org/wiki/PPU_registers
- NESdev PPU rendering: https://www.nesdev.org/wiki/PPU_rendering
- NESdev cycle reference: https://www.nesdev.org/wiki/Cycle_reference_chart
- NESdev controller reading: https://www.nesdev.org/wiki/Controller_reading
- NESdev iNES format: https://www.nesdev.org/wiki/INES
- NESdev NROM: https://www.nesdev.org/wiki/NROM
- 6502 opcode reference: https://www.nesdev.org/obelisk-6502-guide/reference.html
- Klaus Dormann tests: https://github.com/Klaus2m5/6502_65C02_functional_tests

## Mario RL Surface

Mario reward and info should match the existing learning ecosystem before we
optimize. The useful RAM values are already established by `gym-super-mario-bros`
and Data Crystal:

- x position: `ram[0x006D] * 256 + ram[0x0086]`
- time: decimal digits at `0x07F8..0x07FA`
- coins: decimal digits at `0x07ED..0x07EE`
- world/stage/area: `0x075F`, `0x075C`, `0x0760`
- status: `0x0756`
- player state/death: `0x000E`, `0x00B5`

The baseline reward is progress plus time delta plus death penalty:

```text
r = x_delta + time_delta + death_penalty
```

with reset/death x jumps filtered out.

Sources:

- Mario RAM map: https://datacrystal.tcrf.net/wiki/Super_Mario_Bros./RAM_map
- nes-py: https://github.com/Kautenja/nes-py
- gym-super-mario-bros reward/API: https://github.com/Kautenja/gym-super-mario-bros

## RL API Compatibility

Gymnasium single-env API returns `(obs, reward, terminated, truncated, info)`.
SB3 vector environments intentionally use a Gym 0.21-like VecEnv API:
`reset()` returns only observations, and `step(actions)` returns
`obs, rewards, dones, infos`. SB3 also expects automatic reset behavior and
`terminal_observation` in `infos` when an episode ends.

Sources:

- Gymnasium Env API: https://gymnasium.farama.org/api/env/
- SB3 VecEnv API: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
