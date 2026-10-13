#pragma once

#include <cstdint>

#include "nesle/cuda/state.cuh"

#ifdef __CUDACC__
// The banking helpers sit in the PPU pattern-fetch path (one per pattern table
// byte per pixel) as well as the CPU bus, so they are force-inlined for the
// same reason batch_render.cuh does it: nvcc's heuristics otherwise leave a
// call in the innermost loop.
#define NESLE_CUDA_MAPPER_HD __host__ __device__
#define NESLE_CUDA_MAPPER_INLINE __forceinline__
#else
#define NESLE_CUDA_MAPPER_HD
#define NESLE_CUDA_MAPPER_INLINE inline
#endif

// Cartridge banking shared by the bus and the renderer. Mapper 0 (NROM) keeps
// its historical zero-indirection path: no per-env register load, no second
// dependent memory access. Every banked board reduces to the same shape
// described by CartridgeView (switchable window at the bottom of $8000-$FFFF,
// fixed window on top), so the device code below is a single branch on the
// batch-uniform `bank_kind` rather than a per-mapper dispatch.

namespace nesle::cuda {

// Compile-time cartridge specialization for the bus and PPU-fetch paths.
//
// bank_kind is batch-uniform, so the console step kernel can pick the banking
// shape once per launch and hand it down as a template argument instead of
// re-testing it on every PRG byte, CHR byte and nametable fetch. Measured on
// Super Mario Bros. (mapper 0): dispatching on the runtime field instead costs
// 34% of console-step throughput, because the compiler cannot drop the tests and
// the accessors then have to thread `env` through the innermost loop.
//
//   kBankModeRuntime  dispatch on cart.bank_kind (default, so callers that do
//                     not know the cartridge keep working unchanged)
//   kBankModeNrom     compile-time mapper 0: the pre-mapper expressions, no tests
//   kBankModeBanked   compile-time banked board: no NROM test at all
enum BankMode : std::uint8_t {
    kBankModeRuntime = 0,
    kBankModeNrom = 1,
    kBankModeBanked = 2,
};

// True when Mode proves the cartridge has no bank registers, so the NROM
// expressions can be emitted without a runtime test.
NESLE_CUDA_MAPPER_HD constexpr bool bank_mode_is_nrom(BankMode mode) noexcept {
    return mode == kBankModeNrom;
}

// ---------------------------------------------------------------- PRG reads

// NROM: the whole 16 or 32 KB image sits at $8000. Kept byte-identical to the
// pre-UxROM read (including the 16 KB special case) so mapper 0 ROMs keep
// their exact behavior and their hot-path cost.
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE std::uint8_t read_nrom_prg(const CartridgeView& cart,
                                                       std::uint16_t address) {
    auto index = static_cast<std::uint32_t>(address - 0x8000);
    if (cart.prg_rom_size == 16u * 1024u) {
        index &= 0x3FFFu;
    } else {
        index &= 0x7FFFu;
    }
    return cart.prg_rom[index];
}

template <BankMode Mode = kBankModeRuntime>
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE std::uint8_t read_prg(const BatchBuffers& buffers,
                                                  std::uint32_t env,
                                                  std::uint16_t address) {
    const CartridgeView& cart = buffers.cart;
    if constexpr (bank_mode_is_nrom(Mode)) {
        return read_nrom_prg(cart, address);
    } else {
        if constexpr (Mode != kBankModeBanked) {
            if (cart.bank_kind == kBankingNone) {
                return read_nrom_prg(cart, address);
            }
        }
        std::uint32_t index;
        // The window sits at the bottom of CPU space. address - $8000 wraps to
        // a huge value for the fixed half, so one unsigned compare covers both.
        const std::uint32_t offset = static_cast<std::uint32_t>(address) - cart.prg_window_start;
        if (offset <= cart.prg_window_mask) {
            const std::uint32_t bank = buffers.mapper.prg_bank[env] & cart.prg_bank_mask;
            index = (bank << cart.prg_window_shift) + offset;
        } else {
            index = cart.prg_fixed_base + (address & cart.prg_fixed_mask);
        }

        // prg_rom_size is the padded (power of two) size, so the mask wraps
        // into the padding rather than reading out of range.
        return cart.prg_rom[index & cart.prg_rom_mask];
    }
}

// ------------------------------------------------------------ mapper writes

// A write anywhere in $8000-$FFFF latches the switchable bank. UxROM boards
// ignore the address.
template <BankMode Mode = kBankModeRuntime>
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void write_mapper_register(BatchBuffers& buffers,
                                                       std::uint32_t env,
                                                       std::uint16_t address,
                                                       std::uint8_t value) {
    const CartridgeView& cart = buffers.cart;
    if constexpr (bank_mode_is_nrom(Mode)) {
        return;  // mapper 0 has no registers; the write goes nowhere
    } else {
        if constexpr (Mode != kBankModeBanked) {
            if (cart.bank_kind == kBankingNone) {
                return;
            }
        }
        if (address < 0x8000) {
            // The mapper register is in the CPU space; a write to $6000-$7FFF
            // reaches only WRAM.
            return;
        }

        std::uint8_t latched = value;
        if (cart.bus_conflicts != 0) {
            // The cartridge's mask ROM drives 0s more strongly than the CPU data
            // bus, so only bits the ROM byte already has set survive the write.
            latched = static_cast<std::uint8_t>(value & read_prg<Mode>(buffers, env, address));
        }

        buffers.mapper.prg_bank[env] =
            static_cast<std::uint8_t>((latched >> cart.prg_bank_shift) & cart.prg_bank_mask);
    }
}

// ------------------------------------------------------------------ CHR I/O

template <BankMode Mode = kBankModeRuntime>
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE std::uint8_t read_chr(const BatchBuffers& buffers,
                                                  std::uint32_t env,
                                                  std::uint16_t address) {
    const CartridgeView& cart = buffers.cart;
    if (cart.chr_rom == nullptr) {
        // CHR RAM board: reads back what the game uploaded. 8 KB, mirrored
        // through $1FFF, which is what every mapper 2/94 board wires.
        if (buffers.ppu.chr_ram == nullptr) {
            return 0;
        }
        return buffers.ppu.chr_ram[static_cast<std::uint64_t>(env) * kChrRamBytes +
                                   (address & 0x1FFFu)];
    }
    // Fixed CHR wiring. The modulo is the pre-UxROM behavior and handles
    // non-power-of-two CHR ROM images, which some NROM dumps still ship.
    return cart.chr_rom[address % cart.chr_rom_size];
}


NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void write_chr(BatchBuffers& buffers,
                                           std::uint32_t env,
                                           std::uint16_t address,
                                           std::uint8_t value) {
    if (buffers.cart.chr_rom != nullptr) {
        return;  // CHR ROM board: the write goes nowhere
    }
    if (buffers.ppu.chr_ram == nullptr) {
        return;
    }
    buffers.ppu.chr_ram[static_cast<std::uint64_t>(env) * kChrRamBytes + (address & 0x1FFFu)] =
        value;
}

// --------------------------------------------------------------- mirroring

// Runtime arrangement for this env. No board here changes mirroring at run
// time, so this is the header's arrangement; kept as a function so the read
// sites stay uniform. The template parameter matches the other accessors so
// call sites can pass their BankMode through unchanged.
template <BankMode Mode = kBankModeRuntime>
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE std::uint8_t env_nametable_arrangement(
    const BatchBuffers& buffers, std::uint32_t) {
    return buffers.cart.nametable_arrangement;
}

NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE std::uint16_t mirror_nametable_address(std::uint8_t arrangement,
                                                                   std::uint16_t address) {
    const auto index = static_cast<std::uint16_t>((address - 0x2000) & 0x0FFF);
    switch (arrangement) {
        case kNametableFourScreen:
            // Only 2 KB of CIRAM is allocated per env, so a four-screen cart
            // folds to vertical instead of running off the end of its block.
            return static_cast<std::uint16_t>(index & 0x07FF);
        case kNametableHorizontal:
            return static_cast<std::uint16_t>((index & 0x03FF) | ((index & 0x0800) >> 1));
        case kNametableSingleScreenLower:
            return static_cast<std::uint16_t>(index & 0x03FF);
        case kNametableSingleScreenUpper:
            return static_cast<std::uint16_t>((index & 0x03FF) | 0x0400);
        default:
            return static_cast<std::uint16_t>(index & 0x07FF);
    }
}

// ------------------------------------------------------------------ resets

// Power-on state. UxROM boards latch bank 0 in the switchable window; the fixed
// window is not a register at all (it is the last page of the image). FCEUX
// save states carry no mapper registers, so a snapshot restore starts here too.
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void reset_mapper_state(BatchBuffers& buffers, std::uint32_t env) {
    const MapperStateSoA& map = buffers.mapper;
    if (map.prg_bank != nullptr) {
        map.prg_bank[env] = 0;
    }
}

}  // namespace nesle::cuda

#undef NESLE_CUDA_MAPPER_HD
#undef NESLE_CUDA_MAPPER_INLINE
