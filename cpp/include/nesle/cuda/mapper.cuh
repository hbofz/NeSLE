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

NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE std::uint8_t read_prg(const BatchBuffers& buffers,
                                                  std::uint32_t env,
                                                  std::uint16_t address) {
    const CartridgeView& cart = buffers.cart;
    if (cart.bank_kind == kBankingNone) {
        return read_nrom_prg(cart, address);
    }
    std::uint32_t index;
    // The window can sit at the bottom of CPU space (every board but mapper
    // 180) or on top of it. address - prg_window_start wraps to a huge value
    // for whichever side is not the window, so one unsigned compare covers both.
    const std::uint32_t offset = static_cast<std::uint32_t>(address) - cart.prg_window_start;
    if (offset <= cart.prg_window_mask) {
        const std::uint32_t bank = buffers.mapper.prg_bank[env] & cart.prg_bank_mask;
        index = (bank << cart.prg_window_shift) + offset;
    } else {
        index = cart.prg_fixed_base + (address & cart.prg_fixed_mask);
    }

    // prg_rom_size is the padded (power of two) size, so the mask is a wrap

    // into the padding rather than an out-of-range read.
    return cart.prg_rom[index & cart.prg_rom_mask];
}

// ------------------------------------------------------------ mapper writes

// A write anywhere in $8000-$FFFF (or, for NINA-001, $7FFD-$7FFF) latches the
// switchable bank. UxROM boards ignore the address, except NINA-001.
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE void write_mapper_register(BatchBuffers& buffers,
                                                       std::uint32_t env,
                                                       std::uint16_t address,
                                                       std::uint8_t value) {
    const CartridgeView& cart = buffers.cart;
    if (cart.bank_kind == kBankingNone) {
        return;
    }
    if (address < 0x8000 && cart.bank_kind != kBankingNina8k) {
        // UxROM boards keep their registers in the CPU space; a write to
        // $6000-$7FFF reaches only WRAM.
        return;
    }

    std::uint8_t latched = value;
    if (cart.bus_conflicts != 0 && address >= 0x8000) {
        // The cartridge's mask ROM drives 0s more strongly than the CPU data
        // bus, so only bits the ROM byte already has set survive the write.
        latched = static_cast<std::uint8_t>(value & read_prg(buffers, env, address));
    }


    if (cart.bank_kind == kBankingNina8k) {
        switch (address) {
            case 0x7FFD:
                buffers.mapper.prg_bank[env] = latched;
                break;
            case 0x7FFE:
                buffers.mapper.chr_bank[env] = latched;
                break;
            case 0x7FFF:
                buffers.mapper.chr_bank_hi[env] = latched;
                break;
            default:
                break;
        }
        return;
    }

    buffers.mapper.prg_bank[env] =
        static_cast<std::uint8_t>((latched >> cart.prg_bank_shift) & cart.prg_bank_mask);

    if (cart.mapper == kMapperUnrom512) {
        // UNROM 512 multiplexes CHR-RAM and one-screen mirroring into the same
        // register: bits 5-6 pick the 8 KB CHR page, bit 7 picks the CIRAM
        // page the single screen shows.
        buffers.mapper.chr_bank[env] =
            static_cast<std::uint8_t>((latched >> 5) & cart.chr_bank_mask);
        if (cart.mapper_mirroring != 0 && buffers.mapper.nametable_arrangement != nullptr) {
            buffers.mapper.nametable_arrangement[env] = static_cast<std::uint8_t>(
                kNametableSingleScreenLower + ((latched >> 7) & 1));
        }
    }
}

// ------------------------------------------------------------------ CHR I/O

NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE std::uint8_t read_chr(const BatchBuffers& buffers,
                                                  std::uint32_t env,
                                                  std::uint16_t address) {
    const CartridgeView& cart = buffers.cart;
    if (cart.chr_rom == nullptr) {
        // CHR RAM board: reads back what the game uploaded. 8 KB, mirrored
        // through $1FFF, which is what every mapper 2/11/30/94 board wires.
        if (buffers.ppu.chr_ram == nullptr) {
            return 0;
        }
        const std::uint32_t page =
            (cart.bank_kind == kBankingNone)
                ? 0u
                : (static_cast<std::uint32_t>(buffers.mapper.chr_bank[env] & cart.chr_bank_mask)
                   << 13);
        return buffers.ppu.chr_ram[static_cast<std::uint64_t>(env) * kChrRamBytes +
                                   ((page + address) & 0x1FFFu)];
    }

    if (cart.bank_kind == kBankingNina8k) {
        // Two independent 4 KB windows over CHR ROM. describe_mapper only
        // accepts power-of-two NINA CHR ROM, so the mask is exact.
        const std::uint32_t window =
            (address < 0x1000 ? buffers.mapper.chr_bank[env] : buffers.mapper.chr_bank_hi[env]) &
            0x0Fu;
        return cart.chr_rom[((window << 12) | (address & 0x0FFFu)) & (cart.chr_rom_size - 1u)];
    }
    if (cart.mapper == kMapperUnrom512 && cart.chr_bank_mask != 0) {
        const std::uint32_t page =
            static_cast<std::uint32_t>(buffers.mapper.chr_bank[env] & cart.chr_bank_mask);
        return cart.chr_rom[((page << 13) | (address & 0x1FFFu)) & (cart.chr_rom_size - 1u)];
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

// Runtime arrangement for this env, or the header's when the board cannot
// change mirroring. The per-env array is nullptr for every mapper except
// UNROM 512, so the load never happens on the NROM path.
NESLE_CUDA_MAPPER_HD NESLE_CUDA_MAPPER_INLINE std::uint8_t env_nametable_arrangement(const BatchBuffers& buffers,
                                                                  std::uint32_t env) {
    if (buffers.mapper.nametable_arrangement != nullptr) {
        return buffers.mapper.nametable_arrangement[env];
    }
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
    if (map.chr_bank != nullptr) {
        map.chr_bank[env] = 0;
    }
    if (map.chr_bank_hi != nullptr) {
        map.chr_bank_hi[env] = 0;
    }
    if (map.nametable_arrangement != nullptr) {
        map.nametable_arrangement[env] = buffers.cart.nametable_arrangement;
    }
}

}  // namespace nesle::cuda

#undef NESLE_CUDA_MAPPER_HD
#undef NESLE_CUDA_MAPPER_INLINE
