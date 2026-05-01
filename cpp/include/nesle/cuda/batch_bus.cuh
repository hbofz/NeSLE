#pragma once

#include <cstdint>

#include "nesle/cuda/state.cuh"

#ifdef __CUDACC__
#define NESLE_CUDA_HD __host__ __device__
#else
#define NESLE_CUDA_HD
#endif

namespace nesle::cuda {

NESLE_CUDA_HD inline std::uint8_t* env_prg_ram(BatchBuffers& buffers, std::uint32_t env) {
    return buffers.cpu.prg_ram + static_cast<std::uint64_t>(env) * kPrgRamBytes;
}

NESLE_CUDA_HD inline const std::uint8_t* env_prg_ram(const BatchBuffers& buffers,
                                                     std::uint32_t env) {
    return buffers.cpu.prg_ram + static_cast<std::uint64_t>(env) * kPrgRamBytes;
}

NESLE_CUDA_HD inline std::uint8_t read_nrom_prg(const CartridgeView& cart,
                                                std::uint16_t address) {
    auto index = static_cast<std::uint32_t>(address - 0x8000);
    if (cart.prg_rom_size == 16u * 1024u) {
        index &= 0x3FFFu;
    } else {
        index &= 0x7FFFu;
    }
    return cart.prg_rom[index];
}

NESLE_CUDA_HD inline std::uint16_t bus_mirror_palette_address(std::uint16_t address) {
    auto index = static_cast<std::uint16_t>(address & 0x001F);
    if (index == 0x10 || index == 0x14 || index == 0x18 || index == 0x1C) {
        index = static_cast<std::uint16_t>(index - 0x10);
    }
    return index;
}

NESLE_CUDA_HD inline std::uint16_t bus_mirror_nametable_address(const CartridgeView& cart,
                                                                std::uint16_t address) {
    const auto index = static_cast<std::uint16_t>((address - 0x2000) & 0x0FFF);
    if (cart.nametable_arrangement == kNametableFourScreen) {
        return index;
    }
    if (cart.nametable_arrangement == kNametableHorizontal) {
        return static_cast<std::uint16_t>((index & 0x03FF) | ((index & 0x0800) >> 1));
    }
    return static_cast<std::uint16_t>(index & 0x07FF);
}

NESLE_CUDA_HD inline std::uint8_t bus_ppu_memory_read(BatchBuffers& buffers,
                                                      std::uint32_t env,
                                                      std::uint16_t address) {
    address = static_cast<std::uint16_t>(address & 0x3FFF);
    if (address < 0x2000) {
        if (buffers.cart.chr_rom != nullptr && buffers.cart.chr_rom_size != 0) {
            return buffers.cart.chr_rom[address % buffers.cart.chr_rom_size];
        }
        return 0;
    }
    if (address < 0x3F00) {
        if (address >= 0x3000) {
            address = static_cast<std::uint16_t>(address - 0x1000);
        }
        return buffers.ppu.nametable_ram[
            static_cast<std::uint64_t>(env) * kNametableRamBytes +
            bus_mirror_nametable_address(buffers.cart, address)];
    }
    return buffers.ppu.palette_ram[
        static_cast<std::uint64_t>(env) * kPaletteRamBytes + bus_mirror_palette_address(address)];
}

NESLE_CUDA_HD inline void bus_ppu_memory_write(BatchBuffers& buffers,
                                               std::uint32_t env,
                                               std::uint16_t address,
                                               std::uint8_t value) {
    address = static_cast<std::uint16_t>(address & 0x3FFF);
    if (address < 0x2000) {
        return;
    }
    if (address < 0x3F00) {
        if (address >= 0x3000) {
            address = static_cast<std::uint16_t>(address - 0x1000);
        }
        buffers.ppu.nametable_ram[static_cast<std::uint64_t>(env) * kNametableRamBytes +
                                  bus_mirror_nametable_address(buffers.cart, address)] = value;
        return;
    }
    buffers.ppu.palette_ram[static_cast<std::uint64_t>(env) * kPaletteRamBytes +
                            bus_mirror_palette_address(address)] = value;
}

NESLE_CUDA_HD inline void bus_increment_vram_address(BatchBuffers& buffers, std::uint32_t env) {
    if (buffers.ppu.v != nullptr) {
        buffers.ppu.v[env] =
            static_cast<std::uint16_t>(buffers.ppu.v[env] + ((buffers.ppu.ctrl[env] & 0x04) != 0 ? 32 : 1));
    }
}

NESLE_CUDA_HD inline void latch_controller1(BatchBuffers& buffers, std::uint32_t env) {
    buffers.cpu.controller1_shift[env] = buffers.action_masks[env];
    buffers.cpu.controller1_shift_count[env] = 0;
}

NESLE_CUDA_HD inline std::uint8_t read_controller1(BatchBuffers& buffers, std::uint32_t env) {
    if (buffers.cpu.controller1_strobe[env] != 0) {
        return static_cast<std::uint8_t>(0x40 | (buffers.action_masks[env] & 0x01));
    }

    std::uint8_t bit = 1;
    if (buffers.cpu.controller1_shift_count[env] < 8) {
        bit = static_cast<std::uint8_t>(buffers.cpu.controller1_shift[env] & 0x01);
        buffers.cpu.controller1_shift[env] =
            static_cast<std::uint8_t>(buffers.cpu.controller1_shift[env] >> 1);
        ++buffers.cpu.controller1_shift_count[env];
    }
    return static_cast<std::uint8_t>(0x40 | bit);
}

NESLE_CUDA_HD inline void write_controller_strobe(BatchBuffers& buffers,
                                                  std::uint32_t env,
                                                  std::uint8_t value) {
    const auto next_strobe = static_cast<std::uint8_t>(value & 0x01);
    if (next_strobe != 0 || buffers.cpu.controller1_strobe[env] != 0) {
        latch_controller1(buffers, env);
    }
    buffers.cpu.controller1_strobe[env] = next_strobe;
}

NESLE_CUDA_HD inline void batch_oam_dma(BatchBuffers& buffers,
                                        std::uint32_t env,
                                        std::uint8_t page) {
    const auto base = static_cast<std::uint16_t>(page << 8);
    auto* oam = env_oam(buffers, env);
    for (std::uint16_t i = 0; i < kOamBytes; ++i) {
        oam[static_cast<std::uint8_t>(buffers.ppu.oam_addr[env] + i)] =
            env_cpu_ram(buffers, env)[(base + i) & 0x07FF];
    }
    if (buffers.cpu.pending_dma_cycles != nullptr) {
        buffers.cpu.pending_dma_cycles[env] += 513;
    }
}

NESLE_CUDA_HD inline std::uint8_t batch_cpu_read(BatchBuffers& buffers,
                                                 std::uint32_t env,
                                                 std::uint16_t address) {
    if (address < 0x2000) {
        return env_cpu_ram(buffers, env)[address & 0x07FF];
    }
    if (address < 0x4000) {
        const auto reg = static_cast<std::uint16_t>((address - 0x2000) & 0x0007);
        if (reg == 2) {
            const auto open_bus = buffers.ppu.open_bus != nullptr ? buffers.ppu.open_bus[env] : 0;
            const auto value = static_cast<std::uint8_t>((buffers.ppu.status[env] & 0xE0) |
                                                         (open_bus & 0x1F));
            buffers.ppu.status[env] = static_cast<std::uint8_t>(buffers.ppu.status[env] & 0x7F);
            if (buffers.ppu.nmi_pending != nullptr) {
                buffers.ppu.nmi_pending[env] = 0;
            }
            if (buffers.ppu.w != nullptr) {
                buffers.ppu.w[env] = 0;
            }
            if (buffers.ppu.open_bus != nullptr) {
                buffers.ppu.open_bus[env] = value;
            }
            return value;
        }
        if (reg == 4) {
            return env_oam(buffers, env)[buffers.ppu.oam_addr[env]];
        }
        if (reg == 7 && buffers.ppu.v != nullptr && buffers.ppu.read_buffer != nullptr) {
            const auto ppu_address = static_cast<std::uint16_t>(buffers.ppu.v[env] & 0x3FFF);
            std::uint8_t value = 0;
            if (ppu_address >= 0x3F00) {
                value = bus_ppu_memory_read(buffers, env, ppu_address);
                buffers.ppu.read_buffer[env] =
                    bus_ppu_memory_read(buffers, env, static_cast<std::uint16_t>(ppu_address - 0x1000));
            } else {
                value = buffers.ppu.read_buffer[env];
                buffers.ppu.read_buffer[env] = bus_ppu_memory_read(buffers, env, ppu_address);
            }
            bus_increment_vram_address(buffers, env);
            if (buffers.ppu.open_bus != nullptr) {
                buffers.ppu.open_bus[env] = value;
            }
            return value;
        }
        return 0;
    }
    if (address == 0x4016) {
        return read_controller1(buffers, env);
    }
    if (address == 0x4017) {
        return 0x41;
    }
    if (address >= 0x6000 && address < 0x8000) {
        return env_prg_ram(buffers, env)[address - 0x6000];
    }
    if (address >= 0x8000) {
        return read_nrom_prg(buffers.cart, address);
    }
    return 0;
}

NESLE_CUDA_HD inline void batch_cpu_write(BatchBuffers& buffers,
                                          std::uint32_t env,
                                          std::uint16_t address,
                                          std::uint8_t value) {
    if (address < 0x2000) {
        env_cpu_ram(buffers, env)[address & 0x07FF] = value;
        return;
    }
    if (address < 0x4000) {
        const auto reg = static_cast<std::uint16_t>((address - 0x2000) & 0x0007);
        if (reg == 0) {
            if ((value & 0x80) != 0 && (buffers.ppu.ctrl[env] & 0x80) == 0 &&
                (buffers.ppu.status[env] & 0x80) != 0) {
                if (buffers.ppu.nmi_pending != nullptr) {
                    buffers.ppu.nmi_pending[env] = 1;
                }
            }
            buffers.ppu.ctrl[env] = value;
            if (buffers.ppu.t != nullptr) {
                buffers.ppu.t[env] =
                    static_cast<std::uint16_t>((buffers.ppu.t[env] & 0xF3FF) | ((value & 0x03) << 10));
            }
        } else if (reg == 1) {
            buffers.ppu.mask[env] = value;
        } else if (reg == 3) {
            buffers.ppu.oam_addr[env] = value;
        } else if (reg == 4) {
            env_oam(buffers, env)[buffers.ppu.oam_addr[env]] = value;
            ++buffers.ppu.oam_addr[env];
        } else if (reg == 5 && buffers.ppu.t != nullptr && buffers.ppu.w != nullptr) {
            if (buffers.ppu.w[env] == 0) {
                if (buffers.ppu.x != nullptr) {
                    buffers.ppu.x[env] = static_cast<std::uint8_t>(value & 0x07);
                }
                if (buffers.ppu.scroll_x != nullptr) {
                    buffers.ppu.scroll_x[env] = value;
                }
                buffers.ppu.t[env] =
                    static_cast<std::uint16_t>((buffers.ppu.t[env] & 0xFFE0) | (value >> 3));
                buffers.ppu.w[env] = 1;
            } else {
                if (buffers.ppu.scroll_y != nullptr) {
                    buffers.ppu.scroll_y[env] = value;
                }
                buffers.ppu.t[env] =
                    static_cast<std::uint16_t>((buffers.ppu.t[env] & 0x8FFF) | ((value & 0x07) << 12));
                buffers.ppu.t[env] =
                    static_cast<std::uint16_t>((buffers.ppu.t[env] & 0xFC1F) | ((value & 0xF8) << 2));
                buffers.ppu.w[env] = 0;
            }
        } else if (reg == 6 && buffers.ppu.t != nullptr && buffers.ppu.w != nullptr) {
            if (buffers.ppu.w[env] == 0) {
                buffers.ppu.t[env] =
                    static_cast<std::uint16_t>((buffers.ppu.t[env] & 0x00FF) | ((value & 0x3F) << 8));
                buffers.ppu.w[env] = 1;
            } else {
                buffers.ppu.t[env] =
                    static_cast<std::uint16_t>((buffers.ppu.t[env] & 0x7F00) | value);
                if (buffers.ppu.v != nullptr) {
                    buffers.ppu.v[env] = buffers.ppu.t[env];
                }
                buffers.ppu.w[env] = 0;
            }
        } else if (reg == 7 && buffers.ppu.v != nullptr) {
            bus_ppu_memory_write(buffers, env, buffers.ppu.v[env], value);
            bus_increment_vram_address(buffers, env);
        }
        if (buffers.ppu.open_bus != nullptr) {
            buffers.ppu.open_bus[env] = value;
        }
        return;
    }
    if (address == 0x4014) {
        batch_oam_dma(buffers, env, value);
        return;
    }
    if (address == 0x4016) {
        write_controller_strobe(buffers, env, value);
        return;
    }
    if (address >= 0x6000 && address < 0x8000) {
        env_prg_ram(buffers, env)[address - 0x6000] = value;
    }
}

}  // namespace nesle::cuda

#undef NESLE_CUDA_HD
