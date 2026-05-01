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
            const auto value = buffers.ppu.status[env];
            buffers.ppu.status[env] = static_cast<std::uint8_t>(buffers.ppu.status[env] & 0x7F);
            if (buffers.ppu.nmi_pending != nullptr) {
                buffers.ppu.nmi_pending[env] = 0;
            }
            if (buffers.ppu.w != nullptr) {
                buffers.ppu.w[env] = 0;
            }
            return value;
        }
        if (reg == 4) {
            return env_oam(buffers, env)[buffers.ppu.oam_addr[env]];
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
        } else if (reg == 1) {
            buffers.ppu.mask[env] = value;
        } else if (reg == 3) {
            buffers.ppu.oam_addr[env] = value;
        } else if (reg == 4) {
            env_oam(buffers, env)[buffers.ppu.oam_addr[env]] = value;
            ++buffers.ppu.oam_addr[env];
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
