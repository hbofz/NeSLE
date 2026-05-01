#pragma once

#include <cstdint>

#include "nesle/cuda/state.cuh"

#ifdef __CUDACC__
#define NESLE_CUDA_RENDER_HD __host__ __device__
#else
#define NESLE_CUDA_RENDER_HD
#endif

namespace nesle::cuda {

constexpr std::uint8_t kNesPaletteRgb[64 * 3] = {
    0x54, 0x54, 0x54, 0x00, 0x1e, 0x74, 0x08, 0x10, 0x90, 0x30, 0x00, 0x88,
    0x44, 0x00, 0x64, 0x5c, 0x00, 0x30, 0x54, 0x04, 0x00, 0x3c, 0x18, 0x00,
    0x20, 0x2a, 0x00, 0x08, 0x3a, 0x00, 0x00, 0x40, 0x00, 0x00, 0x3c, 0x00,
    0x00, 0x32, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x98, 0x96, 0x98, 0x08, 0x4c, 0xc4, 0x30, 0x32, 0xec, 0x5c, 0x1e, 0xe4,
    0x88, 0x14, 0xb0, 0xa0, 0x14, 0x64, 0x98, 0x22, 0x20, 0x78, 0x3c, 0x00,
    0x54, 0x5a, 0x00, 0x28, 0x72, 0x00, 0x08, 0x7c, 0x00, 0x00, 0x76, 0x28,
    0x00, 0x66, 0x78, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xec, 0xee, 0xec, 0x4c, 0x9a, 0xec, 0x78, 0x7c, 0xec, 0xb0, 0x62, 0xec,
    0xe4, 0x54, 0xec, 0xec, 0x58, 0xb4, 0xec, 0x6a, 0x64, 0xd4, 0x88, 0x20,
    0xa0, 0xaa, 0x00, 0x74, 0xc4, 0x00, 0x4c, 0xd0, 0x20, 0x38, 0xcc, 0x6c,
    0x38, 0xb4, 0xcc, 0x3c, 0x3c, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xec, 0xee, 0xec, 0xa8, 0xcc, 0xec, 0xbc, 0xbc, 0xec, 0xd4, 0xb2, 0xec,
    0xec, 0xae, 0xec, 0xec, 0xae, 0xd4, 0xec, 0xb4, 0xb0, 0xe4, 0xc4, 0x90,
    0xcc, 0xd2, 0x78, 0xb4, 0xde, 0x78, 0xa8, 0xe2, 0x90, 0x98, 0xe2, 0xb4,
    0xa0, 0xd6, 0xe4, 0xa0, 0xa2, 0xa0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

NESLE_CUDA_RENDER_HD inline std::uint8_t* env_frame_rgb(BatchBuffers& buffers,
                                                        std::uint32_t env) {
    return buffers.frames_rgb +
           static_cast<std::uint64_t>(env) * kFrameWidth * kFrameHeight * kRgbChannels;
}

NESLE_CUDA_RENDER_HD inline const std::uint8_t* env_nametable_ram(
    const BatchBuffers& buffers,
    std::uint32_t env) {
    return buffers.ppu.nametable_ram +
           static_cast<std::uint64_t>(env) * kNametableRamBytes;
}

NESLE_CUDA_RENDER_HD inline const std::uint8_t* env_palette_ram(
    const BatchBuffers& buffers,
    std::uint32_t env) {
    return buffers.ppu.palette_ram + static_cast<std::uint64_t>(env) * kPaletteRamBytes;
}

NESLE_CUDA_RENDER_HD inline std::uint16_t mirror_batch_palette_address(
    std::uint16_t address) {
    auto index = static_cast<std::uint16_t>(address & 0x001F);
    if (index == 0x10 || index == 0x14 || index == 0x18 || index == 0x1C) {
        index = static_cast<std::uint16_t>(index - 0x10);
    }
    return index;
}

NESLE_CUDA_RENDER_HD inline std::uint16_t mirror_batch_nametable_address(
    const CartridgeView& cart,
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

NESLE_CUDA_RENDER_HD inline std::uint8_t batch_ppu_memory_read(const BatchBuffers& buffers,
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
        return env_nametable_ram(buffers, env)[mirror_batch_nametable_address(buffers.cart, address)];
    }
    return env_palette_ram(buffers, env)[mirror_batch_palette_address(address)];
}

NESLE_CUDA_RENDER_HD inline std::uint8_t batch_palette_entry(const BatchBuffers& buffers,
                                                             std::uint32_t env,
                                                             std::uint16_t index) {
    auto value = batch_ppu_memory_read(buffers, env, static_cast<std::uint16_t>(0x3F00 + index));
    if ((buffers.ppu.mask[env] & 0x01) != 0) {
        value = static_cast<std::uint8_t>(value & 0x30);
    }
    return static_cast<std::uint8_t>(value & 0x3F);
}

NESLE_CUDA_RENDER_HD inline void write_batch_rgb(std::uint8_t* frame,
                                                 std::uint32_t pixel,
                                                 std::uint8_t palette_index) {
    const auto rgb = static_cast<std::uint16_t>((palette_index & 0x3F) * 3);
    frame[pixel * 3] = kNesPaletteRgb[rgb];
    frame[pixel * 3 + 1] = kNesPaletteRgb[rgb + 1];
    frame[pixel * 3 + 2] = kNesPaletteRgb[rgb + 2];
}

NESLE_CUDA_RENDER_HD inline std::uint8_t batch_pattern_pixel(const BatchBuffers& buffers,
                                                             std::uint32_t env,
                                                             std::uint16_t tile_base,
                                                             std::uint8_t fine_x,
                                                             std::uint8_t fine_y) {
    const auto low =
        batch_ppu_memory_read(buffers, env, static_cast<std::uint16_t>(tile_base + fine_y));
    const auto high =
        batch_ppu_memory_read(buffers, env, static_cast<std::uint16_t>(tile_base + fine_y + 8));
    const auto bit = static_cast<std::uint8_t>(7 - fine_x);
    return static_cast<std::uint8_t>(((low >> bit) & 0x01) | (((high >> bit) & 0x01) << 1));
}

NESLE_CUDA_RENDER_HD inline std::uint8_t batch_background_color(const BatchBuffers& buffers,
                                                                std::uint32_t env,
                                                                std::uint16_t x,
                                                                std::uint16_t y) {
    if ((buffers.ppu.mask[env] & 0x08) == 0 || ((buffers.ppu.mask[env] & 0x02) == 0 && x < 8)) {
        return 0;
    }

    const auto scroll_x = buffers.ppu.x != nullptr ? buffers.ppu.x[env] : 0;
    const auto scroll_y = static_cast<std::uint8_t>(
        buffers.ppu.t != nullptr ? ((buffers.ppu.t[env] >> 5) & 0x1F) * 8 : 0);
    const auto world_y = static_cast<unsigned>(scroll_y) + static_cast<unsigned>(y);
    const auto coarse_y = static_cast<std::uint8_t>((world_y % 240) / 8);
    const auto fine_y = static_cast<std::uint8_t>(world_y & 0x07);
    const auto base_nametable = static_cast<std::uint8_t>(buffers.ppu.ctrl[env] & 0x03);
    const auto nt_y = static_cast<std::uint8_t>(((base_nametable >> 1) + (world_y / 240)) & 0x01);
    const auto world_x = static_cast<unsigned>(scroll_x) + static_cast<unsigned>(x);
    const auto coarse_x = static_cast<std::uint8_t>((world_x & 0xFF) / 8);
    const auto fine_x = static_cast<std::uint8_t>(world_x & 0x07);
    const auto nt_x = static_cast<std::uint8_t>(((base_nametable & 0x01) + (world_x / 256)) & 0x01);
    const auto nametable = static_cast<std::uint8_t>(nt_x | (nt_y << 1));
    const auto nametable_base = static_cast<std::uint16_t>(0x2000 + nametable * 0x0400);
    const auto pattern_base = (buffers.ppu.ctrl[env] & 0x10) != 0 ? 0x1000 : 0x0000;
    const auto tile = batch_ppu_memory_read(
        buffers,
        env,
        static_cast<std::uint16_t>(nametable_base + coarse_y * 32 + coarse_x));
    const auto color = batch_pattern_pixel(
        buffers,
        env,
        static_cast<std::uint16_t>(pattern_base + static_cast<std::uint16_t>(tile) * 16),
        fine_x,
        fine_y);
    if (color == 0) {
        return 0;
    }

    const auto attribute = batch_ppu_memory_read(
        buffers,
        env,
        static_cast<std::uint16_t>(nametable_base + 0x03C0 + (coarse_y / 4) * 8 + (coarse_x / 4)));
    const auto shift = static_cast<std::uint8_t>(((coarse_y & 0x02) << 1) | (coarse_x & 0x02));
    const auto palette = static_cast<std::uint8_t>((attribute >> shift) & 0x03);
    return batch_palette_entry(buffers, env, static_cast<std::uint16_t>(palette * 4 + color));
}

NESLE_CUDA_RENDER_HD inline std::uint8_t batch_sprite_pattern_pixel(const BatchBuffers& buffers,
                                                                    std::uint32_t env,
                                                                    std::uint8_t tile,
                                                                    std::uint8_t attributes,
                                                                    std::uint8_t pixel_x,
                                                                    std::uint8_t pixel_y) {
    if ((attributes & 0x40) != 0) {
        pixel_x = static_cast<std::uint8_t>(7 - pixel_x);
    }

    if ((buffers.ppu.ctrl[env] & 0x20) != 0) {
        if ((attributes & 0x80) != 0) {
            pixel_y = static_cast<std::uint8_t>(15 - pixel_y);
        }
        const auto pattern_base = static_cast<std::uint16_t>((tile & 0x01) ? 0x1000 : 0x0000);
        const auto tile_number = static_cast<std::uint8_t>((tile & 0xFE) + (pixel_y / 8));
        return batch_pattern_pixel(
            buffers,
            env,
            static_cast<std::uint16_t>(pattern_base + tile_number * 16),
            pixel_x,
            static_cast<std::uint8_t>(pixel_y & 0x07));
    }

    if ((attributes & 0x80) != 0) {
        pixel_y = static_cast<std::uint8_t>(7 - pixel_y);
    }
    const auto pattern_base = (buffers.ppu.ctrl[env] & 0x08) != 0 ? 0x1000 : 0x0000;
    return batch_pattern_pixel(
        buffers,
        env,
        static_cast<std::uint16_t>(pattern_base + static_cast<std::uint16_t>(tile) * 16),
        pixel_x,
        pixel_y);
}

NESLE_CUDA_RENDER_HD inline void render_batch_rgb_frame_env(BatchBuffers& buffers,
                                                            std::uint32_t env) {
    auto* frame = env_frame_rgb(buffers, env);
    const auto backdrop = batch_palette_entry(buffers, env, 0);
    for (std::uint32_t pixel = 0; pixel < kFrameWidth * kFrameHeight; ++pixel) {
        write_batch_rgb(frame, pixel, backdrop);
    }

    for (std::uint16_t y = 0; y < kFrameHeight; ++y) {
        for (std::uint16_t x = 0; x < kFrameWidth; ++x) {
            const auto color = batch_background_color(buffers, env, x, y);
            if (color != 0) {
                write_batch_rgb(frame, y * kFrameWidth + x, color);
            }
        }
    }

    if ((buffers.ppu.mask[env] & 0x10) == 0 || buffers.ppu.oam == nullptr) {
        return;
    }

    const bool show_left = (buffers.ppu.mask[env] & 0x04) != 0;
    const auto sprite_height = static_cast<std::uint8_t>((buffers.ppu.ctrl[env] & 0x20) != 0 ? 16 : 8);
    const auto* oam = env_oam(buffers, env);
    for (int sprite = 63; sprite >= 0; --sprite) {
        const auto base = static_cast<std::uint16_t>(sprite * 4);
        const auto top = static_cast<int>(oam[base]) + 1;
        const auto tile = oam[base + 1];
        const auto attributes = oam[base + 2];
        const auto left = static_cast<int>(oam[base + 3]);
        const auto palette = static_cast<std::uint8_t>(attributes & 0x03);
        const bool behind_background = (attributes & 0x20) != 0;

        for (int sy = 0; sy < sprite_height; ++sy) {
            const auto y = top + sy;
            if (y < 0 || y >= kFrameHeight) {
                continue;
            }

            for (int sx = 0; sx < 8; ++sx) {
                const auto x = left + sx;
                if (x < 0 || x >= kFrameWidth || (!show_left && x < 8)) {
                    continue;
                }

                if (behind_background &&
                    batch_background_color(
                        buffers,
                        env,
                        static_cast<std::uint16_t>(x),
                        static_cast<std::uint16_t>(y)) != 0) {
                    continue;
                }

                const auto color = batch_sprite_pattern_pixel(
                    buffers,
                    env,
                    tile,
                    attributes,
                    static_cast<std::uint8_t>(sx),
                    static_cast<std::uint8_t>(sy));
                if (color == 0) {
                    continue;
                }
                const auto palette_value =
                    batch_palette_entry(buffers, env, static_cast<std::uint16_t>(0x10 + palette * 4 + color));
                write_batch_rgb(frame, static_cast<std::uint32_t>(y * kFrameWidth + x), palette_value);
            }
        }
    }
}

}  // namespace nesle::cuda

#undef NESLE_CUDA_RENDER_HD
