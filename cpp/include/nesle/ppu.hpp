#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "nesle/rom.hpp"

namespace nesle {

class Ppu {
public:
    static constexpr std::size_t kOamBytes = 256;
    static constexpr std::size_t kPpuAddressSpaceBytes = 16 * 1024;
    static constexpr std::size_t kScreenWidth = 256;
    static constexpr std::size_t kScreenHeight = 240;
    static constexpr std::size_t kRgbFrameBytes = kScreenWidth * kScreenHeight * 3;
    static constexpr std::uint16_t kDotsPerScanline = 341;
    static constexpr std::uint16_t kScanlinesPerFrame = 262;
    static constexpr std::int16_t kVblankStartScanline = 241;
    static constexpr std::int16_t kPreRenderScanline = 261;
    static constexpr std::uint16_t kVblankFlagDot = 1;
    static constexpr std::int16_t kCoarseSpriteZeroHitScanline = 30;
    static constexpr std::uint16_t kCoarseSpriteZeroHitDot = 1;

    struct StepResult {
        std::uint32_t cycles = 0;
        std::uint32_t frames_completed = 0;
        bool nmi_started = false;
    };

    struct RenderState {
        std::uint8_t ctrl = 0;
        std::uint8_t mask = 0;
        std::uint8_t status = 0;
        std::uint8_t scroll_x = 0;
        std::uint8_t scroll_y = 0;
        std::span<const std::uint8_t> palette_ram;
        std::span<const std::uint8_t> oam;
        std::span<const std::uint8_t> nametable_ram;
    };

    using RgbFrame = std::array<std::uint8_t, kRgbFrameBytes>;

    [[nodiscard]] StepResult step(std::uint32_t ppu_cycles) noexcept {
        StepResult result;
        result.cycles = ppu_cycles;

        for (std::uint32_t i = 0; i < ppu_cycles; ++i) {
            ++dot_;
            if (dot_ >= kDotsPerScanline) {
                dot_ = 0;
                ++scanline_;
                if (scanline_ >= kScanlinesPerFrame) {
                    scanline_ = 0;
                    ++frame_;
                    ++result.frames_completed;
                }
            }

            if (scanline_ == kVblankStartScanline && dot_ == kVblankFlagDot) {
                const bool had_nmi = nmi_pending_;
                set_vblank(true);
                result.nmi_started = result.nmi_started || (!had_nmi && nmi_pending_);
            } else if (scanline_ == kCoarseSpriteZeroHitScanline &&
                       dot_ == kCoarseSpriteZeroHitDot && rendering_enabled()) {
                status_ |= 0x40;
            } else if (scanline_ == kPreRenderScanline && dot_ == kVblankFlagDot) {
                set_vblank(false);
                status_ &= 0x1F;
            }
        }

        return result;
    }

    void configure_cartridge(std::span<const std::uint8_t> chr_rom,
                             NametableArrangement arrangement) noexcept {
        chr_rom_ = chr_rom;
        nametable_arrangement_ = arrangement;
    }

    [[nodiscard]] std::uint8_t read_register(std::uint16_t index) noexcept {
        switch (index & 0x0007) {
            case 0x02:
                return read_status();
            case 0x04:
                return oam_[oam_addr_];
            case 0x07:
                return read_data();
            default:
                return open_bus_;
        }
    }

    void write_register(std::uint16_t index, std::uint8_t value) noexcept {
        open_bus_ = value;
        switch (index & 0x0007) {
            case 0x00:
                if ((value & 0x80) != 0 && !nmi_enabled() && (status_ & 0x80) != 0) {
                    nmi_pending_ = true;
                }
                ctrl_ = value;
                t_ = static_cast<std::uint16_t>((t_ & 0xF3FF) | ((value & 0x03) << 10));
                break;
            case 0x01:
                mask_ = value;
                break;
            case 0x03:
                oam_addr_ = value;
                break;
            case 0x04:
                oam_[oam_addr_] = value;
                oam_addr_ = static_cast<std::uint8_t>(oam_addr_ + 1);
                break;
            case 0x05:
                write_scroll(value);
                break;
            case 0x06:
                write_address(value);
                break;
            case 0x07:
                write_data(value);
                break;
            default:
                break;
        }
    }

    void write_oam_dma(std::uint8_t value) noexcept {
        oam_[oam_addr_] = value;
        oam_addr_ = static_cast<std::uint8_t>(oam_addr_ + 1);
    }

    void set_vblank(bool enabled) noexcept {
        const bool was_in_vblank = (status_ & 0x80) != 0;
        if (enabled) {
            status_ |= 0x80;
            if (!was_in_vblank && nmi_enabled()) {
                nmi_pending_ = true;
            }
        } else {
            status_ &= 0x7F;
            nmi_pending_ = false;
        }
    }

    [[nodiscard]] bool nmi_enabled() const noexcept {
        return (ctrl_ & 0x80) != 0;
    }

    [[nodiscard]] bool rendering_enabled() const noexcept {
        return (mask_ & 0x18) != 0;
    }

    [[nodiscard]] bool nmi_pending() const noexcept {
        return nmi_pending_;
    }

    void clear_nmi_pending() noexcept {
        nmi_pending_ = false;
    }

    [[nodiscard]] std::uint8_t ctrl() const noexcept {
        return ctrl_;
    }

    [[nodiscard]] std::uint8_t mask() const noexcept {
        return mask_;
    }

    [[nodiscard]] std::uint8_t status() const noexcept {
        return status_;
    }

    [[nodiscard]] std::uint8_t oam_addr() const noexcept {
        return oam_addr_;
    }

    [[nodiscard]] std::uint16_t vram_address() const noexcept {
        return v_;
    }

    [[nodiscard]] std::uint8_t fine_x() const noexcept {
        return x_;
    }

    [[nodiscard]] bool write_latch() const noexcept {
        return w_;
    }

    [[nodiscard]] const std::array<std::uint8_t, kOamBytes>& oam() const noexcept {
        return oam_;
    }

    [[nodiscard]] std::uint8_t ppu_read(std::uint16_t address) const noexcept {
        return read_ppu_memory(address);
    }

    [[nodiscard]] std::int16_t scanline() const noexcept {
        return scanline_;
    }

    [[nodiscard]] std::uint16_t dot() const noexcept {
        return dot_;
    }

    [[nodiscard]] std::uint64_t frame() const noexcept {
        return frame_;
    }

    [[nodiscard]] RgbFrame render_rgb_frame() const noexcept {
        RgbFrame output{};
        std::array<bool, kScreenWidth * kScreenHeight> background_opaque{};
        const auto backdrop = palette_rgb(read_palette_entry(0));
        for (std::size_t i = 0; i < kScreenWidth * kScreenHeight; ++i) {
            output[i * 3] = backdrop[0];
            output[i * 3 + 1] = backdrop[1];
            output[i * 3 + 2] = backdrop[2];
        }

        if ((mask_ & 0x08) != 0) {
            render_background(output, background_opaque);
        }
        if ((mask_ & 0x10) != 0) {
            render_sprites(output, background_opaque);
        }
        return output;
    }

    void load_render_state(const RenderState& state) noexcept {
        ctrl_ = state.ctrl;
        mask_ = state.mask;
        status_ = state.status;
        scroll_x_ = state.scroll_x;
        scroll_y_ = state.scroll_y;
        if (state.palette_ram.size() >= palette_ram_.size()) {
            for (std::size_t i = 0; i < palette_ram_.size(); ++i) {
                palette_ram_[i] = state.palette_ram[i];
            }
        }
        if (state.oam.size() >= oam_.size()) {
            for (std::size_t i = 0; i < oam_.size(); ++i) {
                oam_[i] = state.oam[i];
            }
        }
        if (state.nametable_ram.size() >= 2 * 1024) {
            const auto limit = state.nametable_ram.size() < nametable_ram_.size()
                                   ? state.nametable_ram.size()
                                   : nametable_ram_.size();
            for (std::size_t i = 0; i < limit; ++i) {
                nametable_ram_[i] = state.nametable_ram[i];
            }
        }
    }

private:
    using Rgb = std::array<std::uint8_t, 3>;

    static constexpr std::array<Rgb, 64> kNesPalette = {{
        Rgb{0x54, 0x54, 0x54}, Rgb{0x00, 0x1e, 0x74}, Rgb{0x08, 0x10, 0x90}, Rgb{0x30, 0x00, 0x88},
        Rgb{0x44, 0x00, 0x64}, Rgb{0x5c, 0x00, 0x30}, Rgb{0x54, 0x04, 0x00}, Rgb{0x3c, 0x18, 0x00},
        Rgb{0x20, 0x2a, 0x00}, Rgb{0x08, 0x3a, 0x00}, Rgb{0x00, 0x40, 0x00}, Rgb{0x00, 0x3c, 0x00},
        Rgb{0x00, 0x32, 0x3c}, Rgb{0x00, 0x00, 0x00}, Rgb{0x00, 0x00, 0x00}, Rgb{0x00, 0x00, 0x00},
        Rgb{0x98, 0x96, 0x98}, Rgb{0x08, 0x4c, 0xc4}, Rgb{0x30, 0x32, 0xec}, Rgb{0x5c, 0x1e, 0xe4},
        Rgb{0x88, 0x14, 0xb0}, Rgb{0xa0, 0x14, 0x64}, Rgb{0x98, 0x22, 0x20}, Rgb{0x78, 0x3c, 0x00},
        Rgb{0x54, 0x5a, 0x00}, Rgb{0x28, 0x72, 0x00}, Rgb{0x08, 0x7c, 0x00}, Rgb{0x00, 0x76, 0x28},
        Rgb{0x00, 0x66, 0x78}, Rgb{0x00, 0x00, 0x00}, Rgb{0x00, 0x00, 0x00}, Rgb{0x00, 0x00, 0x00},
        Rgb{0xec, 0xee, 0xec}, Rgb{0x4c, 0x9a, 0xec}, Rgb{0x78, 0x7c, 0xec}, Rgb{0xb0, 0x62, 0xec},
        Rgb{0xe4, 0x54, 0xec}, Rgb{0xec, 0x58, 0xb4}, Rgb{0xec, 0x6a, 0x64}, Rgb{0xd4, 0x88, 0x20},
        Rgb{0xa0, 0xaa, 0x00}, Rgb{0x74, 0xc4, 0x00}, Rgb{0x4c, 0xd0, 0x20}, Rgb{0x38, 0xcc, 0x6c},
        Rgb{0x38, 0xb4, 0xcc}, Rgb{0x3c, 0x3c, 0x3c}, Rgb{0x00, 0x00, 0x00}, Rgb{0x00, 0x00, 0x00},
        Rgb{0xec, 0xee, 0xec}, Rgb{0xa8, 0xcc, 0xec}, Rgb{0xbc, 0xbc, 0xec}, Rgb{0xd4, 0xb2, 0xec},
        Rgb{0xec, 0xae, 0xec}, Rgb{0xec, 0xae, 0xd4}, Rgb{0xec, 0xb4, 0xb0}, Rgb{0xe4, 0xc4, 0x90},
        Rgb{0xcc, 0xd2, 0x78}, Rgb{0xb4, 0xde, 0x78}, Rgb{0xa8, 0xe2, 0x90}, Rgb{0x98, 0xe2, 0xb4},
        Rgb{0xa0, 0xd6, 0xe4}, Rgb{0xa0, 0xa2, 0xa0}, Rgb{0x00, 0x00, 0x00}, Rgb{0x00, 0x00, 0x00},
    }};

    [[nodiscard]] static std::uint16_t mirror_ppu_address(std::uint16_t address) noexcept {
        return static_cast<std::uint16_t>(address & 0x3FFF);
    }

    [[nodiscard]] static std::uint16_t mirror_palette_address(std::uint16_t address) noexcept {
        auto index = static_cast<std::uint16_t>(address & 0x001F);
        if (index == 0x10 || index == 0x14 || index == 0x18 || index == 0x1C) {
            index = static_cast<std::uint16_t>(index - 0x10);
        }
        return index;
    }

    [[nodiscard]] std::uint16_t mirror_nametable_address(std::uint16_t address) const noexcept {
        const auto index = static_cast<std::uint16_t>((address - 0x2000) & 0x0FFF);
        if (nametable_arrangement_ == NametableArrangement::FourScreen) {
            return index;
        }

        if (nametable_arrangement_ == NametableArrangement::Vertical) {
            return static_cast<std::uint16_t>(index & 0x07FF);
        }

        return static_cast<std::uint16_t>((index & 0x03FF) | ((index & 0x0800) >> 1));
    }

    [[nodiscard]] std::uint8_t read_ppu_memory(std::uint16_t address) const noexcept {
        address = mirror_ppu_address(address);
        if (address < 0x2000) {
            if (!chr_rom_.empty()) {
                return chr_rom_[address % chr_rom_.size()];
            }
            return chr_ram_[address & 0x1FFF];
        }
        if (address < 0x3F00) {
            if (address >= 0x3000) {
                address = static_cast<std::uint16_t>(address - 0x1000);
            }
            return nametable_ram_[mirror_nametable_address(address)];
        }
        return palette_ram_[mirror_palette_address(address)];
    }

    void write_ppu_memory(std::uint16_t address, std::uint8_t value) noexcept {
        address = mirror_ppu_address(address);
        if (address < 0x2000) {
            if (chr_rom_.empty()) {
                chr_ram_[address & 0x1FFF] = value;
            }
            return;
        }
        if (address < 0x3F00) {
            if (address >= 0x3000) {
                address = static_cast<std::uint16_t>(address - 0x1000);
            }
            nametable_ram_[mirror_nametable_address(address)] = value;
            return;
        }
        palette_ram_[mirror_palette_address(address)] = value;
    }

    [[nodiscard]] std::uint8_t read_status() noexcept {
        const auto value = static_cast<std::uint8_t>((status_ & 0xE0) | (open_bus_ & 0x1F));
        status_ &= 0x7F;
        nmi_pending_ = false;
        w_ = false;
        open_bus_ = value;
        return value;
    }

    [[nodiscard]] std::uint8_t read_data() noexcept {
        const auto address = mirror_ppu_address(v_);
        std::uint8_t value = 0;
        if (address >= 0x3F00) {
            value = read_ppu_memory(address);
            read_buffer_ = read_ppu_memory(static_cast<std::uint16_t>(address - 0x1000));
        } else {
            value = read_buffer_;
            read_buffer_ = read_ppu_memory(address);
        }
        increment_vram_address();
        open_bus_ = value;
        return value;
    }

    void write_data(std::uint8_t value) noexcept {
        write_ppu_memory(v_, value);
        increment_vram_address();
    }

    void write_scroll(std::uint8_t value) noexcept {
        if (!w_) {
            x_ = static_cast<std::uint8_t>(value & 0x07);
            t_ = static_cast<std::uint16_t>((t_ & 0xFFE0) | (value >> 3));
            scroll_x_ = value;
            w_ = true;
        } else {
            t_ = static_cast<std::uint16_t>((t_ & 0x8FFF) | ((value & 0x07) << 12));
            t_ = static_cast<std::uint16_t>((t_ & 0xFC1F) | ((value & 0xF8) << 2));
            scroll_y_ = value;
            w_ = false;
        }
    }

    void write_address(std::uint8_t value) noexcept {
        if (!w_) {
            t_ = static_cast<std::uint16_t>((t_ & 0x00FF) | ((value & 0x3F) << 8));
            w_ = true;
        } else {
            t_ = static_cast<std::uint16_t>((t_ & 0x7F00) | value);
            v_ = t_;
            w_ = false;
        }
    }

    void increment_vram_address() noexcept {
        v_ = static_cast<std::uint16_t>(v_ + ((ctrl_ & 0x04) != 0 ? 32 : 1));
    }

    [[nodiscard]] std::uint8_t read_palette_entry(std::uint16_t index) const noexcept {
        auto value = read_ppu_memory(static_cast<std::uint16_t>(0x3F00 + index));
        if ((mask_ & 0x01) != 0) {
            value &= 0x30;
        }
        return static_cast<std::uint8_t>(value & 0x3F);
    }

    [[nodiscard]] Rgb palette_rgb(std::uint8_t index) const noexcept {
        return kNesPalette[index & 0x3F];
    }

    void write_rgb(RgbFrame& output, std::size_t pixel, const Rgb& rgb) const noexcept {
        output[pixel * 3] = rgb[0];
        output[pixel * 3 + 1] = rgb[1];
        output[pixel * 3 + 2] = rgb[2];
    }

    [[nodiscard]] std::uint8_t pattern_pixel(std::uint16_t tile_base,
                                             std::uint8_t fine_x,
                                             std::uint8_t fine_y) const noexcept {
        const auto low = read_ppu_memory(static_cast<std::uint16_t>(tile_base + fine_y));
        const auto high = read_ppu_memory(static_cast<std::uint16_t>(tile_base + fine_y + 8));
        const auto bit = static_cast<std::uint8_t>(7 - fine_x);
        return static_cast<std::uint8_t>(((low >> bit) & 0x01) | (((high >> bit) & 0x01) << 1));
    }

    void render_background(RgbFrame& output,
                           std::array<bool, kScreenWidth * kScreenHeight>& opaque) const noexcept {
        const bool show_left = (mask_ & 0x02) != 0;
        const std::uint16_t pattern_base = (ctrl_ & 0x10) != 0 ? 0x1000 : 0x0000;
        const std::uint8_t base_nametable = static_cast<std::uint8_t>(ctrl_ & 0x03);

        for (std::size_t y = 0; y < kScreenHeight; ++y) {
            const auto world_y = static_cast<unsigned>(scroll_y_) + static_cast<unsigned>(y);
            const auto coarse_y = static_cast<std::uint8_t>((world_y % 240) / 8);
            const auto fine_y = static_cast<std::uint8_t>(world_y & 0x07);
            const auto nt_y = static_cast<std::uint8_t>(((base_nametable >> 1) + (world_y / 240)) & 0x01);

            for (std::size_t x = 0; x < kScreenWidth; ++x) {
                if (!show_left && x < 8) {
                    continue;
                }

                const auto world_x = static_cast<unsigned>(scroll_x_) + static_cast<unsigned>(x);
                const auto coarse_x = static_cast<std::uint8_t>((world_x & 0xFF) / 8);
                const auto fine_x = static_cast<std::uint8_t>(world_x & 0x07);
                const auto nt_x = static_cast<std::uint8_t>(((base_nametable & 0x01) + (world_x / 256)) & 0x01);
                const auto nametable = static_cast<std::uint8_t>(nt_x | (nt_y << 1));
                const auto nametable_base = static_cast<std::uint16_t>(0x2000 + nametable * 0x0400);
                const auto tile = read_ppu_memory(static_cast<std::uint16_t>(
                    nametable_base + coarse_y * 32 + coarse_x));
                const auto color = pattern_pixel(
                    static_cast<std::uint16_t>(pattern_base + static_cast<std::uint16_t>(tile) * 16),
                    fine_x,
                    fine_y);
                if (color == 0) {
                    continue;
                }

                const auto attribute = read_ppu_memory(static_cast<std::uint16_t>(
                    nametable_base + 0x03C0 + (coarse_y / 4) * 8 + (coarse_x / 4)));
                const auto shift = static_cast<std::uint8_t>(((coarse_y & 0x02) << 1) |
                                                             (coarse_x & 0x02));
                const auto palette = static_cast<std::uint8_t>((attribute >> shift) & 0x03);
                const auto palette_value =
                    read_palette_entry(static_cast<std::uint16_t>(palette * 4 + color));
                const auto pixel = y * kScreenWidth + x;
                opaque[pixel] = true;
                write_rgb(output, pixel, palette_rgb(palette_value));
            }
        }
    }

    [[nodiscard]] std::uint8_t sprite_pattern_pixel(std::uint8_t tile,
                                                    std::uint8_t attributes,
                                                    std::uint8_t pixel_x,
                                                    std::uint8_t pixel_y) const noexcept {
        if ((attributes & 0x40) != 0) {
            pixel_x = static_cast<std::uint8_t>(7 - pixel_x);
        }

        const bool tall_sprite = (ctrl_ & 0x20) != 0;
        if (tall_sprite) {
            if ((attributes & 0x80) != 0) {
                pixel_y = static_cast<std::uint8_t>(15 - pixel_y);
            }
            const auto pattern_base = static_cast<std::uint16_t>((tile & 0x01) ? 0x1000 : 0x0000);
            const auto tile_number = static_cast<std::uint8_t>((tile & 0xFE) + (pixel_y / 8));
            return pattern_pixel(static_cast<std::uint16_t>(pattern_base + tile_number * 16),
                                 pixel_x,
                                 static_cast<std::uint8_t>(pixel_y & 0x07));
        }

        if ((attributes & 0x80) != 0) {
            pixel_y = static_cast<std::uint8_t>(7 - pixel_y);
        }
        const auto pattern_base = (ctrl_ & 0x08) != 0 ? 0x1000 : 0x0000;
        return pattern_pixel(static_cast<std::uint16_t>(pattern_base + tile * 16), pixel_x, pixel_y);
    }

    void render_sprites(RgbFrame& output,
                        const std::array<bool, kScreenWidth * kScreenHeight>& background_opaque) const noexcept {
        const bool show_left = (mask_ & 0x04) != 0;
        const auto sprite_height = static_cast<std::uint8_t>((ctrl_ & 0x20) != 0 ? 16 : 8);

        for (int sprite = 63; sprite >= 0; --sprite) {
            const auto base = static_cast<std::size_t>(sprite * 4);
            const auto top = static_cast<int>(oam_[base]) + 1;
            const auto tile = oam_[base + 1];
            const auto attributes = oam_[base + 2];
            const auto left = static_cast<int>(oam_[base + 3]);
            const auto palette = static_cast<std::uint8_t>(attributes & 0x03);
            const bool behind_background = (attributes & 0x20) != 0;

            for (int sy = 0; sy < sprite_height; ++sy) {
                const auto y = top + sy;
                if (y < 0 || y >= static_cast<int>(kScreenHeight)) {
                    continue;
                }

                for (int sx = 0; sx < 8; ++sx) {
                    const auto x = left + sx;
                    if (x < 0 || x >= static_cast<int>(kScreenWidth) || (!show_left && x < 8)) {
                        continue;
                    }

                    const auto pixel = static_cast<std::size_t>(y) * kScreenWidth +
                                       static_cast<std::size_t>(x);
                    if (behind_background && background_opaque[pixel]) {
                        continue;
                    }

                    const auto color = sprite_pattern_pixel(tile,
                                                            attributes,
                                                            static_cast<std::uint8_t>(sx),
                                                            static_cast<std::uint8_t>(sy));
                    if (color == 0) {
                        continue;
                    }

                    const auto palette_value =
                        read_palette_entry(static_cast<std::uint16_t>(0x10 + palette * 4 + color));
                    write_rgb(output, pixel, palette_rgb(palette_value));
                }
            }
        }
    }

    std::uint8_t ctrl_ = 0;
    std::uint8_t mask_ = 0;
    std::uint8_t status_ = 0;
    std::uint8_t oam_addr_ = 0;
    std::uint8_t open_bus_ = 0;
    std::uint8_t read_buffer_ = 0;
    std::uint16_t v_ = 0;
    std::uint16_t t_ = 0;
    std::uint8_t x_ = 0;
    bool w_ = false;
    bool nmi_pending_ = false;
    std::uint8_t scroll_x_ = 0;
    std::uint8_t scroll_y_ = 0;
    std::int16_t scanline_ = 0;
    std::uint16_t dot_ = 0;
    std::uint64_t frame_ = 0;
    NametableArrangement nametable_arrangement_ = NametableArrangement::Vertical;
    std::span<const std::uint8_t> chr_rom_{};
    std::array<std::uint8_t, 8 * 1024> chr_ram_{};
    std::array<std::uint8_t, 4 * 1024> nametable_ram_{};
    std::array<std::uint8_t, 32> palette_ram_{};
    std::array<std::uint8_t, kOamBytes> oam_{};
};

}  // namespace nesle
