#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace nesle {

class Ppu {
public:
    static constexpr std::size_t kOamBytes = 256;
    static constexpr std::size_t kPpuAddressSpaceBytes = 16 * 1024;

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
        return vram_[mirror_ppu_address(address)];
    }

private:
    [[nodiscard]] static std::uint16_t mirror_ppu_address(std::uint16_t address) noexcept {
        return static_cast<std::uint16_t>(address & 0x3FFF);
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
            value = vram_[address];
            read_buffer_ = vram_[mirror_ppu_address(static_cast<std::uint16_t>(address - 0x1000))];
        } else {
            value = read_buffer_;
            read_buffer_ = vram_[address];
        }
        increment_vram_address();
        open_bus_ = value;
        return value;
    }

    void write_data(std::uint8_t value) noexcept {
        vram_[mirror_ppu_address(v_)] = value;
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
    std::array<std::uint8_t, kOamBytes> oam_{};
    std::array<std::uint8_t, kPpuAddressSpaceBytes> vram_{};
};

}  // namespace nesle
