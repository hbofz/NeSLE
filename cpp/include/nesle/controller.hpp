#pragma once

#include <cstdint>

namespace nesle {

enum ControllerButton : std::uint8_t {
    ButtonA = 1u << 0,
    ButtonB = 1u << 1,
    ButtonSelect = 1u << 2,
    ButtonStart = 1u << 3,
    ButtonUp = 1u << 4,
    ButtonDown = 1u << 5,
    ButtonLeft = 1u << 6,
    ButtonRight = 1u << 7,
};

class StandardController {
public:
    void set_buttons(std::uint8_t mask) noexcept {
        buttons_ = mask;
        if (strobe_) {
            latch();
        }
    }

    void write_strobe(std::uint8_t value) noexcept {
        const bool next_strobe = (value & 0x01) != 0;
        if (next_strobe || strobe_) {
            latch();
        }
        strobe_ = next_strobe;
    }

    [[nodiscard]] std::uint8_t read() noexcept {
        if (strobe_) {
            return static_cast<std::uint8_t>(0x40 | (buttons_ & 0x01));
        }

        std::uint8_t bit = 1;
        if (shift_count_ < 8) {
            bit = static_cast<std::uint8_t>(shift_register_ & 0x01);
            shift_register_ = static_cast<std::uint8_t>(shift_register_ >> 1);
            ++shift_count_;
        }
        return static_cast<std::uint8_t>(0x40 | bit);
    }

    [[nodiscard]] bool strobe() const noexcept {
        return strobe_;
    }

    [[nodiscard]] std::uint8_t buttons() const noexcept {
        return buttons_;
    }

private:
    void latch() noexcept {
        shift_register_ = buttons_;
        shift_count_ = 0;
    }

    std::uint8_t buttons_ = 0;
    std::uint8_t shift_register_ = 0;
    std::uint8_t shift_count_ = 8;
    bool strobe_ = false;
};

}  // namespace nesle
