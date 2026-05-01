#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

#include "nesle/cuda/batch_render.cuh"
#include "nesle/ppu.hpp"

namespace {

using Frame = std::array<std::uint8_t, nesle::cuda::kFrameWidth * nesle::cuda::kFrameHeight *
                                           nesle::cuda::kRgbChannels>;

nesle::cuda::BatchBuffers make_buffers(std::vector<std::uint8_t>& chr_rom,
                                       std::array<std::uint8_t, 1>& ctrl,
                                       std::array<std::uint8_t, 1>& mask,
                                       std::array<std::uint8_t, 1>& status,
                                       std::array<std::uint8_t, 1>& oam_addr,
                                       std::array<std::uint8_t, 1>& nmi_pending,
                                       std::array<std::int16_t, 1>& scanline,
                                       std::array<std::uint16_t, 1>& dot,
                                       std::array<std::uint64_t, 1>& frame_count,
                                       std::vector<std::uint8_t>& nametable,
                                       std::vector<std::uint8_t>& palette,
                                       std::vector<std::uint8_t>& oam,
                                       Frame& frame) {
    nesle::cuda::BatchBuffers buffers{};
    buffers.ppu.ctrl = ctrl.data();
    buffers.ppu.mask = mask.data();
    buffers.ppu.status = status.data();
    buffers.ppu.oam_addr = oam_addr.data();
    buffers.ppu.nmi_pending = nmi_pending.data();
    buffers.ppu.scanline = scanline.data();
    buffers.ppu.dot = dot.data();
    buffers.ppu.frame = frame_count.data();
    buffers.ppu.nametable_ram = nametable.data();
    buffers.ppu.palette_ram = palette.data();
    buffers.ppu.oam = oam.data();
    buffers.cart.chr_rom = chr_rom.data();
    buffers.cart.chr_rom_size = static_cast<std::uint32_t>(chr_rom.size());
    buffers.cart.nametable_arrangement = nesle::cuda::kNametableVertical;
    buffers.frames_rgb = frame.data();
    return buffers;
}

void assert_frame_matches_cpu(const std::vector<std::uint8_t>& chr_rom,
                              const std::vector<std::uint8_t>& nametable,
                              const std::vector<std::uint8_t>& palette,
                              const std::vector<std::uint8_t>& oam,
                              const Frame& batch_frame,
                              std::uint8_t ctrl,
                              std::uint8_t mask) {
    nesle::Ppu ppu;
    ppu.configure_cartridge(chr_rom, nesle::NametableArrangement::Vertical);
    nesle::Ppu::RenderState state;
    state.ctrl = ctrl;
    state.mask = mask;
    state.palette_ram = palette;
    state.oam = oam;
    state.nametable_ram = nametable;
    ppu.load_render_state(state);
    const auto cpu_frame = ppu.render_rgb_frame();
    for (std::size_t i = 0; i < cpu_frame.size(); ++i) {
        assert(batch_frame[i] == cpu_frame[i]);
    }
}

}  // namespace

int main() {
    {
        std::vector<std::uint8_t> chr_rom(8 * 1024, 0);
        chr_rom[0x0010] = 0x80;
        std::vector<std::uint8_t> nametable(nesle::cuda::kNametableRamBytes, 0);
        std::vector<std::uint8_t> palette(nesle::cuda::kPaletteRamBytes, 0);
        std::vector<std::uint8_t> oam(nesle::cuda::kOamBytes, 0);
        Frame frame{};
        std::array<std::uint8_t, 1> ctrl{0};
        std::array<std::uint8_t, 1> mask{0x0A};
        std::array<std::uint8_t, 1> status{0};
        std::array<std::uint8_t, 1> oam_addr{0};
        std::array<std::uint8_t, 1> nmi_pending{0};
        std::array<std::int16_t, 1> scanline{0};
        std::array<std::uint16_t, 1> dot{0};
        std::array<std::uint64_t, 1> frame_count{0};
        nametable[0] = 0x01;
        palette[1] = 0x21;

        auto buffers = make_buffers(chr_rom,
                                    ctrl,
                                    mask,
                                    status,
                                    oam_addr,
                                    nmi_pending,
                                    scanline,
                                    dot,
                                    frame_count,
                                    nametable,
                                    palette,
                                    oam,
                                    frame);
        nesle::cuda::render_batch_rgb_frame_env(buffers, 0);
        assert_frame_matches_cpu(chr_rom, nametable, palette, oam, frame, ctrl[0], mask[0]);
    }

    {
        std::vector<std::uint8_t> chr_rom(8 * 1024, 0);
        chr_rom[0x0020] = 0x80;
        std::vector<std::uint8_t> nametable(nesle::cuda::kNametableRamBytes, 0);
        std::vector<std::uint8_t> palette(nesle::cuda::kPaletteRamBytes, 0);
        std::vector<std::uint8_t> oam(nesle::cuda::kOamBytes, 0);
        Frame frame{};
        std::array<std::uint8_t, 1> ctrl{0};
        std::array<std::uint8_t, 1> mask{0x14};
        std::array<std::uint8_t, 1> status{0};
        std::array<std::uint8_t, 1> oam_addr{0};
        std::array<std::uint8_t, 1> nmi_pending{0};
        std::array<std::int16_t, 1> scanline{0};
        std::array<std::uint16_t, 1> dot{0};
        std::array<std::uint64_t, 1> frame_count{0};
        palette[0x11] = 0x16;
        oam[0] = 9;
        oam[1] = 2;
        oam[2] = 0;
        oam[3] = 12;

        auto buffers = make_buffers(chr_rom,
                                    ctrl,
                                    mask,
                                    status,
                                    oam_addr,
                                    nmi_pending,
                                    scanline,
                                    dot,
                                    frame_count,
                                    nametable,
                                    palette,
                                    oam,
                                    frame);
        nesle::cuda::render_batch_rgb_frame_env(buffers, 0);
        assert_frame_matches_cpu(chr_rom, nametable, palette, oam, frame, ctrl[0], mask[0]);
    }
}
