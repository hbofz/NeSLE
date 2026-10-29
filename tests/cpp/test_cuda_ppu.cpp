#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "nesle/cuda/batch_bus.cuh"
#include "nesle/cuda/batch_ppu.cuh"
#include "nesle/ppu.hpp"

namespace {

nesle::cuda::BatchBuffers make_buffers(std::vector<std::uint8_t>& ctrl,
                                        std::vector<std::uint8_t>& mask,
                                        std::vector<std::uint8_t>& status,
                                        std::vector<std::uint8_t>& nmi_pending,
                                        std::vector<std::uint32_t>& frame_dot,
                                        std::vector<std::uint64_t>& frame) {
    nesle::cuda::BatchBuffers buffers{};
    buffers.ppu.ctrl = ctrl.data();
    buffers.ppu.mask = mask.data();
    buffers.ppu.status = status.data();
    buffers.ppu.nmi_pending = nmi_pending.data();
    buffers.ppu.frame_dot = frame_dot.data();
    buffers.ppu.frame = frame.data();
    return buffers;
}

void assert_matches_ppu(const nesle::cuda::BatchBuffers& buffers,
                        std::uint32_t env,
                        const nesle::Ppu& ppu) {
    assert(buffers.ppu.status[env] == ppu.status());
    assert(buffers.ppu.nmi_pending[env] == static_cast<std::uint8_t>(ppu.nmi_pending()));
    assert(buffers.ppu.frame_dot[env] ==
           static_cast<std::uint32_t>(ppu.scanline()) *
                   static_cast<std::uint32_t>(nesle::cuda::kPpuDotsPerScanline) +
               static_cast<std::uint32_t>(ppu.dot()));
    assert(buffers.ppu.frame[env] == ppu.frame());
}

}  // namespace

int main() {
    constexpr std::size_t kNumEnvs = 3;
    std::vector<std::uint8_t> ctrl(kNumEnvs, 0);
    std::vector<std::uint8_t> mask(kNumEnvs, 0);
    std::vector<std::uint8_t> status(kNumEnvs, 0);
    std::vector<std::uint8_t> nmi_pending(kNumEnvs, 0);
    std::vector<std::uint32_t> frame_dot(kNumEnvs, 0);
    std::vector<std::uint64_t> frame(kNumEnvs, 0);
    auto buffers = make_buffers(ctrl, mask, status, nmi_pending, frame_dot, frame);

    {
        nesle::Ppu ppu;
        ppu.write_register(0x00, 0x80);
        ctrl[0] = 0x80;

        const auto cycles =
            nesle::Ppu::kVblankStartScanline * nesle::Ppu::kDotsPerScanline +
            nesle::Ppu::kVblankFlagDot;
        const auto ppu_step = ppu.step(cycles);
        const auto batch_step = nesle::cuda::batch_ppu_step_env(buffers, 0, cycles);

        assert(batch_step.cycles == ppu_step.cycles);
        assert(batch_step.frames_completed == ppu_step.frames_completed);
        assert(batch_step.nmi_started == ppu_step.nmi_started);
        assert_matches_ppu(buffers, 0, ppu);
    }

    {
        nesle::Ppu ppu;
        ppu.write_register(0x00, 0x80);
        ctrl[1] = 0x80;
        const auto cycles_to_frame =
            nesle::Ppu::kScanlinesPerFrame * nesle::Ppu::kDotsPerScanline;
        const auto ppu_step = ppu.step(cycles_to_frame);
        const auto batch_step = nesle::cuda::batch_ppu_step_env(buffers, 1, cycles_to_frame);

        assert(batch_step.frames_completed == ppu_step.frames_completed);
        assert(batch_step.nmi_started == ppu_step.nmi_started);
        assert(batch_step.frames_completed == 1);
        assert_matches_ppu(buffers, 1, ppu);
    }

    {
        nesle::Ppu ppu;
        ppu.write_register(0x01, 0x18);
        mask[2] = 0x18;
        const auto hit_cycles =
            nesle::Ppu::kCoarseSpriteZeroHitScanline * nesle::Ppu::kDotsPerScanline +
            nesle::Ppu::kCoarseSpriteZeroHitDot;
        (void)ppu.step(hit_cycles);
        (void)nesle::cuda::batch_ppu_step_env(buffers, 2, hit_cycles);
        assert_matches_ppu(buffers, 2, ppu);
        assert((status[2] & 0x40) != 0);

        const auto clear_cycles =
            (nesle::Ppu::kPreRenderScanline - nesle::Ppu::kCoarseSpriteZeroHitScanline) *
            nesle::Ppu::kDotsPerScanline;
        (void)ppu.step(clear_cycles);
        (void)nesle::cuda::batch_ppu_step_env(buffers, 2, clear_cycles);
        assert_matches_ppu(buffers, 2, ppu);
        assert((status[2] & 0x40) == 0);
    }

    assert(frame[0] == 0);
    assert(frame[1] == 1);
    assert(frame[2] == 0);

    {
        // SMB's NMI, through the bus: its VRAM update writes $2000 with the
        // playfield nametable, then $2006 = $3F00, $0000. On hardware that
        // leaves t, and so the status bar, on nametable 0; the frame-start
        // latch must follow t, not PPUCTRL.
        std::vector<std::uint8_t> c(1, 0), m(1, 0x1E), st(1, 0), nmi(1, 0), w(1, 0), x(1, 0), open_bus(1, 0);
        std::vector<std::uint8_t> sx(1, 0), sy(1, 0), lat_sx(1, 0xFF), lat_sy(1, 0xFF), lat_ctrl(1, 0xFF);
        std::vector<std::uint16_t> t(1, 0), v(1, 0);
        std::vector<std::uint32_t> dot(1, 0);
        std::vector<std::uint64_t> fr(1, 0);
        auto smb = make_buffers(c, m, st, nmi, dot, fr);
        smb.ppu.w = w.data();
        smb.ppu.open_bus = open_bus.data();
        smb.ppu.x = x.data();
        smb.ppu.t = t.data();
        smb.ppu.v = v.data();
        smb.ppu.scroll_x = sx.data();
        smb.ppu.scroll_y = sy.data();
        smb.ppu.lat_scroll_x = lat_sx.data();
        smb.ppu.lat_scroll_y = lat_sy.data();
        smb.ppu.lat_ctrl = lat_ctrl.data();
        auto write = [&](std::uint16_t address, std::uint8_t value) {
            nesle::cuda::batch_cpu_write(smb, 0, address, value);
        };
        constexpr std::uint8_t kMirror = 0x11;  // SMB's $2000 mirror in its NMI: playfield on nametable 1

        write(0x2000, kMirror & 0x7E);  // NMI entry: status bar on nametable 0
        write(0x2005, 0);
        write(0x2005, 0);
        write(0x2006, 0x20);  // a VRAM buffer entry
        write(0x2006, 0x65);
        write(0x2000, kMirror | 0x04);
        write(0x2006, 0x3F);
        write(0x2006, 0x00);
        write(0x2006, 0x00);
        write(0x2006, 0x00);
        write(0x2005, 0);  // InitScroll
        write(0x2005, 0);
        const auto pre_render =
            nesle::Ppu::kPreRenderScanline * nesle::Ppu::kDotsPerScanline + nesle::Ppu::kVblankFlagDot;
        (void)nesle::cuda::batch_ppu_step_env(smb, 0, pre_render);
        assert(c[0] == 0x15);         // PPUCTRL still selects the playfield nametable
        assert(lat_ctrl[0] == 0x14);  // the status bar is latched from nametable 0
        assert(lat_sx[0] == 0 && lat_sy[0] == 0);

        write(0x2000, kMirror | 0x80);  // NMI exit, no $2006 afterwards: t keeps nametable 1
        (void)nesle::cuda::batch_ppu_step_env(
            smb, 0, nesle::Ppu::kScanlinesPerFrame * nesle::Ppu::kDotsPerScanline);
        assert(lat_ctrl[0] == 0x91);
    }

    return 0;
}
