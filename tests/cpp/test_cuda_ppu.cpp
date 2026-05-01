#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "nesle/cuda/batch_ppu.cuh"
#include "nesle/ppu.hpp"

namespace {

nesle::cuda::BatchBuffers make_buffers(std::vector<std::uint8_t>& ctrl,
                                        std::vector<std::uint8_t>& mask,
                                        std::vector<std::uint8_t>& status,
                                        std::vector<std::uint8_t>& nmi_pending,
                                        std::vector<std::int16_t>& scanline,
                                        std::vector<std::uint16_t>& dot,
                                        std::vector<std::uint64_t>& frame) {
    nesle::cuda::BatchBuffers buffers{};
    buffers.ppu.ctrl = ctrl.data();
    buffers.ppu.mask = mask.data();
    buffers.ppu.status = status.data();
    buffers.ppu.nmi_pending = nmi_pending.data();
    buffers.ppu.scanline = scanline.data();
    buffers.ppu.dot = dot.data();
    buffers.ppu.frame = frame.data();
    return buffers;
}

void assert_matches_ppu(const nesle::cuda::BatchBuffers& buffers,
                        std::uint32_t env,
                        const nesle::Ppu& ppu) {
    assert(buffers.ppu.status[env] == ppu.status());
    assert(buffers.ppu.nmi_pending[env] == static_cast<std::uint8_t>(ppu.nmi_pending()));
    assert(buffers.ppu.scanline[env] == ppu.scanline());
    assert(buffers.ppu.dot[env] == ppu.dot());
    assert(buffers.ppu.frame[env] == ppu.frame());
}

}  // namespace

int main() {
    constexpr std::size_t kNumEnvs = 3;
    std::vector<std::uint8_t> ctrl(kNumEnvs, 0);
    std::vector<std::uint8_t> mask(kNumEnvs, 0);
    std::vector<std::uint8_t> status(kNumEnvs, 0);
    std::vector<std::uint8_t> nmi_pending(kNumEnvs, 0);
    std::vector<std::int16_t> scanline(kNumEnvs, 0);
    std::vector<std::uint16_t> dot(kNumEnvs, 0);
    std::vector<std::uint64_t> frame(kNumEnvs, 0);
    auto buffers = make_buffers(ctrl, mask, status, nmi_pending, scanline, dot, frame);

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

    return 0;
}
