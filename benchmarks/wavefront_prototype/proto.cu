// Tier-3 prototype (docs/gpu-scaling.md): does opcode-binned "wavefront"
// dispatch beat the status-quo thread-per-env interpreter for the 6502
// workload?
//
// Reuses the REAL production interpreter (cpp/include/nesle/cpu.hpp,
// table-driven core) against a minimal synthetic bus:
//   - 2 KB RAM per instance in global memory, mirrored across 0x0000-0x7FFF
//     (matches the NES layout: zero page / stack / RAM all live here)
//   - one SHARED 32 KB read-only code image at 0x8000-0xFFFF (matches NROM:
//     every real env runs the same SMB PRG; writes >= 0x8000 are dropped,
//     exactly like writes into cartridge ROM)
//   - no PPU, no APU, no controllers: pure CPU dispatch. Real emulation
//     interleaves PPU catch-up after every instruction, so any win measured
//     here is an UPPER BOUND on the real payoff.
//
// The code image is synthesized by sampling the MEASURED opcode distribution
// from real SMB emulation (opcode_hist.inc, gathered via CudaBatch
// .step_profile on the W1-1 snapshot). Operands are randomized within
// addressing-mode constraints. Exclusions (documented in RESULTS.md):
//   - BRK / RTI: never sampled meaningfully in SMB steady state anyway
//   - JSR / RTS: unpaired sampling would pop garbage return addresses and
//     fly off to arbitrary PCs (~5.1% of the real stream)
//   - JMP ($nn) indirect: pointer contents are uncontrolled
//   - illegal/unofficial opcodes (the real core traps on them)
// Branches are patched to forward instruction boundaries within range so the
// PC never leaves the code region and never lands mid-instruction; the image
// ends with JMP back to 0x8000 so streams run forever.
//
// Variant A (status quo): one thread per instance, K x cpu::step().
// Variant B (wavefront):  per instruction, the warp iterates opcode groups —
//   ballot/shuffle binning (sm_61 has no __match_any_sync), each group of
//   lanes with an identical next-opcode executes step() together, so every
//   decode-table dispatch inside a group is uniform. Cheapest credible
//   binning per the plan: warp-local, no global compaction.
//
// Correlated mode: all instances start at the same PC with identical RAM ->
// lanes stay in lockstep forever (best case, CuLE's "right after reset").
// Decorrelated mode: random start boundary + random RAM + random registers
// per instance (worst case, fully drifted population).

#include "nesle/cpu.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace nc = nesle::cpu;

// Measured real-SMB opcode execution counts (see opcode_hist.json).
static const unsigned long long kMeasuredCounts[256] = {
#include "opcode_hist.inc"
};

constexpr int kRamSize = 0x800;          // 2 KB, mirrored below 0x8000
constexpr int kCodeSize = 0x8000;        // 32 KB at 0x8000-0xFFFF
constexpr std::uint16_t kCodeBase = 0x8000;
constexpr int kThreadsPerBlock = 128;    // same shape as the real kernel

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t err__ = (expr);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr,         \
                         __FILE__, __LINE__, cudaGetErrorString(err__));        \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Bus: identical semantics on host (validation) and device (benchmark).
// ---------------------------------------------------------------------------

struct DeviceBus {
    std::uint8_t* __restrict__ ram;        // per-instance 2 KB
    const std::uint8_t* __restrict__ code; // shared 32 KB image
    __device__ std::uint8_t read(std::uint16_t addr) const {
        return addr < 0x8000 ? ram[addr & (kRamSize - 1)] : code[addr & (kCodeSize - 1)];
    }
    __device__ void write(std::uint16_t addr, std::uint8_t value) {
        if (addr < 0x8000) {
            ram[addr & (kRamSize - 1)] = value;
        }
    }
};

struct HostBus {
    std::uint8_t* ram;
    const std::uint8_t* code;
    std::uint8_t read(std::uint16_t addr) const {
        return addr < 0x8000 ? ram[addr & (kRamSize - 1)] : code[addr & (kCodeSize - 1)];
    }
    void write(std::uint16_t addr, std::uint8_t value) {
        if (addr < 0x8000) {
            ram[addr & (kRamSize - 1)] = value;
        }
    }
};

// ---------------------------------------------------------------------------
// Synthetic code generation from the measured distribution.
// ---------------------------------------------------------------------------

static int operand_len(nc::AddrMode mode) {
    switch (mode) {
        case nc::AddrMode::Implied:
        case nc::AddrMode::Accumulator:
            return 0;
        case nc::AddrMode::Immediate:
        case nc::AddrMode::ZeroPage:
        case nc::AddrMode::ZeroPageX:
        case nc::AddrMode::ZeroPageY:
        case nc::AddrMode::IndirectX:
        case nc::AddrMode::IndirectY:
        case nc::AddrMode::Relative:
            return 1;
        default:  // Absolute, AbsoluteX, AbsoluteY, Indirect
            return 2;
    }
}

static bool op_excluded(std::uint8_t opcode) {
    const nc::DecodeEntry e = nc::detail::kDecodeTable.entries[opcode];
    if ((e.flags & nc::FlagIllegal) != 0) return true;
    if (e.op == nc::Op::BRK || e.op == nc::Op::JSR || e.op == nc::Op::RTS ||
        e.op == nc::Op::RTI) {
        return true;
    }
    if (e.mode == nc::AddrMode::Indirect) return true;  // JMP ($nn)
    return false;
}

struct Image {
    std::vector<std::uint8_t> code;   // kCodeSize bytes
    std::vector<int> boundaries;      // instruction start offsets, ascending
    std::vector<int> static_count;    // per-opcode emitted count (256)
    unsigned long long included_total = 0;
    unsigned long long excluded_total = 0;
};

static Image generate_image(std::uint64_t seed) {
    std::mt19937_64 rng(seed);

    // Build the sampling table over included opcodes.
    std::vector<std::uint8_t> ops;
    std::vector<unsigned long long> cum;
    Image img;
    img.static_count.assign(256, 0);
    for (int op = 0; op < 256; ++op) {
        const unsigned long long c = kMeasuredCounts[op];
        if (c == 0) continue;
        if (op_excluded(static_cast<std::uint8_t>(op))) {
            img.excluded_total += c;
            continue;
        }
        img.included_total += c;
        ops.push_back(static_cast<std::uint8_t>(op));
        cum.push_back(img.included_total);
    }

    auto sample_opcode = [&]() {
        std::uniform_int_distribution<unsigned long long> pick(0, img.included_total - 1);
        const auto r = pick(rng);
        const auto it = std::upper_bound(cum.begin(), cum.end(), r);
        return ops[static_cast<std::size_t>(it - cum.begin())];
    };
    auto rnd8 = [&]() { return static_cast<std::uint8_t>(rng() & 0xFF); };

    img.code.assign(kCodeSize, 0xEA);  // NOP filler (never reached)
    std::vector<std::size_t> branch_idx;  // boundary indices needing patch
    std::vector<std::size_t> jmp_idx;

    int pos = 0;
    const int limit = kCodeSize - 3;  // leave room for the wrap JMP
    while (true) {
        const std::uint8_t opcode = sample_opcode();
        const nc::DecodeEntry e = nc::detail::kDecodeTable.entries[opcode];
        const int len = 1 + operand_len(e.mode);
        if (pos + len > limit) break;

        img.boundaries.push_back(pos);
        img.code[static_cast<std::size_t>(pos)] = opcode;
        img.static_count[opcode] += 1;

        const bool touches_memory =
            (e.flags & (nc::FlagStore | nc::FlagRmw)) != 0;
        switch (e.mode) {
            case nc::AddrMode::Implied:
            case nc::AddrMode::Accumulator:
                break;
            case nc::AddrMode::Immediate:
            case nc::AddrMode::ZeroPage:
            case nc::AddrMode::ZeroPageX:
            case nc::AddrMode::ZeroPageY:
            case nc::AddrMode::IndirectX:
            case nc::AddrMode::IndirectY:
                // ZP operand / immediate / ZP pointer: any byte is safe.
                // Indirect stores may resolve to >= 0x8000; the bus drops
                // those writes exactly like real NROM hardware does.
                img.code[static_cast<std::size_t>(pos) + 1] = rnd8();
                break;
            case nc::AddrMode::Absolute:
            case nc::AddrMode::AbsoluteX:
            case nc::AddrMode::AbsoluteY: {
                // Stores/RMW aim at RAM so they do real work (base + index
                // stays < 0x8000); reads may target anywhere including ROM.
                std::uint16_t base;
                if (touches_memory) {
                    base = static_cast<std::uint16_t>(rng() % 0x7700u);
                } else {
                    base = static_cast<std::uint16_t>(rng() % 0xFF00u);
                }
                img.code[static_cast<std::size_t>(pos) + 1] =
                    static_cast<std::uint8_t>(base & 0xFF);
                img.code[static_cast<std::size_t>(pos) + 2] =
                    static_cast<std::uint8_t>(base >> 8);
                if (e.op == nc::Op::JMP) {
                    jmp_idx.push_back(img.boundaries.size() - 1);
                }
                break;
            }
            case nc::AddrMode::Relative:
                img.code[static_cast<std::size_t>(pos) + 1] = 0;  // patched below
                branch_idx.push_back(img.boundaries.size() - 1);
                break;
            case nc::AddrMode::Indirect:
                break;  // excluded, unreachable
        }
        pos += len;
    }

    // Terminal wrap: JMP $8000 keeps every stream cycling forever.
    img.boundaries.push_back(pos);
    img.code[static_cast<std::size_t>(pos)] = 0x4C;
    img.code[static_cast<std::size_t>(pos) + 1] = 0x00;
    img.code[static_cast<std::size_t>(pos) + 2] = 0x80;

    // Patch branches: forward target boundary with offset in [0, 127] so the
    // PC always lands on an instruction start inside the region. Forward-only
    // (plus the wrap JMP) guarantees no degenerate infinite micro-loops while
    // still exercising taken/not-taken divergence.
    for (const std::size_t bi : branch_idx) {
        const int base = img.boundaries[bi] + 2;  // PC after the branch fetch
        std::size_t hi = bi + 1;
        while (hi + 1 < img.boundaries.size() && img.boundaries[hi + 1] - base <= 127) {
            ++hi;
        }
        std::uniform_int_distribution<std::size_t> pick(bi + 1, hi);
        const int target = img.boundaries[pick(rng)];
        img.code[static_cast<std::size_t>(img.boundaries[bi]) + 1] =
            static_cast<std::uint8_t>(target - base);
    }

    // Patch sampled JMPs: forward boundary anywhere ahead (worst case the
    // terminal wrap).
    for (const std::size_t ji : jmp_idx) {
        std::uniform_int_distribution<std::size_t> pick(ji + 1, img.boundaries.size() - 1);
        const std::uint16_t target =
            static_cast<std::uint16_t>(kCodeBase + img.boundaries[pick(rng)]);
        img.code[static_cast<std::size_t>(img.boundaries[ji]) + 1] =
            static_cast<std::uint8_t>(target & 0xFF);
        img.code[static_cast<std::size_t>(img.boundaries[ji]) + 2] =
            static_cast<std::uint8_t>(target >> 8);
    }

    return img;
}

// ---------------------------------------------------------------------------
// Host-side validation: run the generated image through the same interpreter
// on the CPU. Catches generator bugs (illegal opcode -> throw instead of a
// device trap) and yields the DYNAMIC opcode mix for the report.
// ---------------------------------------------------------------------------

static bool validate_on_host(const Image& img, std::uint64_t seed, long long steps,
                             std::vector<unsigned long long>& dyn_hist) {
    std::mt19937_64 rng(seed);
    std::vector<std::uint8_t> ram(kRamSize);
    for (auto& b : ram) b = static_cast<std::uint8_t>(rng() & 0xFF);

    nc::CpuState st;
    st.a = static_cast<std::uint8_t>(rng() & 0xFF);
    st.x = static_cast<std::uint8_t>(rng() & 0xFF);
    st.y = static_cast<std::uint8_t>(rng() & 0xFF);
    st.pc = static_cast<std::uint16_t>(
        kCodeBase + img.boundaries[rng() % img.boundaries.size()]);

    HostBus bus{ram.data(), img.code.data()};
    try {
        for (long long i = 0; i < steps; ++i) {
            const auto r = nc::step(st, bus);
            dyn_hist[r.opcode] += 1;
            if (st.pc < kCodeBase) {
                std::fprintf(stderr, "validation: PC left code region: 0x%04X\n", st.pc);
                return false;
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "validation: interpreter threw: %s\n", e.what());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Kernels.
// ---------------------------------------------------------------------------

// Variant A: status quo. One thread per instance; every lane just steps its
// own CPU. Warp divergence is whatever the hardware makes of the decode
// switches with 32 unrelated opcodes in flight.
__global__ void kernel_baseline(nc::CpuState* states, std::uint8_t* ram,
                                const std::uint8_t* __restrict__ code, int n, int k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    nc::CpuState s = states[i];
    DeviceBus bus{ram + static_cast<std::size_t>(i) * kRamSize, code};
#pragma unroll 1
    for (int step = 0; step < k; ++step) {
        (void)nc::step(s, bus);
    }
    states[i] = s;
}

// Variant B: wavefront / opcode-binned. Per instruction, every lane peeks its
// next opcode; the warp then iterates over the distinct bin keys present
// (ballot/shuffle binning — Pascal sm_61 has no __match_any_sync) and each
// group executes step() with a warp-uniform key. group_count accumulates
// bins/instruction for the report. Requires n % 32 == 0 (full warps).
//
// Two binning granularities:
//   BinByOp = false: key = raw opcode      -> every dispatch in the group is
//                    fully uniform (mode + op + cycles), max convergence,
//                    most groups.
//   BinByOp = true:  key = operation class -> fewer, larger groups; the op
//                    switch is uniform but the addressing-mode switch may
//                    still diverge within a group.
template <bool BinByOp>
__global__ void kernel_wavefront(nc::CpuState* states, std::uint8_t* ram,
                                 const std::uint8_t* __restrict__ code, int n, int k,
                                 unsigned long long* group_count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    nc::CpuState s = states[i];
    DeviceBus bus{ram + static_cast<std::size_t>(i) * kRamSize, code};
    constexpr unsigned kFull = 0xFFFFFFFFu;
    unsigned long long groups = 0;
#pragma unroll 1
    for (int step = 0; step < k; ++step) {
        const std::uint8_t op = bus.read(s.pc);  // peek, no side effects
        int key = op;
        if constexpr (BinByOp) {
            key = static_cast<int>(nc::detail::decode(op).op);
        }
        unsigned pending = kFull;
        while (pending) {
            const int leader = __ffs(pending) - 1;
            const int leader_key = __shfl_sync(kFull, key, leader);
            const unsigned group = __ballot_sync(kFull, key == leader_key);
            if (key == leader_key) {
                (void)nc::step(s, bus);
            }
            pending &= ~group;
            ++groups;
        }
    }
    states[i] = s;
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(group_count, groups);
    }
}

// ---------------------------------------------------------------------------
// Harness.
// ---------------------------------------------------------------------------

struct InitData {
    std::vector<nc::CpuState> states;
    std::vector<std::uint8_t> ram;
};

static InitData make_init(const Image& img, int n, bool correlated, std::uint64_t seed) {
    InitData init;
    init.states.resize(static_cast<std::size_t>(n));
    init.ram.assign(static_cast<std::size_t>(n) * kRamSize, 0);
    std::mt19937_64 rng(seed);
    for (int i = 0; i < n; ++i) {
        nc::CpuState st;
        if (correlated) {
            st.pc = kCodeBase;  // boundary 0; identical RAM (zeros) and regs
        } else {
            st.a = static_cast<std::uint8_t>(rng() & 0xFF);
            st.x = static_cast<std::uint8_t>(rng() & 0xFF);
            st.y = static_cast<std::uint8_t>(rng() & 0xFF);
            st.pc = static_cast<std::uint16_t>(
                kCodeBase + img.boundaries[rng() % img.boundaries.size()]);
            std::uint8_t* r = init.ram.data() + static_cast<std::size_t>(i) * kRamSize;
            for (int b = 0; b < kRamSize; b += 8) {
                const std::uint64_t v = rng();
                std::memcpy(r + b, &v, 8);
            }
        }
        init.states[static_cast<std::size_t>(i)] = st;
    }
    return init;
}

struct DeviceBuffers {
    nc::CpuState* states = nullptr;
    std::uint8_t* ram = nullptr;
    std::uint8_t* code = nullptr;
    unsigned long long* group_count = nullptr;
};

static void upload_init(const DeviceBuffers& dev, const InitData& init, int n) {
    CUDA_CHECK(cudaMemcpy(dev.states, init.states.data(),
                          static_cast<std::size_t>(n) * sizeof(nc::CpuState),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.ram, init.ram.data(),
                          static_cast<std::size_t>(n) * kRamSize,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dev.group_count, 0, sizeof(unsigned long long)));
}

struct RunResult {
    double instr_per_sec = 0.0;
    double avg_groups = 0.0;  // wavefront only: mean opcode bins per warp-instruction
};

enum class Variant { Baseline, BinByOpcode, BinByOpClass };

static RunResult run_variant(Variant variant, const DeviceBuffers& dev, const Image& img,
                             int n, bool correlated, int k, int repeats,
                             std::uint64_t init_seed) {
    const InitData init = make_init(img, n, correlated, init_seed);
    upload_init(dev, init, n);

    const int blocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
    auto launch = [&]() {
        switch (variant) {
            case Variant::Baseline:
                kernel_baseline<<<blocks, kThreadsPerBlock>>>(dev.states, dev.ram,
                                                              dev.code, n, k);
                break;
            case Variant::BinByOpcode:
                kernel_wavefront<false><<<blocks, kThreadsPerBlock>>>(
                    dev.states, dev.ram, dev.code, n, k, dev.group_count);
                break;
            case Variant::BinByOpClass:
                kernel_wavefront<true><<<blocks, kThreadsPerBlock>>>(
                    dev.states, dev.ram, dev.code, n, k, dev.group_count);
                break;
        }
        CUDA_CHECK(cudaGetLastError());
    };

    // Warmup launch (also decorrelates "decorrelated" mode further, and keeps
    // correlated mode in perfect lockstep — deterministic identical streams).
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(dev.group_count, 0, sizeof(unsigned long long)));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int r = 0; r < repeats; ++r) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    RunResult out;
    const double total_instr =
        static_cast<double>(n) * static_cast<double>(k) * static_cast<double>(repeats);
    out.instr_per_sec = total_instr / (static_cast<double>(ms) * 1e-3);
    if (variant != Variant::Baseline) {
        unsigned long long groups = 0;
        CUDA_CHECK(cudaMemcpy(&groups, dev.group_count, sizeof(groups),
                              cudaMemcpyDeviceToHost));
        const double warp_rounds = static_cast<double>(n / 32) *
                                   static_cast<double>(k) * static_cast<double>(repeats);
        out.avg_groups = static_cast<double>(groups) / warp_rounds;
    }
    return out;
}

// Both variants must compute the exact same thing: run each once from the
// same init and compare final CPU states and RAM byte-for-byte.
static bool verify_equivalence(const DeviceBuffers& dev, const Image& img, int n,
                               bool correlated, int k, std::uint64_t init_seed) {
    const InitData init = make_init(img, n, correlated, init_seed);
    const int blocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;

    std::vector<nc::CpuState> states_a(static_cast<std::size_t>(n));
    std::vector<nc::CpuState> states_b(static_cast<std::size_t>(n));
    std::vector<std::uint8_t> ram_a(static_cast<std::size_t>(n) * kRamSize);
    std::vector<std::uint8_t> ram_b(static_cast<std::size_t>(n) * kRamSize);

    upload_init(dev, init, n);
    kernel_baseline<<<blocks, kThreadsPerBlock>>>(dev.states, dev.ram, dev.code, n, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(states_a.data(), dev.states,
                          states_a.size() * sizeof(nc::CpuState), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ram_a.data(), dev.ram, ram_a.size(), cudaMemcpyDeviceToHost));

    for (const bool bin_by_op : {false, true}) {
        upload_init(dev, init, n);
        if (bin_by_op) {
            kernel_wavefront<true><<<blocks, kThreadsPerBlock>>>(
                dev.states, dev.ram, dev.code, n, k, dev.group_count);
        } else {
            kernel_wavefront<false><<<blocks, kThreadsPerBlock>>>(
                dev.states, dev.ram, dev.code, n, k, dev.group_count);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(states_b.data(), dev.states,
                              states_b.size() * sizeof(nc::CpuState),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ram_b.data(), dev.ram, ram_b.size(), cudaMemcpyDeviceToHost));

        for (int i = 0; i < n; ++i) {
            const auto& a = states_a[static_cast<std::size_t>(i)];
            const auto& b = states_b[static_cast<std::size_t>(i)];
            if (a.pc != b.pc || a.a != b.a || a.x != b.x || a.y != b.y || a.sp != b.sp ||
                a.p != b.p || a.cycles != b.cycles) {
                std::fprintf(stderr,
                             "MISMATCH instance %d (bin_by_op=%d): A pc=%04X a=%02X "
                             "cyc=%llu | B pc=%04X a=%02X cyc=%llu\n",
                             i, bin_by_op ? 1 : 0, a.pc, a.a,
                             static_cast<unsigned long long>(a.cycles), b.pc, b.a,
                             static_cast<unsigned long long>(b.cycles));
                return false;
            }
        }
        if (std::memcmp(ram_a.data(), ram_b.data(), ram_a.size()) != 0) {
            std::fprintf(stderr, "MISMATCH: RAM contents differ (bin_by_op=%d)\n",
                         bin_by_op ? 1 : 0);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    int k = 1024;     // instructions per instance per launch
    int repeats = 5;  // timed launches
    if (argc > 1) k = std::atoi(argv[1]);
    if (argc > 2) repeats = std::atoi(argv[2]);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s (sm_%d%d), %d SMs\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);
    std::printf("config: K=%d instructions/launch, %d timed launches, %d threads/block\n\n",
                k, repeats, kThreadsPerBlock);

    // --- Generate and validate the synthetic stream --------------------------
    const Image img = generate_image(42);
    const double excl_pct = 100.0 * static_cast<double>(img.excluded_total) /
                            static_cast<double>(img.excluded_total + img.included_total);
    std::printf("code image: %zu instructions in %d bytes; excluded opcodes = %.2f%% "
                "of the measured stream (JSR/RTS/BRK/RTI/JMP-indirect/illegal)\n",
                img.boundaries.size(), kCodeSize, excl_pct);

    std::vector<unsigned long long> dyn_hist(256, 0);
    bool ok = true;
    for (std::uint64_t seed = 100; seed < 103; ++seed) {
        ok = ok && validate_on_host(img, seed, 1000000, dyn_hist);
    }
    if (!ok) {
        std::fprintf(stderr, "host validation FAILED — aborting\n");
        return 1;
    }
    std::printf("host validation: 3x1,000,000 instructions OK (PC stayed in code "
                "region, no illegal opcodes)\n");

    unsigned long long dyn_total = 0;
    for (const auto c : dyn_hist) dyn_total += c;
    std::vector<int> order(256);
    for (int i = 0; i < 256; ++i) order[static_cast<std::size_t>(i)] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return dyn_hist[a] > dyn_hist[b]; });
    std::printf("\nsynthetic dynamic mix vs measured target (top 12):\n");
    std::printf("  op    synthetic%%   target%%\n");
    for (int r = 0; r < 12; ++r) {
        const int op = order[static_cast<std::size_t>(r)];
        std::printf("  0x%02X  %8.3f%%  %8.3f%%\n", op,
                    100.0 * static_cast<double>(dyn_hist[op]) / static_cast<double>(dyn_total),
                    100.0 * static_cast<double>(kMeasuredCounts[op]) /
                        static_cast<double>(img.included_total));
    }

    // --- Device setup ---------------------------------------------------------
    const std::vector<int> sizes = {4096, 32768, 131072};
    const int max_n = sizes.back();

    DeviceBuffers dev;
    CUDA_CHECK(cudaMalloc(&dev.states, static_cast<std::size_t>(max_n) * sizeof(nc::CpuState)));
    CUDA_CHECK(cudaMalloc(&dev.ram, static_cast<std::size_t>(max_n) * kRamSize));
    CUDA_CHECK(cudaMalloc(&dev.code, kCodeSize));
    CUDA_CHECK(cudaMalloc(&dev.group_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(dev.code, img.code.data(), kCodeSize, cudaMemcpyHostToDevice));

    // --- Equivalence check ------------------------------------------------------
    for (const bool correlated : {true, false}) {
        if (!verify_equivalence(dev, img, 4096, correlated, 256, 7777)) {
            std::fprintf(stderr, "A/B equivalence FAILED (%s) — aborting\n",
                         correlated ? "correlated" : "decorrelated");
            return 1;
        }
    }
    std::printf("\nA/B equivalence: identical final states + RAM at 4096 instances "
                "(correlated and decorrelated), 256 instr each\n");

    // --- Benchmark ---------------------------------------------------------------
    // A  = baseline thread-per-instance
    // B1 = wavefront, bin by raw opcode (fully uniform dispatch per group)
    // B2 = wavefront, bin by operation class (fewer, larger groups)
    std::printf("\n%-9s %-13s %12s %12s %12s %7s %7s %9s %9s\n", "n", "streams",
                "A instr/s", "B1 instr/s", "B2 instr/s", "B1/A", "B2/A", "bins(B1)",
                "bins(B2)");
    for (const int n : sizes) {
        for (const bool correlated : {true, false}) {
            const std::uint64_t seed = 900 + static_cast<std::uint64_t>(n) +
                                       (correlated ? 1 : 0);
            const RunResult a =
                run_variant(Variant::Baseline, dev, img, n, correlated, k, repeats, seed);
            const RunResult b1 = run_variant(Variant::BinByOpcode, dev, img, n, correlated,
                                             k, repeats, seed);
            const RunResult b2 = run_variant(Variant::BinByOpClass, dev, img, n, correlated,
                                             k, repeats, seed);
            std::printf("%-9d %-13s %12.3e %12.3e %12.3e %7.3f %7.3f %9.2f %9.2f\n", n,
                        correlated ? "correlated" : "decorrelated", a.instr_per_sec,
                        b1.instr_per_sec, b2.instr_per_sec,
                        b1.instr_per_sec / a.instr_per_sec,
                        b2.instr_per_sec / a.instr_per_sec, b1.avg_groups, b2.avg_groups);
        }
    }

    CUDA_CHECK(cudaFree(dev.states));
    CUDA_CHECK(cudaFree(dev.ram));
    CUDA_CHECK(cudaFree(dev.code));
    CUDA_CHECK(cudaFree(dev.group_count));
    return 0;
}
