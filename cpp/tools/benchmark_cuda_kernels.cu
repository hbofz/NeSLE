#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "nesle/cuda/batch_step.cuh"
#include "nesle/cuda/kernels.cuh"

namespace {

void check(cudaError_t error, const char* label) {
    if (error != cudaSuccess) {
        std::cerr << label << ": " << cudaGetErrorString(error) << '\n';
        std::exit(1);
    }
}

std::vector<std::uint32_t> parse_env_counts(const std::string& value) {
    std::vector<std::uint32_t> counts;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            counts.push_back(static_cast<std::uint32_t>(std::stoul(item)));
        }
    }
    return counts;
}

void set_digits(std::vector<std::uint8_t>& ram, std::size_t base, int value) {
    value = std::max(0, std::min(999, value));
    ram[base + nesle::cuda::kMarioTimeDigits] = static_cast<std::uint8_t>(value / 100);
    ram[base + nesle::cuda::kMarioTimeDigits + 1] = static_cast<std::uint8_t>((value / 10) % 10);
    ram[base + nesle::cuda::kMarioTimeDigits + 2] = static_cast<std::uint8_t>(value % 10);
}

void seed_mario_ram(std::vector<std::uint8_t>& ram,
                    std::vector<int>& previous_x,
                    std::vector<int>& previous_time) {
    for (std::uint32_t env = 0; env < previous_x.size(); ++env) {
        const auto base = static_cast<std::size_t>(env) * nesle::cuda::kCpuRamBytes;
        const auto x = 40 + static_cast<int>(env % 64);
        previous_x[env] = x;
        previous_time[env] = 400;
        ram[base + nesle::cuda::kMarioXPage] = 0;
        ram[base + nesle::cuda::kMarioXScreen] = static_cast<std::uint8_t>(x + 2);
        ram[base + nesle::cuda::kMarioYViewport] = 1;
        ram[base + nesle::cuda::kMarioPlayerState] = 8;
        ram[base + nesle::cuda::kMarioLives] = 2;
        set_digits(ram, base, 399);
    }
}

template <typename T>
T* copy_to_device(const std::vector<T>& host, const char* label) {
    T* device = nullptr;
    check(cudaMalloc(&device, host.size() * sizeof(T)), label);
    check(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), label);
    return device;
}

float milliseconds_between(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0F;
    check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    return ms;
}

}  // namespace

int main(int argc, char** argv) {
    std::string env_counts_arg = "1024,4096,8192,16384";
    int step_iterations = 1000;
    int render_iterations = 50;
    int warmup_iterations = 10;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--env-counts" && i + 1 < argc) {
            env_counts_arg = argv[++i];
        } else if (arg == "--step-iterations" && i + 1 < argc) {
            step_iterations = std::stoi(argv[++i]);
        } else if (arg == "--render-iterations" && i + 1 < argc) {
            render_iterations = std::stoi(argv[++i]);
        } else if (arg == "--warmup-iterations" && i + 1 < argc) {
            warmup_iterations = std::stoi(argv[++i]);
        } else {
            std::cerr << "unknown argument: " << arg << '\n';
            return 2;
        }
    }

    const auto counts = parse_env_counts(env_counts_arg);
    if (counts.empty()) {
        std::cerr << "at least one env count is required\n";
        return 2;
    }

    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::cout << "cuda_kernel_benchmark device=\"" << prop.name << "\"\n";

    for (const auto num_envs : counts) {
        std::vector<std::uint8_t> host_ram(static_cast<std::size_t>(num_envs) * nesle::cuda::kCpuRamBytes, 0);
        std::vector<int> host_previous_x(num_envs, 0);
        std::vector<int> host_previous_time(num_envs, 400);
        std::vector<float> host_rewards(num_envs, 0.0F);
        std::vector<std::uint8_t> host_done(num_envs, 0);
        seed_mario_ram(host_ram, host_previous_x, host_previous_time);

        std::vector<std::uint8_t> host_ctrl(num_envs, 0);
        std::vector<std::uint8_t> host_mask(num_envs, 0);
        std::vector<std::uint8_t> host_palette(static_cast<std::size_t>(num_envs) * nesle::cuda::kPaletteRamBytes, 0);
        std::vector<std::uint8_t> host_frames(
            static_cast<std::size_t>(num_envs) * nesle::cuda::kFrameWidth * nesle::cuda::kFrameHeight *
                nesle::cuda::kRgbChannels,
            0);

        auto* device_ram = copy_to_device(host_ram, "ram");
        auto* device_previous_x = copy_to_device(host_previous_x, "previous_x");
        auto* device_previous_time = copy_to_device(host_previous_time, "previous_time");
        auto* device_rewards = copy_to_device(host_rewards, "rewards");
        auto* device_done = copy_to_device(host_done, "done");
        auto* device_ctrl = copy_to_device(host_ctrl, "ppu_ctrl");
        auto* device_mask = copy_to_device(host_mask, "ppu_mask");
        auto* device_palette = copy_to_device(host_palette, "palette");
        auto* device_frames = copy_to_device(host_frames, "frames");

        nesle::cuda::BatchBuffers buffers{};
        buffers.cpu.ram = device_ram;
        buffers.previous_mario_x = device_previous_x;
        buffers.previous_mario_time = device_previous_time;
        buffers.rewards = device_rewards;
        buffers.done = device_done;
        buffers.ppu.ctrl = device_ctrl;
        buffers.ppu.mask = device_mask;
        buffers.ppu.palette_ram = device_palette;
        buffers.frames_rgb = device_frames;

        const nesle::cuda::StepConfig config{num_envs, 1, false};
        for (int i = 0; i < warmup_iterations; ++i) {
            nesle::cuda::launch_step_kernel(buffers, config, nullptr);
        }
        check(cudaDeviceSynchronize(), "step warmup");

        cudaEvent_t start{};
        cudaEvent_t stop{};
        check(cudaEventCreate(&start), "cudaEventCreate start");
        check(cudaEventCreate(&stop), "cudaEventCreate stop");
        check(cudaEventRecord(start), "cudaEventRecord step start");
        for (int i = 0; i < step_iterations; ++i) {
            nesle::cuda::launch_step_kernel(buffers, config, nullptr);
        }
        check(cudaEventRecord(stop), "cudaEventRecord step stop");
        check(cudaEventSynchronize(stop), "cudaEventSynchronize step");
        const auto step_ms = milliseconds_between(start, stop);

        for (int i = 0; i < warmup_iterations; ++i) {
            nesle::cuda::launch_render_kernel(buffers, config, nullptr);
        }
        check(cudaDeviceSynchronize(), "render warmup");
        check(cudaEventRecord(start), "cudaEventRecord render start");
        for (int i = 0; i < render_iterations; ++i) {
            nesle::cuda::launch_render_kernel(buffers, config, nullptr);
        }
        check(cudaEventRecord(stop), "cudaEventRecord render stop");
        check(cudaEventSynchronize(stop), "cudaEventSynchronize render");
        const auto render_ms = milliseconds_between(start, stop);
        check(cudaEventDestroy(start), "cudaEventDestroy start");
        check(cudaEventDestroy(stop), "cudaEventDestroy stop");

        const auto step_seconds = static_cast<double>(step_ms) / 1000.0;
        const auto render_seconds = static_cast<double>(render_ms) / 1000.0;
        const auto env_steps_per_sec =
            (static_cast<double>(num_envs) * static_cast<double>(step_iterations)) / step_seconds;
        const auto render_frames_per_sec =
            (static_cast<double>(num_envs) * static_cast<double>(render_iterations)) / render_seconds;

        std::cout << "cuda_kernel_result envs=" << num_envs
                  << " step_iterations=" << step_iterations
                  << " render_iterations=" << render_iterations
                  << " env_steps_per_sec=" << env_steps_per_sec
                  << " render_frames_per_sec=" << render_frames_per_sec << '\n';

        check(cudaFree(device_ram), "cudaFree ram");
        check(cudaFree(device_previous_x), "cudaFree previous_x");
        check(cudaFree(device_previous_time), "cudaFree previous_time");
        check(cudaFree(device_rewards), "cudaFree rewards");
        check(cudaFree(device_done), "cudaFree done");
        check(cudaFree(device_ctrl), "cudaFree ctrl");
        check(cudaFree(device_mask), "cudaFree mask");
        check(cudaFree(device_palette), "cudaFree palette");
        check(cudaFree(device_frames), "cudaFree frames");
    }
}
