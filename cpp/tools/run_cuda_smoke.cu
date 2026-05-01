#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
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

void set_digits(std::vector<std::uint8_t>& ram,
                std::size_t base,
                int hundreds,
                int tens,
                int ones) {
    ram[base + nesle::cuda::kMarioTimeDigits] = static_cast<std::uint8_t>(hundreds);
    ram[base + nesle::cuda::kMarioTimeDigits + 1] = static_cast<std::uint8_t>(tens);
    ram[base + nesle::cuda::kMarioTimeDigits + 2] = static_cast<std::uint8_t>(ones);
}

void seed_mario_ram(std::vector<std::uint8_t>& ram,
                    std::size_t env,
                    int x_page,
                    int x_screen,
                    int timer,
                    std::uint8_t player_state,
                    std::uint8_t y_viewport) {
    const auto base = env * nesle::cuda::kCpuRamBytes;
    ram[base + nesle::cuda::kMarioXPage] = static_cast<std::uint8_t>(x_page);
    ram[base + nesle::cuda::kMarioXScreen] = static_cast<std::uint8_t>(x_screen);
    ram[base + nesle::cuda::kMarioYViewport] = y_viewport;
    ram[base + nesle::cuda::kMarioPlayerState] = player_state;
    ram[base + nesle::cuda::kMarioLives] = 2;
    set_digits(ram, base, timer / 100, (timer / 10) % 10, timer % 10);
}

}  // namespace

int main() {
    constexpr std::uint32_t kNumEnvs = 3;

    std::vector<std::uint8_t> host_ram(kNumEnvs * nesle::cuda::kCpuRamBytes, 0);
    std::vector<int> host_previous_x = {40, 80, 48};
    std::vector<int> host_previous_time = {400, 399, 400};
    std::vector<std::uint8_t> host_previous_dying(kNumEnvs, 0);
    std::vector<float> host_rewards(kNumEnvs, 0.0F);
    std::vector<std::uint8_t> host_done(kNumEnvs, 0);

    seed_mario_ram(host_ram, 0, 0, 43, 399, 8, 1);
    seed_mario_ram(host_ram, 1, 0, 90, 398, 8, 1);
    seed_mario_ram(host_ram, 2, 0, 50, 399, 0x0B, 1);

    std::uint8_t* device_ram = nullptr;
    int* device_previous_x = nullptr;
    int* device_previous_time = nullptr;
    std::uint8_t* device_previous_dying = nullptr;
    float* device_rewards = nullptr;
    std::uint8_t* device_done = nullptr;

    check(cudaMalloc(&device_ram, host_ram.size()), "cudaMalloc ram");
    check(cudaMalloc(&device_previous_x, host_previous_x.size() * sizeof(int)),
          "cudaMalloc previous_x");
    check(cudaMalloc(&device_previous_time, host_previous_time.size() * sizeof(int)),
          "cudaMalloc previous_time");
    check(cudaMalloc(&device_previous_dying, host_previous_dying.size()),
          "cudaMalloc previous_dying");
    check(cudaMalloc(&device_rewards, host_rewards.size() * sizeof(float)), "cudaMalloc rewards");
    check(cudaMalloc(&device_done, host_done.size()), "cudaMalloc done");

    check(cudaMemcpy(device_ram, host_ram.data(), host_ram.size(), cudaMemcpyHostToDevice),
          "cudaMemcpy ram");
    check(cudaMemcpy(device_previous_x,
                     host_previous_x.data(),
                     host_previous_x.size() * sizeof(int),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy previous_x");
    check(cudaMemcpy(device_previous_time,
                     host_previous_time.data(),
                     host_previous_time.size() * sizeof(int),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy previous_time");
    check(cudaMemcpy(device_previous_dying,
                     host_previous_dying.data(),
                     host_previous_dying.size(),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy previous_dying");
    check(cudaMemcpy(device_rewards,
                     host_rewards.data(),
                     host_rewards.size() * sizeof(float),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy rewards");
    check(cudaMemcpy(device_done, host_done.data(), host_done.size(), cudaMemcpyHostToDevice),
          "cudaMemcpy done");

    nesle::cuda::BatchBuffers buffers{};
    buffers.cpu.ram = device_ram;
    buffers.previous_mario_x = device_previous_x;
    buffers.previous_mario_time = device_previous_time;
    buffers.previous_mario_dying = device_previous_dying;
    buffers.rewards = device_rewards;
    buffers.done = device_done;

    nesle::cuda::launch_step_kernel(
        buffers,
        nesle::cuda::StepConfig{kNumEnvs, 1, false},
        nullptr);
    check(cudaGetLastError(), "launch_step_kernel");
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    check(cudaMemcpy(host_rewards.data(),
                     device_rewards,
                     host_rewards.size() * sizeof(float),
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy rewards back");
    check(cudaMemcpy(host_done.data(), device_done, host_done.size(), cudaMemcpyDeviceToHost),
          "cudaMemcpy done back");
    check(cudaMemcpy(host_previous_x.data(),
                     device_previous_x,
                     host_previous_x.size() * sizeof(int),
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy previous_x back");

    check(cudaFree(device_ram), "cudaFree ram");
    check(cudaFree(device_previous_x), "cudaFree previous_x");
    check(cudaFree(device_previous_time), "cudaFree previous_time");
    check(cudaFree(device_previous_dying), "cudaFree previous_dying");
    check(cudaFree(device_rewards), "cudaFree rewards");
    check(cudaFree(device_done), "cudaFree done");

    if (host_rewards[0] != 2.0F || host_rewards[1] != -1.0F || host_rewards[2] != -24.0F) {
        std::cerr << "unexpected rewards: "
                  << host_rewards[0] << ", "
                  << host_rewards[1] << ", "
                  << host_rewards[2] << '\n';
        return 1;
    }
    if (host_done[0] != 0 || host_done[1] != 0 || host_done[2] != 1) {
        std::cerr << "unexpected done flags: "
                  << static_cast<int>(host_done[0]) << ", "
                  << static_cast<int>(host_done[1]) << ", "
                  << static_cast<int>(host_done[2]) << '\n';
        return 1;
    }
    if (host_previous_x[0] != 43 || host_previous_x[1] != 90 || host_previous_x[2] != 50) {
        std::cerr << "previous x baselines were not advanced\n";
        return 1;
    }

    std::cout << "cuda_smoke envs=" << kNumEnvs
              << " rewards=" << host_rewards[0] << "," << host_rewards[1] << "," << host_rewards[2]
              << " done=" << static_cast<int>(host_done[0]) << ","
              << static_cast<int>(host_done[1]) << "," << static_cast<int>(host_done[2])
              << '\n';
    return 0;
}
