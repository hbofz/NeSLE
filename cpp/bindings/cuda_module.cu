#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "nesle/cuda/batch_step.cuh"
#include "nesle/cuda/kernels.cuh"

namespace py = pybind11;

namespace {

constexpr std::uint32_t kFrameBytes =
    nesle::cuda::kFrameWidth * nesle::cuda::kFrameHeight * nesle::cuda::kRgbChannels;

void check_cuda(cudaError_t error, const char* label) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(error));
    }
}

template <typename T>
T* cuda_alloc(std::size_t count, const char* label) {
    T* ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, count * sizeof(T)), label);
    return ptr;
}

template <typename T>
void copy_to_device(T* device, const std::vector<T>& host, const char* label) {
    check_cuda(
        cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice),
        label);
}

void write_time_digits(std::vector<std::uint8_t>& ram, std::size_t base, int value) {
    value = std::max(0, std::min(999, value));
    ram[base + nesle::cuda::kMarioTimeDigits] = static_cast<std::uint8_t>(value / 100);
    ram[base + nesle::cuda::kMarioTimeDigits + 1] =
        static_cast<std::uint8_t>((value / 10) % 10);
    ram[base + nesle::cuda::kMarioTimeDigits + 2] = static_cast<std::uint8_t>(value % 10);
}

__global__ void apply_actions_kernel(nesle::cuda::BatchBuffers buffers,
                                     const std::uint8_t* actions,
                                     std::uint32_t num_envs,
                                     std::uint32_t frameskip,
                                     std::uint32_t* step_counts) {
    const auto env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= num_envs) {
        return;
    }

    auto* ram = nesle::cuda::env_cpu_ram(buffers, env);
    const auto action = actions[env];
    int x_delta = 0;
    if ((action & 0x80) != 0) {
        x_delta += 1 + ((action & 0x02) != 0 ? 1 : 0);
    }
    if ((action & 0x40) != 0) {
        x_delta -= 1;
    }

    auto x = static_cast<int>(ram[nesle::cuda::kMarioXPage]) * 0x100 +
             static_cast<int>(ram[nesle::cuda::kMarioXScreen]);
    x = max(0, min(0xFFFF, x + x_delta));
    ram[nesle::cuda::kMarioXPage] = static_cast<std::uint8_t>((x >> 8) & 0xFF);
    ram[nesle::cuda::kMarioXScreen] = static_cast<std::uint8_t>(x & 0xFF);

    const auto cadence = max(1u, 24u / max(1u, frameskip));
    const auto step = step_counts[env]++;
    if (step % cadence == 0) {
        auto time = nesle::cuda::read_bcd_digits(ram, nesle::cuda::kMarioTimeDigits, 3);
        time = max(0, time - 1);
        ram[nesle::cuda::kMarioTimeDigits] = static_cast<std::uint8_t>(time / 100);
        ram[nesle::cuda::kMarioTimeDigits + 1] = static_cast<std::uint8_t>((time / 10) % 10);
        ram[nesle::cuda::kMarioTimeDigits + 2] = static_cast<std::uint8_t>(time % 10);
    }
}

class CudaBatchBinding {
public:
    CudaBatchBinding(std::uint32_t num_envs, std::uint32_t frameskip)
        : num_env_(num_envs),
          frameskip_(frameskip) {
        if (num_env_ == 0) {
            throw std::invalid_argument("num_envs must be positive");
        }
        if (frameskip_ == 0) {
            throw std::invalid_argument("frameskip must be positive");
        }
        allocate();
        reset();
    }

    CudaBatchBinding(const CudaBatchBinding&) = delete;
    CudaBatchBinding& operator=(const CudaBatchBinding&) = delete;

    ~CudaBatchBinding() {
        release();
    }

    py::array_t<std::uint8_t> reset() {
        std::vector<std::uint8_t> ram(static_cast<std::size_t>(num_env_) * nesle::cuda::kCpuRamBytes, 0);
        std::vector<int> previous_x(num_env_, 0);
        std::vector<int> previous_time(num_env_, 400);
        std::vector<std::uint8_t> previous_dying(num_env_, 0);
        std::vector<float> rewards(num_env_, 0.0F);
        std::vector<std::uint8_t> done(num_env_, 0);
        std::vector<std::uint32_t> step_counts(num_env_, 0);
        std::vector<std::uint8_t> ctrl(num_env_, 0);
        std::vector<std::uint8_t> mask(num_env_, 0);
        std::vector<std::uint8_t> palette(static_cast<std::size_t>(num_env_) *
                                              nesle::cuda::kPaletteRamBytes,
                                          0);

        for (std::uint32_t env = 0; env < num_env_; ++env) {
            const auto base = static_cast<std::size_t>(env) * nesle::cuda::kCpuRamBytes;
            ram[base + nesle::cuda::kMarioXPage] = 1;
            ram[base + nesle::cuda::kMarioXScreen] = 2;
            ram[base + nesle::cuda::kMarioYViewport] = 1;
            ram[base + nesle::cuda::kMarioLives] = 2;
            ram[base + nesle::cuda::kMarioPlayerState] = 0;
            write_time_digits(ram, base, 400);
            previous_x[env] = 0x100 + 2;
        }

        copy_to_device(device_ram_, ram, "reset ram");
        copy_to_device(device_previous_x_, previous_x, "reset previous_x");
        copy_to_device(device_previous_time_, previous_time, "reset previous_time");
        copy_to_device(device_previous_dying_, previous_dying, "reset previous_dying");
        copy_to_device(device_rewards_, rewards, "reset rewards");
        copy_to_device(device_done_, done, "reset done");
        copy_to_device(device_step_counts_, step_counts, "reset step_counts");
        copy_to_device(device_ppu_ctrl_, ctrl, "reset ppu ctrl");
        copy_to_device(device_ppu_mask_, mask, "reset ppu mask");
        copy_to_device(device_palette_, palette, "reset palette");
        render_device();
        return render();
    }

    py::dict step(py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> actions) {
        const auto view = actions.request();
        if (view.ndim != 1 || static_cast<std::uint32_t>(view.shape[0]) != num_env_) {
            throw std::invalid_argument("actions must have shape (num_envs,)");
        }
        check_cuda(cudaMemcpy(device_actions_,
                              view.ptr,
                              static_cast<std::size_t>(num_env_) * sizeof(std::uint8_t),
                              cudaMemcpyHostToDevice),
                   "copy actions");

        constexpr int kThreads = 256;
        const auto blocks = static_cast<int>((num_env_ + kThreads - 1) / kThreads);
        apply_actions_kernel<<<blocks, kThreads>>>(buffers_, device_actions_, num_env_, frameskip_, device_step_counts_);
        check_cuda(cudaGetLastError(), "apply_actions_kernel");
        nesle::cuda::launch_step_kernel(buffers_, nesle::cuda::StepConfig{num_env_, frameskip_, false}, nullptr);
        check_cuda(cudaGetLastError(), "launch_step_kernel");
        render_device();
        check_cuda(cudaDeviceSynchronize(), "cuda step synchronize");

        py::array_t<float> rewards(static_cast<py::ssize_t>(num_env_));
        py::array_t<std::uint8_t> dones(static_cast<py::ssize_t>(num_env_));
        check_cuda(cudaMemcpy(rewards.mutable_data(),
                              device_rewards_,
                              static_cast<std::size_t>(num_env_) * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "copy rewards");
        check_cuda(cudaMemcpy(dones.mutable_data(),
                              device_done_,
                              static_cast<std::size_t>(num_env_) * sizeof(std::uint8_t),
                              cudaMemcpyDeviceToHost),
                   "copy dones");

        py::dict out;
        out["obs"] = render();
        out["rewards"] = rewards;
        out["dones"] = dones;
        return out;
    }

    py::array_t<std::uint8_t> render() const {
        py::array_t<std::uint8_t> out(std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(num_env_),
            nesle::cuda::kFrameHeight,
            nesle::cuda::kFrameWidth,
            nesle::cuda::kRgbChannels,
        });
        check_cuda(cudaMemcpy(out.mutable_data(),
                              device_frames_,
                              static_cast<std::size_t>(num_env_) * kFrameBytes,
                              cudaMemcpyDeviceToHost),
                   "copy frames");
        return out;
    }

    std::string name() const {
        return "cuda";
    }

private:
    void allocate() {
        device_ram_ = cuda_alloc<std::uint8_t>(
            static_cast<std::size_t>(num_env_) * nesle::cuda::kCpuRamBytes,
            "cudaMalloc ram");
        device_previous_x_ = cuda_alloc<int>(num_env_, "cudaMalloc previous_x");
        device_previous_time_ = cuda_alloc<int>(num_env_, "cudaMalloc previous_time");
        device_previous_dying_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc previous_dying");
        device_rewards_ = cuda_alloc<float>(num_env_, "cudaMalloc rewards");
        device_done_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc done");
        device_actions_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc actions");
        device_step_counts_ = cuda_alloc<std::uint32_t>(num_env_, "cudaMalloc step_counts");
        device_ppu_ctrl_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc ppu ctrl");
        device_ppu_mask_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc ppu mask");
        device_palette_ = cuda_alloc<std::uint8_t>(
            static_cast<std::size_t>(num_env_) * nesle::cuda::kPaletteRamBytes,
            "cudaMalloc palette");
        device_frames_ = cuda_alloc<std::uint8_t>(
            static_cast<std::size_t>(num_env_) * kFrameBytes,
            "cudaMalloc frames");

        buffers_.cpu.ram = device_ram_;
        buffers_.previous_mario_x = device_previous_x_;
        buffers_.previous_mario_time = device_previous_time_;
        buffers_.previous_mario_dying = device_previous_dying_;
        buffers_.rewards = device_rewards_;
        buffers_.done = device_done_;
        buffers_.ppu.ctrl = device_ppu_ctrl_;
        buffers_.ppu.mask = device_ppu_mask_;
        buffers_.ppu.palette_ram = device_palette_;
        buffers_.frames_rgb = device_frames_;
    }

    void release() noexcept {
        cudaFree(device_ram_);
        cudaFree(device_previous_x_);
        cudaFree(device_previous_time_);
        cudaFree(device_previous_dying_);
        cudaFree(device_rewards_);
        cudaFree(device_done_);
        cudaFree(device_actions_);
        cudaFree(device_step_counts_);
        cudaFree(device_ppu_ctrl_);
        cudaFree(device_ppu_mask_);
        cudaFree(device_palette_);
        cudaFree(device_frames_);
    }

    void render_device() const {
        nesle::cuda::launch_render_kernel(buffers_, nesle::cuda::StepConfig{num_env_, frameskip_, true}, nullptr);
        check_cuda(cudaGetLastError(), "launch_render_kernel");
    }

    std::uint32_t num_env_ = 0;
    std::uint32_t frameskip_ = 0;
    nesle::cuda::BatchBuffers buffers_{};
    std::uint8_t* device_ram_ = nullptr;
    int* device_previous_x_ = nullptr;
    int* device_previous_time_ = nullptr;
    std::uint8_t* device_previous_dying_ = nullptr;
    float* device_rewards_ = nullptr;
    std::uint8_t* device_done_ = nullptr;
    std::uint8_t* device_actions_ = nullptr;
    std::uint32_t* device_step_counts_ = nullptr;
    std::uint8_t* device_ppu_ctrl_ = nullptr;
    std::uint8_t* device_ppu_mask_ = nullptr;
    std::uint8_t* device_palette_ = nullptr;
    std::uint8_t* device_frames_ = nullptr;
};

}  // namespace

PYBIND11_MODULE(_cuda_core, m) {
    m.doc() = "CUDA NeSLE batch helpers";

    py::class_<CudaBatchBinding>(m, "CudaBatch")
        .def(py::init<std::uint32_t, std::uint32_t>())
        .def("reset", &CudaBatchBinding::reset)
        .def("step", &CudaBatchBinding::step)
        .def("render", &CudaBatchBinding::render)
        .def_property_readonly("name", &CudaBatchBinding::name);
}
