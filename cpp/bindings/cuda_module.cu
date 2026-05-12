#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <optional>
#include <sstream>

#include "nesle/cuda/batch_step.cuh"
#include "nesle/cuda/kernels.cuh"
#include "nesle/fcs.hpp"
#include "nesle/rom.hpp"

namespace py = pybind11;

namespace {

constexpr std::uint32_t kFrameBytes =
    nesle::cuda::kFrameWidth * nesle::cuda::kFrameHeight * nesle::cuda::kRgbChannels;

class CudaDeviceArrayView {
public:
    CudaDeviceArrayView(std::uintptr_t ptr,
                        std::vector<py::ssize_t> shape,
                        std::string typestr,
                        bool read_only = false)
        : ptr_(ptr),
          shape_(std::move(shape)),
          typestr_(std::move(typestr)),
          read_only_(read_only) {}

    py::dict cuda_array_interface() const {
        py::dict out;
        py::tuple shape(shape_.size());
        for (std::size_t i = 0; i < shape_.size(); ++i) {
            shape[i] = shape_[i];
        }
        out["shape"] = shape;
        out["typestr"] = typestr_;
        out["data"] = py::make_tuple(ptr_, read_only_);
        out["version"] = 3;
        return out;
    }

private:
    std::uintptr_t ptr_ = 0;
    std::vector<py::ssize_t> shape_;
    std::string typestr_;
    bool read_only_ = false;
};

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

std::vector<std::uint8_t> bytes_to_vector(const py::bytes& bytes) {
    const std::string raw = bytes;
    return {raw.begin(), raw.end()};
}

std::uint8_t cuda_nametable_arrangement(nesle::NametableArrangement arrangement) {
    switch (arrangement) {
        case nesle::NametableArrangement::Vertical:
            return nesle::cuda::kNametableVertical;
        case nesle::NametableArrangement::Horizontal:
            return nesle::cuda::kNametableHorizontal;
        case nesle::NametableArrangement::FourScreen:
            return nesle::cuda::kNametableFourScreen;
    }
    return nesle::cuda::kNametableVertical;
}

std::uint16_t reset_vector_from_prg(const std::vector<std::uint8_t>& prg_rom) {
    if (prg_rom.size() != 16u * 1024u && prg_rom.size() != 32u * 1024u) {
        throw std::invalid_argument("CUDA console mode currently supports NROM PRG sizes only");
    }
    const auto base = prg_rom.size() == 16u * 1024u ? 0x3FFCu : 0x7FFCu;
    return static_cast<std::uint16_t>(prg_rom[base] | (static_cast<std::uint16_t>(prg_rom[base + 1]) << 8));
}

std::uintptr_t cuda_array_pointer(const py::object& object,
                                  std::uint32_t expected_len,
                                  std::string* typestr_out) {
    if (!py::hasattr(object, "__cuda_array_interface__")) {
        throw std::invalid_argument(
            "expected a CUDA tensor/array exposing __cuda_array_interface__");
    }
    py::dict iface = object.attr("__cuda_array_interface__").cast<py::dict>();
    const auto shape = iface["shape"].cast<py::tuple>();
    if (shape.size() != 1 || shape[0].cast<std::uint32_t>() != expected_len) {
        std::ostringstream msg;
        msg << "CUDA action array must have shape (" << expected_len << ",)";
        throw std::invalid_argument(msg.str());
    }
    const auto typestr = iface["typestr"].cast<std::string>();
    if (typestr_out != nullptr) {
        *typestr_out = typestr;
    }
    const auto data = iface["data"].cast<py::tuple>();
    if (data.size() < 1) {
        throw std::invalid_argument("__cuda_array_interface__ data field is malformed");
    }
    return data[0].cast<std::uintptr_t>();
}

__global__ void copy_int64_actions_kernel(std::uint8_t* dst,
                                          const long long* src,
                                          std::uint32_t num_envs) {
    const auto env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < num_envs) {
        dst[env] = static_cast<std::uint8_t>(src[env] & 0xFF);
    }
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

__global__ void poke_cpu_ram_kernel(nesle::cuda::BatchBuffers buffers,
                                    std::uint32_t num_envs,
                                    std::uint16_t address,
                                    std::uint8_t value) {
    const auto env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < num_envs) {
        nesle::cuda::env_cpu_ram(buffers, env)[address & 0x07FF] = value;
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

    CudaBatchBinding(std::uint32_t num_envs, std::uint32_t frameskip, const py::bytes& rom_bytes)
        : num_env_(num_envs),
          frameskip_(frameskip),
          rom_(nesle::parse_ines(bytes_to_vector(rom_bytes))),
          use_console_(true) {
        if (num_env_ == 0) {
            throw std::invalid_argument("num_envs must be positive");
        }
        if (frameskip_ == 0) {
            throw std::invalid_argument("frameskip must be positive");
        }
        if (!rom_.metadata.is_nrom()) {
            throw std::invalid_argument("CUDA console mode currently supports mapper 0/NROM ROMs");
        }
        if (rom_.prg_rom.empty()) {
            throw std::invalid_argument("CUDA console mode requires PRG ROM bytes");
        }
        allocate();
        upload_rom();
        reset();
    }

    CudaBatchBinding(std::uint32_t num_envs,
                     std::uint32_t frameskip,
                     const py::bytes& rom_bytes,
                     const py::bytes& snapshot_bytes)
        : num_env_(num_envs),
          frameskip_(frameskip),
          rom_(nesle::parse_ines(bytes_to_vector(rom_bytes))),
          use_console_(true) {
        validate_basics();
        const std::string snapshot_raw = snapshot_bytes;
        snapshots_.push_back(nesle::fcs::parse(snapshot_raw));
        std::vector<std::uint8_t> env_to_level(num_env_, 0);  // all envs use slot 0
        allocate();
        upload_rom();
        upload_snapshot_bank(env_to_level);
        reset();
    }

    CudaBatchBinding(std::uint32_t num_envs,
                     std::uint32_t frameskip,
                     const py::bytes& rom_bytes,
                     const std::vector<py::bytes>& snapshot_bytes_list,
                     py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>
                         env_to_level)
        : num_env_(num_envs),
          frameskip_(frameskip),
          rom_(nesle::parse_ines(bytes_to_vector(rom_bytes))),
          use_console_(true) {
        validate_basics();
        if (snapshot_bytes_list.empty()) {
            throw std::invalid_argument("snapshot_bytes_list must contain at least one snapshot");
        }
        if (snapshot_bytes_list.size() > 255) {
            throw std::invalid_argument("at most 255 snapshot levels supported per CudaBatch");
        }
        for (const auto& sb : snapshot_bytes_list) {
            const std::string raw = sb;
            snapshots_.push_back(nesle::fcs::parse(raw));
        }
        const auto view = env_to_level.request();
        if (view.ndim != 1 || static_cast<std::uint32_t>(view.shape[0]) != num_env_) {
            throw std::invalid_argument("env_to_level must have shape (num_envs,)");
        }
        std::vector<std::uint8_t> env_to_level_host(num_env_);
        const auto* src = static_cast<const std::uint8_t*>(view.ptr);
        for (std::uint32_t e = 0; e < num_env_; ++e) {
            if (src[e] >= snapshots_.size()) {
                throw std::invalid_argument(
                    "env_to_level[" + std::to_string(e) + "]=" + std::to_string(src[e]) +
                    " out of range (have " + std::to_string(snapshots_.size()) + " levels)");
            }
            env_to_level_host[e] = src[e];
        }
        allocate();
        upload_rom();
        upload_snapshot_bank(env_to_level_host);
        reset();
    }

    CudaBatchBinding(const CudaBatchBinding&) = delete;
    CudaBatchBinding& operator=(const CudaBatchBinding&) = delete;

    ~CudaBatchBinding() {
        release();
    }

    py::array_t<std::uint8_t> reset() {
        if (use_console_) {
            if (!snapshots_.empty()) {
                reset_console_from_snapshot();
            } else {
                reset_console();
            }
            render_device();
            return render();
        }

        std::vector<std::uint8_t> ram(static_cast<std::size_t>(num_env_) * nesle::cuda::kCpuRamBytes, 0);
        std::vector<int> previous_x(num_env_, 0);
        std::vector<int> previous_time(num_env_, 400);
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
        copy_to_device(device_rewards_, rewards, "reset rewards");
        copy_to_device(device_done_, done, "reset done");
        copy_to_device(device_step_counts_, step_counts, "reset step_counts");
        copy_to_device(device_ppu_ctrl_, ctrl, "reset ppu ctrl");
        copy_to_device(device_ppu_mask_, mask, "reset ppu mask");
        copy_to_device(device_palette_, palette, "reset palette");
        render_device();
        return render();
    }

    CudaDeviceArrayView reset_device() {
        if (use_console_) {
            if (!snapshots_.empty()) {
                reset_console_from_snapshot();
            } else {
                reset_console();
            }
        } else {
            reset();
        }
        check_cuda(cudaDeviceSynchronize(), "reset_device synchronize");
        return ram_device();
    }

    py::dict step(py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> actions,
                  bool render_frame = true,
                  bool copy_obs = true) {
        const auto view = actions.request();
        if (view.ndim != 1 || static_cast<std::uint32_t>(view.shape[0]) != num_env_) {
            throw std::invalid_argument("actions must have shape (num_envs,)");
        }
        check_cuda(cudaMemcpy(device_actions_,
                              view.ptr,
                              static_cast<std::size_t>(num_env_) * sizeof(std::uint8_t),
                              cudaMemcpyHostToDevice),
                   "copy actions");

        if (use_console_) {
            nesle::cuda::launch_console_step_kernel(
                buffers_,
                nesle::cuda::StepConfig{num_env_, frameskip_, false},
                max_instructions_per_frame_,
                {},
                nullptr);
            check_cuda(cudaGetLastError(), "launch_console_step_kernel");
        } else {
            constexpr int kThreads = 256;
            const auto blocks = static_cast<int>((num_env_ + kThreads - 1) / kThreads);
            apply_actions_kernel<<<blocks, kThreads>>>(
                buffers_,
                device_actions_,
                num_env_,
                frameskip_,
                device_step_counts_);
            check_cuda(cudaGetLastError(), "apply_actions_kernel");
            nesle::cuda::launch_step_kernel(buffers_, nesle::cuda::StepConfig{num_env_, frameskip_, false}, nullptr);
            check_cuda(cudaGetLastError(), "launch_step_kernel");
        }
        if (render_frame || copy_obs) {
            render_device();
        }
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
        if (copy_obs) {
            out["obs"] = render();
        }
        out["rewards"] = rewards;
        out["dones"] = dones;
        return out;
    }

    py::dict step_device(const py::object& actions, bool auto_reset = true, bool synchronize = true) {
        std::string typestr;
        const auto ptr = cuda_array_pointer(actions, num_env_, &typestr);
        if (typestr == "|u1" || typestr == "<u1") {
            check_cuda(cudaMemcpy(device_actions_,
                                  reinterpret_cast<const void*>(ptr),
                                  static_cast<std::size_t>(num_env_) * sizeof(std::uint8_t),
                                  cudaMemcpyDeviceToDevice),
                       "copy cuda uint8 actions");
        } else if (typestr == "<i8" || typestr == "|i8") {
            constexpr int kThreads = 256;
            const auto blocks = static_cast<int>((num_env_ + kThreads - 1) / kThreads);
            copy_int64_actions_kernel<<<blocks, kThreads>>>(
                device_actions_,
                reinterpret_cast<const long long*>(ptr),
                num_env_);
            check_cuda(cudaGetLastError(), "copy_int64_actions_kernel");
        } else {
            throw std::invalid_argument(
                "CUDA actions must be uint8 masks or int64 values already encoded as masks");
        }

        if (use_console_) {
            nesle::cuda::launch_console_step_kernel(
                buffers_,
                nesle::cuda::StepConfig{num_env_, frameskip_, false},
                max_instructions_per_frame_,
                {},
                nullptr);
            check_cuda(cudaGetLastError(), "launch_console_step_kernel");
        } else {
            nesle::cuda::launch_step_kernel(
                buffers_, nesle::cuda::StepConfig{num_env_, frameskip_, false}, nullptr);
            check_cuda(cudaGetLastError(), "launch_step_kernel");
        }

        check_cuda(cudaMemcpy(device_last_rewards_,
                              device_rewards_,
                              static_cast<std::size_t>(num_env_) * sizeof(float),
                              cudaMemcpyDeviceToDevice),
                   "preserve device rewards");
        check_cuda(cudaMemcpy(device_last_done_,
                              device_done_,
                              static_cast<std::size_t>(num_env_) * sizeof(std::uint8_t),
                              cudaMemcpyDeviceToDevice),
                   "preserve device dones");

        if (auto_reset) {
            if (!snapshots_.empty()) {
                nesle::cuda::launch_snapshot_reset_envs_kernel(
                    buffers_, snapshot_template_, device_last_done_, num_env_, nullptr);
                check_cuda(cudaGetLastError(), "launch_snapshot_reset_envs_kernel");
            } else {
                nesle::cuda::launch_reset_envs_kernel(
                    buffers_, device_last_done_, num_env_, use_console_, nullptr);
                check_cuda(cudaGetLastError(), "launch_reset_envs_kernel");
            }
        }
        if (synchronize) {
            check_cuda(cudaDeviceSynchronize(), "cuda device step synchronize");
        }

        py::dict out;
        out["rewards"] = rewards_device();
        out["dones"] = last_done_device();
        out["ram"] = ram_device();
        return out;
    }

    py::dict step_stats(py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> actions) {
        if (!use_console_) {
            throw std::runtime_error("step_stats is only available for CUDA console mode");
        }
        const auto view = actions.request();
        if (view.ndim != 1 || static_cast<std::uint32_t>(view.shape[0]) != num_env_) {
            throw std::invalid_argument("actions must have shape (num_envs,)");
        }
        check_cuda(cudaMemcpy(device_actions_,
                              view.ptr,
                              static_cast<std::size_t>(num_env_) * sizeof(std::uint8_t),
                              cudaMemcpyHostToDevice),
                   "copy actions");
        nesle::cuda::launch_console_step_kernel(
            buffers_,
            nesle::cuda::StepConfig{num_env_, frameskip_, false},
            max_instructions_per_frame_,
            nesle::cuda::ConsoleStepStats{
                device_stat_instructions_,
                device_stat_frames_completed_,
                device_stat_budget_hits_,
            },
            nullptr);
        check_cuda(cudaGetLastError(), "launch_console_step_stats_kernel");
        check_cuda(cudaDeviceSynchronize(), "cuda step stats synchronize");

        py::array_t<float> rewards(static_cast<py::ssize_t>(num_env_));
        py::array_t<std::uint8_t> dones(static_cast<py::ssize_t>(num_env_));
        py::array_t<std::uint64_t> instructions(static_cast<py::ssize_t>(num_env_));
        py::array_t<std::uint32_t> frames_completed(static_cast<py::ssize_t>(num_env_));
        py::array_t<std::uint32_t> budget_hits(static_cast<py::ssize_t>(num_env_));
        check_cuda(cudaMemcpy(rewards.mutable_data(),
                              device_rewards_,
                              static_cast<std::size_t>(num_env_) * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "copy stats rewards");
        check_cuda(cudaMemcpy(dones.mutable_data(),
                              device_done_,
                              static_cast<std::size_t>(num_env_) * sizeof(std::uint8_t),
                              cudaMemcpyDeviceToHost),
                   "copy stats dones");
        check_cuda(cudaMemcpy(instructions.mutable_data(),
                              device_stat_instructions_,
                              static_cast<std::size_t>(num_env_) * sizeof(std::uint64_t),
                              cudaMemcpyDeviceToHost),
                   "copy stats instructions");
        check_cuda(cudaMemcpy(frames_completed.mutable_data(),
                              device_stat_frames_completed_,
                              static_cast<std::size_t>(num_env_) * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost),
                   "copy stats frames completed");
        check_cuda(cudaMemcpy(budget_hits.mutable_data(),
                              device_stat_budget_hits_,
                              static_cast<std::size_t>(num_env_) * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost),
                   "copy stats budget hits");

        py::dict out;
        out["rewards"] = rewards;
        out["dones"] = dones;
        out["instructions"] = instructions;
        out["frames_completed"] = frames_completed;
        out["budget_hits"] = budget_hits;
        return out;
    }

    py::dict step_profile(py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> actions) {
        if (!use_console_) {
            throw std::runtime_error("step_profile is only available for CUDA console mode");
        }
        const auto view = actions.request();
        if (view.ndim != 1 || static_cast<std::uint32_t>(view.shape[0]) != num_env_) {
            throw std::invalid_argument("actions must have shape (num_envs,)");
        }
        check_cuda(cudaMemset(device_profile_opcode_counts_, 0, kOpcodeProfileBytes),
                   "clear opcode profile");
        check_cuda(cudaMemset(device_profile_pc_counts_, 0, kPcProfileBytes),
                   "clear pc profile");
        check_cuda(cudaMemcpy(device_actions_,
                              view.ptr,
                              static_cast<std::size_t>(num_env_) * sizeof(std::uint8_t),
                              cudaMemcpyHostToDevice),
                   "copy actions");
        nesle::cuda::launch_console_step_kernel(
            buffers_,
            nesle::cuda::StepConfig{num_env_, frameskip_, false},
            max_instructions_per_frame_,
            nesle::cuda::ConsoleStepStats{
                device_stat_instructions_,
                device_stat_frames_completed_,
                device_stat_budget_hits_,
                device_profile_opcode_counts_,
                device_profile_pc_counts_,
            },
            nullptr);
        check_cuda(cudaGetLastError(), "launch_console_step_profile_kernel");
        check_cuda(cudaDeviceSynchronize(), "cuda step profile synchronize");

        py::array_t<float> rewards(static_cast<py::ssize_t>(num_env_));
        py::array_t<std::uint8_t> dones(static_cast<py::ssize_t>(num_env_));
        py::array_t<std::uint64_t> instructions(static_cast<py::ssize_t>(num_env_));
        py::array_t<std::uint32_t> frames_completed(static_cast<py::ssize_t>(num_env_));
        py::array_t<std::uint32_t> budget_hits(static_cast<py::ssize_t>(num_env_));
        py::array_t<unsigned long long> opcode_counts(256);
        py::array_t<unsigned long long> pc_counts(65536);
        check_cuda(cudaMemcpy(rewards.mutable_data(),
                              device_rewards_,
                              static_cast<std::size_t>(num_env_) * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "copy profile rewards");
        check_cuda(cudaMemcpy(dones.mutable_data(),
                              device_done_,
                              static_cast<std::size_t>(num_env_) * sizeof(std::uint8_t),
                              cudaMemcpyDeviceToHost),
                   "copy profile dones");
        check_cuda(cudaMemcpy(instructions.mutable_data(),
                              device_stat_instructions_,
                              static_cast<std::size_t>(num_env_) * sizeof(std::uint64_t),
                              cudaMemcpyDeviceToHost),
                   "copy profile instructions");
        check_cuda(cudaMemcpy(frames_completed.mutable_data(),
                              device_stat_frames_completed_,
                              static_cast<std::size_t>(num_env_) * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost),
                   "copy profile frames completed");
        check_cuda(cudaMemcpy(budget_hits.mutable_data(),
                              device_stat_budget_hits_,
                              static_cast<std::size_t>(num_env_) * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost),
                   "copy profile budget hits");
        check_cuda(cudaMemcpy(opcode_counts.mutable_data(),
                              device_profile_opcode_counts_,
                              kOpcodeProfileBytes,
                              cudaMemcpyDeviceToHost),
                   "copy opcode profile");
        check_cuda(cudaMemcpy(pc_counts.mutable_data(),
                              device_profile_pc_counts_,
                              kPcProfileBytes,
                              cudaMemcpyDeviceToHost),
                   "copy pc profile");

        py::dict out;
        out["rewards"] = rewards;
        out["dones"] = dones;
        out["instructions"] = instructions;
        out["frames_completed"] = frames_completed;
        out["budget_hits"] = budget_hits;
        out["opcode_counts"] = opcode_counts;
        out["pc_counts"] = pc_counts;
        return out;
    }

    py::array_t<std::uint8_t> render() {
        // Always re-rasterize before the memcpy. Previously this was a const memcpy of
        // device_frames_, which silently returned whatever was last written by a step()
        // with render_frame=True — turning the high-throughput step(render_frame=False)
        // path into a "frozen frame" footgun. Re-rendering is one kernel launch; cheap.
        render_device();
        check_cuda(cudaDeviceSynchronize(), "render synchronize");
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

    py::array_t<std::uint8_t> ram() const {
        py::array_t<std::uint8_t> out(std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(num_env_),
            nesle::cuda::kCpuRamBytes,
        });
        check_cuda(cudaMemcpy(out.mutable_data(),
                              device_ram_,
                              static_cast<std::size_t>(num_env_) * nesle::cuda::kCpuRamBytes,
                              cudaMemcpyDeviceToHost),
                   "copy ram");
        return out;
    }

    CudaDeviceArrayView ram_device() const {
        return CudaDeviceArrayView(
            reinterpret_cast<std::uintptr_t>(device_ram_),
            {
                static_cast<py::ssize_t>(num_env_),
                nesle::cuda::kCpuRamBytes,
            },
            "|u1",
            false);
    }

    CudaDeviceArrayView rewards_device() const {
        return CudaDeviceArrayView(
            reinterpret_cast<std::uintptr_t>(device_last_rewards_),
            {static_cast<py::ssize_t>(num_env_)},
            "<f4",
            false);
    }

    CudaDeviceArrayView last_done_device() const {
        return CudaDeviceArrayView(
            reinterpret_cast<std::uintptr_t>(device_last_done_),
            {static_cast<py::ssize_t>(num_env_)},
            "|u1",
            false);
    }

    py::array_t<std::uint8_t> oam() const {
        py::array_t<std::uint8_t> out(std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(num_env_),
            nesle::cuda::kOamBytes,
        });
        check_cuda(cudaMemcpy(out.mutable_data(),
                              device_oam_,
                              static_cast<std::size_t>(num_env_) * nesle::cuda::kOamBytes,
                              cudaMemcpyDeviceToHost),
                   "copy oam");
        return out;
    }

    void reset_envs(py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> mask) {
        const auto view = mask.request();
        if (view.ndim != 1 || static_cast<std::uint32_t>(view.shape[0]) != num_env_) {
            throw std::invalid_argument("mask must have shape (num_envs,)");
        }
        check_cuda(cudaMemcpy(device_reset_mask_,
                              view.ptr,
                              static_cast<std::size_t>(num_env_) * sizeof(std::uint8_t),
                              cudaMemcpyHostToDevice),
                   "copy reset mask");
        if (!snapshots_.empty()) {
            nesle::cuda::launch_snapshot_reset_envs_kernel(
                buffers_, snapshot_template_, device_reset_mask_, num_env_, nullptr);
            check_cuda(cudaGetLastError(), "launch_snapshot_reset_envs_kernel");
        } else {
            nesle::cuda::launch_reset_envs_kernel(
                buffers_, device_reset_mask_, num_env_, use_console_, nullptr);
            check_cuda(cudaGetLastError(), "launch_reset_envs_kernel");
        }
        check_cuda(cudaDeviceSynchronize(), "reset_envs synchronize");
    }

    void poke_ram(std::uint16_t address, std::uint8_t value) {
        constexpr int kThreads = 256;
        const auto blocks = static_cast<int>((num_env_ + kThreads - 1) / kThreads);
        poke_cpu_ram_kernel<<<blocks, kThreads>>>(buffers_, num_env_, address, value);
        check_cuda(cudaGetLastError(), "poke_cpu_ram_kernel");
        check_cuda(cudaDeviceSynchronize(), "poke_ram synchronize");
    }

    std::string name() const {
        return use_console_ ? "cuda-console" : "cuda";
    }

    bool has_snapshot() const noexcept {
        return !snapshots_.empty();
    }

    std::uint32_t num_levels() const noexcept {
        return snapshot_template_.num_levels;
    }

private:
    void validate_basics() {
        if (num_env_ == 0) {
            throw std::invalid_argument("num_envs must be positive");
        }
        if (frameskip_ == 0) {
            throw std::invalid_argument("frameskip must be positive");
        }
        if (!rom_.metadata.is_nrom()) {
            throw std::invalid_argument("CUDA console mode currently supports mapper 0/NROM ROMs");
        }
        if (rom_.prg_rom.empty()) {
            throw std::invalid_argument("CUDA console mode requires PRG ROM bytes");
        }
    }

public:

private:
    void allocate() {
        device_pc_ = cuda_alloc<std::uint16_t>(num_env_, "cudaMalloc pc");
        device_a_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc a");
        device_x_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc x");
        device_y_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc y");
        device_sp_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc sp");
        device_p_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc p");
        device_cycles_ = cuda_alloc<std::uint64_t>(num_env_, "cudaMalloc cycles");
        device_cpu_nmi_pending_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc cpu nmi pending");
        device_irq_pending_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc irq pending");
        device_ram_ = cuda_alloc<std::uint8_t>(
            static_cast<std::size_t>(num_env_) * nesle::cuda::kCpuRamBytes,
            "cudaMalloc ram");
        device_prg_ram_ = cuda_alloc<std::uint8_t>(
            static_cast<std::size_t>(num_env_) * nesle::cuda::kPrgRamBytes,
            "cudaMalloc prg ram");
        device_controller_shift_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc controller shift");
        device_controller_shift_count_ = cuda_alloc<std::uint8_t>(
            num_env_,
            "cudaMalloc controller shift count");
        device_controller_strobe_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc controller strobe");
        device_pending_dma_cycles_ = cuda_alloc<std::uint32_t>(num_env_, "cudaMalloc pending dma");
        device_previous_x_ = cuda_alloc<int>(num_env_, "cudaMalloc previous_x");
        device_previous_time_ = cuda_alloc<int>(num_env_, "cudaMalloc previous_time");
        device_rewards_ = cuda_alloc<float>(num_env_, "cudaMalloc rewards");
        device_done_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc done");
        device_last_rewards_ = cuda_alloc<float>(num_env_, "cudaMalloc last rewards");
        device_last_done_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc last done");
        device_actions_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc actions");
        device_step_counts_ = cuda_alloc<std::uint32_t>(num_env_, "cudaMalloc step_counts");
        device_ppu_ctrl_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc ppu ctrl");
        device_ppu_mask_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc ppu mask");
        device_ppu_status_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc ppu status");
        device_ppu_oam_addr_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc ppu oam addr");
        device_ppu_nmi_pending_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc ppu nmi pending");
        device_ppu_scanline_ = cuda_alloc<std::int16_t>(num_env_, "cudaMalloc ppu scanline");
        device_ppu_dot_ = cuda_alloc<std::uint16_t>(num_env_, "cudaMalloc ppu dot");
        device_ppu_frame_ = cuda_alloc<std::uint64_t>(num_env_, "cudaMalloc ppu frame");
        device_ppu_v_ = cuda_alloc<std::uint16_t>(num_env_, "cudaMalloc ppu v");
        device_ppu_t_ = cuda_alloc<std::uint16_t>(num_env_, "cudaMalloc ppu t");
        device_ppu_x_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc ppu x");
        device_ppu_w_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc ppu w");
        device_ppu_open_bus_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc ppu open bus");
        device_ppu_read_buffer_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc ppu read buffer");
        device_ppu_scroll_x_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc ppu scroll x");
        device_ppu_scroll_y_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc ppu scroll y");
        device_nametable_ = cuda_alloc<std::uint8_t>(
            static_cast<std::size_t>(num_env_) * nesle::cuda::kNametableRamBytes,
            "cudaMalloc nametable");
        device_palette_ = cuda_alloc<std::uint8_t>(
            static_cast<std::size_t>(num_env_) * nesle::cuda::kPaletteRamBytes,
            "cudaMalloc palette");
        device_oam_ = cuda_alloc<std::uint8_t>(
            static_cast<std::size_t>(num_env_) * nesle::cuda::kOamBytes,
            "cudaMalloc oam");
        device_frames_ = cuda_alloc<std::uint8_t>(
            static_cast<std::size_t>(num_env_) * kFrameBytes,
            "cudaMalloc frames");
        device_reset_mask_ = cuda_alloc<std::uint8_t>(num_env_, "cudaMalloc reset mask");
        device_stat_instructions_ =
            cuda_alloc<std::uint64_t>(num_env_, "cudaMalloc stat instructions");
        device_stat_frames_completed_ =
            cuda_alloc<std::uint32_t>(num_env_, "cudaMalloc stat frames completed");
        device_stat_budget_hits_ =
            cuda_alloc<std::uint32_t>(num_env_, "cudaMalloc stat budget hits");
        device_profile_opcode_counts_ =
            cuda_alloc<unsigned long long>(256, "cudaMalloc opcode profile");
        device_profile_pc_counts_ =
            cuda_alloc<unsigned long long>(65536, "cudaMalloc pc profile");

        buffers_.cpu.pc = device_pc_;
        buffers_.cpu.a = device_a_;
        buffers_.cpu.x = device_x_;
        buffers_.cpu.y = device_y_;
        buffers_.cpu.sp = device_sp_;
        buffers_.cpu.p = device_p_;
        buffers_.cpu.cycles = device_cycles_;
        buffers_.cpu.nmi_pending = device_cpu_nmi_pending_;
        buffers_.cpu.irq_pending = device_irq_pending_;
        buffers_.cpu.ram = device_ram_;
        buffers_.cpu.prg_ram = device_prg_ram_;
        buffers_.cpu.controller1_shift = device_controller_shift_;
        buffers_.cpu.controller1_shift_count = device_controller_shift_count_;
        buffers_.cpu.controller1_strobe = device_controller_strobe_;
        buffers_.cpu.pending_dma_cycles = device_pending_dma_cycles_;
        buffers_.action_masks = device_actions_;
        buffers_.previous_mario_x = device_previous_x_;
        buffers_.previous_mario_time = device_previous_time_;
        buffers_.rewards = device_rewards_;
        buffers_.done = device_done_;
        buffers_.ppu.ctrl = device_ppu_ctrl_;
        buffers_.ppu.mask = device_ppu_mask_;
        buffers_.ppu.status = device_ppu_status_;
        buffers_.ppu.oam_addr = device_ppu_oam_addr_;
        buffers_.ppu.nmi_pending = device_ppu_nmi_pending_;
        buffers_.ppu.scanline = device_ppu_scanline_;
        buffers_.ppu.dot = device_ppu_dot_;
        buffers_.ppu.frame = device_ppu_frame_;
        buffers_.ppu.v = device_ppu_v_;
        buffers_.ppu.t = device_ppu_t_;
        buffers_.ppu.x = device_ppu_x_;
        buffers_.ppu.w = device_ppu_w_;
        buffers_.ppu.open_bus = device_ppu_open_bus_;
        buffers_.ppu.read_buffer = device_ppu_read_buffer_;
        buffers_.ppu.scroll_x = device_ppu_scroll_x_;
        buffers_.ppu.scroll_y = device_ppu_scroll_y_;
        buffers_.ppu.nametable_ram = device_nametable_;
        buffers_.ppu.palette_ram = device_palette_;
        buffers_.ppu.oam = device_oam_;
        buffers_.frames_rgb = device_frames_;
    }

    void release() noexcept {
        cudaFree(device_pc_);
        cudaFree(device_a_);
        cudaFree(device_x_);
        cudaFree(device_y_);
        cudaFree(device_sp_);
        cudaFree(device_p_);
        cudaFree(device_cycles_);
        cudaFree(device_cpu_nmi_pending_);
        cudaFree(device_irq_pending_);
        cudaFree(device_ram_);
        cudaFree(device_prg_ram_);
        cudaFree(device_controller_shift_);
        cudaFree(device_controller_shift_count_);
        cudaFree(device_controller_strobe_);
        cudaFree(device_pending_dma_cycles_);
        cudaFree(device_previous_x_);
        cudaFree(device_previous_time_);
        cudaFree(device_rewards_);
        cudaFree(device_done_);
        cudaFree(device_last_rewards_);
        cudaFree(device_last_done_);
        cudaFree(device_actions_);
        cudaFree(device_step_counts_);
        cudaFree(device_ppu_ctrl_);
        cudaFree(device_ppu_mask_);
        cudaFree(device_ppu_status_);
        cudaFree(device_ppu_oam_addr_);
        cudaFree(device_ppu_nmi_pending_);
        cudaFree(device_ppu_scanline_);
        cudaFree(device_ppu_dot_);
        cudaFree(device_ppu_frame_);
        cudaFree(device_ppu_v_);
        cudaFree(device_ppu_t_);
        cudaFree(device_ppu_x_);
        cudaFree(device_ppu_w_);
        cudaFree(device_ppu_open_bus_);
        cudaFree(device_ppu_read_buffer_);
        cudaFree(device_ppu_scroll_x_);
        cudaFree(device_ppu_scroll_y_);
        cudaFree(device_nametable_);
        cudaFree(device_palette_);
        cudaFree(device_oam_);
        cudaFree(device_prg_rom_);
        cudaFree(device_chr_rom_);
        cudaFree(device_frames_);
        cudaFree(device_reset_mask_);
        cudaFree(device_stat_instructions_);
        cudaFree(device_stat_frames_completed_);
        cudaFree(device_stat_budget_hits_);
        cudaFree(device_profile_opcode_counts_);
        cudaFree(device_profile_pc_counts_);
        cudaFree(device_snapshot_cpu_ram_);
        cudaFree(device_snapshot_prg_ram_);
        cudaFree(device_snapshot_nametable_);
        cudaFree(device_snapshot_palette_);
        cudaFree(device_snapshot_oam_);
        cudaFree(device_snap_pc_);
        cudaFree(device_snap_a_);
        cudaFree(device_snap_x_);
        cudaFree(device_snap_y_);
        cudaFree(device_snap_sp_);
        cudaFree(device_snap_p_);
        cudaFree(device_snap_cycles_);
        cudaFree(device_snap_ppu_ctrl_);
        cudaFree(device_snap_ppu_mask_);
        cudaFree(device_snap_ppu_status_);
        cudaFree(device_snap_ppu_oam_addr_);
        cudaFree(device_snap_ppu_open_bus_);
        cudaFree(device_snap_ppu_read_buffer_);
        cudaFree(device_snap_ppu_x_);
        cudaFree(device_snap_ppu_w_);
        cudaFree(device_snap_ppu_v_);
        cudaFree(device_snap_ppu_t_);
        cudaFree(device_env_to_level_);
    }

    void upload_rom() {
        if (rom_.prg_rom.empty()) {
            return;
        }
        device_prg_rom_ = cuda_alloc<std::uint8_t>(rom_.prg_rom.size(), "cudaMalloc prg rom");
        copy_to_device(device_prg_rom_, rom_.prg_rom, "copy prg rom");
        buffers_.cart.prg_rom = device_prg_rom_;
        buffers_.cart.prg_rom_size = static_cast<std::uint32_t>(rom_.prg_rom.size());
        if (!rom_.chr_rom.empty()) {
            device_chr_rom_ = cuda_alloc<std::uint8_t>(rom_.chr_rom.size(), "cudaMalloc chr rom");
            copy_to_device(device_chr_rom_, rom_.chr_rom, "copy chr rom");
            buffers_.cart.chr_rom = device_chr_rom_;
            buffers_.cart.chr_rom_size = static_cast<std::uint32_t>(rom_.chr_rom.size());
        }
        buffers_.cart.mapper = static_cast<std::uint8_t>(rom_.metadata.mapper);
        buffers_.cart.nametable_arrangement = cuda_nametable_arrangement(rom_.metadata.nametable_arrangement);
    }

    void reset_console() {
        const auto env_count = static_cast<std::size_t>(num_env_);
        std::vector<std::uint16_t> pc(env_count, reset_vector_from_prg(rom_.prg_rom));
        std::vector<std::uint8_t> bytes(env_count, 0);
        std::vector<std::uint8_t> sp(env_count, 0xFD);
        std::vector<std::uint8_t> p(env_count, 0x24);
        std::vector<std::uint64_t> cycles(env_count, 7);
        std::vector<std::uint8_t> shift_count(env_count, 8);
        std::vector<std::uint8_t> ram(env_count * nesle::cuda::kCpuRamBytes, 0);
        std::vector<std::uint8_t> prg_ram(env_count * nesle::cuda::kPrgRamBytes, 0);
        std::vector<std::uint32_t> pending_dma(env_count, 0);
        std::vector<int> previous_x(env_count, 0);
        std::vector<int> previous_time(env_count, 0);
        std::vector<float> rewards(env_count, 0.0F);
        std::vector<std::uint8_t> done(env_count, 0);
        std::vector<std::uint32_t> step_counts(env_count, 0);
        std::vector<std::int16_t> scanline(env_count, 0);
        std::vector<std::uint16_t> dot(env_count, 0);
        std::vector<std::uint64_t> frame(env_count, 0);
        std::vector<std::uint8_t> nametable(env_count * nesle::cuda::kNametableRamBytes, 0);
        std::vector<std::uint8_t> palette(env_count * nesle::cuda::kPaletteRamBytes, 0);
        std::vector<std::uint8_t> oam(env_count * nesle::cuda::kOamBytes, 0);

        copy_to_device(device_pc_, pc, "reset pc");
        copy_to_device(device_a_, bytes, "reset a");
        copy_to_device(device_x_, bytes, "reset x");
        copy_to_device(device_y_, bytes, "reset y");
        copy_to_device(device_sp_, sp, "reset sp");
        copy_to_device(device_p_, p, "reset p");
        copy_to_device(device_cycles_, cycles, "reset cycles");
        copy_to_device(device_cpu_nmi_pending_, bytes, "reset cpu nmi pending");
        copy_to_device(device_irq_pending_, bytes, "reset irq pending");
        copy_to_device(device_ram_, ram, "reset ram");
        copy_to_device(device_prg_ram_, prg_ram, "reset prg ram");
        copy_to_device(device_controller_shift_, bytes, "reset controller shift");
        copy_to_device(device_controller_shift_count_, shift_count, "reset controller shift count");
        copy_to_device(device_controller_strobe_, bytes, "reset controller strobe");
        copy_to_device(device_pending_dma_cycles_, pending_dma, "reset pending dma");
        copy_to_device(device_previous_x_, previous_x, "reset previous_x");
        copy_to_device(device_previous_time_, previous_time, "reset previous_time");
        copy_to_device(device_rewards_, rewards, "reset rewards");
        copy_to_device(device_done_, done, "reset done");
        copy_to_device(device_actions_, bytes, "reset actions");
        copy_to_device(device_step_counts_, step_counts, "reset step_counts");
        copy_to_device(device_ppu_ctrl_, bytes, "reset ppu ctrl");
        copy_to_device(device_ppu_mask_, bytes, "reset ppu mask");
        copy_to_device(device_ppu_status_, bytes, "reset ppu status");
        copy_to_device(device_ppu_oam_addr_, bytes, "reset ppu oam addr");
        copy_to_device(device_ppu_nmi_pending_, bytes, "reset ppu nmi pending");
        copy_to_device(device_ppu_scanline_, scanline, "reset ppu scanline");
        copy_to_device(device_ppu_dot_, dot, "reset ppu dot");
        copy_to_device(device_ppu_frame_, frame, "reset ppu frame");
        copy_to_device(device_ppu_v_, dot, "reset ppu v");
        copy_to_device(device_ppu_t_, dot, "reset ppu t");
        copy_to_device(device_ppu_x_, bytes, "reset ppu x");
        copy_to_device(device_ppu_w_, bytes, "reset ppu w");
        copy_to_device(device_ppu_open_bus_, bytes, "reset ppu open bus");
        copy_to_device(device_ppu_read_buffer_, bytes, "reset ppu read buffer");
        copy_to_device(device_ppu_scroll_x_, bytes, "reset ppu scroll x");
        copy_to_device(device_ppu_scroll_y_, bytes, "reset ppu scroll y");
        copy_to_device(device_nametable_, nametable, "reset nametable");
        copy_to_device(device_palette_, palette, "reset palette");
        copy_to_device(device_oam_, oam, "reset oam");
    }

    void upload_snapshot_bank(const std::vector<std::uint8_t>& env_to_level_host) {
        if (snapshots_.empty()) {
            return;
        }
        const auto n = static_cast<std::uint32_t>(snapshots_.size());

        // Bulk array buffers: num_levels * kind_bytes, contiguous.
        device_snapshot_cpu_ram_ =
            cuda_alloc<std::uint8_t>(n * nesle::cuda::kCpuRamBytes, "snap cpu_ram");
        device_snapshot_prg_ram_ =
            cuda_alloc<std::uint8_t>(n * nesle::cuda::kPrgRamBytes, "snap prg_ram");
        device_snapshot_nametable_ =
            cuda_alloc<std::uint8_t>(n * nesle::cuda::kNametableRamBytes, "snap nametable");
        device_snapshot_palette_ =
            cuda_alloc<std::uint8_t>(n * nesle::cuda::kPaletteRamBytes, "snap palette");
        device_snapshot_oam_ =
            cuda_alloc<std::uint8_t>(n * nesle::cuda::kOamBytes, "snap oam");

        // Per-level scalar arrays.
        device_snap_pc_ = cuda_alloc<std::uint16_t>(n, "snap pc");
        device_snap_a_ = cuda_alloc<std::uint8_t>(n, "snap a");
        device_snap_x_ = cuda_alloc<std::uint8_t>(n, "snap x");
        device_snap_y_ = cuda_alloc<std::uint8_t>(n, "snap y");
        device_snap_sp_ = cuda_alloc<std::uint8_t>(n, "snap sp");
        device_snap_p_ = cuda_alloc<std::uint8_t>(n, "snap p");
        device_snap_cycles_ = cuda_alloc<std::uint64_t>(n, "snap cycles");
        device_snap_ppu_ctrl_ = cuda_alloc<std::uint8_t>(n, "snap ppu_ctrl");
        device_snap_ppu_mask_ = cuda_alloc<std::uint8_t>(n, "snap ppu_mask");
        device_snap_ppu_status_ = cuda_alloc<std::uint8_t>(n, "snap ppu_status");
        device_snap_ppu_oam_addr_ = cuda_alloc<std::uint8_t>(n, "snap ppu_oam_addr");
        device_snap_ppu_open_bus_ = cuda_alloc<std::uint8_t>(n, "snap ppu_open_bus");
        device_snap_ppu_read_buffer_ = cuda_alloc<std::uint8_t>(n, "snap ppu_read_buffer");
        device_snap_ppu_x_ = cuda_alloc<std::uint8_t>(n, "snap ppu_x");
        device_snap_ppu_w_ = cuda_alloc<std::uint8_t>(n, "snap ppu_w");
        device_snap_ppu_v_ = cuda_alloc<std::uint16_t>(n, "snap ppu_v");
        device_snap_ppu_t_ = cuda_alloc<std::uint16_t>(n, "snap ppu_t");

        // Per-env level map.
        device_env_to_level_ = cuda_alloc<std::uint8_t>(num_env_, "snap env_to_level");

        // Stage host-side per-level arrays then memcpy to device.
        std::vector<std::uint16_t> h_pc(n);
        std::vector<std::uint8_t> h_a(n), h_x(n), h_y(n), h_sp(n), h_p(n);
        std::vector<std::uint64_t> h_cycles(n);
        std::vector<std::uint8_t> h_ppu_ctrl(n), h_ppu_mask(n), h_ppu_status(n);
        std::vector<std::uint8_t> h_ppu_oam_addr(n), h_ppu_open_bus(n), h_ppu_read_buffer(n);
        std::vector<std::uint8_t> h_ppu_x(n), h_ppu_w(n);
        std::vector<std::uint16_t> h_ppu_v(n), h_ppu_t(n);

        for (std::uint32_t i = 0; i < n; ++i) {
            const auto& s = snapshots_[i];
            check_cuda(cudaMemcpy(device_snapshot_cpu_ram_ + i * nesle::cuda::kCpuRamBytes,
                                  s.cpu_ram.data(), nesle::cuda::kCpuRamBytes,
                                  cudaMemcpyHostToDevice),
                       "copy snap cpu_ram slot");
            check_cuda(cudaMemcpy(device_snapshot_prg_ram_ + i * nesle::cuda::kPrgRamBytes,
                                  s.prg_ram.data(), nesle::cuda::kPrgRamBytes,
                                  cudaMemcpyHostToDevice),
                       "copy snap prg_ram slot");
            check_cuda(cudaMemcpy(device_snapshot_nametable_ + i * nesle::cuda::kNametableRamBytes,
                                  s.nametable_ram.data(), nesle::cuda::kNametableRamBytes,
                                  cudaMemcpyHostToDevice),
                       "copy snap nametable slot");
            check_cuda(cudaMemcpy(device_snapshot_palette_ + i * nesle::cuda::kPaletteRamBytes,
                                  s.palette_ram.data(), nesle::cuda::kPaletteRamBytes,
                                  cudaMemcpyHostToDevice),
                       "copy snap palette slot");
            check_cuda(cudaMemcpy(device_snapshot_oam_ + i * nesle::cuda::kOamBytes,
                                  s.oam.data(), nesle::cuda::kOamBytes,
                                  cudaMemcpyHostToDevice),
                       "copy snap oam slot");
            h_pc[i] = s.pc;
            h_a[i] = s.a; h_x[i] = s.x; h_y[i] = s.y; h_sp[i] = s.sp; h_p[i] = s.p;
            h_cycles[i] = s.cycles;
            h_ppu_ctrl[i] = s.ppu_ctrl; h_ppu_mask[i] = s.ppu_mask; h_ppu_status[i] = s.ppu_status;
            h_ppu_oam_addr[i] = s.ppu_oam_addr;
            h_ppu_open_bus[i] = s.ppu_open_bus;
            h_ppu_read_buffer[i] = s.ppu_read_buffer;
            h_ppu_x[i] = s.ppu_x; h_ppu_w[i] = s.ppu_w;
            h_ppu_v[i] = s.ppu_v; h_ppu_t[i] = s.ppu_t;
        }
        copy_to_device(device_snap_pc_, h_pc, "snap pc upload");
        copy_to_device(device_snap_a_, h_a, "snap a upload");
        copy_to_device(device_snap_x_, h_x, "snap x upload");
        copy_to_device(device_snap_y_, h_y, "snap y upload");
        copy_to_device(device_snap_sp_, h_sp, "snap sp upload");
        copy_to_device(device_snap_p_, h_p, "snap p upload");
        copy_to_device(device_snap_cycles_, h_cycles, "snap cycles upload");
        copy_to_device(device_snap_ppu_ctrl_, h_ppu_ctrl, "snap ppu_ctrl upload");
        copy_to_device(device_snap_ppu_mask_, h_ppu_mask, "snap ppu_mask upload");
        copy_to_device(device_snap_ppu_status_, h_ppu_status, "snap ppu_status upload");
        copy_to_device(device_snap_ppu_oam_addr_, h_ppu_oam_addr, "snap ppu_oam_addr upload");
        copy_to_device(device_snap_ppu_open_bus_, h_ppu_open_bus, "snap ppu_open_bus upload");
        copy_to_device(device_snap_ppu_read_buffer_, h_ppu_read_buffer, "snap ppu_read_buffer upload");
        copy_to_device(device_snap_ppu_x_, h_ppu_x, "snap ppu_x upload");
        copy_to_device(device_snap_ppu_w_, h_ppu_w, "snap ppu_w upload");
        copy_to_device(device_snap_ppu_v_, h_ppu_v, "snap ppu_v upload");
        copy_to_device(device_snap_ppu_t_, h_ppu_t, "snap ppu_t upload");
        copy_to_device(device_env_to_level_, env_to_level_host, "snap env_to_level upload");

        snapshot_template_.cpu_ram = device_snapshot_cpu_ram_;
        snapshot_template_.prg_ram = device_snapshot_prg_ram_;
        snapshot_template_.nametable_ram = device_snapshot_nametable_;
        snapshot_template_.palette_ram = device_snapshot_palette_;
        snapshot_template_.oam = device_snapshot_oam_;
        snapshot_template_.pc = device_snap_pc_;
        snapshot_template_.a = device_snap_a_;
        snapshot_template_.x = device_snap_x_;
        snapshot_template_.y = device_snap_y_;
        snapshot_template_.sp = device_snap_sp_;
        snapshot_template_.p = device_snap_p_;
        snapshot_template_.cycles = device_snap_cycles_;
        snapshot_template_.ppu_ctrl = device_snap_ppu_ctrl_;
        snapshot_template_.ppu_mask = device_snap_ppu_mask_;
        snapshot_template_.ppu_status = device_snap_ppu_status_;
        snapshot_template_.ppu_oam_addr = device_snap_ppu_oam_addr_;
        snapshot_template_.ppu_open_bus = device_snap_ppu_open_bus_;
        snapshot_template_.ppu_read_buffer = device_snap_ppu_read_buffer_;
        snapshot_template_.ppu_x = device_snap_ppu_x_;
        snapshot_template_.ppu_w = device_snap_ppu_w_;
        snapshot_template_.ppu_v = device_snap_ppu_v_;
        snapshot_template_.ppu_t = device_snap_ppu_t_;
        snapshot_template_.env_to_level = device_env_to_level_;
        snapshot_template_.num_levels = n;
    }

    void reset_console_from_snapshot() {
        // Use the snapshot-restore kernel with an all-1s mask so every env is restored
        // in a single launch — no host-side replication of snapshot arrays needed.
        std::vector<std::uint8_t> mask(num_env_, 1);
        check_cuda(cudaMemcpy(device_reset_mask_, mask.data(),
                              mask.size(), cudaMemcpyHostToDevice),
                   "upload snapshot reset mask");
        // Also clear per-step bookkeeping that the kernel doesn't touch.
        std::vector<std::uint8_t> zeros(num_env_, 0);
        std::vector<std::uint32_t> step_counts(num_env_, 0);
        copy_to_device(device_actions_, zeros, "reset actions (snapshot)");
        copy_to_device(device_step_counts_, step_counts, "reset step_counts (snapshot)");
        nesle::cuda::launch_snapshot_reset_envs_kernel(
            buffers_, snapshot_template_, device_reset_mask_, num_env_, nullptr);
        check_cuda(cudaGetLastError(), "launch_snapshot_reset_envs_kernel (initial)");
        check_cuda(cudaDeviceSynchronize(), "snapshot reset synchronize");
    }

    void render_device() const {
        nesle::cuda::launch_render_kernel(buffers_, nesle::cuda::StepConfig{num_env_, frameskip_, true}, nullptr);
        check_cuda(cudaGetLastError(), "launch_render_kernel");
    }

    std::uint32_t num_env_ = 0;
    std::uint32_t frameskip_ = 0;
    std::uint64_t max_instructions_per_frame_ = 200'000;
    nesle::RomImage rom_{};
    bool use_console_ = false;
    nesle::cuda::BatchBuffers buffers_{};
    std::uint16_t* device_pc_ = nullptr;
    std::uint8_t* device_a_ = nullptr;
    std::uint8_t* device_x_ = nullptr;
    std::uint8_t* device_y_ = nullptr;
    std::uint8_t* device_sp_ = nullptr;
    std::uint8_t* device_p_ = nullptr;
    std::uint64_t* device_cycles_ = nullptr;
    std::uint8_t* device_cpu_nmi_pending_ = nullptr;
    std::uint8_t* device_irq_pending_ = nullptr;
    std::uint8_t* device_ram_ = nullptr;
    std::uint8_t* device_prg_ram_ = nullptr;
    std::uint8_t* device_controller_shift_ = nullptr;
    std::uint8_t* device_controller_shift_count_ = nullptr;
    std::uint8_t* device_controller_strobe_ = nullptr;
    std::uint32_t* device_pending_dma_cycles_ = nullptr;
    int* device_previous_x_ = nullptr;
    int* device_previous_time_ = nullptr;
    float* device_rewards_ = nullptr;
    std::uint8_t* device_done_ = nullptr;
    float* device_last_rewards_ = nullptr;
    std::uint8_t* device_last_done_ = nullptr;
    std::uint8_t* device_actions_ = nullptr;
    std::uint32_t* device_step_counts_ = nullptr;
    std::uint8_t* device_ppu_ctrl_ = nullptr;
    std::uint8_t* device_ppu_mask_ = nullptr;
    std::uint8_t* device_ppu_status_ = nullptr;
    std::uint8_t* device_ppu_oam_addr_ = nullptr;
    std::uint8_t* device_ppu_nmi_pending_ = nullptr;
    std::int16_t* device_ppu_scanline_ = nullptr;
    std::uint16_t* device_ppu_dot_ = nullptr;
    std::uint64_t* device_ppu_frame_ = nullptr;
    std::uint16_t* device_ppu_v_ = nullptr;
    std::uint16_t* device_ppu_t_ = nullptr;
    std::uint8_t* device_ppu_x_ = nullptr;
    std::uint8_t* device_ppu_w_ = nullptr;
    std::uint8_t* device_ppu_open_bus_ = nullptr;
    std::uint8_t* device_ppu_read_buffer_ = nullptr;
    std::uint8_t* device_ppu_scroll_x_ = nullptr;
    std::uint8_t* device_ppu_scroll_y_ = nullptr;
    std::uint8_t* device_nametable_ = nullptr;
    std::uint8_t* device_palette_ = nullptr;
    std::uint8_t* device_oam_ = nullptr;
    std::uint8_t* device_prg_rom_ = nullptr;
    std::uint8_t* device_chr_rom_ = nullptr;
    std::uint8_t* device_frames_ = nullptr;
    std::uint8_t* device_reset_mask_ = nullptr;
    std::uint64_t* device_stat_instructions_ = nullptr;
    std::uint32_t* device_stat_frames_completed_ = nullptr;
    std::uint32_t* device_stat_budget_hits_ = nullptr;
    unsigned long long* device_profile_opcode_counts_ = nullptr;
    unsigned long long* device_profile_pc_counts_ = nullptr;
    static constexpr std::size_t kOpcodeProfileBytes = 256 * sizeof(unsigned long long);
    static constexpr std::size_t kPcProfileBytes = 65536 * sizeof(unsigned long long);
    std::vector<nesle::fcs::StateSnapshot> snapshots_;
    nesle::cuda::SnapshotTemplate snapshot_template_{};
    std::uint8_t* device_snapshot_cpu_ram_ = nullptr;
    std::uint8_t* device_snapshot_prg_ram_ = nullptr;
    std::uint8_t* device_snapshot_nametable_ = nullptr;
    std::uint8_t* device_snapshot_palette_ = nullptr;
    std::uint8_t* device_snapshot_oam_ = nullptr;
    std::uint16_t* device_snap_pc_ = nullptr;
    std::uint8_t* device_snap_a_ = nullptr;
    std::uint8_t* device_snap_x_ = nullptr;
    std::uint8_t* device_snap_y_ = nullptr;
    std::uint8_t* device_snap_sp_ = nullptr;
    std::uint8_t* device_snap_p_ = nullptr;
    std::uint64_t* device_snap_cycles_ = nullptr;
    std::uint8_t* device_snap_ppu_ctrl_ = nullptr;
    std::uint8_t* device_snap_ppu_mask_ = nullptr;
    std::uint8_t* device_snap_ppu_status_ = nullptr;
    std::uint8_t* device_snap_ppu_oam_addr_ = nullptr;
    std::uint8_t* device_snap_ppu_open_bus_ = nullptr;
    std::uint8_t* device_snap_ppu_read_buffer_ = nullptr;
    std::uint8_t* device_snap_ppu_x_ = nullptr;
    std::uint8_t* device_snap_ppu_w_ = nullptr;
    std::uint16_t* device_snap_ppu_v_ = nullptr;
    std::uint16_t* device_snap_ppu_t_ = nullptr;
    std::uint8_t* device_env_to_level_ = nullptr;
};

}  // namespace

PYBIND11_MODULE(_cuda_core, m) {
    m.doc() = "CUDA NeSLE batch helpers";

    py::class_<CudaDeviceArrayView>(m, "CudaDeviceArrayView")
        .def_property_readonly("__cuda_array_interface__",
                               &CudaDeviceArrayView::cuda_array_interface);

    m.def(
        "parse_fcs_state",
        [](const py::bytes& data) {
            const std::string raw = data;
            const auto snapshot = nesle::fcs::parse(raw);
            py::dict out;
            out["pc"] = snapshot.pc;
            out["a"] = snapshot.a;
            out["x"] = snapshot.x;
            out["y"] = snapshot.y;
            out["sp"] = snapshot.sp;
            out["p"] = snapshot.p;
            out["cycles"] = snapshot.cycles;
            out["ppu_ctrl"] = snapshot.ppu_ctrl;
            out["ppu_mask"] = snapshot.ppu_mask;
            out["ppu_status"] = snapshot.ppu_status;
            out["ppu_oam_addr"] = snapshot.ppu_oam_addr;
            out["ppu_open_bus"] = snapshot.ppu_open_bus;
            out["ppu_read_buffer"] = snapshot.ppu_read_buffer;
            out["ppu_x"] = snapshot.ppu_x;
            out["ppu_w"] = snapshot.ppu_w;
            out["ppu_v"] = snapshot.ppu_v;
            out["ppu_t"] = snapshot.ppu_t;
            out["cpu_ram"] = py::bytes(
                reinterpret_cast<const char*>(snapshot.cpu_ram.data()),
                snapshot.cpu_ram.size());
            out["prg_ram"] = py::bytes(
                reinterpret_cast<const char*>(snapshot.prg_ram.data()),
                snapshot.prg_ram.size());
            out["nametable_ram"] = py::bytes(
                reinterpret_cast<const char*>(snapshot.nametable_ram.data()),
                snapshot.nametable_ram.size());
            out["palette_ram"] = py::bytes(
                reinterpret_cast<const char*>(snapshot.palette_ram.data()),
                snapshot.palette_ram.size());
            out["oam"] = py::bytes(
                reinterpret_cast<const char*>(snapshot.oam.data()),
                snapshot.oam.size());
            return out;
        },
        py::arg("data"),
        "Parse an already-decompressed FCEUX FCS save state into a dict of fields.");

    py::class_<CudaBatchBinding>(m, "CudaBatch")
        .def(py::init<std::uint32_t, std::uint32_t>())
        .def(py::init<std::uint32_t, std::uint32_t, const py::bytes&>())
        .def(py::init<std::uint32_t, std::uint32_t, const py::bytes&, const py::bytes&>(),
             py::arg("num_envs"),
             py::arg("frameskip"),
             py::arg("rom_bytes"),
             py::arg("snapshot_bytes"))
        .def(py::init<std::uint32_t, std::uint32_t, const py::bytes&,
                      const std::vector<py::bytes>&,
                      py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>>(),
             py::arg("num_envs"),
             py::arg("frameskip"),
             py::arg("rom_bytes"),
             py::arg("snapshot_bytes_list"),
             py::arg("env_to_level"))
        .def("reset", &CudaBatchBinding::reset)
        .def("reset_device", &CudaBatchBinding::reset_device)
        .def("step",
             &CudaBatchBinding::step,
             py::arg("actions"),
             py::arg("render_frame") = true,
             py::arg("copy_obs") = true)
        .def("step_device",
             &CudaBatchBinding::step_device,
             py::arg("actions"),
             py::arg("auto_reset") = true,
             py::arg("synchronize") = true)
        .def("step_stats", &CudaBatchBinding::step_stats, py::arg("actions"))
        .def("step_profile", &CudaBatchBinding::step_profile, py::arg("actions"))
        .def("render", &CudaBatchBinding::render)
        .def("ram", &CudaBatchBinding::ram)
        .def("ram_device", &CudaBatchBinding::ram_device)
        .def("oam", &CudaBatchBinding::oam)
        .def("reset_envs", &CudaBatchBinding::reset_envs, py::arg("mask"))
        .def("poke_ram", &CudaBatchBinding::poke_ram, py::arg("address"), py::arg("value"))
        .def_property_readonly("name", &CudaBatchBinding::name)
        .def_property_readonly("has_snapshot",
                               [](const CudaBatchBinding& self) { return self.has_snapshot(); })
        .def_property_readonly("num_levels",
                               [](const CudaBatchBinding& self) { return self.num_levels(); });
}
