/// \file model_load_cancel.cpp
/// \brief Cooperative model-load cancellation and Linux PLT hooks for prebuilt NPU/Q4NX libraries.
#include "utils/model_load_cancel.hpp"

#include <atomic>
#include <cstddef>
#include <string>

#if !defined(_WIN32)
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include "tensor_utils/safe_tensors.hpp"
#include "utils/debug_utils.hpp"
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#endif

namespace flm {

namespace {
std::atomic<bool> g_load_cancel_requested{false};
std::atomic<uint64_t> g_readahead_generation{0};
thread_local bool tl_inside_model_load = false;
thread_local const std::atomic<bool>* tl_token_cancelled = nullptr;

#if !defined(_WIN32)
constexpr size_t kLargeModelThresholdBytes = 4ULL * 1024ULL * 1024ULL * 1024ULL; // 4 GiB
constexpr size_t kPageSize = 4096;

size_t align_page_down(size_t value) noexcept {
    return value & ~(kPageSize - 1);
}

std::string resolve_q4nx_file_path(const std::string& path_or_dir) {
    std::error_code ec;
    std::filesystem::path p(path_or_dir);
    if (std::filesystem::is_directory(p, ec)) {
        p /= "model.q4nx";
    }
    return p.string();
}

struct MmapWeightFile {
    int fd = -1;
    const uint8_t* base = nullptr;
    size_t file_size = 0;
    size_t half_size = 0;
    size_t indexed_tensor_count = 0;
    std::unordered_map<std::string, size_t> tensor_index;

    ~MmapWeightFile() {
        if (base && base != MAP_FAILED && file_size > 0) {
            munmap(const_cast<uint8_t*>(base), file_size);
        }
        if (fd >= 0) {
            close(fd);
        }
    }
};

std::mutex g_mmap_mutex;
std::unordered_map<std::string, std::shared_ptr<MmapWeightFile>> g_mmap_files;

std::shared_ptr<MmapWeightFile> get_or_open_mmap_file(const std::string& raw_path) {
    const std::string file_path = resolve_q4nx_file_path(raw_path);
    if (file_path.empty()) return nullptr;

    std::lock_guard<std::mutex> lock(g_mmap_mutex);
    auto it = g_mmap_files.find(file_path);
    if (it != g_mmap_files.end()) {
        return it->second;
    }

    int fd = open(file_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return nullptr;
    }
    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        return nullptr;
    }
    const size_t sz = static_cast<size_t>(st.st_size);
    void* mapped = mmap(nullptr, sz, PROT_READ, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        close(fd);
        return nullptr;
    }

    auto entry = std::make_shared<MmapWeightFile>();
    entry->fd = fd;
    entry->base = static_cast<const uint8_t*>(mapped);
    entry->file_size = sz;
    entry->half_size = align_page_down(sz / 2);
    g_mmap_files.emplace(file_path, entry);
    return entry;
}
#endif
} // namespace

void prewarm_half_model_weight_file(const std::string& model_dir) noexcept {
#if !defined(_WIN32)
    try {
        auto entry = get_or_open_mmap_file(model_dir);
        if (!entry || entry->half_size == 0) return;

        // Advise kernel that the first 50% will be needed and drop the second 50% if >= 4 GiB.
        posix_fadvise(entry->fd, 0, static_cast<off_t>(entry->half_size), POSIX_FADV_WILLNEED);
        madvise(const_cast<uint8_t*>(entry->base), entry->half_size, MADV_WILLNEED);
        if (entry->file_size >= kLargeModelThresholdBytes && entry->file_size > entry->half_size) {
            posix_fadvise(
                entry->fd,
                static_cast<off_t>(entry->half_size),
                static_cast<off_t>(entry->file_size - entry->half_size),
                POSIX_FADV_DONTNEED);
        }

        // Touch pages in the first 50% so they are faulted into active RAM page cache.
        volatile uint64_t sink = 0;
        constexpr size_t kStride = 64 * 1024;
        for (size_t offset = 0; offset < entry->half_size; offset += kStride) {
            sink += entry->base[offset];
        }
        (void)sink;
        header_print("FLM", "Prewarmed first 50% (" << (entry->half_size / (1024 * 1024))
                            << " MiB / " << (entry->file_size / (1024 * 1024))
                            << " MiB) of " << resolve_q4nx_file_path(model_dir) << " in RAM cache");
    } catch (...) {
    }
#else
    (void)model_dir;
#endif
}

void start_second_half_readahead(const std::string& model_dir) noexcept {
#if !defined(_WIN32)
    try {
        auto entry = get_or_open_mmap_file(model_dir);
        if (!entry || entry->file_size <= entry->half_size) return;

        const uint64_t gen = g_readahead_generation.fetch_add(1, std::memory_order_acq_rel) + 1;
        std::thread([entry, gen]() {
            const size_t start = entry->half_size;
            const size_t len = entry->file_size - start;
            posix_fadvise(entry->fd, static_cast<off_t>(start), static_cast<off_t>(len), POSIX_FADV_WILLNEED);
            madvise(const_cast<uint8_t*>(entry->base + start), len, MADV_WILLNEED);

            volatile uint64_t sink = 0;
            constexpr size_t kChunk = 4 * 1024 * 1024; // 4 MiB slices
            constexpr size_t kStep = 64 * 1024;
            for (size_t pos = start; pos < entry->file_size; pos += kChunk) {
                if (g_load_cancel_requested.load(std::memory_order_acquire) ||
                    g_readahead_generation.load(std::memory_order_acquire) != gen) {
                    return;
                }
                const size_t end = std::min(entry->file_size, pos + kChunk);
                for (size_t off = pos; off < end; off += kStep) {
                    sink += entry->base[off];
                }
            }
            (void)sink;
        }).detach();
    } catch (...) {
    }
#else
    (void)model_dir;
#endif
}

void trim_second_half_page_cache(const std::string& model_dir) noexcept {
#if !defined(_WIN32)
    try {
        auto entry = get_or_open_mmap_file(model_dir);
        if (!entry || entry->file_size < kLargeModelThresholdBytes || entry->file_size <= entry->half_size) {
            return;
        }
        // Invalidate in-flight readahead and release the second 50% back to the OS.
        g_readahead_generation.fetch_add(1, std::memory_order_acq_rel);
        const size_t start = entry->half_size;
        const size_t len = entry->file_size - start;
        madvise(const_cast<uint8_t*>(entry->base + start), len, MADV_DONTNEED);
        posix_fadvise(entry->fd, static_cast<off_t>(start), static_cast<off_t>(len), POSIX_FADV_DONTNEED);
    } catch (...) {
    }
#else
    (void)model_dir;
#endif
}

void request_model_load_cancel() noexcept {
    g_load_cancel_requested.store(true, std::memory_order_release);
}

bool is_inside_model_load() noexcept {
    return tl_inside_model_load;
}

bool is_model_load_cancelled() noexcept {
    if (!tl_inside_model_load) {
        return false;
    }
    if (g_load_cancel_requested.load(std::memory_order_acquire)) {
        return true;
    }
    if (tl_token_cancelled && tl_token_cancelled->load(std::memory_order_acquire)) {
        return true;
    }
    return false;
}

ScopedModelLoadCancelContext::ScopedModelLoadCancelContext(
    const std::atomic<bool>* token_cancelled) noexcept
    : prev_inside_(tl_inside_model_load), prev_token_(tl_token_cancelled) {
    g_load_cancel_requested.store(false, std::memory_order_release);
    tl_token_cancelled = token_cancelled;
    tl_inside_model_load = true;
}

ScopedModelLoadCancelContext::~ScopedModelLoadCancelContext() noexcept {
    tl_inside_model_load = prev_inside_;
    tl_token_cancelled = prev_token_;
    if (!prev_inside_) {
        g_load_cancel_requested.store(false, std::memory_order_release);
    }
}

} // namespace flm

#if !defined(_WIN32)
// Interpose weight loading, BO syncing, and xclbin registration while a model load is
// active so prebuilt lib*_npu.so engines abort cleanly between tensors/BOs instead of
// blocking the server for minutes when a different request arrives or the client disconnects.
// Also serves tensor reads directly from a cached read-only mmap + O(1) hash index to avoid
// O(N^2) linear scans and per-tensor ifstream seekg/read syscalls.

namespace {
bool try_mmap_load_weights(
    const std::string& model_path,
    const std::vector<tensor_metadata>& tensors_data,
    bytes& weight_buffer,
    const std::string& weights_name,
    size_t byte_offset,
    bool allow_auto_resize,
    std::string& out_dtype) {
    if (model_path.empty() || tensors_data.empty()) return false;
    auto entry = flm::get_or_open_mmap_file(model_path);
    if (!entry || !entry->base || entry->file_size == 0) return false;

    size_t tensor_idx = 0;
    {
        std::lock_guard<std::mutex> lock(flm::g_mmap_mutex);
        if (entry->indexed_tensor_count != tensors_data.size()) {
            entry->tensor_index.clear();
            entry->tensor_index.reserve(tensors_data.size());
            for (size_t i = 0; i < tensors_data.size(); ++i) {
                entry->tensor_index.emplace(tensors_data[i].name, i);
            }
            entry->indexed_tensor_count = tensors_data.size();
        }
        auto it = entry->tensor_index.find(weights_name);
        if (it == entry->tensor_index.end()) {
            return false;
        }
        tensor_idx = it->second;
    }

    if (tensor_idx >= tensors_data.size()) return false;
    const tensor_metadata& meta = tensors_data[tensor_idx];
    if (meta.offsets.empty()) return false;
    const size_t file_offset = meta.offsets[0];
    const size_t byte_size = meta.byte_size;
    if (file_offset > entry->file_size || byte_size > entry->file_size - file_offset) {
        return false;
    }

    if (allow_auto_resize) {
        if (weight_buffer.size() == 0) {
            if (byte_size == 0) return false;
            weight_buffer.resize(byte_size);
        } else if (weight_buffer.size() != byte_size) {
            return false;
        }
    } else {
        if (weight_buffer.size() == 0 || byte_offset > weight_buffer.size() ||
            byte_size > weight_buffer.size() - byte_offset) {
            return false;
        }
    }

    uint8_t* dst = weight_buffer.data() + byte_offset;
    const uint8_t* src = entry->base + file_offset;
    constexpr size_t kCopySlice = 16 * 1024 * 1024; // 16 MiB slices for fast cancellation checks
    size_t copied = 0;
    while (copied < byte_size) {
        if (flm::is_model_load_cancelled()) {
            throw flm::ModelLoadCancelledError("Model load cancelled during weight read");
        }
        const size_t n = std::min(kCopySlice, byte_size - copied);
        std::memcpy(dst + copied, src + copied, n);
        copied += n;
    }
    if (flm::is_model_load_cancelled()) {
        throw flm::ModelLoadCancelledError("Model load cancelled during weight read");
    }
    out_dtype = meta.dtype;
    return true;
}
} // namespace

std::string SafeTensors::load_weights(bytes& weight_buffer, std::string weights_name) {
    if (flm::is_model_load_cancelled()) {
        throw flm::ModelLoadCancelledError("Model load cancelled during weight read");
    }
    std::string dtype;
    if (try_mmap_load_weights(this->model_path, this->tensors_data, weight_buffer, weights_name, 0, true, dtype)) {
        return dtype;
    }
    using Fn = std::string (*)(SafeTensors*, bytes&, std::string);
    static Fn real_fn = reinterpret_cast<Fn>(dlsym(
        RTLD_NEXT,
        "_ZN11SafeTensors12load_weightsER5bytesNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE"));
    if (!real_fn) {
        throw std::runtime_error("Failed to resolve SafeTensors::load_weights");
    }
    std::string result = real_fn(this, weight_buffer, std::move(weights_name));
    if (flm::is_model_load_cancelled()) {
        throw flm::ModelLoadCancelledError("Model load cancelled during weight read");
    }
    return result;
}

std::string SafeTensors::load_weights(bytes& weight_buffer, std::string weights_name, size_t byte_offset) {
    if (flm::is_model_load_cancelled()) {
        throw flm::ModelLoadCancelledError("Model load cancelled during weight read");
    }
    std::string dtype;
    if (try_mmap_load_weights(this->model_path, this->tensors_data, weight_buffer, weights_name, byte_offset, false, dtype)) {
        return dtype;
    }
    using Fn = std::string (*)(SafeTensors*, bytes&, std::string, size_t);
    static Fn real_fn = reinterpret_cast<Fn>(dlsym(
        RTLD_NEXT,
        "_ZN11SafeTensors12load_weightsER5bytesNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEm"));
    if (!real_fn) {
        throw std::runtime_error("Failed to resolve SafeTensors::load_weights(offset)");
    }
    std::string result = real_fn(this, weight_buffer, std::move(weights_name), byte_offset);
    if (flm::is_model_load_cancelled()) {
        throw flm::ModelLoadCancelledError("Model load cancelled during weight read");
    }
    return result;
}

void xrt::bo::sync(xclBOSyncDirection dir, size_t sz, size_t offset) {
    if (flm::is_model_load_cancelled()) {
        throw flm::ModelLoadCancelledError("Model load cancelled during BO sync");
    }
    using Fn = void (*)(xrt::bo*, xclBOSyncDirection, size_t, size_t);
    static Fn real_fn = reinterpret_cast<Fn>(dlsym(
        RTLD_NEXT, "_ZN3xrt2bo4syncE18xclBOSyncDirectionmm"));
    if (!real_fn) {
        throw std::runtime_error("Failed to resolve xrt::bo::sync");
    }
    real_fn(this, dir, sz, offset);
    if (flm::is_model_load_cancelled()) {
        throw flm::ModelLoadCancelledError("Model load cancelled during BO sync");
    }
}

xrt::uuid xrt::device::register_xclbin(const xrt::xclbin& xclbin) {
    if (flm::is_model_load_cancelled()) {
        throw flm::ModelLoadCancelledError("Model load cancelled during xclbin registration");
    }
    using Fn = xrt::uuid (*)(xrt::device*, const xrt::xclbin&);
    static Fn real_fn = reinterpret_cast<Fn>(dlsym(
        RTLD_NEXT, "_ZN3xrt6device15register_xclbinERKNS_6xclbinE"));
    if (!real_fn) {
        throw std::runtime_error("Failed to resolve xrt::device::register_xclbin");
    }
    xrt::uuid result = real_fn(this, xclbin);
    if (flm::is_model_load_cancelled()) {
        throw flm::ModelLoadCancelledError("Model load cancelled during xclbin registration");
    }
    return result;
}
#endif
