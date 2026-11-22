/// \file model_load_cancel.cpp
/// \brief Cooperative model-load cancellation and Linux PLT hooks for prebuilt NPU/Q4NX libraries.
#include "utils/model_load_cancel.hpp"

#include <atomic>
#include <cstddef>
#include <string>

#if !defined(_WIN32)
#include <dlfcn.h>
#include "tensor_utils/safe_tensors.hpp"
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#endif

namespace flm {

namespace {
std::atomic<bool> g_load_cancel_requested{false};
thread_local bool tl_inside_model_load = false;
thread_local const std::atomic<bool>* tl_token_cancelled = nullptr;
} // namespace

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

std::string SafeTensors::load_weights(bytes& weight_buffer, std::string weights_name) {
    if (flm::is_model_load_cancelled()) {
        throw flm::ModelLoadCancelledError("Model load cancelled during weight read");
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
