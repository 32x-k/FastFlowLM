/// \file model_load_cancel.hpp
/// \brief Cooperative cancellation for in-progress model downloads and NPU model loads.
#pragma once

#include <atomic>
#include <stdexcept>
#include <string>

namespace flm {

class ModelLoadCancelledError : public std::runtime_error {
public:
    explicit ModelLoadCancelledError(const std::string& msg = "Model load cancelled")
        : std::runtime_error(msg) {}
};

/// Signal the active model load (if any) to abort as soon as possible.
void request_model_load_cancel() noexcept;

/// True when the active model load on this thread has been cancelled either by a
/// newer competing request or because the originating client disconnected.
bool is_model_load_cancelled() noexcept;

/// True while this thread is inside a ScopedModelLoadCancelContext.
bool is_inside_model_load() noexcept;

/// Throw ModelLoadCancelledError if the active model load was cancelled.
inline void throw_if_model_load_cancelled() {
    if (is_model_load_cancelled()) {
        throw ModelLoadCancelledError("Model load cancelled");
    }
}

/// RAII scope active for the duration of a single model download + NPU load.
class ScopedModelLoadCancelContext {
public:
    explicit ScopedModelLoadCancelContext(const std::atomic<bool>* token_cancelled) noexcept;
    ~ScopedModelLoadCancelContext() noexcept;

    ScopedModelLoadCancelContext(const ScopedModelLoadCancelContext&) = delete;
    ScopedModelLoadCancelContext& operator=(const ScopedModelLoadCancelContext&) = delete;

private:
    bool prev_inside_;
    const std::atomic<bool>* prev_token_;
};

} // namespace flm
