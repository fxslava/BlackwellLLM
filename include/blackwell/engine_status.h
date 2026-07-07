#pragma once
// The Hybrid Error Doctrine's status vocabulary (Roadmap #2).
//
// Two error tiers, split by phase:
//   INIT tier (ctors / factories / setup): C++ exceptions. Errors there are
//     rare (VRAM exhaustion at load), RAII members make a throwing ctor
//     leak-free, and the DLL boundary maps them to HRESULTs.
//   RUNTIME tier (decode hot loop, per-token lifecycle): zero-cost status
//     codes -- no throw, no unwind-latency spike mid-generation. CUDA failures
//     become EngineStatus values that ripple up by return value and are
//     translated ONCE: to HRESULT at the COM boundary, or to an engine_error
//     by the exception-tier C++ facade wrappers (white-box consumers).
//
// This header is deliberately CUDA-free and windows.h-free: the status enum is
// part of the white-box C++ surface; the HRESULT mapping lives with the
// boundary (src/core/engine_com.cpp).
#include <cstdint>
#include <stdexcept>
#include <string>

namespace blackwell {

enum class EngineStatus : int32_t {
    Success          = 0,
    CudaRuntimeError = 1,  // CUDA API / kernel failure inside the decode path
    OutOfVram        = 2,  // device allocation failure (cudaErrorMemoryAllocation)
    InvalidArgument  = 3,  // caller error: pos/seq_id outside the resolved plan
    InvalidConfig    = 4,  // checkpoint/plan integrity violation discovered late
    StateMismatch    = 5,  // operation illegal in the current engine state
};

inline const char* to_string(EngineStatus status) noexcept {
    switch (status) {
        case EngineStatus::Success:          return "Success";
        case EngineStatus::CudaRuntimeError: return "CudaRuntimeError";
        case EngineStatus::OutOfVram:        return "OutOfVram";
        case EngineStatus::InvalidArgument:  return "InvalidArgument";
        case EngineStatus::InvalidConfig:    return "InvalidConfig";
        case EngineStatus::StateMismatch:    return "StateMismatch";
    }
    return "UnknownEngineStatus";
}

// Carries an EngineStatus across an exception-tier surface. The runtime hot
// loop NEVER raises this (it is status-only, end to end); the remaining
// throwers are init/admin surfaces -- e.g. the coordinator's AOT-warmup facet
// (EnginePrefillCoordinator::prefill), whose offline compilation contract is
// exception-based. Derives from std::runtime_error to match existing catch sites.
class engine_error : public std::runtime_error {
public:
    engine_error(EngineStatus status, const std::string& what_arg)
        : std::runtime_error(what_arg), status_(status) {}
    EngineStatus status() const noexcept { return status_; }

private:
    EngineStatus status_;
};

} // namespace blackwell
