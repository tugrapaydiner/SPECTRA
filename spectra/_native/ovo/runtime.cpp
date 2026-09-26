// Public bridge. The retained numerical implementation is included unchanged.
#include "exact_tables_20260926/engine.cpp"
#include <cfenv>

extern "C" {
int sp_svm_abi() { return 1; }

// All ABI pointers must refer to valid nonoverlapping caller-owned buffers.
// Sessions are mutable: callers must serialize these functions and destruction.
int sp_svm_batch(void* handle, const float* inputs, int rows, int features,
                 int schedule, int hint, int* output, int capacity) {
    return lo::protect([&] {
        lo::require(handle, "null session");
        auto& engine = *static_cast<et::Engine*>(handle);
        // Even a rejected call cannot leave an earlier request's certificate live.
        engine.valid_certificate = false;
        lo::require(rows >= 0 && rows <= 65536 && features == 16 && capacity == rows,
                    "invalid batch geometry");
        lo::require(schedule >= 0 && schedule <= 5 && hint >= -1 && hint < engine.c,
                    "invalid batch schedule or hint");
        lo::require(std::fegetround() == FE_TONEAREST, "round-to-nearest required");
        lo::require(rows == 0 || (inputs && output), "null batch buffers");
        // Validate the entire input before writing any output, including late NaNs.
        for (size_t i = 0; i < size_t(rows) * 16; ++i)
            lo::require(std::isfinite(inputs[i]), "nonfinite batch input");
        for (int row = 0; row < rows; ++row) {
            try {
                output[row] = engine.run2(inputs + size_t(row) * 16, 16, schedule, hint);
            } catch (...) {
                engine.valid_certificate = false;
                throw;
            }
        }
        // A batch returns classes, not a certificate implicitly tied to its last row.
        engine.valid_certificate = false;
    });
}

int sp_svm_single(void* handle, const float* input, int features, int schedule,
                  int hint, int* output, uint64_t* stats, int capacity) {
    if (handle) static_cast<et::Engine*>(handle)->valid_certificate = false;
    if (std::fegetround() != FE_TONEAREST)
        return lo::protect([] { throw std::invalid_argument("round-to-nearest required"); });
    return et_run(handle, input, features, schedule, hint, output, stats, capacity);
}
}

// New ABI is additive. Existing Session and historical engine symbols are intact.
#include "shared.hpp"
