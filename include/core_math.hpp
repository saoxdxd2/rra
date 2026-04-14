#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rra::core_math {

/**
 * @brief Universal Attractor Function (Linear Interpolation / EMA)
 * Pulls 'current' towards 'target' by a 'force' coefficient [0.0, 1.0].
 * Optimizes to a single FMA (Fused Multiply-Add) instruction.
 */
inline float apply_attractor(float current, float target, float force) {
    // NaN Guard: If current is corrupt, snap to target immediately
    float is_finite = static_cast<float>(current * 0.0f == 0.0f);
    float val = (current * is_finite) + (target * (1.0f - is_finite));
    return val + force * (target - val);
}

template <typename T>
inline T clamp(T value, T lo, T hi) {
    // Branchless clamp using min/max which compile to CMOV on x64
    return (std::min)((std::max)(value, lo), hi);
}

inline double ema(double prev, double sample, double decay, bool initialized) {
    // Branchless initialization: if not initialized, w1=1.0, w2=0.0
    // else w1=decay, w2=(1.0-decay)
    double w1 = initialized ? decay : 1.0;
    double w2 = initialized ? (1.0 - decay) : 0.0;
    return (w1 * sample) + (w2 * prev);
}

void vector_activation(float* values, std::size_t size, float clip_value = 10.0f);

// Canonical lowercase name — AVX-512 LIF kernel.
void fused_lif_kernel(
    const float* input,
    const float* prev_state,
    float* next_state,
    float* spikes,
    std::size_t size,
    float decay,
    float threshold
);

/**
 * @brief Adaptive LIF kernel with per-neuron thresholds.
 */
void fused_lif_kernel_adaptive(
    const float* input,
    const float* prev_state,
    float* next_state,
    float* spikes,
    const float* thresholds,
    std::size_t size,
    float decay
);


[[nodiscard]] float bf16_to_f32(uint16_t b);

void rms_norm_inplace(
    float* x,
    std::size_t total_size,
    std::size_t rows,
    std::size_t cols,
    float eps
);

[[nodiscard]] double calculate_sparsity(const std::vector<float>& x, float threshold = 1e-7f);

} // namespace rra::core_math
