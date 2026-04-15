#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <span>
#include <concepts>

namespace rra::core_math {

/**
 * @brief C++20 Concepts for Neural Math constraints.
 */
template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template <typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

/**
 * @brief Universal Attractor Function (Linear Interpolation / EMA)
 */
template <FloatingPoint T>
inline T apply_attractor(T current, T target, T force) {
    // NaN Guard: If current is corrupt, snap to target immediately
    T is_finite = static_cast<T>(current * 0.0f == 0.0f);
    T val = (current * is_finite) + (target * (1.0 - is_finite));
    return val + force * (target - val);
}

template <Arithmetic T>
inline T clamp(T value, T lo, T hi) {
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

[[nodiscard]] double calculate_sparsity(std::span<const float> x, float threshold = 1e-7f);

/**
 * @brief High-Performance Dot Product with AVX-512.
 */
float dot_product(const float* a, const float* b, std::size_t size);

/**
 * @brief Softmax on a row of scores with AVX-512 stabilization.
 */
void softmax_inplace(float* x, std::size_t size);


/**
 * @brief High-Performance Causal Conv1d.
 */
void causal_conv1d(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    std::size_t size,
    std::size_t kernel_size
);

} // namespace rra::core_math
