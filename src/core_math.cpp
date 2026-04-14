#include "core_math.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <Eigen/Dense>

namespace rra::core_math {

/**
 * @brief Optimized ReLU-style activation with clipping.
 * Eigen's .cwiseMax/Min is highly vectorized.
 */
void vector_activation(float* values, std::size_t size, float clip_value) {
    if (!values || size == 0) return;
    
    float limit = (clip_value > 0.0f) ? clip_value : 10.0f;

    Eigen::Map<Eigen::ArrayXf> arr(values, size);
    // fused in-place operation
    arr = arr.max(0.0f).min(limit);
}

/**
 * @brief Fused LIF Kernel. 
 * Optimized to avoid intermediate temporaries using Eigen's .select()
 */
void fused_lif_kernel(
    const float* __restrict input,
    const float* __restrict prev_state,
    float* __restrict next_state,
    float* __restrict spikes,
    std::size_t size,
    float decay,
    float threshold
) {
    if (!input || !prev_state || !next_state || !spikes || size == 0) return;

    const float d = std::clamp(decay, 0.0f, 1.0f);

    for (std::size_t i = 0; i < size; ++i) {
        float mem = (prev_state[i] * d) + input[i];
        bool spiked = mem >= threshold;
        spikes[i] = static_cast<float>(spiked);
        next_state[i] = spiked ? 0.0f : mem;
    }
}

/**
 * @brief Adaptive Fused LIF Kernel.
 */
void fused_lif_kernel_adaptive(
    const float* __restrict input,
    const float* __restrict prev_state,
    float* __restrict next_state,
    float* __restrict spikes,
    const float* __restrict thresholds,
    std::size_t size,
    float decay
) {
    if (!input || !prev_state || !next_state || !spikes || !thresholds || size == 0) return;

    const float d = std::clamp(decay, 0.0f, 1.0f);

    for (std::size_t i = 0; i < size; ++i) {
        float mem = (prev_state[i] * d) + input[i];
        bool spiked = mem >= thresholds[i];
        spikes[i] = static_cast<float>(spiked);
        next_state[i] = spiked ? 0.0f : mem;
    }
}

/**
 * @brief Faster BF16 conversion using bit-casting.
 */
float bf16_to_f32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16U;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

/**
 * @brief In-place RMS Norm.
 * Pre-calculates the scaling factor to perform one multiplication per element.
 */
void rms_norm_inplace(
    float* x,
    std::size_t total_size,
    std::size_t rows,
    std::size_t cols,
    float eps
) {
    if (!x || rows == 0 || cols == 0 || total_size < (rows * cols)) return;
    
    float eps_is_finite = static_cast<float>(eps * 0.0f == 0.0f);
    const float safe_eps = (eps * eps_is_finite) + (1e-5f * (1.0f - eps_is_finite));
    const float epsilon = std::max(1e-12f, safe_eps);

    for (std::size_t r = 0; r < rows; ++r) {
        Eigen::Map<Eigen::ArrayXf> row_arr(x + (r * cols), cols);
        
        float mean_sq = row_arr.square().mean();
        
        // Standard RMS Norm (Removed Discontinuous Gated Logic)
        float scale = 1.0f / std::sqrt(mean_sq + epsilon);
        row_arr *= scale;
    }
}

/**
 * @brief Sparsity check with NaN safety.
 */
double calculate_sparsity(const std::vector<float>& x, float threshold) {
    if (x.empty()) return 1.0;

    const float th = std::max(0.0f, std::isfinite(threshold) ? threshold : 1e-7f);
    std::size_t zero_like = 0;
    
    // Using a pointer for faster traversal in non-simd environments
    const float* data = x.data();
    const std::size_t n = x.size();

    for (std::size_t i = 0; i < n; ++i) {
        float val = data[i];
        // Branchless check for finiteness over threshold
        bool is_finite = std::isfinite(val);
        bool is_zero = (std::abs(val) <= th);
        zero_like += static_cast<std::size_t>(!is_finite || is_zero);
    }
    return static_cast<double>(zero_like) / static_cast<double>(n);
}

} // namespace rra::core_math