#include "cafe.hpp"
#include <numeric>
#include <cmath>
#include <algorithm>

namespace s4m::cafe {

void CafeField::apply_optimal_transport(float total_mass) {
    size_t num_chunks = chunks_.size();
    if (num_chunks == 0) return;

    std::vector<float> demands(num_chunks);
    float sum_demand = 0.0f;

    // 1. Calculate Target Demand based on prediction error gradients
    // μ evolves via Wasserstein transport on error field: μ_i <- normalize(exp(error_i))
    for (size_t i = 0; i < num_chunks; ++i) {
        // energy_gradient represents the local prediction error (CrossEntropy loss)
        // Bound the exponent to prevent overflow
        float error_i = std::min(10.0f, chunks_[i].energy_gradient);
        float d = std::exp(error_i);
        
        demands[i] = d;
        sum_demand += d;
    }

    // 2. Normalize and exact 1D Transport
    // Setting the normalized demand directly ensures the spatial CDF of \mu 
    // perfectly matches the target CDF derived from the demand.
    for (size_t i = 0; i < num_chunks; ++i) {
        chunks_[i].mu = total_mass * (demands[i] / sum_demand);
    }
}

s4m::Tensor CafeField::to_tensor() const {
    size_t seq_len = chunks_.size();
    size_t d_model = CHUNK_SIZE * 16; // 32 * 16 = 512
    s4m::Tensor t({seq_len, d_model});
    
    for (size_t i = 0; i < seq_len; ++i) {
        float* dst = t.ptr() + i * d_model;
        for (size_t j = 0; j < CHUNK_SIZE; ++j) {
            _mm512_storeu_ps(dst + j * 16, chunks_[i].state[j].data);
        }
    }
    return t;
}

void CafeField::apply_delta(const s4m::Tensor& delta) {
    size_t seq_len = chunks_.size();
    size_t d_model = CHUNK_SIZE * 16;
    
    for (size_t i = 0; i < seq_len; ++i) {
        const float* src = delta.ptr() + i * d_model;
        for (size_t j = 0; j < CHUNK_SIZE; ++j) {
            __m512 d = _mm512_loadu_ps(src + j * 16);
            chunks_[i].state[j].data = _mm512_add_ps(chunks_[i].state[j].data, d);
        }
    }
}

} // namespace s4m::cafe
