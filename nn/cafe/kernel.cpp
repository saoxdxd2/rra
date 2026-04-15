#include "cafe.hpp"
#include <cmath>
#include <immintrin.h>
#include <algorithm>

namespace s4m::cafe {

void CafeField::apply_stencil_convolution() {
    size_t num_chunks = chunks_.size();
    if (num_chunks == 0) return;

    // Temporary buffer for simultaneous update
    std::vector<FieldChunk> next_state = chunks_;

    for (size_t i = 0; i < num_chunks; ++i) {
        float mu = chunks_[i].mu;
        // The spread of the convolution depends on the local compute resolution (mu).
        float sigma = std::max(0.1f, mu);
        
        // Stencil weights: w(r, mu) = exp(-r^2 / (2 * sigma^2))
        float w0 = 1.0f; // r = 0
        float w1 = std::exp(-1.0f / (2.0f * sigma * sigma)); // r = 1

        // Normalization factor
        float norm = w0;
        if (i > 0) norm += w1;
        if (i < num_chunks - 1) norm += w1;
        
        w0 /= norm;
        float w1_norm = w1 / norm;

        __m512 mw0 = _mm512_set1_ps(w0);
        __m512 mw1 = _mm512_set1_ps(w1_norm);

        for (size_t v = 0; v < CHUNK_SIZE; ++v) {
            __m512 center = chunks_[i].state[v].data;
            __m512 left = (i > 0) ? chunks_[i - 1].state[v].data : _mm512_setzero_ps();
            __m512 right = (i < num_chunks - 1) ? chunks_[i + 1].state[v].data : _mm512_setzero_ps();

            // center * w0
            __m512 res = _mm512_mul_ps(center, mw0);
            
            // + left * w1
            if (i > 0) {
                res = _mm512_fmadd_ps(left, mw1, res);
            }
            
            // + right * w1
            if (i < num_chunks - 1) {
                res = _mm512_fmadd_ps(right, mw1, res);
            }

            next_state[i].state[v].data = res;
        }
    }

    chunks_ = std::move(next_state);
}

} // namespace s4m::cafe
