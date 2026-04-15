// include/sao_core.hpp
// SC-V8: Terminal Bit-Algebraic Solver

#pragma once

#include <cstdint>
#include <cstddef>
#include <immintrin.h>
#include <algorithm>
#include <stdexcept>
#include <bit>
#include <cmath>
#include <vector>
#include <cstring>
#include "core_math.hpp"
#include "neural_types.hpp"
#include "neuron.hpp"

namespace s4m::core {

inline int pop64(uint64_t x) {
#ifdef _MSC_VER
    return static_cast<int>(__popcnt64(static_cast<unsigned __int64>(x)));
#else
    return static_cast<int>(std::popcount(x));
#endif
}

inline uint64_t rotl64(uint64_t x, int s) {
#ifdef _MSC_VER
    return _rotl64(x, s);
#else
    return std::rotl(x, s);
#endif
}

struct ManifoldView {
    float* voltage; float* base_threshold; float* leak_rates; float* target_rates;
    float* eligibility_trace; float* sensory_pressure; float* gradient;
    uint64_t* active_chunks_read; uint64_t* active_chunks_write;
    uint8_t* spike_state; uint64_t* spike_packed; float* firing_rate_ema; float* adaptive_thresholds;
    uint64_t* basis_pool; uint64_t* query_seeds_planes; BitVector512* latent_state;
    float* spikes; float* neuron_gain; float* gain_gradient; float* context_modulation;
    uint64_t* codebook; uint64_t* slates; uint64_t* proposal_buffer; uint8_t* proposal_count;

    float* policy_weights; float* policy_gates; float* policy_interactions;
    size_t num_neurons; uint32_t tick;
};

/**
 * @brief SC-V8: Direct Attractor Solver (M4R Peak Optimization)
 * Resolves the manifold state in GF(2) via cascading XOR row-reduction.
 */
namespace solver_v8 {

/**
 * @brief Terminal M4R Solver
 * Projects sensory input onto the bit-basis to find the global attractor.
 */
inline BitVector512 resolve_attractor(const BitVector512& input, const uint64_t* planes, size_t n) {
    BitVector512 x = input;
    // 8 macro-steps: process 64-bit words of surprise
    for (int word_idx = 0; word_idx < 8; ++word_idx) {
        while (uint64_t mask = x.data[word_idx]) {
            int b = _tzcnt_u64(mask);
            int b_global = (word_idx << 6) | b;
            const uint64_t* row = &planes[b_global * 8];
            // Elimination Stacking: XOR basis row bits-in-parallel
            for (int w = word_idx; w < 8; ++w) x.data[w] ^= row[w];
        }
    }
    return x;
}

/**
 * @brief Basis Rank Expansion (Gaussian Learning)
 */
inline void update_basis(uint64_t* planes, const BitVector512& pattern, size_t n) {
    BitVector512 r = pattern;
    for (int b = 0; b < 512; ++b) {
        if ((r.data[b >> 6] >> (b & 63)) & 1) {
            uint64_t* row = &planes[b * 8];
            if (row[b >> 6] == 0) {
                for(int w=0; w<8; ++w) row[w] = r.data[w];
                return;
            }
            for(int w=0; w<8; ++w) r.data[w] ^= row[w];
        }
    }
}

} // namespace solver_v8

struct HolographicFrame { BitVector512 emv, imv; };

/**
 * @brief SC-V8: Holographic VSA Binding
 */
inline void vsa_bind(HolographicFrame* f, const uint8_t* bytes, size_t n, const uint64_t* sigs) {
    alignas(64) int16_t cnt[512] = {0};
    for (size_t k = 0; k < n; ++k) {
        uint8_t b = bytes[k];
        const __m512i* sig_ptr = reinterpret_cast<const __m512i*>(&sigs[b * 16]);
        __m512i v_se = _mm512_load_si512(sig_ptr);
        __m512i v_si = _mm512_load_si512(sig_ptr + 1);
        alignas(64) uint64_t se_arr[8], si_arr[8];
        _mm512_store_si512(se_arr, v_se); _mm512_store_si512(si_arr, v_si);
        for (int w = 0; w < 8; ++w) {
            for (int bit = 0; bit < 64; ++bit) {
                int idx = w * 64 + bit;
                cnt[idx] += ((se_arr[w] >> bit) & 1) - ((si_arr[w] >> bit) & 1);
            }
        }
    }
    for (int i = 0; i < 8; ++i) {
        uint64_t m_emv = 0, m_imv = 0;
        for (int b = 0; b < 64; ++b) {
            int16_t v = cnt[i * 64 + b];
            if (v > 1) m_emv |= (1ULL << b);
            else if (v < -1) m_imv |= (1ULL << b);
        }
        f->emv.data[i] = m_emv; f->imv.data[i] = m_imv;
    }
}

} // namespace s4m::core
