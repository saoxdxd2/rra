#pragma once

#include <immintrin.h>
#include <cstdint>
#include "neural_types.hpp"

namespace s4m::cafe {

/**
 * @brief The DualVector holds 16 elements (512 bits) of Magnitude and Phase.
 * Magnitude is the absolute value (continuous energy, propagated by CAFE).
 * Phase is the sign bit (+1 or -1) (discrete state, aligned by NIS).
 * This completely avoids float->bit->float quantization loss.
 */
struct DualVector {
    __m512 data;

    DualVector() : data(_mm512_setzero_ps()) {}
    explicit DualVector(__m512 d) : data(d) {}

    static DualVector loadu(const float* ptr) {
        return DualVector(_mm512_loadu_ps(ptr));
    }

    void storeu(float* ptr) const {
        _mm512_storeu_ps(ptr, data);
    }

    /**
     * @brief Extracts the signs of the continuous field as a 16-bit mask.
     * 1 = negative, 0 = positive.
     */
    __mmask16 extract_phase_mask() const {
        // castps_si512 preserves bits, movepi32_mask extracts the most significant bit (sign bit) of each 32-bit element.
        return _mm512_movepi32_mask(_mm512_castps_si512(data));
    }

    /**
     * @brief Extracts the continuous magnitudes (absolute values), clearing the signs.
     */
    __m512 extract_magnitude() const {
        __m512i mask = _mm512_set1_epi32(0x7FFFFFFF); // Clear the MSB
        return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(data), mask));
    }

    /**
     * @brief Injects a discrete phase mask into the continuous magnitude field.
     * Retains the current magnitude (energy) but overwrites the phase (routing).
     */
    void inject_phase(__mmask16 phase_mask) {
        __m512 mags = extract_magnitude();
        // Create sign bits: mask bit 1 -> 0x80000000, 0 -> 0x00000000
        __m512i signs = _mm512_maskz_set1_epi32(phase_mask, 0x80000000);
        data = _mm512_or_ps(mags, _mm512_castsi512_ps(signs));
    }
};

/**
 * @brief Extracts the phase of an array of 32 DualVectors (512 floats) into a BitVector512.
 */
inline s4m::BitVector512 extract_to_bitvector(const DualVector* vecs) {
    s4m::BitVector512 bv;
    // Each DualVector gives a 16-bit mask. 4 DualVectors = 64 bits (1 uint64).
    for (int i = 0; i < 8; ++i) {
        uint64_t chunk = 0;
        chunk |= static_cast<uint64_t>(vecs[i * 4 + 0].extract_phase_mask());
        chunk |= static_cast<uint64_t>(vecs[i * 4 + 1].extract_phase_mask()) << 16;
        chunk |= static_cast<uint64_t>(vecs[i * 4 + 2].extract_phase_mask()) << 32;
        chunk |= static_cast<uint64_t>(vecs[i * 4 + 3].extract_phase_mask()) << 48;
        bv.data[i] = chunk;
    }
    return bv;
}

/**
 * @brief Injects a BitVector512 into an array of 32 DualVectors.
 */
inline void inject_from_bitvector(DualVector* vecs, const s4m::BitVector512& bv) {
    for (int i = 0; i < 8; ++i) {
        uint64_t chunk = bv.data[i];
        vecs[i * 4 + 0].inject_phase(static_cast<__mmask16>(chunk));
        vecs[i * 4 + 1].inject_phase(static_cast<__mmask16>(chunk >> 16));
        vecs[i * 4 + 2].inject_phase(static_cast<__mmask16>(chunk >> 32));
        vecs[i * 4 + 3].inject_phase(static_cast<__mmask16>(chunk >> 48));
    }
}

} // namespace s4m::cafe
