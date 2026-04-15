#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <immintrin.h>
#include <array>
#include <bit>
#include <algorithm>

/**
 * @namespace s4m
 * @brief Unified namespace for the $D^D$ Ternary Manifold.
 */
namespace s4m {

/**
 * @brief Memory Alignment Allocator for SIMD Manifolds
 */
template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;
    template <typename U> struct rebind { using other = AlignedAllocator<U, Alignment>; };
    AlignedAllocator() noexcept = default;
    template <typename U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        void* ptr = nullptr;
#if defined(_WIN32) || defined(__MINGW32__)
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
#else
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) ptr = nullptr;
#endif
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }
    void deallocate(T* p, std::size_t) noexcept {
#if defined(_WIN32) || defined(__MINGW32__)
        _aligned_free(p);
#else
        free(p);
#endif
    }
    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

template <typename T>
using aligned_vector = std::vector<T, AlignedAllocator<T, 32>>;

/**
 * @brief SC-V8: 512-bit Multivector Primitive.
 * Optimized for AVX-512 and branchless execution.
 */
struct alignas(64) BitVector512 {
    union {
        uint64_t data[8];
        __m512i  v512;
    };

    BitVector512() : v512(_mm512_setzero_si512()) {}

    // Mix-identity derivation (Titan Basis)
    // 0x9E3779B97F4A7C15ULL = (2^64 / φ)
    BitVector512(uint64_t base) {
        alignas(64) uint64_t tmp[8];
        for(int i=0; i<8; ++i) tmp[i] = base ^ (static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL);
        v512 = _mm512_load_epi64(tmp);
    }

    // Bitwise Operators - Branchless SIMD
    inline BitVector512 operator&(const BitVector512& o) const {
        BitVector512 r; r.v512 = _mm512_and_si512(v512, o.v512); return r;
    }
    inline BitVector512 operator^(const BitVector512& o) const {
        BitVector512 r; r.v512 = _mm512_xor_si512(v512, o.v512); return r;
    }
    inline BitVector512 operator|(const BitVector512& o) const {
        BitVector512 r; r.v512 = _mm512_or_si512(v512, o.v512); return r;
    }

    // Branchless Equality
    inline bool operator==(const BitVector512& o) const {
        return _mm512_cmpeq_epi64_mask(v512, o.v512) == 0xFF;
    }
    inline bool is_null() const {
        return _mm512_test_epi64_mask(v512, v512) == 0;
    }

    // SC-V8: Fast holographic mixing (No branching, pure bit-op)
    inline uint64_t mix_bits() const {
        uint64_t h = data[0];
        h ^= data[1] + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= data[2] + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= data[3] + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= data[4] + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= data[5] + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= data[6] + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= data[7] + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }

    // AVX-512 Population Count (VNNI-adjacent)
    inline int popcount() const {
        return (int)_mm512_reduce_add_epi64(_mm512_popcnt_epi64(v512));
    }

    // Extreme performance intersection popcount
    inline int popcount_and(const BitVector512& o) const {
        return (int)_mm512_reduce_add_epi64(_mm512_popcnt_epi64(_mm512_and_si512(v512, o.v512)));
    }

    // Surprise distance (XOR popcount)
    inline int popcount_xor(const BitVector512& o) const {
        return (int)_mm512_reduce_add_epi64(_mm512_popcnt_epi64(_mm512_xor_si512(v512, o.v512)));
    }
};

enum class TickMode : uint8_t {
    Standard = 0,
    Cognitive = 1,
    Retrograde = 2
};

enum class FunctionalRegion : uint8_t {
    Integrator = 0,
    Input      = 1,
    Output     = 2,
    Modulator  = 3,
    Limbic     = 4,
    Thalamic   = 5
};

struct EngineConfig {
    float default_threshold = 0.5f;
    float default_decay = 0.85f;
    float homeos_target_rate = 0.05f;
    float plasticity_learning_rate = 0.005f;
};

struct NodeParams {
    uint64_t spatial_id;
    uint8_t type_id;
    uint8_t region_id;
};

struct WeightStats {
    size_t count = 0;           // Number of active synapse entries
    float mean_weight = 0.0f;    // Average fill ratio of 512-bit vectors (0.0-1.0)
    size_t total_entries = 0;     // Total possible entries (PHASOR_MEM_SIZE)
    float sparsity = 1.0f;       // Ratio of inactive entries (1.0 = empty, 0.0 = full)
};

struct alignas(16) InputEvent {
    uint32_t x = 0, y = 0, z = 0, t = 0;
    float current = 0.0f;
    uint32_t origin_id = 0;
};

/**
 * @brief Simple Tensor structure for pure C++ implementation.
 */
struct Tensor {
    std::vector<float> data;
    std::vector<size_t> shape;

    Tensor() = default;
    Tensor(std::vector<size_t> s) : shape(s) {
        size_t size = 1;
        for (auto dim : shape) size *= dim;
        data.assign(size, 0.0f);
    }

    size_t size() const { return data.size(); }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

} // namespace s4m