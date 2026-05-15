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
#include <array>

namespace rra::nn::topology {

// ---------------------------------------------------------
// Core 512-bit Vector Symbolic Architecture (VSA)
// ---------------------------------------------------------
struct BitVector512 {
    union {
        __m512i v512;
        uint64_t data[8];
    };

    BitVector512() {
        for(int i=0; i<8; ++i) data[i] = 0;
    }

    BitVector512(const BitVector512& other) {
        for(int i=0; i<8; ++i) data[i] = other.data[i];
    }

    BitVector512& operator=(const BitVector512& other) {
        if (this != &other) {
            for(int i=0; i<8; ++i) data[i] = other.data[i];
        }
        return *this;
    }

    BitVector512 operator^(const BitVector512& other) const {
        BitVector512 res;
        for (int i=0; i<8; ++i) res.data[i] = data[i] ^ other.data[i];
        return res;
    }

    BitVector512& operator^=(const BitVector512& other) {
        for (int i=0; i<8; ++i) data[i] ^= other.data[i];
        return *this;
    }

    int popcount() const {
        int cnt = 0;
        for (int i = 0; i < 8; ++i) {
#ifdef _MSC_VER
            cnt += static_cast<int>(__popcnt64(data[i]));
#else
            cnt += std::popcount(data[i]);
#endif
        }
        return cnt;
    }

    int popcount_xor(const BitVector512& other) const {
        int cnt = 0;
        for (int i = 0; i < 8; ++i) {
#ifdef _MSC_VER
            cnt += static_cast<int>(__popcnt64(data[i] ^ other.data[i]));
#else
            cnt += std::popcount(data[i] ^ other.data[i]);
#endif
        }
        return cnt;
    }
};

// ---------------------------------------------------------
// SC-V8: Direct Attractor Solver (O(1) Backprop)
// ---------------------------------------------------------
namespace solver_v8 {
    BitVector512 resolve_attractor(const BitVector512& input, const uint64_t* planes, size_t n = 512);
    void update_basis(uint64_t* planes, const BitVector512& pattern, size_t n = 512);
}

// ---------------------------------------------------------
// Titan Multivector Encoding & Gaussian Binder
// ---------------------------------------------------------
struct Coord4D { float x, y, z, w; };

struct ByteField {
    std::array<Coord4D, 256> coordinates;
    std::array<BitVector512, 256> morton_keys;
};

// Maps 4D space to 512-bit thermometer mask
BitVector512 titan_encode_512(float x, float y, float z, float w);

class GaussianBinder {
public:
    explicit GaussianBinder(int corusion_shift) : corusion_shift_(corusion_shift) {}

    void bind(const ByteField& field, uint8_t byte, float surprise = 0.0f);
    BitVector512 snap_to_titan() const;
    void reset();

    Coord4D current_coord() const { return current_; }
private:
    Coord4D current_ = {0.5f, 0.5f, 0.5f, 0.5f};
    int corusion_shift_;
};

void seed_grid(ByteField& field);

// Decode Spiking neurons back into Byte Tokens using AVX-512 majority vote
uint8_t value_centric_decode(
    const uint8_t* spike_state, 
    const BitVector512* neuron_seeds, 
    const BitVector512* byte_keys, 
    int num_neurons, 
    float* confidence_out = nullptr
);

} // namespace rra::nn::topology
