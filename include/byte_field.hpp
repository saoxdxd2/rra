#pragma once

#include <cstdint>
#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <cmath>
#include <atomic>
#include "neural_types.hpp"

namespace s4m::gnf {

static constexpr int    NUM_BYTES       = 256;
static constexpr int    NUM_DIMS        = 4;
static constexpr int    BASIS_ELEMENTS  = 16;
static constexpr int    THERMO_BITS     = 32; // Increased from 4 to 32 bits for semantic depth
static constexpr int    MORTON_LEVELS   = 4;
static constexpr float  MOMENTUM_BETA   = 0.9f;

struct Coord4D { float x, y, z, w; };

struct ByteField {
    std::array<Coord4D, NUM_BYTES> coordinates;
    std::array<BitVector512, NUM_BYTES> morton_keys;
    std::array<float, NUM_BYTES> energies;
};

// Titan Multivector Encoding (512-bit)
BitVector512 titan_encode_512(float x, float y, float z, float w);

/**
 * @brief Multi-Resolution Gaussian Binder
 */
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

uint8_t value_centric_decode(const uint8_t* spike_state, const BitVector512* neuron_seeds, const BitVector512* byte_keys, int num_neurons, float* confidence_out = nullptr);

} // namespace s4m::gnf
