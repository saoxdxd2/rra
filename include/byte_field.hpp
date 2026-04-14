#pragma once
// ============================================================================
// byte_field.hpp — Geometric Neural Field (GNF) Core Module
//
// Provides:
//   1. 4^4 Perfect Grid Seed for 256 byte embeddings in 4D space
//   2. 64-bit Morton encoder (16 bits/axis) via hardware PDEP
//   3. GNF(B,N,V) activation: YOLO soft-boundary with base-2 exp decay
//   4. Geometric Output Anchors (8-bit receptor zones)
//   5. Momentum Manifold (4D velocity vectors)
//   6. Multi-resolution hierarchical Morton activation (4 × 16-bit passes)
// ============================================================================

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <immintrin.h>   // _pdep_u64, __popcnt64

namespace rra::gnf {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int    NUM_BYTES       = 256;
static constexpr int    NUM_DIMS        = 4;
static constexpr int    GRID_SIDE       = 4;       // 4^4 = 256
static constexpr int    OUTPUT_BITS     = 8;       // 8-bit byte prediction
static constexpr int    MORTON_LEVELS   = 4; // Hierarchical depth levels
static constexpr int    BITS_PER_LEVEL  = 16;       // 16 bits per axis = 64-bit key
static constexpr float  MOMENTUM_BETA   = 0.9f; // Velocity EMA decay
static constexpr float  DEFAULT_TAU     = 4.0f;     // Initial temperature
static constexpr int    DEFAULT_RADIUS  = 16;       // Sparsity cutoff (Hamming bits)

// ---------------------------------------------------------------------------
// 4D Coordinate
// ---------------------------------------------------------------------------
struct Coord4D {
    float x = 0.0f, y = 0.0f, z = 0.0f, w = 0.0f;

    float& operator[](int i) { return (&x)[i]; }
    const float& operator[](int i) const { return (&x)[i]; }
};

// ---------------------------------------------------------------------------
// ByteField — The Embedding Manifold
// ---------------------------------------------------------------------------
struct ByteField {
    // Learned 4D coordinates for each of 256 byte values
    std::array<Coord4D, NUM_BYTES>  coords;

    // 4D momentum velocity vectors (Principle 6)
    std::array<Coord4D, NUM_BYTES>  velocity;

    // Precomputed 64-bit Morton keys (refreshed when coords change)
    std::array<uint64_t, NUM_BYTES> morton_keys;

    // 8 fixed Output Anchor receptor zones (Principle 3 readout)
    std::array<Coord4D, OUTPUT_BITS> output_anchors;
    std::array<uint64_t, OUTPUT_BITS> output_anchor_mortons;
};

// ---------------------------------------------------------------------------
// GNF Activation Result (per-byte routing)
// ---------------------------------------------------------------------------
struct GNFResult {
    float    activation;          // Weighted sum of membrane values
    uint32_t top_neuron_idx;      // Index of closest neuron (for diagnostics)
};

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

/// @brief Seed all 256 byte coordinates onto the 4^4 equidistant grid.
///        byte_index → base-4 digits → (X, Y, Z, W) coordinates.
///        Also initializes the 8 output anchors at maximally spaced positions.
void seed_grid(ByteField& field);

/// @brief Recompute all 256 Morton keys from current float coordinates.
///        Call after any coordinate mutation (evolution, momentum update).
void refresh_morton_keys(ByteField& field);

/// @brief Encode a single 4D float coordinate into a 64-bit Morton Z-key.
///        Quantizes each axis to 16-bit uint, then bit-interleaves via PDEP.
uint64_t morton_encode_4d(float x, float y, float z, float w);

/// @brief The complete GNF(B, N, V) activation function.
///
///   1. D_i = POPCNT(M_B XOR M_Ni)
///   2. D_min = min(D_j)
///   3. delta_i = D_i - D_min
///   4. w_i = 2^{-(delta_i / tau)}  if delta_i <= R, else 0
///   5. Output = sum( (w_i / sum_w) * V_i )
///
/// @param byte_morton   The Morton key of the input byte.
/// @param neuron_mortons Array of neuron Morton keys.
/// @param neuron_values  Array of neuron membrane potentials (V_i).
/// @param num_neurons    Number of neurons.
/// @param tau            Temperature scalar (anneals with CE loss).
/// @param radius         Sparsity cutoff in Hamming distance bits.
/// @return GNFResult with the weighted activation and closest neuron index.
/// @param dopamine_out   Optional. If non-null, receives per-neuron credit signal:
///                        dopamine_out[i] += exp2(-d_i * inv_tau)
///                        Callers inject this directly into the engine's dopamine array
///                        for geometrically-grounded, high-dimensional credit assignment.
GNFResult gnf_activate(
    uint64_t        byte_morton,
    const uint64_t* neuron_mortons,
    const float*    neuron_values,
    int             num_neurons,
    float           tau,
    int             radius,
    float*          dopamine_out = nullptr
);

/// @brief Hierarchical Multi-Resolution GNF activation (Principle 7).
///        Processes the 64-bit Morton key in 4 × 16-bit passes:
///        Level 0 (bits 63-48): Global context routing
///        Level 1 (bits 47-32): Structural pattern routing
///        Level 2 (bits 31-16): Word-level pattern routing
///        Level 3 (bits 15-0):  Character-level exact routing
///
///   w_i = prod_{L=0}^{3} 2^{-(delta_i^L / tau_L)}
///
/// @param byte_morton   Full 64-bit Morton key.
/// @param neuron_mortons Array of neuron Morton keys.
/// @param neuron_values  Array of membrane potentials.
/// @param num_neurons    Number of neurons.
/// @param tau_levels     Temperature per level [4 values].
/// @param radius         Sparsity cutoff.
/// @return GNFResult.
GNFResult gnf_activate_hierarchical(
    uint64_t        byte_morton,
    const uint64_t* neuron_mortons,
    const float*    neuron_values,
    int             num_neurons,
    const float     tau_levels[MORTON_LEVELS],
    int             radius
);

/// @brief Read the 8-bit output prediction from the geometric anchors.
///        For each of the 8 output anchors, finds the closest neuron
///        by Morton proximity and reads its membrane potential.
///
/// @param field          The ByteField with output anchor Morton keys.
/// @param neuron_mortons Array of neuron Morton keys.
/// @param neuron_values  Array of membrane potentials.
/// @param num_neurons    Number of neurons.
/// @param tau            Temperature for weighted readout.
/// @param radius         Sparsity cutoff.
/// @return 8-element array of GNFResult (one per output bit).
std::array<GNFResult, OUTPUT_BITS> read_output_anchors(
    const ByteField& field,
    const uint64_t*  neuron_mortons,
    const float*     neuron_values,
    int              num_neurons,
    float            tau,
    int              radius
);

/// @brief Apply momentum-damped velocity update to a single byte coordinate.
///        velocity = β * velocity + (1 - β) * gradient_signal
///        coord   += velocity
void apply_momentum(ByteField& field, int byte_idx, const Coord4D& gradient);

/// @brief Save ByteField to a binary checkpoint file.
///        Format: [magic][version][coords][velocity][output_anchors]
bool save_field(const ByteField& field, const std::string& path);

/// @brief Load ByteField from a binary checkpoint file.
///        Automatically refreshes Morton keys after loading.
bool load_field(ByteField& field, const std::string& path);

} // namespace rra::gnf
