// ============================================================================
// byte_field.cpp — Geometric Neural Field (GNF) Implementation
//
// Implements:
//   1. 4^4 Grid Seed (base-4 decomposition → equidistant 4D coordinates)
//   2. 64-bit Morton encoder (16 bits/axis, hardware PDEP)
//   3. GNF(B,N,V): YOLO soft-boundary activation with base-2 exp decay
//   4. Hierarchical multi-resolution GNF (4 × 16-bit Morton passes)
//   5. Geometric output anchor readout
//   6. Momentum manifold velocity updates
// ============================================================================

#include "byte_field.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <limits>

namespace rra::gnf {

// ---------------------------------------------------------------------------
// seed_grid — Initialize the 4^4 equidistant grid
// ---------------------------------------------------------------------------
void seed_grid(ByteField& field) {
    // Map each byte index (0–255) to base-4 digits → (X, Y, Z, W)
    // Normalize to [0, 1] range: digit / (GRID_SIDE - 1) → {0.0, 0.333, 0.667, 1.0}
    for (int b = 0; b < NUM_BYTES; ++b) {
        int rem = b;
        for (int d = NUM_DIMS - 1; d >= 0; --d) {
            int digit = rem % GRID_SIDE;
            rem /= GRID_SIDE;
            field.coords[b][d] = static_cast<float>(digit) / static_cast<float>(GRID_SIDE - 1);
        }
    }

    // Zero-initialize velocity vectors
    for (int b = 0; b < NUM_BYTES; ++b) {
        field.velocity[b] = {0.0f, 0.0f, 0.0f, 0.0f};
    }

    // Place 8 output anchors at maximally spaced corners of the 4D unit hypercube.
    // Use the first 8 vertices of a 4D hypercube (binary decomposition of 0–7).
    for (int i = 0; i < OUTPUT_BITS; ++i) {
        field.output_anchors[i].x = (i & 1) ? 1.0f : 0.0f;
        field.output_anchors[i].y = (i & 2) ? 1.0f : 0.0f;
        field.output_anchors[i].z = (i & 4) ? 1.0f : 0.0f;
        field.output_anchors[i].w = 0.5f;  // Center on T-axis
    }

    // Compute initial Morton keys
    refresh_morton_keys(field);
}

// ---------------------------------------------------------------------------
// morton_encode_4d — Quantize + bit-interleave 4 floats into uint64_t
// ---------------------------------------------------------------------------
// Deleted unused av_mix hash due to MSB Unrolling requirements

uint64_t morton_encode_4d(float x, float y, float z, float w) {
    // Clamp to [0, 1] then quantize to 16-bit unsigned integer
    auto quantize = [](float v) -> uint32_t {
        v = std::max(0.0f, std::min(1.0f, v));
        return static_cast<uint32_t>(v * 65535.0f);
    };

    uint32_t qx = quantize(x);
    uint32_t qy = quantize(y);
    uint32_t qz = quantize(z);
    
    // BPTT Geometry Unrolling: DO NOT avalanche/shuffle time! 
    // It must remain strictly sequential.
    uint32_t qw = quantize(w); 

    // PDEP 3D Spatial Morton (Lower 48 bits)
    // X -> bits 0, 3, 6, 9... (mask: 0x0000249249249249)
    // Y -> bits 1, 4, 7, 10... (mask: 0x0000492492492492)
    // Z -> bits 2, 5, 8, 11... (mask: 0x0000924924924924)
    uint64_t spatial_morton = 
        _pdep_u64(qx, 0x0000249249249249ULL) |
        _pdep_u64(qy, 0x0000492492492492ULL) |
        _pdep_u64(qz, 0x0000924924924924ULL);

    // Time (w) gets absolute MSB priority to simulate Unrolled Causality
    return (static_cast<uint64_t>(qw) << 48) | spatial_morton;
}

// ---------------------------------------------------------------------------
// refresh_morton_keys — Rebuild all Morton keys after coordinate changes
// ---------------------------------------------------------------------------
void refresh_morton_keys(ByteField& field) {
    for (int b = 0; b < NUM_BYTES; ++b) {
        field.morton_keys[b] = morton_encode_4d(
            field.coords[b].x, field.coords[b].y,
            field.coords[b].z, field.coords[b].w
        );
    }
    for (int i = 0; i < OUTPUT_BITS; ++i) {
        field.output_anchor_mortons[i] = morton_encode_4d(
            field.output_anchors[i].x, field.output_anchors[i].y,
            field.output_anchors[i].z, field.output_anchors[i].w
        );
    }
}

// ---------------------------------------------------------------------------
// gnf_activate — The complete GNF(B, N, V) activation function
//
//   D_i     = POPCNT(M_B XOR M_Ni)
//   D_min   = min(D_j)
//   delta_i = D_i - D_min
//   w_i     = 2^{-(delta_i / tau)}   if delta_i <= R, else 0
//   Output  = sum( (w_i / sum_w) * V_i )
// ---------------------------------------------------------------------------
GNFResult gnf_activate(
    uint64_t        byte_morton,
    const uint64_t* neuron_mortons,
    const float*    neuron_values,
    int             num_neurons,
    float           tau,
    int             radius,
    float*          dopamine_out
) {
    // Guard: stack arrays are fixed at 512
    // Step 1: Compute all Hamming distances
    // Use thread_local to remove the 512 ceiling with 0 heap cost
    thread_local std::vector<int> t_distances;
    if (t_distances.size() < static_cast<std::size_t>(num_neurons)) {
        t_distances.resize(num_neurons);
    }
    int* distances = t_distances.data();

    int d_min = 64;
    uint32_t closest_idx = 0;

#if defined(__AVX2__)
    __m256i v_byte_morton = _mm256_set1_epi64x(byte_morton);
    __m256i v_lut = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    );
    __m256i v_mask = _mm256_set1_epi8(0x0F);

    int i = 0;
    for (; i <= num_neurons - 4; i += 4) {
        __m256i v_neu = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&neuron_mortons[i]));
        __m256i v_xor = _mm256_xor_si256(v_byte_morton, v_neu);
        
        __m256i v_lo = _mm256_and_si256(v_xor, v_mask);
        __m256i v_hi = _mm256_and_si256(_mm256_srli_epi16(v_xor, 4), v_mask);
        __m256i v_pop = _mm256_add_epi8(_mm256_shuffle_epi8(v_lut, v_lo),
                                        _mm256_shuffle_epi8(v_lut, v_hi));
        __m256i v_sum = _mm256_sad_epu8(v_pop, _mm256_setzero_si256());
        
        alignas(32) int64_t counts[4];
        _mm256_store_si256(reinterpret_cast<__m256i*>(counts), v_sum);
        
        for (int j = 0; j < 4; ++j) {
            int d = static_cast<int>(counts[j]);
            distances[i+j] = d;
            
            // Branchless update
            bool is_closer = (d < d_min);
            d_min = is_closer ? d : d_min;
            closest_idx = is_closer ? static_cast<uint32_t>(i + j) : closest_idx;
        }
    }
    // Remainder using scalar __popcnt64
    for (; i < num_neurons; ++i) {
        uint64_t xor_val = byte_morton ^ neuron_mortons[i];
        int d = static_cast<int>(__popcnt64(xor_val));
        distances[i] = d;
        
        bool is_closer = (d < d_min);
        d_min = is_closer ? d : d_min;
        closest_idx = is_closer ? static_cast<uint32_t>(i) : closest_idx;
    }
#else
    for (int i = 0; i < num_neurons; ++i) {
        uint64_t xor_val = byte_morton ^ neuron_mortons[i];
        int d = static_cast<int>(__popcnt64(xor_val));
        distances[i] = d;
        
        bool is_closer = (d < d_min);
        d_min = is_closer ? d : d_min;
        closest_idx = is_closer ? static_cast<uint32_t>(i) : closest_idx;
    }
#endif

    // Steps 2-4: Anchor shift + base-2 weight + sparsity gate
    float sum_w = 0.0f;
    float weighted_val = 0.0f;
    const float inv_tau = 1.0f / (std::max)(tau, 0.01f);

    for (int i = 0; i < num_neurons; ++i) {
        int delta = distances[i] - d_min;

        // Branchless Sparsity gate: prune distant neurons using a mask
        float mask = static_cast<float>(delta <= radius);

        // Base-2 YOLO boundary weight: w = 2^{-(delta / tau)}
        float w = exp2f(-static_cast<float>(delta) * inv_tau) * mask;

        sum_w += w;
        weighted_val += w * neuron_values[i];
    }

    float output = (sum_w > 1e-9f) ? (weighted_val / sum_w) : 0.0f;

    // NaN/Inf guard: silent corruption trap
    if (!std::isfinite(output)) output = 0.0f;

    // --- Geometric Credit Assignment (DDQN bypass) ---
    // Feed the GNF routing weight back as per-neuron dopamine.
    // Neurons physically closest to the input byte in Morton space get the
    // strongest reward signal. This replaces scalar global dopamine entirely.
    if (dopamine_out) {
        for (int i = 0; i < num_neurons; ++i) {
            int delta = distances[i] - d_min;
            float credit = exp2f(-static_cast<float>(delta) * inv_tau)
                           * static_cast<float>(delta <= radius);
            dopamine_out[i] += credit;
        }
    }

    return {output, closest_idx};
}

// ---------------------------------------------------------------------------
// gnf_activate_hierarchical — Multi-resolution 4 × 16-bit Morton passes
//
//   For each level L (0..3):
//     Extract the 16-bit slice from both Morton keys
//     d_L = POPCNT(slice_B XOR slice_N)
//     w_i *= 2^{-(delta_L / tau_L)}
// ---------------------------------------------------------------------------
GNFResult gnf_activate_hierarchical(
    uint64_t        byte_morton,
    const uint64_t* neuron_mortons,
    const float*    neuron_values,
    int             num_neurons,
    const float     tau_levels[MORTON_LEVELS],
    int             radius
) {
    // For each neuron, compute the product of per-level weights
    thread_local std::vector<float> t_weights;
    thread_local std::vector<int> t_total_dist;
    if (t_weights.size() < static_cast<std::size_t>(num_neurons)) {
        t_weights.resize(num_neurons);
        t_total_dist.resize(num_neurons);
    }
    float* weights = t_weights.data();
    int* total_dist = t_total_dist.data();

    int   d_min_total = 64 * MORTON_LEVELS;
    uint32_t closest_idx = 0;

    // First pass: compute total hierarchical distance for anchor shift
    for (int i = 0; i < num_neurons; ++i) {
        uint64_t xor_val = byte_morton ^ neuron_mortons[i];
        // Note: The sum of popcnts of 16-bit slices of a 64-bit value 
        // is exactly the 64-bit popcnt! So we can just use __popcnt64.
        int total_d = static_cast<int>(__popcnt64(xor_val));

        total_dist[i] = total_d;
        
        // Branchless update
        bool is_closer = (total_d < d_min_total);
        d_min_total = is_closer ? total_d : d_min_total;
        closest_idx = is_closer ? static_cast<uint32_t>(i) : closest_idx;
    }

    // Second pass: compute per-level product weights with anchor shift
    float sum_w = 0.0f;
    float weighted_val = 0.0f;

    for (int i = 0; i < num_neurons; ++i) {
        int delta_total = total_dist[i] - d_min_total;
        
        // Branchless Sparsity gate mask
        float mask = static_cast<float>(delta_total <= radius);

        uint64_t xor_val = byte_morton ^ neuron_mortons[i];
        float w = 1.0f;

        for (int L = 0; L < MORTON_LEVELS; ++L) {
            int shift = (3 - L) * BITS_PER_LEVEL;
            uint16_t slice = static_cast<uint16_t>((xor_val >> shift) & 0xFFFF);
            int d_L = __popcnt(slice);

            float inv_tau_L = 1.0f / (std::max)(tau_levels[L], 0.01f);
            w *= exp2f(-static_cast<float>(d_L) * inv_tau_L);
        }

        // Apply sparsity mask
        w *= mask;

        weights[i] = w;
        sum_w += w;
        weighted_val += w * neuron_values[i];
    }

    float output = (sum_w > 1e-9f) ? (weighted_val / sum_w) : 0.0f;
    if (!std::isfinite(output)) output = 0.0f;
    return {output, closest_idx};
}

// ---------------------------------------------------------------------------
// read_output_anchors — Read 8-bit prediction from geometric receptor zones
// ---------------------------------------------------------------------------
std::array<GNFResult, OUTPUT_BITS> read_output_anchors(
    const ByteField& field,
    const uint64_t*  neuron_mortons,
    const float*     neuron_values,
    int              num_neurons,
    float            tau,
    int              radius
) {
    std::array<GNFResult, OUTPUT_BITS> results;
    for (int bit = 0; bit < OUTPUT_BITS; ++bit) {
        results[bit] = gnf_activate(
            field.output_anchor_mortons[bit],
            neuron_mortons,
            neuron_values,
            num_neurons,
            tau,
            radius
        );
    }
    return results;
}

// ---------------------------------------------------------------------------
// apply_momentum — Momentum manifold velocity update
// ---------------------------------------------------------------------------
void apply_momentum(ByteField& field, int byte_idx, const Coord4D& gradient) {
    // NaN guard: skip entire update if gradient contains NaN
    for (int d = 0; d < NUM_DIMS; ++d) {
        if (!std::isfinite(gradient[d])) return;
    }

    for (int d = 0; d < NUM_DIMS; ++d) {
        // Safe bounded gradients
        float grad = std::max(-10.0f, std::min(10.0f, gradient[d]));

        float new_vel = MOMENTUM_BETA * field.velocity[byte_idx][d] +
                        (1.0f - MOMENTUM_BETA) * grad;

        // Velocity clamping and NaN guard
        if (!std::isfinite(new_vel)) {
            new_vel = 0.0f;
        } else {
            new_vel = std::max(-5.0f, std::min(5.0f, new_vel));
        }

        field.velocity[byte_idx][d] = new_vel;
        field.coords[byte_idx][d] += new_vel;

        // Clamp to [0, 1] manifold bounds
        field.coords[byte_idx][d] = std::max(0.0f, std::min(1.0f, field.coords[byte_idx][d]));
    }
}

// ---------------------------------------------------------------------------
// Checkpoint Persistence API
// ---------------------------------------------------------------------------

// Magic header for format validation: "GNF" (Geometric Neural Field)
static constexpr uint32_t GNF_MAGIC = 0x474E4600;
static constexpr uint32_t GNF_VERSION = 1;

namespace {
template <typename T>
void write_rle(std::ofstream& out, const T* data, size_t count) {
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(data);
    size_t byte_count = count * sizeof(T);
    if (byte_count == 0) return;
    size_t i = 0;
    while (i < byte_count) {
        uint8_t run_len = 1;
        while (i + run_len < byte_count && run_len < 255 && ptr[i] == ptr[i + run_len]) {
            run_len++;
        }
        out.put(static_cast<char>(run_len));
        out.put(static_cast<char>(ptr[i]));
        i += run_len;
    }
}
}

bool save_field(const ByteField& field, const std::string& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;

    // Header
    out.write(reinterpret_cast<const char*>(&GNF_MAGIC), sizeof(GNF_MAGIC));
    out.write(reinterpret_cast<const char*>(&GNF_VERSION), sizeof(GNF_VERSION));

    // Data (RLE Compressed)
    write_rle(out, field.coords.data(), field.coords.size());
    write_rle(out, field.velocity.data(), field.velocity.size());
    write_rle(out, field.output_anchors.data(), field.output_anchors.size());

    return out.good();
}

namespace {
template <typename T>
void read_rle(std::ifstream& in, T* data, size_t count) {
    uint8_t* ptr = reinterpret_cast<uint8_t*>(data);
    size_t byte_count = count * sizeof(T);
    size_t i = 0;
    while (i < byte_count && in.good()) {
        char run_len, val;
        in.get(run_len);
        if (!in.good()) return;
        in.get(val);
        if (!in.good()) return;
        for (uint8_t j = 0; j < static_cast<uint8_t>(run_len) && i < byte_count; ++j) {
            ptr[i++] = static_cast<uint8_t>(val);
        }
    }
}
}

bool load_field(ByteField& field, const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    // Validate Header
    uint32_t magic = 0, version = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != GNF_MAGIC || version != GNF_VERSION) {
        return false;
    }

    // Load Data (RLE Compressed)
    read_rle(in, field.coords.data(), field.coords.size());
    read_rle(in, field.velocity.data(), field.velocity.size());
    read_rle(in, field.output_anchors.data(), field.output_anchors.size());

    if (in.good() || in.eof()) {
        // Rebuild 64-bit Morton keys from the loaded floats
        refresh_morton_keys(field);
        return true;
    }
    return false;
}

} // namespace rra::gnf
