#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include "dual_vector.hpp"
#include "neural_types.hpp"

namespace s4m::cafe {

// The resolution of a spatial chunk.
// 32 DualVectors = 512 floats. This maps perfectly to 1 BitVector512 when phase is extracted.
constexpr size_t CHUNK_SIZE = 32; 

/**
 * @brief Represents a localized spatial chunk of the continuous CAFE PDE field.
 * Sorted by 1D Morton Z-curve for optimal locality.
 */
struct alignas(64) FieldChunk {
    // Continuous states (Magnitudes + Phases)
    DualVector state[CHUNK_SIZE]; 
    
    // Compute resolution (local mass \mu)
    float mu; 
    
    // Morton spatial index
    uint64_t morton_code;

    // Local energy/prediction error for optimal transport
    float energy_gradient;
};

class CafeField {
public:
    explicit CafeField(size_t num_chunks) : chunks_(num_chunks) {
        for (size_t i = 0; i < num_chunks; ++i) {
            chunks_[i].morton_code = i; // Simplified 1D Morton for linear sequence
            chunks_[i].mu = 1.0f; // Uniform initial compute distribution
            chunks_[i].energy_gradient = 0.0f;
        }
    }

    // AVX-512 Spatial Convolution across Morton Grid
    void apply_stencil_convolution();

    // 1D Optimal Transport (CDF Matching) for \mu mass
    void apply_optimal_transport(float total_mass);

    // Tensor Marshalling for Spectral Geometrics
    s4m::Tensor to_tensor() const;
    void apply_delta(const s4m::Tensor& delta);

    std::vector<FieldChunk>& get_chunks() { return chunks_; }
    const std::vector<FieldChunk>& get_chunks() const { return chunks_; }

private:
    std::vector<FieldChunk> chunks_;
};

} // namespace s4m::cafe
