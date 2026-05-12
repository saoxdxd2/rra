#pragma once

#include "cafe/cafe.hpp"
#include "aether/aether.hpp"
#include "../include/nis_engine.hpp"
#include <vector>
#include <memory>
#include <array>

namespace s4m {

class HybridBlock {
public:
    HybridBlock(size_t num_chunks);

    // Forward pass processes the entire CAFE field through one layer of depth.
    void forward(core::NISEngine& nis_engine);

    // The continuous PDE field
    cafe::CafeField cafe_field;
    
    // AETHER Spectral Field Operator
    std::unique_ptr<rra::nn::aether::SpectralGeometricPropagator> aether_propagator;
};

class HybridEngine {
public:
    HybridEngine(size_t num_blocks, size_t num_chunks);

    // Forward pass through the deep stack of HybridBlocks
    void forward();

    // Wire prediction errors to trigger NIS basis updates and CAFE \mu routing
    void backward(const std::vector<float>& chunk_energy_gradients);

    // Ingest a batch of bytes into the input layer
    void ingest(const uint8_t* data, size_t len);

    // Predict the next byte from the output layer
    uint8_t predict() const;
    std::array<float, 256> get_logits() const;

    std::vector<HybridBlock> blocks;
    core::NISEngine global_memory;
};

} // namespace s4m
