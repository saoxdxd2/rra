#pragma once

#include <vector>
#include <cstdint>
#include "neural_types.hpp"
#include "synapse.hpp"

namespace rra::nis_engine {

struct NodeParams {
    // 1. Topology (Where does it live in the spatial manifold?)
    uint64_t spatial_id;
    
    // 2. Identity (Signal polarity and type)
    // 0 = Excitatory, 1 = Inhibitory, 2 = Modulatory
    uint8_t type_id; 
    
    // 3. Signal Physics
    float base_threshold;   // Firing sensitivity
    float leak_rate;        // Voltage decay rate
    float plasticity;       // Learning rate coefficient
    float metabolic_cap;    // Energy buffer capacity
    bool frozen = false;    // If true, parameters cannot be mutated
    uint8_t region_id = 0;  // Explicit functional region index
    
    // 4. Structural Initial State (Genome Only)
    struct InitialConnection {
        uint32_t target_idx;
        float weight;
    };
    std::vector<InitialConnection> initial_weights;
};

struct Node {
    uint64_t spatial_id;

    // Packed Identity & Routing
    enum class NodeType : uint8_t { Excitatory = 0, Inhibitory = 1, Modulatory = 2, Context = 3 };
    NodeType type = NodeType::Excitatory;
    FunctionalRegion region = FunctionalRegion::Integrator;

    uint16_t group_id = 0;
    bool is_lateral_projection = false;
    bool frozen = false;
    bool is_immortal = false; // Phase 11: Protect anchors from apoptosis/swapping

    // Contiguous Engine Pool routing offsets
    uint32_t pool_syn_offset = 0xFFFFFFFF;
    uint32_t pool_syn_count = 0;
    float desired_1d_pos = 0.0f;

    // Runtime Physics
    float base_threshold = 1.0f;
    float leak_rate = 0.5f;
    float plasticity = 0.01f;
    float metabolic_cap = 1.0f;
};

} // namespace rra::nis_engine
