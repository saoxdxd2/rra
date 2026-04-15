#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include "neural_types.hpp"

namespace s4m::core {

/**
 * @brief Persistent Node Identity — Simplified
 */
struct Node {
    enum NodeType : uint8_t { Excitatory = 1, Inhibitory = 2 };

    uint64_t spatial_id = 0;
    NodeType type = Excitatory;
    FunctionalRegion region = FunctionalRegion::Integrator;

    float base_threshold = 0.5f;
    float leak_rate = 0.85f;
    float target_rate = 0.05f;
    float plasticity = 0.005f;
    bool frozen = false;

    // Meta-Consistency Field (Layer-3)
    float consistency_score = 1.0f;
    float temporal_penalty = 0.0f;
    uint64_t prev_state_hash = 0;
    uint32_t cluster_id = 0;
};

} // namespace s4m::core