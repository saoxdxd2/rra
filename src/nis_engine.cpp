#include "nis_engine.hpp"

namespace s4m::core {

BitVector512 NISEngine::map_inference(const BitVector512& query) const {
    BitVector512 attractor = query; // Default to self if memory is empty
    int min_distance = 513;
    
    for (const auto& mem_state : memory_bank) {
        int dist = query.popcount_xor(mem_state);
        if (dist < min_distance) {
            min_distance = dist;
            attractor = mem_state;
        }
    }
    return attractor;
}

void NISEngine::expand_memory(const BitVector512& state) {
    bool is_novel = true;
    for (const auto& mem_state : memory_bank) {
        if (state.popcount_xor(mem_state) < 32) { // Allow some noise threshold
            is_novel = false;
            break;
        }
    }
    if (is_novel) {
        memory_bank.push_back(state);
    }
}

} // namespace s4m::core
