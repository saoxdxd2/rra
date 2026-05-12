#pragma once

#include <vector>
#include "sao_core.hpp"

namespace s4m::core {

class NISEngine {
public:
    NISEngine() = default;
    ~NISEngine() = default;

    // MAP Inference: Finds the nearest global memory attractor via Hamming distance
    BitVector512 map_inference(const BitVector512& query) const;

    // Online Vector-Symbolic Memory Expansion
    void expand_memory(const BitVector512& state);

    // Get the current number of learned states
    size_t get_memory_size() const { return memory_bank.size(); }

private:
    std::vector<BitVector512> memory_bank;
};

} // namespace s4m::core