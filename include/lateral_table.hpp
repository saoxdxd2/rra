#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <immintrin.h>
#include <algorithm>

namespace rra::gnf {

struct LateralTable {
    static constexpr int K = 8;

    // Fixed array sizes for maximum AVX2 footprint efficiency
    std::vector<std::array<uint16_t, K>> neighbors;
    std::vector<std::array<float, K>> weights;
    std::vector<std::array<float, K>> perturbations; // Stores Forward Gradient previous ε

    void initialize(size_t num_neurons) {
        neighbors.assign(num_neurons, {0});
        weights.assign(num_neurons, {0.0f});
        perturbations.assign(num_neurons, {0.0f});
    }

    // Rebuilds the neighbor list based on Morton distance.
    void rebuild(const std::vector<uint64_t>& morton_keys) {
        if (morton_keys.empty()) return;
        
        size_t n = morton_keys.size();
        if (neighbors.size() != n) initialize(n);

        struct ScoredNeighbor {
            uint16_t id;
            int dist;
            bool operator<(const ScoredNeighbor& o) const { return dist < o.dist; }
        };

        for (size_t i = 0; i < n; ++i) {
            std::vector<ScoredNeighbor> candidates;
            candidates.reserve(n - 1);
            
            for (size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                int dist = static_cast<int>(__popcnt64(morton_keys[i] ^ morton_keys[j]));
                candidates.push_back({static_cast<uint16_t>(j), dist});
            }
            
            // Partially sort to get the top K closest
            std::nth_element(candidates.begin(), candidates.begin() + K, candidates.end());
            
            for (size_t k = 0; k < K; ++k) {
                // Keep existing weights, just update the topological target
                neighbors[i][k] = candidates[k].id;
            }
        }
    }

    // Enforces spectral radius constraint to prevent exploding states
    void clamp_spectral_radius() {
        for (size_t i = 0; i < weights.size(); ++i) {
            float row_norm_sq = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                row_norm_sq += weights[i][k] * weights[i][k];
            }
            
            float row_norm = std::sqrt(row_norm_sq);
            // Branchless spectral scaling: if row_norm > 0.95, scale = 0.95/row_norm, else 1.0
            float scale = 0.95f / (std::max)(0.95f, row_norm);
            for (size_t k = 0; k < K; ++k) {
                weights[i][k] *= scale;
            }
        }
    }
};

} // namespace rra::gnf
