#include "cafe.hpp"
#include <numeric>
#include <cmath>
#include <algorithm>

namespace s4m::cafe {

void CafeField::apply_optimal_transport(float total_mass) {
    size_t num_chunks = chunks_.size();
    if (num_chunks == 0) return;

    std::vector<float> demands(num_chunks);
    float sum_demand = 0.0f;

    // 1. Calculate Target Demand based on prediction error gradients
    for (size_t i = 0; i < num_chunks; ++i) {
        // D_i = max(0, dE/d\mu_i)
        // energy_gradient represents the local prediction error
        float d = std::max(0.0f, chunks_[i].energy_gradient);
        
        // Add a tiny epsilon to ensure no chunk has absolute zero compute resolution
        d += 1e-6f; 
        demands[i] = d;
        sum_demand += d;
    }

    // 2. Normalize and exact 1D Transport
    // Setting the normalized demand directly ensures the spatial CDF of \mu 
    // perfectly matches the target CDF derived from the demand.
    for (size_t i = 0; i < num_chunks; ++i) {
        chunks_[i].mu = total_mass * (demands[i] / sum_demand);
    }
}

} // namespace s4m::cafe
