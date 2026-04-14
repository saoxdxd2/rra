#pragma once

#include <cstdint>
#include <cstddef>
#include <immintrin.h>
#include <algorithm>
#include "core_math.hpp"

namespace sao1 {

/**
 * @brief Memory Layout for the Ternary Manifold.
 */
struct sao_manifold {
    float* voltage;             
    float* base_threshold;      
    float* eligibility_trace;   
    uint32_t* last_spike_tick;  
    float* sensory_pressure;    
    float* dopamine;            
    float* dopamine_back;       
    uint64_t* local_ex_masks;   
    uint64_t* local_in_masks;   
    uint8_t* spike_state;       // WRITE buffer for current tick
    uint8_t* spike_state_read;  // READ buffer for previous tick
    uint8_t* refractory_timer;  
    float* leak_rates;
    float* local_weights;      // N*64 float weights
    float* firing_rate_ema;
    float* adaptive_thresholds;
    const float* region_biases;
    const uint8_t* node_regions;
    size_t num_neurons;
    uint32_t current_tick;
};

// MaskTable removed due to NaN casting issues and obsolescence
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

/**
 * Pillar 3: The Physics Kernel (Gated SSM / Linear RNN Update)
 */
inline void sao_tick_physics(sao_manifold* m, const float* injected_currents) {
    
    #pragma omp parallel for if(m->num_neurons > 10000)
    for (int i = 0; i < static_cast<int>(m->num_neurons); ++i) {
        // Signals are driven by SPIKES from the consistent snapshot
        float input_sum = 0.0f;
        uint64_t ex = m->local_ex_masks[i];
        uint64_t inh = m->local_in_masks[i];

        // 1. Local Spatial Shortcut (Dense Neighborhood Cache)
        // NOTE ON TOPOLOGY vs TIME:
        // The local 64-bit masks are a *spatial* cache rebuilt by compile_topology()
        // from the paged synapse pool after each Barycenter sort. They encode the
        // ~64 physically-nearest neighbors post-sort, NOT a strict temporal arrow-of-time.
        // True temporal causality is encoded in the MSB Morton key hierarchy: neurons
        // at the same time-depth cluster together in RAM, so spatial neighbors ARE
        // (approximately) temporally correlated after sorting. Long-range learned
        // associations that survive across sort boundaries live in the paged pool
        // and are processed in execute_cycle() via pointer-chasing (wormholes).
        // This dual-path is the correct model: local mask = fast spatial band,
        // paged pool = exact learned topology.
        for (int j = 0; j < 64; ++j) {
            int neighbor_idx = i - j;
            if (neighbor_idx >= 0) {
                float s_nb = static_cast<float>(m->spike_state_read[neighbor_idx]);
                float is_ex = static_cast<float>((ex >> j) & 1ULL);
                float is_in = static_cast<float>((inh >> j) & 1ULL);
                input_sum += s_nb * (is_ex - is_in);
            }
        }

        // 2. Add Thalamic Bias & External Context
        float m_bias = m->region_biases ? m->region_biases[m->node_regions[i]] : 1.0f;
        float total_drive = (injected_currents[i] + m->sensory_pressure[i] + input_sum * m_bias);

        float v = m->voltage[i];
        float d = m->dopamine[i];
        float t = m->base_threshold[i];

        // --- 1. THE ORGANIC GATE (Dopamine-Modulated Leak) ---
        // If dopamine is high, retention approaches 0.99 (valves close).
        // If dopamine is low, retention is 0.80 (valves open).
        v += total_drive;
        float retention = 0.80f + (0.19f * std::max(0.0f, std::min(d, 1.0f)));
        v *= retention;
        
        // --- 4. Smooth Homeostatic Plasticity (Log-scaled PD control)
        // Unified formula: no hard if/else discontinuity.
        // rate_error is large-and-negative when neuron is dead -> threshold drops quickly.
        // std::log1p prevents explosive gain at very low firing rates.
        // Clamp prevents runaway in both directions.
        float target_rate = 0.02f;
        float rate = m->firing_rate_ema[i];
        float rate_error = target_rate - rate;
        // Logarithmic damping: sensitivity proportional to log(1+|error|) to prevent seizures
        float damped_error = std::copysign(std::log1pf(std::abs(rate_error) * 20.0f), rate_error);
        m->adaptive_thresholds[i] -= damped_error * 0.003f;
        m->adaptive_thresholds[i] = std::clamp(m->adaptive_thresholds[i], 0.1f, 5.0f);
        
        // Soft Reset: preserve excess energy above threshold (avoids destroying
        // high-frequency temporal precision). Hard reset to 0 would lose the 0.5
        // "overshoot" if threshold=1.0 and mem=1.5.
        float threshold_total = t + m->adaptive_thresholds[i];
        bool spiked = v >= threshold_total;
        float f_spiked = static_cast<float>(spiked);
        m->voltage[i] = spiked ? (v - threshold_total) : v;
        m->voltage[i] = std::max(m->voltage[i], -2.0f); // prevent negative runaway
        m->spike_state[i] = spiked ? 1 : 0;
        
        // Threshold jumps to pinch variance, then relaxes to 1.0f base gravity
        t += f_spiked * 0.5f;
        m->base_threshold[i] = 1.0f + (t - 1.0f) * 0.95f;

        // --- 3. THE ORGANIC OBJECTIVE (Gas Temporal Dissipation) ---
        m->dopamine[i] = d * 0.90f;

        // 4. Eligibility Trace for Dopamine Learning
        m->eligibility_trace[i] = rra::core_math::apply_attractor(m->eligibility_trace[i], f_spiked, 0.1f);
        if (spiked) m->last_spike_tick[i] = m->current_tick;
        
        // Decay sensory pressure
        m->sensory_pressure[i] *= 0.92f;
        
        // Update EMA
        m->firing_rate_ema[i] = (m->firing_rate_ema[i] * 0.99f) + (f_spiked * 0.01f);
    }
}

} // namespace sao1
