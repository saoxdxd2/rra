#pragma once

#include <cstdint>
#include <vector>
#include <string>

namespace rra::nis_engine {

enum class TickMode : uint8_t {
    Standard = 0,
    Cognitive = 1,
    Retrograde = 2
};

enum class FunctionalRegion : uint8_t {
    Input = 0,
    Integrator = 1,
    Output = 2,
    Context = 3,
    Modulatory = 4
};

enum class Emotion : uint8_t {
    Neutral = 0,
    Curious = 1,
    Fear = 2,
    Joy = 3
};

struct EngineConfig {
    float   physics_voltage_decay      = 0.985f;
    int32_t physics_max_k_ticks       = 20;
    float   physics_confidence_threshold = 0.50f;
    float   physics_sparsity_cap       = 2.0f;
    float   physics_logit_gain         = 5.0f;
    float   physics_prob_scale         = 0.48f;
    
    float   homeos_target_rate         = 0.01f;
    int32_t homeos_refractory         = 3;
    
    float   default_threshold          = 1.0f;
    float   default_decay              = 0.5f;
    
    float   plasticity_learning_rate  = 0.10f;
    float   plasticity_momentum       = 0.05f;

    int32_t train_context_window      = 32;
    int32_t train_per_capacity        = 1000;
    float   train_lr                  = 0.01f;
    float   train_is_min_ce           = 0.1f;
    int64_t train_metrics_interval    = 1000;
};

struct MetabolicStats {
    double stability = 0.0;
    double energy = 0.0;
    double coherence = 0.0;
};

struct WeightStats {
    size_t count = 0;
    float min_weight = 0.0f;
    float max_weight = 0.0f;
    float mean_weight = 0.0f;
};

/**
 * @brief Raw input event from the spatial manifold.
 */
struct alignas(16) InputEvent {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;
    uint32_t t = 0;
    float current = 0.0f;
    uint32_t origin_id = 0;
};

} // namespace rra::nis_engine
