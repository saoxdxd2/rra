#ifndef BRAIN_ISA_CONFIG_H
#define BRAIN_ISA_CONFIG_H

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <immintrin.h>
#include <type_traits>

namespace NIS {

    // -----------------------------------------------------------------------
    // Core Utilities
    // -----------------------------------------------------------------------
    template <typename T>
    inline T clamp(T value, T lo, T hi) {
        if constexpr (std::is_floating_point_v<T>) {
            if (!std::isfinite(value)) return lo;
        }
        if (value < lo) return lo;
        if (value > hi) return hi;
        return value;
    }

    inline double ema(double prev, double sample, double decay, bool initialized) {
        sample = std::isfinite(sample) ? sample : 0.0;
        decay = clamp(decay, 1e-12, 1.0);
        if (!initialized) return sample;
        return ((1.0 - decay) * prev) + (decay * sample);
    }

    // -----------------------------------------------------------------------
    // Engine Constants (sao1 Pillarized Architecture)
    // -----------------------------------------------------------------------
    
    // Physical Bounds
    static constexpr float   PHYSICS_THRESHOLD_MIN      = 0.1f;
    static constexpr float   PHYSICS_THRESHOLD_MAX      = 5.0f;
    static constexpr float   PHYSICS_WEIGHT_MIN         = -10.0f;
    static constexpr float   PHYSICS_WEIGHT_MAX         = 10.0f;
    static constexpr float   PHYSICS_VOLTAGE_DECAY      = 0.985f; // Relaxed (0.95 -> 0.985)
    static constexpr float   GNF_TAU_LEVELS[4]          = {0.95f, 0.90f, 0.80f, 0.50f};
    static constexpr int32_t GNF_MORTON_LEVELS          = 4;
    static constexpr float   GNF_MOMENTUM_BETA          = 0.95f;
    static constexpr int32_t GNF_RADIUS                 = 24;
    static constexpr int32_t GNF_LATERAL_K              = 8;  // Matches LateralTable::K
    static constexpr float   GNF_LATERAL_SPECTRAL_MAX   = 0.95f;
    static constexpr int32_t UEI_SSM_HIDDEN_DIM         = 64;
    static constexpr int32_t HOPFIELD_LOCAL_RADIUS      = 24;
    
    // Cognitive Cycle
    static constexpr int32_t PHYSICS_MAX_K_TICKS       = 20;
    static constexpr float   PHYSICS_CONFIDENCE_THRESHOLD = 0.50f;
    static constexpr float   PHYSICS_SPARSITY_CAP       = 2.0f;
    static constexpr float   PHYSICS_LOGIT_GAIN         = 5.0f;
    static constexpr float   PHYSICS_PROB_SCALE         = 0.48f;
    
    // Homeostasis & Plasticity
    static constexpr float   HOMEOS_DECAY              = 0.995f;
    static constexpr float   HOMEOS_STRENGTH           = 0.05f;
    static constexpr int32_t HOMEOS_REFRACTORY         = 3;
    
    static constexpr float   PLASTICITY_FIRING_EMA     = 0.95f;
    static constexpr float   PLASTICITY_SURPRISE_EMA   = 0.90f;
    static constexpr float   PLASTICITY_MOMENTUM       = 0.05f;
    static constexpr float   PLASTICITY_LEARNING_RATE  = 0.10f;
    
    // -----------------------------------------------------------------------
    // Training & Manifold Constants
    // -----------------------------------------------------------------------
    static constexpr int32_t TRAIN_CONTEXT_WINDOW      = 32;
    static constexpr int32_t TRAIN_PER_CAPACITY        = 1000;
    static constexpr int32_t TRAIN_REPLAY_BATCH        = 4;
    static constexpr float   TRAIN_BOOTSTRAP_WEIGHT    = 5.0f;
    
    static constexpr float   TRAIN_SURPRISE_SCALE      = 0.25f;
    static constexpr float   TRAIN_GATING_BIAS         = 1.5f;
    static constexpr float   TRAIN_GATING_SLOPE        = 4.0f;
    static constexpr float   TRAIN_INJECTION_GAIN      = 150.0f; // Boosted (100 -> 150)
    
    static constexpr float   TRAIN_IS_ALPHA_LIMIT      = 1.0f;
    static constexpr float   TRAIN_IS_ALPHA_STEPS      = 250000.0f;
    static constexpr float   TRAIN_IS_MIN_CE           = 0.1f;
    
    // Metabolic Governance
    static constexpr int64_t TRAIN_APOPTOSIS_INTERVAL  = 10000;
    static constexpr int64_t TRAIN_METRICS_INTERVAL    = 1000;
    static constexpr int64_t TRAIN_GROWTH_INTERVAL     = 50000;
    static constexpr int32_t TRAIN_GROWTH_COUNT        = 256;
    static constexpr float   TRAIN_STAGNATION_EMA_DIFF = 0.05f;
    
    // Maintenance
    static constexpr int64_t TRAIN_PROBE_INTERVAL      = 5000;
    static constexpr int64_t TRAIN_CHECKPOINT_INTERVAL = 50000;

    // -----------------------------------------------------------------------
    // Hardware Morton Jump (PDEP)
    // -----------------------------------------------------------------------
    inline uint32_t av_mix(uint32_t k) {
        k ^= k >> 16;
        k *= 0x85ebca6b;
        k ^= k >> 13;
        k *= 0xc2b2ae35;
        k ^= k >> 16;
        return k;
    }

    inline uint64_t morton_jump_4d(uint32_t x, uint32_t y, uint32_t z, uint32_t t) {
        t = av_mix(t);
        return _pdep_u64(x, 0x1111111111111111ULL) | 
               _pdep_u64(y, 0x2222222222222222ULL) | 
               _pdep_u64(z, 0x4444444444444444ULL) | 
               _pdep_u64(t, 0x8888888888888888ULL);
    }

} // namespace NIS

#endif // BRAIN_ISA_CONFIG_H
