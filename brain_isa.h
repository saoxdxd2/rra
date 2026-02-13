#ifndef BRAIN_ISA_H
#define BRAIN_ISA_H

#include <cstdint>
#include <immintrin.h> // For _pdep_u64

namespace NIS {

    /**
     * Neural Instruction Set (NIS) Opcodes
     * Maps 8-bit tokens to AVX-512/ASM operations
     */
    static constexpr uint8_t OP_NOP     = 0x00;
    static constexpr uint8_t OP_ADD     = 0x01;  // vaddps
    static constexpr uint8_t OP_SCALE   = 0x02;  // vmulps
    static constexpr uint8_t OP_GATE    = 0x03;  // vblendvps
    static constexpr uint8_t OP_REFLECT = 0x04;  // subps (sign inversion)
    static constexpr uint8_t OP_JMP     = 0x05;  // Morton spatial jump

    /**
     * Structural Dimensions (Fixed Bio-Wiring)
     */
    static constexpr int64_t L           = 32;
    static constexpr int64_t R           = 8;
    static constexpr int64_t WORKING_DIM = 512;
    static constexpr int64_t C           = 4;
    static constexpr int64_t MEMORY_DEPTH = 5;

    /**
     * Cognitive Cycle Constraints
     */
    static constexpr int64_t H_CYCLES    = 2;
    static constexpr int64_t L_CYCLES    = 4;
    static constexpr float RMS_NORM_EPS  = 1e-5f;
    static constexpr float ROPE_THETA    = 10000.0f;
    static constexpr float HALT_THRESHOLD = 0.9f;

    /**
     * Optimizer & Training (Static Firmware Defaults)
     */
    static constexpr int64_t BATCH_SIZE     = 64;
    static constexpr int64_t SEQ_LEN        = 512;
    static constexpr float LEARNING_RATE  = 1e-4f;
    static constexpr int64_t EPOCHS         = 10;
    static constexpr int64_t SEED           = 1337;

    static constexpr float ADEMAMIX_BETA1_FAST = 0.9f;
    static constexpr float ADEMAMIX_BETA1_SLOW = 0.9999f;
    static constexpr float ADEMAMIX_BETA2      = 0.999f;
    static constexpr float WEIGHT_DECAY        = 0.0f;
    
    /**
     * Initialization (Biological Primitives)
     */
    static constexpr float INIT_SCALE          = 0.05f;
    static constexpr float DECAY_INIT_OFFSET   = 2.0f;
    static constexpr float DECAY_INIT_SCALE    = 0.1f;
    static constexpr float DELAY_INIT_STD      = 0.5f;
    static constexpr float DELAY_MIN           = 0.1f;
    static constexpr float DELAY_MAX           = 20.0f;
    static constexpr float RAM_INIT_SCALE      = 0.1f;
    static constexpr float DELREC_INIT_MAX     = 5.0f;

    /**
     * Neuron Physics (LIF)
     */
    static constexpr float LIF_DECAY           = 0.9f;
    static constexpr float LIF_THRESHOLD       = 1.0f;
    static constexpr int64_t H_CYCLE_THRESHOLD   = 2;

    /**
     * Metabolic Governance & Survival
     */
    static constexpr bool GLOBAL_BACKPROP     = false;
    static constexpr float LOCAL_LR_RATIO      = 0.5f;
    static constexpr float MES_LOCAL_L1        = 0.01f;
    static constexpr float SURPRISE_REWIRE_THRESHOLD = 0.8f;
    static constexpr float DISSONANCE_PENALTY  = 0.1f;
    static constexpr float METABOLIC_TAX_RATE  = 0.01f;

    static constexpr float SURVIVAL_GAMMA      = 0.01f;
    static constexpr int64_t SURVIVAL_UPDATE_EVERY = 50;
    static constexpr float TARGET_SPARSITY     = 0.9f;
    static constexpr float LAMBDA_COST         = 0.01f;
    static constexpr float LAMBDA_STABILITY    = 0.01f;
    static constexpr float LAMBDA_ENERGY       = 0.01f;

    static constexpr float PARAM_COST_SCALE    = 1e-6f;
    static constexpr float MEMORY_COST_SCALE   = 1e-4f;
    static constexpr float FAST_PATH_COST      = 0.001f;

    /**
     * Engagement & Transparency
     */
    static constexpr float BYPASS_H_DECAY      = 0.99f;
    static constexpr float CURIOSITY_EXPLORE_PROB = 0.05f;
    static constexpr float ENGAGEMENT_THRESHOLD_MIN = 0.05f;
    static constexpr float ENGAGEMENT_THRESHOLD_MAX = 0.95f;
    static constexpr float EFFICIENCY_BONUS_CAP     = 0.05f;

    /**
     * Cache & LGH Metadata
     */
    static constexpr int64_t CACHE_HASH_BITS     = 20;
    static constexpr int64_t LGH_CURVE_LENGTH    = 96;
    static constexpr float LGH_TRACE_DECAY     = 0.90f;
    static constexpr float LGH_TRACE_GAIN      = 0.20f;
    static constexpr float LGH_FOCUS_STRENGTH  = 0.35f;
    static constexpr float LGH_FOCUS_SHARPNESS = 2.0f;

    /**
     * HPC Metadata
     */
    static constexpr int64_t HPC_HIDDEN          = 256;
    static constexpr float HPC_TARGET_ERROR    = 0.05f;
    static constexpr float HPC_HALT_GAIN       = 0.35f;
    static constexpr float HPC_SURPRISE_THRESHOLD = 0.20f;
    static constexpr float HPC_TEMPORAL_THRESHOLD = 0.08f;

    /**
     * SIMD & Alignment Physics
     */
    static constexpr int64_t SIMD_WIDTH = 16; // 512 bits
    static constexpr int64_t ALIGNMENT  = 64;

    /**
     * Metadata & Runtime Strategy (Bio-BIOS)
     */
    static constexpr int64_t TRAIN_SAMPLES_PER_EPOCH = 40000;
    static constexpr int64_t VAL_SAMPLES_PER_EPOCH   = 4000;
    static constexpr int64_t DREAM_REPLAY_BATCHES    = 2;
    static constexpr bool MES_ENABLED             = true;
    static constexpr float COHERENCE_WEIGHT        = 0.01f;
    static constexpr float AGC_CLIP_FACTOR         = 0.1f;

    static constexpr int64_t CPP_OMP_THREADS         = 8;
    static constexpr int64_t TORCH_NUM_THREADS       = 2;
    static constexpr int64_t TORCH_INTEROP_THREADS   = 1;

    static constexpr float TRAIN_LOSS_EMA_DECAY    = 0.98f;
    static constexpr int64_t SPARSITY_LOG_EVERY      = 50;
    static constexpr float KNOWLEDGE_GAMMA         = 0.01f;
    static constexpr int64_t IMPORTANCE_EVERY        = 200;
    static constexpr float IMPORTANCE_RATIO        = 0.3f;
    static constexpr int64_t GATE_UPDATE_EVERY       = 50;
    static constexpr float RAM_PRUNE_FRACTION      = 0.2f;
    static constexpr float RAM_CRITICAL_THRESHOLD  = 0.95f;
    static constexpr int64_t SLEEP_INTERVAL_STEPS    = 1000;

    static constexpr float SPARSITY_THRESHOLD      = 0.5f;
    static constexpr int64_t MAX_LOG_ENTRIES         = 1000;
    static constexpr float EPSILON                 = 1e-8f;
    static constexpr float WARM_UP_STEPS           = 2000.0f;
    static constexpr float SCALE_CONSTANT          = 0.1f;
    static constexpr bool STRICT_CPU_ONLY         = false;

    // Hardware Morton Jump (PDEP)
    inline uint64_t morton_jump_4d(uint32_t x, uint32_t y, uint32_t z, uint32_t t) {
        // Parallel Bits Deposit (PDEP) interleaves bits in 1 clock cycle
        // Requires AVX2/BMI2 support (Haswell+)
        return _pdep_u64(x, 0x1111111111111111ULL) | 
               _pdep_u64(y, 0x2222222222222222ULL) | 
               _pdep_u64(z, 0x4444444444444444ULL) | 
               _pdep_u64(t, 0x8888888888888888ULL);
    }
}

#endif // BRAIN_ISA_H
