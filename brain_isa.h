#ifndef BRAIN_ISA_H
#define BRAIN_ISA_H

/**
 * Neural Instruction Set (NIS) Opcodes
 * Maps 8-bit tokens to AVX-512/ASM operations
 */
#define NIS_OP_NOP     0x00
#define NIS_OP_ADD     0x01  // vaddps
#define NIS_OP_SCALE   0x02  // vmulps
#define NIS_OP_GATE    0x03  // vblendvps
#define NIS_OP_REFLECT 0x04  // subps (sign inversion)
#define NIS_OP_JMP     0x05  // Morton spatial jump

/**
 * Structural Dimensions (Fixed Bio-Wiring)
 */
#define NIS_L           32
#define NIS_R           8
#define NIS_WORKING_DIM 512
#define NIS_C           4
#define NIS_MEMORY_DEPTH 5

/**
 * Cognitive Cycle Constraints
 */
#define NIS_H_CYCLES    2
#define NIS_L_CYCLES    4
#define NIS_RMS_NORM_EPS 1e-5f
#define NIS_ROPE_THETA   10000.0f
#define NIS_HALT_THRESHOLD 0.9f

/**
 * Optimizer & Training (Static Firmware Defaults)
 */
#define NIS_BATCH_SIZE     64
#define NIS_SEQ_LEN        512
#define NIS_LEARNING_RATE  1e-4f
#define NIS_EPOCHS         10
#define NIS_SEED           1337

#define NIS_ADEMAMIX_BETA1_FAST 0.9f
#define NIS_ADEMAMIX_BETA1_SLOW 0.9999f
#define NIS_ADEMAMIX_BETA2      0.999f
#define NIS_WEIGHT_DECAY        0.0f
#define NIS_AGC_CLIP_FACTOR     0.1f

/**
 * Initialization (Biological Primitives)
 */
#define NIS_INIT_SCALE          0.05f
#define NIS_DECAY_INIT_OFFSET   2.0f
#define NIS_DECAY_INIT_SCALE    0.1f
#define NIS_DELAY_INIT_STD      0.5f
#define NIS_DELAY_MIN           0.1f
#define NIS_DELAY_MAX           20.0f
#define NIS_RAM_INIT_SCALE      0.1f
#define NIS_DELREC_INIT_MAX     5.0f

/**
 * Neuron Physics (LIF)
 */
#define NIS_LIF_DECAY           0.9f
#define NIS_LIF_THRESHOLD       1.0f
#define NIS_H_CYCLE_THRESHOLD   2

/**
 * Metabolic Governance & Survival
 */
#define NIS_GLOBAL_BACKPROP     0 // boolean
#define NIS_LOCAL_LR_RATIO      0.5f
#define NIS_MES_LOCAL_L1        0.01f
#define NIS_SURPRISE_REWIRE_THRESHOLD 0.8f
#define NIS_DISSONANCE_PENALTY  0.1f
#define NIS_METABOLIC_TAX_RATE  0.01f

#define NIS_SURVIVAL_GAMMA      0.01f
#define NIS_SURVIVAL_UPDATE_EVERY 50
#define NIS_TARGET_SPARSITY     0.9f
#define NIS_LAMBDA_COST         0.01f
#define NIS_LAMBDA_STABILITY    0.01f
#define NIS_LAMBDA_ENERGY       0.01f

#define NIS_PARAM_COST_SCALE    1e-6f
#define NIS_MEMORY_COST_SCALE   1e-4f
#define NIS_FAST_PATH_COST      0.001f

/**
 * Engagement & Transparency
 */
#define NIS_BYPASS_H_DECAY      0.99f
#define NIS_CURIOSITY_EXPLORE_PROB 0.05f
#define NIS_ENGAGEMENT_THRESHOLD_MIN 0.05f
#define NIS_ENGAGEMENT_THRESHOLD_MAX 0.95f
#define NIS_EFFICIENCY_BONUS_CAP     0.05f

/**
 * Cache & LGH Metadata
 */
#define NIS_CACHE_HASH_BITS     20
#define NIS_LGH_CURVE_LENGTH    96
#define NIS_LGH_TRACE_DECAY     0.90f
#define NIS_LGH_TRACE_GAIN      0.20f
#define NIS_LGH_FOCUS_STRENGTH  0.35f
#define NIS_LGH_FOCUS_SHARPNESS 2.0f

/**
 * HPC Metadata
 */
#define NIS_HPC_HIDDEN          256
#define NIS_HPC_TARGET_ERROR    0.05f
#define NIS_HPC_HALT_GAIN       0.35f
#define NIS_HPC_SURPRISE_THRESHOLD 0.20f
#define NIS_HPC_TEMPORAL_THRESHOLD 0.08f

/**
 * SIMD & Alignment Physics
 */
#define NIS_SIMD_WIDTH 16 // 512 bits
#define NIS_ALIGNMENT  64

/**
 * Metadata & Runtime Strategy (Bio-BIOS)
 */
#define NIS_TRAIN_SAMPLES_PER_EPOCH 40000
#define NIS_VAL_SAMPLES_PER_EPOCH   4000
#define NIS_DREAM_REPLAY_BATCHES    2
#define NIS_MES_ENABLED             1
#define NIS_COHERENCE_WEIGHT        0.01f
#define NIS_AGC_CLIP_FACTOR         0.1f

#define NIS_CPP_OMP_THREADS         8
#define NIS_TORCH_NUM_THREADS       2
#define NIS_TORCH_INTEROP_THREADS   1

#define NIS_TRAIN_LOSS_EMA_DECAY    0.98f
#define NIS_SPARSITY_LOG_EVERY      50
#define NIS_KNOWLEDGE_GAMMA         0.01f
#define NIS_IMPORTANCE_EVERY        200
#define NIS_IMPORTANCE_RATIO        0.3f
#define NIS_GATE_UPDATE_EVERY       50
#define NIS_RAM_PRUNE_FRACTION      0.2f
#define NIS_RAM_CRITICAL_THRESHOLD  0.95f
#define NIS_SLEEP_INTERVAL_STEPS    1000

#define NIS_SPARSITY_THRESHOLD      0.5f
#define NIS_MAX_LOG_ENTRIES         1000
#define NIS_EPSILON                 1e-8f
#define NIS_WARM_UP_STEPS           2000.0f
#define NIS_SCALE_CONSTANT          0.1f

#endif // BRAIN_ISA_H
