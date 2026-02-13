#ifndef BRAIN_ISA_H
#define BRAIN_ISA_H

/**
 * Neural Instruction Set (NIS) Opcodes
 * Maps 8-bit tokens to AVX-512/ASM operations
 */

#define NIS_OP_NOP     0x00
#define NIS_OP_ADD     0x01  // vaddps (Directly adds the signal)
#define NIS_OP_SCALE   0x02  // vmulps (Scales by hardcoded constant)
#define NIS_OP_GATE    0x03  // vblendvps (Conditional gating)
#define NIS_OP_REFLECT 0x04  // Logical bit-shift/inversion
#define NIS_OP_JMP     0x05  // Morton spatial jump

/**
 * Architectural Constants (Migrated from config.py)
 */
#define NIS_L           32
#define NIS_R           8
#define NIS_WORKING_DIM 512
#define NIS_C           4
#define NIS_MEMORY_DEPTH 5

#define NIS_H_CYCLES    2
#define NIS_L_CYCLES    4

#define NIS_RMS_NORM_EPS 1e-5f
#define NIS_ROPE_THETA   10000.0f
#define NIS_HALT_THRESHOLD 0.9f

// Metabolic / Training Constants
#define NIS_LOCAL_LR_RATIO 0.5f
#define NIS_SURPRISE_REWIRE_THRESHOLD 0.8f
#define NIS_DISSONANCE_PENALTY 0.1f
#define NIS_METABOLIC_TAX_RATE 0.01f

// Opcode specific constants
#define NIS_SCALE_CONSTANT 0.75f // Example scale for NIS_OP_SCALE

// AVX-512 alignment parameters
#define NIS_SIMD_WIDTH 16 // 512 bits / 32 bits (float)
#define NIS_ALIGNMENT  64

#endif // BRAIN_ISA_H
