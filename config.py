import torch
import yaml
import os

# ---------------------------
# NIS ARCHITECTURE OVERRIDE
# ---------------------------
try:
    import cpp_loader
    ISA_AVAILABLE = True
except ImportError:
    cpp_loader = None
    ISA_AVAILABLE = False

def load_config(config_path='conf/config.yaml'):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            import yaml
            return yaml.safe_load(f)
    return {}

_CONFIG_DATA = load_config()

class Config:
    # Architecture (Unified via NIS brain_isa.h)
    L = getattr(cpp_loader, 'NIS_L', _CONFIG_DATA.get('model', {}).get('L', 32)) if ISA_AVAILABLE else _CONFIG_DATA.get('model', {}).get('L', 32)
    R = getattr(cpp_loader, 'NIS_R', _CONFIG_DATA.get('model', {}).get('R', 8)) if ISA_AVAILABLE else _CONFIG_DATA.get('model', {}).get('R', 8)
    WORKING_DIM = getattr(cpp_loader, 'NIS_WORKING_DIM', _CONFIG_DATA.get('model', {}).get('working_dim', 512)) if ISA_AVAILABLE else _CONFIG_DATA.get('model', {}).get('working_dim', 512)
    C = getattr(cpp_loader, 'NIS_C', _CONFIG_DATA.get('model', {}).get('C', 4)) if ISA_AVAILABLE else _CONFIG_DATA.get('model', {}).get('C', 4)
    MEMORY_DEPTH = getattr(cpp_loader, 'NIS_MEMORY_DEPTH', _CONFIG_DATA.get('model', {}).get('memory_depth', 5)) if ISA_AVAILABLE else _CONFIG_DATA.get('model', {}).get('memory_depth', 5)
    
    H_CYCLES = _CONFIG_DATA.get('model', {}).get('H_cycles', 2)
    L_CYCLES = _CONFIG_DATA.get('model', {}).get('L_cycles', 4)
    RMS_NORM_EPS = float(_CONFIG_DATA.get('model', {}).get('rms_norm_eps', 1e-5))
    ROPE_THETA = float(_CONFIG_DATA.get('model', {}).get('rope_theta', 10000.0))
    HALT_THRESHOLD = 0.9 # Dynamic ACT halting threshold
    
    # MES (Metabolic Engram Sculpting)
    MES_ENABLED = True
    GLOBAL_BACKPROP = False
    LOCAL_LR_RATIO = 0.5
    MES_SUPER_KERNEL = bool(_CONFIG_DATA.get('model', {}).get('mes_super_kernel', True))
    MES_LOCAL_L1 = float(_CONFIG_DATA.get('training', {}).get('mes_local_l1', 0.01))
    SURPRISE_REWIRE_THRESHOLD = 0.8
    DISSONANCE_PENALTY = 0.1
    H_CYCLE_THRESHOLD = 2 # Tokens taking > 2 cycles are "Rules"
    
    # Training
    TRAINING_MODE = _CONFIG_DATA.get('training', {}).get('mode', 'hybrid')
    LEARNING_RATE = float(_CONFIG_DATA.get('training', {}).get('lr', 1e-4))
    EPOCHS = _CONFIG_DATA.get('training', {}).get('epochs', 10)
    BATCH_SIZE = _CONFIG_DATA.get('training', {}).get('batch_size', 64)
    SEQ_LEN = _CONFIG_DATA.get('training', {}).get('seq_len', 512) 
    MAX_EPOCHS_PER_PHASE = _CONFIG_DATA.get('training', {}).get('max_epochs_per_phase', 50)
    IMPROVE_STREAK = _CONFIG_DATA.get('training', {}).get('improve_streak_needed', 5)
    IMPROVE_DELTA = float(_CONFIG_DATA.get('training', {}).get('improve_min_delta', 1e-3))
    TRAIN_SAMPLES_PER_EPOCH = _CONFIG_DATA.get('training', {}).get('train_samples_per_epoch', None)
    VAL_SAMPLES_PER_EPOCH = _CONFIG_DATA.get('training', {}).get('val_samples_per_epoch', None)
    
    # Numerical Stability & Spiking
    EPSILON = 1e-8
    CLAMP_VAL = 10.0
    THRESH_SPIKE = 0.5
    SURROGATE_ALPHA = 2.0
    
    # Initialization
    INIT_SCALE = 0.05
    DECAY_INIT_OFFSET = 2.0
    DECAY_INIT_SCALE = 0.1
    DELAY_INIT_STD = 0.5
    DELAY_MIN = 0.1
    DELAY_MAX = 20.0
    RAM_INIT_SCALE = 0.1
    DELREC_INIT_MAX = 5.0
    
    # LIF Neuron
    LIF_DECAY = 0.9
    LIF_THRESHOLD = 1.0
    
    # Survival & Costs
    SURVIVAL_GAMMA = _CONFIG_DATA.get('survival', {}).get('gamma', 0.01)
    SURVIVAL_UPDATE_EVERY = int(_CONFIG_DATA.get('survival', {}).get('update_every', 50))
    TARGET_SPARSITY = _CONFIG_DATA.get('survival', {}).get('target_sparsity', 0.9)
    LAMBDA_COST = _CONFIG_DATA.get('survival', {}).get('lambda_cost', 0.01)
    LAMBDA_STABILITY = _CONFIG_DATA.get('survival', {}).get('lambda_stability', 0.01)
    LAMBDA_ENERGY = _CONFIG_DATA.get('survival', {}).get('lambda_energy', 0.01)
    PARAM_COST_SCALE = 1e-6
    MEMORY_COST_SCALE = 1e-4
    FAST_PATH_COST = 0.001
    PHASE_0_KEEP_RATIO = 1.0
    PHASE_1_KEEP_RATIO = 0.05 # Increased density
    METABOLIC_TAX_RATE = 0.01
    
    # Granular Cost Model (Standard Units)
    COST_OPS_VNNI = 0.1       # Relative cost of INT8 op
    COST_OPS_FP32 = 1.0       # Base cost of FP32 op
    COST_MEM_IO   = 5.0       # Cost of moving 1MB from RAM
    COST_RAM_LOOKUP = 2.0     # Cost of a single WNN RAM lookup
    ENERGY_IDLE_WATTS = 5.0   # Baseline power draw (estimated)
    ENERGY_ACTIVE_WATTS = 15.0 # Max power draw (estimated)
    
    # LearningBrain & Pressure
    WARM_UP_STEPS = _CONFIG_DATA.get('learning', {}).get('warm_up_steps', 2000)
    DYNAMIC_ENERGY_SCALE = 0.5 # Increased from 0.1 to force energy awareness
    COHERENCE_WEIGHT = 0.01
    SURVIVAL_WEIGHT = 1.5 # Increased from 0.5 to force active sparsity/cost minimization
    OMEGA_STEP = 0.05     # Increased from 0.01 to accelerate teacher detachment
    CONFUSION_NORM = 2.0
    
    # Brain Updates
    IMPORTANCE_EVERY = _CONFIG_DATA.get('learning', {}).get('importance_every', 200)
    GATE_UPDATE_EVERY = _CONFIG_DATA.get('learning', {}).get('gate_update_every', 50)
    KNOWLEDGE_GAMMA = _CONFIG_DATA.get('learning', {}).get('knowledge_gamma', 0.01)
    IMPORTANCE_RATIO = _CONFIG_DATA.get('learning', {}).get('importance_sample_ratio', 0.3)
    IMPORTANCE_EMA_DECAY = float(_CONFIG_DATA.get('learning', {}).get('importance_ema_decay', 0.95))
    IMPORTANCE_STD_FACTOR = float(_CONFIG_DATA.get('learning', {}).get('importance_std_factor', 0.25))
    
    # Cache
    CACHE_HASH_BITS = _CONFIG_DATA.get('model', {}).get('cache_hash_bits', 20)
    NEURAL_CACHE_ENABLED = bool(_CONFIG_DATA.get('model', {}).get('neural_cache_enabled', False))
    USE_FUSED_COGNITIVE_CYCLE = bool(_CONFIG_DATA.get('model', {}).get('use_fused_cognitive_cycle', True))
    USE_FORWARD_STACK = bool(_CONFIG_DATA.get('model', {}).get('use_forward_stack', True))
    LGH_ENABLED = bool(_CONFIG_DATA.get('model', {}).get('lgh_enabled', True))
    LGH_REPLACE_FORWARD_STACK = bool(_CONFIG_DATA.get('model', {}).get('lgh_replace_forward_stack', True))
    LGH_CURVE_LENGTH = int(_CONFIG_DATA.get('model', {}).get('lgh_curve_length', 96))
    LGH_CURVE_WRAP = bool(_CONFIG_DATA.get('model', {}).get('lgh_curve_wrap', True))
    LGH_MASK_MIN_KEEP = float(_CONFIG_DATA.get('model', {}).get('lgh_mask_min_keep', 0.10))
    LGH_MASK_MAX_KEEP = float(_CONFIG_DATA.get('model', {}).get('lgh_mask_max_keep', 0.90))
    LGH_MORTON_DEPTH = int(_CONFIG_DATA.get('model', {}).get('lgh_morton_depth', 1))
    LGH_PREFETCH_DISTANCE = int(_CONFIG_DATA.get('model', {}).get('lgh_prefetch_distance', 2))
    LGH_ALIGN_MULTIPLE = int(_CONFIG_DATA.get('model', {}).get('lgh_align_multiple', 64))
    LGH_TEMPORAL_BINS = int(_CONFIG_DATA.get('model', {}).get('lgh_temporal_bins', 16))
    LGH_TEMPORAL_FOLD_ALPHA = float(_CONFIG_DATA.get('model', {}).get('lgh_temporal_fold_alpha', 0.25))
    LGH_WAVE_RADIUS = int(_CONFIG_DATA.get('model', {}).get('lgh_wave_radius', 1))
    LGH_WAVE_DECAY = float(_CONFIG_DATA.get('model', {}).get('lgh_wave_decay', 0.65))
    LGH_TRACE_DECAY = float(_CONFIG_DATA.get('model', {}).get('lgh_trace_decay', 0.90))
    LGH_TRACE_GAIN = float(_CONFIG_DATA.get('model', {}).get('lgh_trace_gain', 0.20))
    LGH_LOW_ENTROPY_FOLD_THRESHOLD = float(_CONFIG_DATA.get('model', {}).get('lgh_low_entropy_fold_threshold', 0.015))
    LGH_FOCUS_STRENGTH = float(_CONFIG_DATA.get('model', {}).get('lgh_focus_strength', 0.35))
    LGH_FOCUS_SHARPNESS = float(_CONFIG_DATA.get('model', {}).get('lgh_focus_sharpness', 2.0))
    AUDIT_PERIOD_STEPS = int(_CONFIG_DATA.get('model', {}).get('audit_period_steps', 25))
    AUDIT_RANDOM_PROB = float(_CONFIG_DATA.get('model', {}).get('audit_random_prob', 0.01))
    # Hierarchical Predictive Coding (HPC)
    HPC_ENABLED = bool(_CONFIG_DATA.get('model', {}).get('hpc_enabled', True))
    HPC_HIDDEN = int(_CONFIG_DATA.get('model', {}).get('hpc_hidden', 256))
    HPC_TARGET_ERROR = float(_CONFIG_DATA.get('model', {}).get('hpc_target_error', 0.05))
    HPC_ERROR_EMA_DECAY = float(_CONFIG_DATA.get('model', {}).get('hpc_error_ema_decay', 0.95))
    HPC_TEMPORAL_FOLDING = bool(_CONFIG_DATA.get('model', {}).get('hpc_temporal_folding', True))
    HPC_FOLD_ALPHA = float(_CONFIG_DATA.get('model', {}).get('hpc_fold_alpha', 0.25))
    HPC_CYCLE_MIN_SCALE = float(_CONFIG_DATA.get('model', {}).get('hpc_cycle_min_scale', 0.5))
    HPC_CYCLE_MAX_SCALE = float(_CONFIG_DATA.get('model', {}).get('hpc_cycle_max_scale', 2.0))
    HPC_H_CYCLES_MAX = int(_CONFIG_DATA.get('model', {}).get('hpc_h_cycles_max', 8))
    HPC_L_CYCLES_MAX = int(_CONFIG_DATA.get('model', {}).get('hpc_l_cycles_max', 8))
    HPC_HALT_GAIN = float(_CONFIG_DATA.get('model', {}).get('hpc_halt_gain', 0.35))
    HPC_MONITOR_EVERY = int(_CONFIG_DATA.get('model', {}).get('hpc_monitor_every', 1))
    HPC_SURPRISE_GATE = bool(_CONFIG_DATA.get('model', {}).get('hpc_surprise_gate', True))
    HPC_SURPRISE_THRESHOLD = float(_CONFIG_DATA.get('model', {}).get('hpc_surprise_threshold', 0.20))
    HPC_SURPRISE_MIN_SCALE = float(_CONFIG_DATA.get('model', {}).get('hpc_surprise_min_scale', 0.35))
    HPC_SURPRISE_SKIP_ENABLED = bool(_CONFIG_DATA.get('model', {}).get('hpc_surprise_skip_enabled', True))
    HPC_SURPRISE_SKIP_SCALE = float(_CONFIG_DATA.get('model', {}).get('hpc_surprise_skip_scale', 0.40))
    HPC_TEMPORAL_GATE_ENABLED = bool(_CONFIG_DATA.get('model', {}).get('hpc_temporal_gate_enabled', True))
    HPC_TEMPORAL_GATE_THRESHOLD = float(_CONFIG_DATA.get('model', {}).get('hpc_temporal_gate_threshold', 0.08))
    HPC_TEMPORAL_GATE_MIN_SCALE = float(_CONFIG_DATA.get('model', {}).get('hpc_temporal_gate_min_scale', 0.55))
    HPC_TEMPORAL_GATE_SKIP_ENABLED = bool(_CONFIG_DATA.get('model', {}).get('hpc_temporal_gate_skip_enabled', True))
    HPC_TEMPORAL_GATE_SKIP_SCALE = float(_CONFIG_DATA.get('model', {}).get('hpc_temporal_gate_skip_scale', 0.40))
    HPC_TEMPORAL_GATE_WINDOW = int(_CONFIG_DATA.get('model', {}).get('hpc_temporal_gate_window', 64))
    EVENT_DRIVEN_MODE = str(_CONFIG_DATA.get('runtime', {}).get('event_driven_mode', 'auto')).lower()
    EVENT_DENSITY_THRESHOLD = float(_CONFIG_DATA.get('runtime', {}).get('event_density_threshold', 0.20))
    TTFS_ENABLED = bool(_CONFIG_DATA.get('runtime', {}).get('ttfs_enabled', True))
    TTFS_SLOPE_THRESHOLD = float(_CONFIG_DATA.get('runtime', {}).get('ttfs_slope_threshold', 0.0))
    RAM_INT8_INFER = bool(_CONFIG_DATA.get('runtime', {}).get('ram_int8_infer', True))
    HPC_LOCAL_LOSS_WEIGHT = float(_CONFIG_DATA.get('training', {}).get('hpc_local_loss_weight', 0.25))
    HPC_SURPRISE_LOSS_WEIGHT = float(_CONFIG_DATA.get('training', {}).get('hpc_surprise_loss_weight', 0.10))
    
    # Trainer & Gating
    REFLEX_DROPOUT_RATE = 0.25
    STRICT_CONFIDENCE = 2.0
    STABILITY_THRESHOLD = 0.1
    RAM_CRITICAL_THRESHOLD = 0.95
    RAM_PRUNE_FRACTION = 0.2
    
    # Virtual Lab & Logging
    MAX_LOG_ENTRIES = 1000
    SPARSITY_THRESHOLD = 0.1
    MYELIN_COST = 0.001
    LAMBDA_COST_SCHEDULE_DEFAULT = 0.1
    PRUNING_IMPORTANCE_WEIGHT = 0.3  # Weight for activation magnitude
    PRUNING_SURPRISE_WEIGHT = 0.7    # Weight for temporal change (surprise)
    SLEEP_INTERVAL_STEPS = 1000     # Less frequent sleep
    DREAM_INTENSITY = 10.0          
    DREAM_REPLAY_BATCHES = 2
    MES_SKIP_STEP = 2               # Only do MES update every 2 steps
    
    # MES Biology
    MES_SURPRISE_WEIGHT = 0.8
    MES_GATING_THRESHOLD = 0.05
    MES_REWIRE_INTERVAL = 100

    # HMI Safety & Homeostasis
    FKBP5_CHAOS_LIMIT = 3.0         # Prevent stress from killing the model
    HEALTH_DECAY = 0.95             # EMA for genome stability
    MIN_HEALTH = 0.1                # Prevent total adaptation loss
    HEALTH_RECOVERY_RATE = 0.001    # Gradual healing over time
    DISSONANCE_CONFIDENCE_THRESHOLD = 0.3 # Require S2 confidence > S1 confidence + gap
    PHENOTYPE_UPDATE_EVERY = int(_CONFIG_DATA.get('runtime', {}).get('phenotype_update_every', 200))
    CONSOLIDATE_EVERY = int(_CONFIG_DATA.get('runtime', {}).get('consolidate_every', 500))
    
    # Transparency Gate (Certainty-Bypass)
    TRANSPARENCY_GATE_ENABLED = True
    ENGAGEMENT_THRESHOLD_MIN = 0.05    # Floor at Ω=0 (curious/sponge)
    ENGAGEMENT_THRESHOLD_MAX = 0.95    # Ceiling at Ω=1 (masterly)
    CURIOSITY_EXPLORE_PROB = 0.05      # Chance to engage bypassed data
    BYPASS_H_DECAY = 0.99              # Light memory decay during bypass
    EFFICIENCY_BONUS_CAP = 0.05        # Prevents training divergence
    
    # AdEMAMix Optimizer (Dual-Momentum for Cognitive Organisms)
    OPTIMIZER_TYPE = _CONFIG_DATA.get('training', {}).get('optimizer', 'ademamix')  # 'adam', 'nadam', 'ademamix'
    ADEMAMIX_BETA1_FAST = 0.9      # Fast EMA for System 1 (context/reflexive)
    ADEMAMIX_BETA1_SLOW = 0.9999   # Slow EMA for System 2 (rules/deliberative)
    ADEMAMIX_BETA2 = 0.999         # Variance estimation decay
    AGC_CLIP_FACTOR = 0.1          # Adaptive Gradient Clipping factor (grad/param norm ratio)
    WEIGHT_DECAY = 0.0             # Decoupled weight decay (AdamW-style)
    USE_SIGN_SGD = bool(_CONFIG_DATA.get('training', {}).get('use_sign_sgd', False))

    # Runtime Threading (0 = keep framework defaults)
    TORCH_NUM_THREADS = int(_CONFIG_DATA.get('runtime', {}).get('torch_num_threads', 0))
    TORCH_INTEROP_THREADS = int(_CONFIG_DATA.get('runtime', {}).get('torch_interop_threads', 0))
    CPP_OMP_THREADS = int(_CONFIG_DATA.get('runtime', {}).get('cpp_omp_threads', 0))
    OMEGA_STEP_UPDATE_EVERY = int(_CONFIG_DATA.get('runtime', {}).get('omega_step_update_every', 250))
    GENOME_STEP_UPDATE_EVERY = int(_CONFIG_DATA.get('runtime', {}).get('genome_step_update_every', 5000))
    TRAIN_LOSS_EMA_DECAY = float(_CONFIG_DATA.get('runtime', {}).get('train_loss_ema_decay', 0.98))
    SPARSITY_LOG_EVERY = int(_CONFIG_DATA.get('runtime', {}).get('sparsity_log_every', 50))
    LGH_THERMAL_FREQ_MIN_GHZ = float(_CONFIG_DATA.get('runtime', {}).get('lgh_thermal_freq_min_ghz', 3.0))
    LGH_THERMAL_EMA_DECAY = float(_CONFIG_DATA.get('runtime', {}).get('lgh_thermal_ema_decay', 0.95))
    LGH_THERMAL_PENALTY_WEIGHT = float(_CONFIG_DATA.get('runtime', {}).get('lgh_thermal_penalty_weight', 0.25))
    LGH_THERMAL_SAMPLE_EVERY_S = float(_CONFIG_DATA.get('runtime', {}).get('lgh_thermal_sample_every_s', 2.0))
    LGH_SIMD_CYCLE_PENALTY_WEIGHT = float(_CONFIG_DATA.get('runtime', {}).get('lgh_simd_cycle_penalty_weight', 0.15))
    LGH_SIMD_STARVATION_THRESHOLD = float(_CONFIG_DATA.get('runtime', {}).get('lgh_simd_starvation_threshold', 1200.0))
    LGH_INT4_UNCERTAINTY_THRESHOLD = float(_CONFIG_DATA.get('runtime', {}).get('lgh_int4_uncertainty_threshold', 0.05))
    LGH_FP32_UNCERTAINTY_THRESHOLD = float(_CONFIG_DATA.get('runtime', {}).get('lgh_fp32_uncertainty_threshold', 0.18))
    RUNTIME_DEVICE = str(_CONFIG_DATA.get('runtime', {}).get('device', 'auto')).lower()
    STRICT_CPU_ONLY = bool(_CONFIG_DATA.get('runtime', {}).get('strict_cpu_only', True))
    VIRTUAL_LAB_ENABLED = bool(_CONFIG_DATA.get('runtime', {}).get('virtual_lab_enabled', False))
    PREFLIGHT_ENABLED = bool(_CONFIG_DATA.get('runtime', {}).get('preflight_enabled', True))
    SEED = int(_CONFIG_DATA.get('runtime', {}).get('seed', 1337))
    
    # Global Constants
    if STRICT_CPU_ONLY:
        DEVICE = torch.device('cpu')
        print(">>> Device: CPU (Strict)")
    else:
        if RUNTIME_DEVICE in ('cpu', 'cuda'):
            if RUNTIME_DEVICE == 'cuda' and not torch.cuda.is_available():
                DEVICE = torch.device('cpu')
                print(">>> Device: CPU (CUDA requested but not available)")
            else:
                DEVICE = torch.device(RUNTIME_DEVICE)
                print(f">>> Device: {DEVICE}")
        else: # 'auto'
            if torch.cuda.is_available():
                DEVICE = torch.device('cuda')
                print(">>> Device: CUDA (Auto-detected)")
            else:
                DEVICE = torch.device('cpu')
                print(">>> Device: CPU (Auto-fallback)")

