import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import psutil
from torch.utils.tensorboard import SummaryWriter
import time
import random
import yaml
import os
from genome import Genome
from morton import MortonBuffer
import math
from types import SimpleNamespace
from optimizers import AdEMAMix
from accelerator import get_accelerator

_ACCEL = get_accelerator()
cpp_loader = _ACCEL.loader


def _cpp_ready(*tensors):
    return _ACCEL.ready(None, *tensors)


def _cpp_has(op_name, *tensors):
    return _ACCEL.has(op_name, *tensors)


def _cpp_try(op_name, *args, tensors=None):
    return _ACCEL.call(op_name, *args, tensors=tensors)



# ---------------------------
# Hyperparameters & Constants
# ---------------------------

# --- COMPUTE WASTE RANKING ($O(N^n)$ Analysis) ---
# 1. RAMTupleLayer.forward (Python): RANK 1 (CRITICAL WASTE)
#    Why: Nested Python loop [M, K] with per-bit spiking logic.
#    Solution: Already uses cpp_loader.dcls_ram_lookup as primary path.
#
# 2. CognitiveOrganism.forward (H-Cycle): RANK 2 (HIGH WASTE)
#    Why: Triple nested loop [H_cycles, L_cycles, L].
#    Target for Speedup: JIT compilation (torch.compile) or C++ layer-stacking.
#
# 3. parallel_scan_optimized: RANK 3 (MEDIUM WASTE)
#    Why: Large memory footprints during cumsum.
# --------------------------------------------------
from config import Config

L = Config.L
R = Config.R
D = Config.D
C = Config.C
DEVICE = Config.DEVICE

# ---------------------------
# Modern Layers (Recurrent Cognitive Architecture)
# ---------------------------
def rms_norm(x, eps=Config.RMS_NORM_EPS):
    """RMSNorm via C++ kernel (fail-fast if unavailable)."""
    return _cpp_try('rms_norm', x.contiguous(), eps, tensors=(x,))


class HomeostaticModule(nn.Module):
    """
    Unified Governance & Homeostasis Module:
    1. Strategic Gating (Engagement/Certainty & Importance).
    2. Metabolic Governance (Usage Tracking & Reliability).
    3. Consolidated Configuration (Scalar Buffers for C++ kernels).
    """
    def __init__(self, L, R, input_dim, hidden_dim=128, device=None):
        super().__init__()
        self.L, self.R = L, R
        self.device = device
        
        # --- 1. Strategic Gating (Architecture from StrategicGatingModule) ---
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.engagement_head = nn.Linear(hidden_dim, 1)   # Logit for Transparency Gate
        self.importance_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )                                                # Probability for imprinting
        
        # --- 2. Metabolic Governance (Migrated from Genome) ---
        self.register_buffer('usage', torch.ones(L, R))
        self.register_buffer('reliability', torch.ones(L, R))
        
        # --- 3. Consolidated configuration for C++ kernels ---
        # 0: gamma, 1: imp_w, 2: surp_w, 3: metabolic_pressure, 4: tps_pressure, 5: noise
        # 6: hpc_error_ema, 7: hpc_last_error, 8: hpc_last_cycle_scale
        # 9: surprise_signal, 10: surprise_cycle_scale
        # 11: temporal_signal, 12: temporal_cycle_scale
        # 13: thermal_penalty_ema, 14: last_freq_ghz, 15: current_engagement_rate
        # 16-20: forward_stack_scalars, 21-23: mes_super_scalars, 24-31: reserved
        self.register_buffer('config_scalars', torch.zeros(32, dtype=torch.float32))
        self.config_scalars[0] = 0.95 # Default gamma
        self.config_scalars[1] = 1.0  # Default importance weight
        self.config_scalars[2] = 1.0  # Default surprise weight
        self.config_scalars[5] = 0.05 # Default noise
        self.config_scalars[15] = 1.0 # Default engagement rate
        
        if device:
            self.to(device)
            
    def update_metabolism(self, H, H_prev=None, bdnf_gamma=None):
        """Update usage scores via C++ kernel survival_update_io."""
        H_c = H.contiguous()
        H_prev_t = H_prev.contiguous() if H_prev is not None else torch.empty(0, device=H.device, dtype=H_c.dtype)
        
        if bdnf_gamma is not None:
            self.config_scalars[0] = float(bdnf_gamma)
            
        # survival_update_io handles the EMA update of self.usage
        from organism import _cpp_try
        out = _cpp_try(
            'survival_update_io',
            H_c,
            self.usage,
            [H_prev_t],
            self.config_scalars[:3], # Points 0, 1, 2
            tensors=(H_c, self.usage, H_prev_t, self.config_scalars)
        )
        # Note: In-place update might be handled by C++ or return value.
        # Genome implementation assigned it: self.usage = out
        if out is not None:
            self.usage = out
            
    def get_metabolic_mask(self, metabolic_pressure, tps_pressure):
        """Generate sparse activation mask via C++ kernel survival_mask_io."""
        self.config_scalars[3] = float(metabolic_pressure)
        self.config_scalars[4] = float(tps_pressure)
        
        from organism import _cpp_try
        out = _cpp_try(
            'survival_mask_io',
            self.usage,
            self.reliability,
            self.config_scalars[3:6], # Points 3, 4, 5
            tensors=(self.usage, self.reliability, self.config_scalars)
        )
        return out

    def forward(self, x):
        """Strategic Gating Forward Pass."""
        features = self.shared(x)
        engagement = self.engagement_head(features)
        importance = self.importance_head(features)
        return engagement, importance




class SwiGLU(nn.Module):
    def __init__(self, dim, expansion=2.0, out_dim=None):
        super().__init__()
        self.out_dim = out_dim or dim
        hidden_dim = int(dim * expansion)
        self.w1 = VNNILinear(dim, hidden_dim, bias=False)
        self.w2 = VNNILinear(hidden_dim, self.out_dim, bias=False)
        self.w3 = VNNILinear(dim, hidden_dim, bias=False)

    def forward(self, x):
        # Handle scaling if hidden_dim != dim
        out = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return out

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=Config.ROPE_THETA, device=DEVICE):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def apply_rotary_pos_emb(x, cos, sin):
    # x: [B, T, R, D]
    # cos, sin: [T, D]
    d = x.shape[-1]
    x1 = x[..., :d//2]
    x2 = x[..., d//2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos.view(1, -1, 1, d)) + (rotated * sin.view(1, -1, 1, d))

# ---------------------------
# Utilities
# ---------------------------
def init_state(L, R, D, C, device=DEVICE, scale=Config.INIT_SCALE):
    return torch.randn(L, R, D, C, device=device) * scale

def parallel_scan_optimized(u, decay):
    """Pruned parallel scan; restricted to recursive updates if Phase 0."""
    return _cpp_try('parallel_scan', u.contiguous(), decay.contiguous(), tensors=(u, decay))

def fused_rms_mean(x_seq):
    return _cpp_try('fused_rms_mean', x_seq.contiguous(), tensors=(x_seq,))

# ---------------------------
# Virtual Lab
# ---------------------------
class VirtualLab:
    def __init__(self, log_dir="logs/virtual_lab", enabled=False):
        self.logs = []
        self.enabled = bool(enabled)
        self.step_count = 0
        self._noise_step = -1
        self._bench_cache_key = None
        self._bench_cache_value = {'total_steps': 0, 'tps': 0.0, 'tps_pressure': 0.0}
        self._log_dir = log_dir
        self.writer = None
        self.process = psutil.Process()
        if self.enabled:
            self._ensure_writer()

    def _ensure_writer(self):
        if self.writer is None:
            self.writer = SummaryWriter(self._log_dir)
            print(f">>> VirtualLab Initialized. Logging to {self._log_dir}")

    def enable(self):
        self.enabled = True
        self._ensure_writer()

    def disable(self):
        self.enabled = False

    def log_step(self, data):
        if (not self.enabled) or (self.writer is None):
            return
        self.step_count += 1
        
        # Throttled OS logging to avoid overhead
        if self.step_count % 10 != 0:
            return

        cpu_usage = self.process.cpu_percent()
        ram_usage = self.process.memory_info().rss / 1e9 # GB

        entry = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    entry[k] = float(v.item())
                elif k == 'mask':
                    # Only compute sparsity if we are logging OS metrics
                    entry['mask_sparsity'] = 1.0 - (v > Config.SPARSITY_THRESHOLD).float().mean().item()
                elif k == 'ram_addresses' and self.step_count % 100 == 0:
                    entry['ram_addresses'] = v.detach().cpu()
            elif isinstance(v, (int, float, bool)):
                entry[k] = float(v)
        
        entry['timestamp'] = time.time()
        entry['step'] = self.step_count
        entry['os_cpu_percent'] = cpu_usage
        entry['os_ram_gb'] = ram_usage
        
        # TensorBoard Logging
        for k, v in entry.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"VirtualLab/{k}", v, self.step_count)
        
        if self._noise_step >= 0 and 'H' in data:
            entry['recovery_norm'] = data['H'].norm().item() / (data['H'].shape[0] ** 0.5)
            
        self.logs.append(entry)
        if len(self.logs) > Config.MAX_LOG_ENTRIES:
            self.logs.pop(0)
        self._bench_cache_key = None



    def get_benchmarks(self):
        """
        Returns high-fidelity benchmarking metrics including LEARNING VS MEMORIZATION diagnostics.
        """
        if not self.logs:
            return {'total_steps': self.step_count, 'tps': 0.0, 'tps_pressure': 0.0}

        first_ts = self.logs[0].get('timestamp', 0.0)
        last_ts = self.logs[-1].get('timestamp', 0.0)
        cache_key = (len(self.logs), first_ts, last_ts)
        if self._bench_cache_key == cache_key and self._bench_cache_value is not None:
            cached = dict(self._bench_cache_value)
            cached['total_steps'] = self.step_count
            return cached
        
        # Filtering logs for valid numerical values
        mask_sparsity_logs = [log['mask_sparsity'] for log in self.logs if 'mask_sparsity' in log]
        cost_logs = [log['cost_step'] for log in self.logs if 'cost_step' in log]
        energy_logs = [log['loss_energy'] for log in self.logs if 'loss_energy' in log]
        loss_logs = [log['loss_task'] for log in self.logs if 'loss_task' in log]
        val_loss_logs = [log['val_loss'] for log in self.logs if 'val_loss' in log]
        address_logs = [log['ram_addresses'] for log in self.logs if 'ram_addresses' in log]
        
        # Timing calculation for TPS
        total_time = self.logs[-1]['timestamp'] - self.logs[0]['timestamp'] if len(self.logs) > 1 else 0.1
        total_tokens = sum([log.get('t', 1) for log in self.logs])
        tps = total_tokens / total_time
        
        # TPS Pressure: Higher when TPS is low compared to target (e.g. 50 TPS)
        target_tps = 50.0
        tps_pressure = max(0.0, 1.0 - (tps / target_tps))
        
        if not mask_sparsity_logs:
            result = {'total_steps': self.step_count, 'tps': tps, 'tps_pressure': tps_pressure}
            self._bench_cache_key = cache_key
            self._bench_cache_value = dict(result)
            return result
        
        avg_loss = sum(loss_logs) / len(loss_logs) if loss_logs else 2.0
        avg_val_loss = sum(val_loss_logs) / len(val_loss_logs) if val_loss_logs else avg_loss
        avg_sparsity = sum(mask_sparsity_logs) / len(mask_sparsity_logs)
        avg_cost = sum(cost_logs) / len(cost_logs) if cost_logs else 0.0
        avg_energy = sum(energy_logs) / len(energy_logs) if energy_logs else 0.0
        
        # BPC Calculation
        avg_bpc = avg_loss / 0.693147
        
        # Generalization Gap
        generalization_gap = avg_val_loss - avg_loss
        is_memorizing = generalization_gap > 0.5
        
        # Address Entropy
        address_entropy = 0.0
        knowledge_coverage = 0.0
        total_possible = 1
        if address_logs:
            all_addresses = torch.cat(address_logs, dim=0)
            if all_addresses.numel() > 0:
                unique_addresses = len(torch.unique(all_addresses))
                total_possible = all_addresses.max().item() + 1
                knowledge_coverage = unique_addresses / max(1, total_possible)
                counts = torch.bincount(all_addresses.flatten().long())
                probs = counts.float() / counts.sum()
                probs = probs[probs > 0]
                address_entropy = -(probs * torch.log2(probs + 1e-9)).sum().item()
        
        # Generalization Score
        entropy_normalized = address_entropy / max(1, np.log2(total_possible) if address_logs else 8)
        generalization_score = (
            (1.0 - min(1.0, abs(generalization_gap)))
            * (entropy_normalized + 0.1)
            * (knowledge_coverage + 0.1)
        ) * 10
        
        # Virtualization Score 2.0
        performance_index = 1.0 / (avg_bpc + Config.EPSILON) * (tps / 100.0)
        resource_index = (avg_cost + avg_energy + Config.EPSILON)
        v_score_2 = performance_index / resource_index
        
        # Mood Metric
        halt_logs = [log['halt'] for log in self.logs if 'halt' in log]
        avg_halt = sum(halt_logs) / len(halt_logs) if halt_logs else 0.5
        
        result = {
            'avg_sparsity': avg_sparsity,
            'avg_cost': avg_cost,
            'avg_energy': avg_energy,
            'avg_bpc': avg_bpc,
            'avg_mood': 1.0 - avg_halt,
            'tps': tps,
            'tps_pressure': tps_pressure,
            'virtualization_score_2.0': v_score_2,
            'total_steps': self.step_count,
            'train_loss': avg_loss,
            'val_loss': avg_val_loss,
            'generalization_gap': generalization_gap,
            'is_memorizing': is_memorizing,
            'address_entropy': address_entropy,
            'knowledge_coverage': knowledge_coverage,
            'generalization_score': generalization_score,
        }
        self._bench_cache_key = cache_key
        self._bench_cache_value = dict(result)
        return result

# ---------------------------
# Spiking & Weightless Layers (C++ ACCELERATED)
# ---------------------------

# Multi-layer RAM lookups are handled by fused C++ kernels (forward_stack_io).


# --- RAM Parameter Storage ---


class VNNILinear(nn.Module):
    """
    Path B: Feed-Forward Projection using INT8 VNNI (Ice Lake)
    Wraps cpp_loader.quantized_matmul for high throughput.
    """
    def __init__(self, in_features, out_features, bias=True, device=DEVICE):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features, device=device) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device))
        else:
            self.register_parameter('bias', None)
        
        # Cache for quantized weights to avoid redundant work
        self.register_buffer('w_q', None)
        self.register_buffer('scale_w', None)
        self._is_quantized = False

    def quantize(self):
        """Pre-quantize weights for high-throughput inference."""
        with torch.no_grad():
            w_max = torch.max(torch.abs(self.weight), dim=1, keepdim=True).values.clamp(min=1e-6)
            self.scale_w = w_max / 127.0
            self.w_q = torch.round(self.weight / self.scale_w).to(torch.int8)
            self._is_quantized = True

    def forward(self, x):
        if _cpp_ready(x) and not self.training: # Use VNNI for inference speedup
            if not self._is_quantized:
                self.quantize()
            
            # Handle tensors [..., D] for C++ kernel which expects (N, D)
            original_shape = x.shape
            if x.dim() > 2:
                x = x.reshape(-1, original_shape[-1])
            
            bias = self.bias if self.bias is not None else torch.zeros(self.out_features, device=x.device)
            y = cpp_loader.quantized_matmul(x.contiguous(), self.w_q, self.scale_w.flatten(), bias)
            
            if len(original_shape) > 2:
                new_shape = list(original_shape[:-1]) + [-1]
                y = y.reshape(new_shape)
            return y
        
        # Reset quantization flag if we return to training
        if self.training:
            self._is_quantized = False
        
        # Fallback to standard Torch path (MKL/OneDNN)
        return F.linear(x, self.weight, self.bias)

class RAMTupleLayer(nn.Module):
    def __init__(self, M, K, D_in, device=DEVICE):
        super().__init__()
        self.M, self.K, self.D_in = M, K, D_in
        self.device = device
        self.ram_int8_infer = bool(Config.RAM_INT8_INFER)
        
        # FIX #2: Log-Normal Delay Initialization
        # Biases toward recent past (t-1, t-2) with long tail for far past
        log_delays = torch.randn(M, K, device=device) * Config.DELAY_INIT_STD
        delays_raw = torch.exp(log_delays)
        # Clamp to reasonable range
        self.delays = nn.Parameter(torch.clamp(delays_raw, Config.DELAY_MIN, Config.DELAY_MAX))
        
        self.ram_tables = nn.Parameter(torch.randn(M, 1 << K, device=device) * Config.RAM_INIT_SCALE)
        self.register_buffer('connections', torch.randint(0, D_in, (M, K), device=device))
        self.register_buffer('ram_tables_q', torch.zeros(M, 1 << K, dtype=torch.int8, device=device))
        self.register_buffer('ram_scales', torch.ones(M, dtype=torch.float32, device=device))
        self._quantized_valid = False

    @torch.no_grad()
    def _refresh_quantized_tables(self):
        tables = self.ram_tables.detach().float()
        scales = tables.abs().amax(dim=1).clamp(min=1e-6) / 127.0
        q = torch.round(tables / scales.unsqueeze(1)).clamp(-127.0, 127.0).to(torch.int8)
        self.ram_scales.copy_(scales)
        self.ram_tables_q.copy_(q)
        self._quantized_valid = True

                                    


# Spiking logic is handled by C++ kernels (fused_lif_ram_lookup / forward_stack_io).

class BaseCognitiveModule(nn.Module):
    """Shared tensor/device plumbing for cognitive modules."""

    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device

    def _to_device(self, x, dtype=None):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if dtype is None:
            return x.to(self.device)
        return x.to(self.device, dtype=dtype)

    @staticmethod
    def _ensure_contig(x):
        if isinstance(x, torch.Tensor):
            return x if x.is_contiguous() else x.contiguous()
        return x

    @staticmethod
    def _require_dim(x, expected, name):
        if x.dim() != expected:
            raise ValueError(f"{name} must have {expected} dims, got {tuple(x.shape)}")

    @staticmethod
    def _require_cpu(*tensors):
        for idx, t in enumerate(tensors):
            if isinstance(t, torch.Tensor) and t.device.type != 'cpu':
                raise RuntimeError(f"Tensor at position {idx} must be on CPU for this path.")

# BaseOrganismLayer removed; utilities moved to BaseCognitiveModule.


class OrganismLevel(BaseCognitiveModule):
    def __init__(self, R, D, C, device=DEVICE, cfg=Config):
        super().__init__(device=device)
        self.R, self.D, self.C = R, D, C
        self.cfg = cfg
        
        # Homeostatic Module: O(1) engagement & importance + metabolic governance
        self.homeostasis = HomeostaticModule(L=1, R=R, input_dim=D*C, device=device)
        self.strategic_gating = self.homeostasis # Alias for backward compatibility
        self.omega_ref = 0.0  # Synced from parent CognitiveOrganism
        self.h_decay_rate = self.cfg.BYPASS_H_DECAY  # Light memory decay during bypass
        
        self.wsnn = RAMTupleLayer(M=D*C, K=8, D_in=D*C, device=device)
        self.halt_head = VNNILinear(D * C, 1).to(device)
        self.firing_rate_ema = torch.tensor(0.5, device=device)
        
        # Decay Parameter: Learned scalar per block, initialized for stability
        self.raw_decay = nn.Parameter(
            (torch.randn(self.R, 1, device=device) * self.cfg.DECAY_INIT_SCALE) 
            + self.cfg.DECAY_INIT_OFFSET
        )
        
        # MES state for C++ optimizer (batched_ademamix_update)
        if self.cfg.MES_ENABLED:
            self.register_buffer('m_fast', torch.zeros_like(self.wsnn.ram_tables))
            self.register_buffer('m_slow', torch.zeros_like(self.wsnn.ram_tables))
            self.register_buffer('v_opt', torch.zeros_like(self.wsnn.ram_tables))
            self.register_buffer('_zero_grad_ram', torch.zeros_like(self.wsnn.ram_tables))
            self.opt_step = 1

                                    



class CognitiveOrganism(BaseCognitiveModule):
    def __init__(self, input_dim, L, R, D=None, C=Config.C, memory_depth=5, device=DEVICE, output_dim=None, vocab_size=None, d_s1=None, d_s2=None):
        super().__init__(device=device)
        self.cfg = SimpleNamespace(
            D_S1=Config.WORKING_DIM // 8, # Derived for bit- latent breadcrumb
            D_S2=Config.WORKING_DIM,
            WORKING_DIM=Config.WORKING_DIM,
            C=Config.C,
            MES_ENABLED=Config.MES_ENABLED,
            LEARNING_RATE=Config.LEARNING_RATE,
            LOCAL_LR_RATIO=Config.LOCAL_LR_RATIO,
            CACHE_HASH_BITS=Config.CACHE_HASH_BITS,
            NEURAL_CACHE_ENABLED=Config.NEURAL_CACHE_ENABLED,
            H_CYCLES=Config.H_CYCLES,
            L_CYCLES=Config.L_CYCLES,
            BYPASS_H_DECAY=Config.BYPASS_H_DECAY,
            DELREC_INIT_MAX=Config.DELREC_INIT_MAX,
            LIF_DECAY=Config.LIF_DECAY,
            LIF_THRESHOLD=Config.LIF_THRESHOLD,
            DECAY_INIT_SCALE=Config.DECAY_INIT_SCALE,
            DECAY_INIT_OFFSET=Config.DECAY_INIT_OFFSET,
        )
        self.exec_cfg = SimpleNamespace(
            use_forward_stack=bool(Config.USE_FORWARD_STACK),
            use_fused_cycle=bool(Config.USE_FUSED_COGNITIVE_CYCLE),
            event_mode=str(Config.EVENT_DRIVEN_MODE).strip().lower(),
            event_density_threshold=float(Config.EVENT_DENSITY_THRESHOLD),
            ttfs_enabled=bool(Config.TTFS_ENABLED),
            ttfs_slope_threshold=float(Config.TTFS_SLOPE_THRESHOLD),
        )
        self.lgh_cfg = SimpleNamespace(
            enabled=bool(Config.LGH_ENABLED),
            replace_forward_stack=bool(Config.LGH_REPLACE_FORWARD_STACK),
            curve_length=max(1, int(Config.LGH_CURVE_LENGTH)),
            curve_wrap=bool(Config.LGH_CURVE_WRAP),
            mask_min_keep=float(Config.LGH_MASK_MIN_KEEP),
            mask_max_keep=float(Config.LGH_MASK_MAX_KEEP),
            morton_depth=max(1, int(Config.LGH_MORTON_DEPTH)),
            prefetch_distance=max(1, int(Config.LGH_PREFETCH_DISTANCE)),
            align_multiple=max(1, int(getattr(Config, 'LGH_ALIGN_MULTIPLE', 64))),
            temporal_bins=max(1, int(getattr(Config, 'LGH_TEMPORAL_BINS', 16))),
            temporal_fold_alpha=float(getattr(Config, 'LGH_TEMPORAL_FOLD_ALPHA', 0.25)),
            wave_radius=max(0, int(getattr(Config, 'LGH_WAVE_RADIUS', 1))),
            wave_decay=float(getattr(Config, 'LGH_WAVE_DECAY', 0.65)),
            trace_decay=float(getattr(Config, 'LGH_TRACE_DECAY', 0.90)),
            trace_gain=float(getattr(Config, 'LGH_TRACE_GAIN', 0.20)),
            low_entropy_fold_threshold=float(getattr(Config, 'LGH_LOW_ENTROPY_FOLD_THRESHOLD', 0.015)),
            focus_strength=float(getattr(Config, 'LGH_FOCUS_STRENGTH', 0.35)),
            focus_sharpness=float(getattr(Config, 'LGH_FOCUS_SHARPNESS', 2.0)),
            thermal_freq_min_ghz=float(Config.LGH_THERMAL_FREQ_MIN_GHZ),
            thermal_ema_decay=float(Config.LGH_THERMAL_EMA_DECAY),
            thermal_penalty_weight=float(Config.LGH_THERMAL_PENALTY_WEIGHT),
            int4_uncertainty_threshold=float(getattr(Config, 'LGH_INT4_UNCERTAINTY_THRESHOLD', 0.05)),
            fp32_uncertainty_threshold=float(getattr(Config, 'LGH_FP32_UNCERTAINTY_THRESHOLD', 0.18)),
        )
        self.hpc_cfg = SimpleNamespace(
            enabled=bool(Config.HPC_ENABLED),
            hidden=int(Config.HPC_HIDDEN),
            target_error=float(Config.HPC_TARGET_ERROR),
            error_ema_decay=float(Config.HPC_ERROR_EMA_DECAY),
            temporal_folding=bool(Config.HPC_TEMPORAL_FOLDING),
            fold_alpha=float(Config.HPC_FOLD_ALPHA),
            cycle_min_scale=float(Config.HPC_CYCLE_MIN_SCALE),
            cycle_max_scale=float(Config.HPC_CYCLE_MAX_SCALE),
            h_cycles_max=int(Config.HPC_H_CYCLES_MAX),
            l_cycles_max=int(Config.HPC_L_CYCLES_MAX),
            halt_gain=float(Config.HPC_HALT_GAIN),
            monitor_every=int(Config.HPC_MONITOR_EVERY),
            local_loss_weight=float(Config.HPC_LOCAL_LOSS_WEIGHT),
            surprise_gate=bool(Config.HPC_SURPRISE_GATE),
            surprise_threshold=float(Config.HPC_SURPRISE_THRESHOLD),
            surprise_min_scale=float(Config.HPC_SURPRISE_MIN_SCALE),
            surprise_skip_enabled=bool(Config.HPC_SURPRISE_SKIP_ENABLED),
            surprise_skip_scale=float(Config.HPC_SURPRISE_SKIP_SCALE),
            temporal_gate_enabled=bool(Config.HPC_TEMPORAL_GATE_ENABLED),
            temporal_gate_threshold=float(Config.HPC_TEMPORAL_GATE_THRESHOLD),
            temporal_gate_min_scale=float(Config.HPC_TEMPORAL_GATE_MIN_SCALE),
            temporal_gate_skip_enabled=bool(Config.HPC_TEMPORAL_GATE_SKIP_ENABLED),
            temporal_gate_skip_scale=float(Config.HPC_TEMPORAL_GATE_SKIP_SCALE),
            temporal_gate_window=max(2, int(Config.HPC_TEMPORAL_GATE_WINDOW)),
            surprise_loss_weight=float(Config.HPC_SURPRISE_LOSS_WEIGHT),
        )
        self.audit_cfg = SimpleNamespace(
            period_steps=int(Config.AUDIT_PERIOD_STEPS),
            random_prob=float(Config.AUDIT_RANDOM_PROB),
        )
        self.importance_cfg = SimpleNamespace(
            std_factor=float(Config.IMPORTANCE_STD_FACTOR),
            ema_decay=float(Config.IMPORTANCE_EMA_DECAY),
        )
        self.survival_cfg = SimpleNamespace(
            update_every=int(Config.SURVIVAL_UPDATE_EVERY),
        )
        self.mes_cfg = SimpleNamespace(
            enabled=bool(Config.MES_ENABLED),
            global_backprop=bool(Config.GLOBAL_BACKPROP),
            super_kernel=bool(Config.MES_SUPER_KERNEL),
            local_l1=float(Config.MES_LOCAL_L1),
            skip_step=max(1, int(Config.MES_SKIP_STEP)),
        )
        self.lifecycle_cfg = SimpleNamespace(
            consolidate_every=max(1, int(Config.CONSOLIDATE_EVERY)),
            phenotype_update_every=max(1, int(Config.PHENOTYPE_UPDATE_EVERY)),
        )
        self.runtime_cfg = SimpleNamespace(
            lif_decay=float(Config.LIF_DECAY),
            lif_threshold=float(Config.LIF_THRESHOLD),
            halt_threshold=float(Config.HALT_THRESHOLD),
            phase0_keep_ratio=float(Config.PHASE_0_KEEP_RATIO),
            fast_path_cost=float(Config.FAST_PATH_COST),
            param_cost_scale=float(Config.PARAM_COST_SCALE),
            dissonance_penalty=float(Config.DISSONANCE_PENALTY),
            dissonance_confidence_threshold=float(Config.DISSONANCE_CONFIDENCE_THRESHOLD),
            metabolic_tax_rate=float(Config.METABOLIC_TAX_RATE),
            omega_step=float(Config.OMEGA_STEP),
        )
        self.lambda_sparsity = Config.LAMBDA_COST # Enforced early init to fix AttributeError
        self.L, self.R, self.C = L, R, C
        self.input_dim = input_dim
        # Bit-Level Modeling always expects 8 bits as output
        self.output_dim = 8
        self.vocab_size = vocab_size
        
        # Consolidated Dimensions
        self.working_dim = d_s2 or D or self.cfg.WORKING_DIM
        self.d_s1 = d_s1 or (self.working_dim // 8) # Fast path usually 1/8th
        self.d_s2 = self.working_dim
        
        # Bit-Level Modeling (BLT Style)
        # We replace the Byte Embedding with a Bit-to-Latent projection
        # Raw Bits: [B, T, 8] -> Latent: [B, T, d_s1 * C]
        self.bit_to_latent = nn.Linear(8, self.d_s1 * C).to(self.device)
        self.register_buffer('byte_bit_shifts', torch.arange(7, -1, -1, dtype=torch.long, device=self.device).view(1, 1, -1))
        if self.cfg.MES_ENABLED:
            self.s1_optimizer = torch.optim.SGD(self.bit_to_latent.parameters(), lr=self.cfg.LEARNING_RATE * self.cfg.LOCAL_LR_RATIO)
            
        # Sensory Noise State (Anti-Cheating / Manifold Expansion)
        self.noise_scale = 0.0
        self.min_noise = 0.0
        self.max_noise = 2.0 # Allow massive noise for 1-batch datasets
        self.cfg_cache_enabled = bool(self.cfg.NEURAL_CACHE_ENABLED)
             
        # Neural Cache (System 1)
        if self.cfg_cache_enabled:
            self.neural_cache = NeuralCache(
                input_dim=self.d_s1 * C, 
                output_dim=self.output_dim, 
                hash_bits=self.cfg.CACHE_HASH_BITS,
                num_tables=4,
                device=self.device
            )
        else:
            self.neural_cache = None

        # Non-Linear Cognitive Bridge: System 1 -> System 2 (The Uploader)
        self.bridge_s1_to_s2 = SwiGLU(self.d_s1 * C, expansion=2.0, out_dim=self.d_s2 * C)
        self.homeostasis = HomeostaticModule(L=L, R=R, input_dim=self.d_s1 * C, device=self.device)
        self.strategic_gating = self.homeostasis # Unified alias
        
        # System 2: Slow/Reasoning Path (D_S2)
        self.levels = nn.ModuleList([OrganismLevel(R, self.d_s2, C, device=device, cfg=self.cfg) for _ in range(L)])
        self.hpc_enabled = self.hpc_cfg.enabled
        self._hpc_dim = self.d_s2 * C
        if self.hpc_enabled and self.L > 1:
            hpc_hidden = max(64, min(self.hpc_cfg.hidden, self._hpc_dim))
            self.hpc_encoder = nn.Linear(self._hpc_dim, hpc_hidden, bias=False).to(self.device)
            self.hpc_decoder = nn.Linear(hpc_hidden, self._hpc_dim, bias=False).to(self.device)
            self.hpc_layer_gain = nn.Parameter(torch.ones(self.L - 1, self._hpc_dim, device=self.device))
            self.hpc_layer_bias = nn.Parameter(torch.zeros(self.L - 1, self._hpc_dim, device=self.device))
        else:
            self.hpc_encoder = None
            self.hpc_decoder = None
            self.register_parameter('hpc_layer_gain', None)
            self.register_parameter('hpc_layer_bias', None)
        
        
        # System 2 Readout (Bit-Level)
        self.readout = VNNILinear(R * self.d_s2 * C, 8).to(self.device)
        self.curve_index_head = nn.Linear(self.d_s1 * C, 1).to(self.device)
        self.hyper_control_head = nn.Sequential(
            nn.Linear(self.d_s1 * C, 128),
            nn.GELU(),
            nn.Linear(128, self.L * self.R)
        ).to(self.device)
        
        # --- COGNITIVE RESONANCE: Surprise Head (World Model) ---
        # Predicts sensory latent (d_s1 * C) from internal state (d_s2 * C)
        self.surprise_head = nn.Linear(self.d_s2 * C, self.d_s1 * C).to(self.device)
        self.surprise_gate = nn.Linear(self.d_s2 * C, 1).to(self.device)
        nn.init.zeros_(self.surprise_gate.weight)
        nn.init.zeros_(self.surprise_gate.bias)
        if self.cfg.MES_ENABLED:
            self.surprise_optimizer = torch.optim.SGD(
                [*self.surprise_head.parameters(), *self.surprise_gate.parameters()],
                lr=self.cfg.LEARNING_RATE * self.cfg.LOCAL_LR_RATIO
            )
        if self.cfg.MES_ENABLED:
            self.readout_optimizer = torch.optim.SGD(
                [*self.readout.parameters(), *self.strategic_gating.parameters()], 
                lr=self.cfg.LEARNING_RATE * self.cfg.LOCAL_LR_RATIO
            )
        if self.cfg.MES_ENABLED and self.hpc_enabled and self.hpc_encoder is not None:
            # Sub-module optimizers are now unified in RRATrainer via model.parameters()
            # but we can provide a master_optimizer for local updates if needed.
            pass


        
        # Cache config values to avoid self.cfg/self.runtime_cfg getattr overhead in hot path
        self.cfg_mes_enabled = bool(Config.MES_ENABLED)
        self.cfg_cache_enabled = bool(Config.NEURAL_CACHE_ENABLED)
        self.cfg_pruning_enabled = True
        self.cfg_h_cycles = int(Config.H_CYCLES)
        self.cfg_l_cycles = int(Config.L_CYCLES)
        self.cfg_dissonance_penalty = float(Config.DISSONANCE_PENALTY)
        self.cfg_dissonance_threshold = float(Config.DISSONANCE_CONFIDENCE_THRESHOLD)
        self.cfg_metabolic_tax_rate = float(Config.METABOLIC_TAX_RATE)
        self.cfg_halt_threshold = float(Config.HALT_THRESHOLD)
        self.cfg_param_cost_scale = float(Config.PARAM_COST_SCALE)
        self.cfg_lif_decay = float(Config.LIF_DECAY)
        self.cfg_lif_threshold = float(Config.LIF_THRESHOLD)
        self.cfg_lgh_enabled = bool(self.lgh_cfg.enabled)
        
        # Pre-allocate contiguous state buffer (System 2 States)
        self.max_batch_size = Config.BATCH_SIZE
        self.H_buffer = torch.zeros(
            self.max_batch_size, L, R, self.d_s2, C, 
            device=device
        ).contiguous()
        
        # Phase 3: Zero-Copy Workspace Allocation (~1GB for B=64, T=512, M=1024)
        M_dim = self.d_s2 * C
        self._lgh_workspace_size = Config.BATCH_SIZE * (
            L * R * M_dim +          # Zone A: H_next
            Config.SEQ_LEN * R * M_dim + # Zone B: out
            Config.SEQ_LEN           # Zone C: halt_probs
        )
        self.register_buffer('_lgh_workspace', torch.zeros(self._lgh_workspace_size, dtype=torch.float32, device=device))
        
        self._init_manifold()
        self.register_buffer('_lgh_thermal_penalty_ema', torch.tensor(0.0, dtype=torch.float32, device=self.device))
        self.register_buffer('_lgh_last_freq_ghz', torch.tensor(0.0, dtype=torch.float32, device=self.device))
        self.register_buffer('_lgh_mdna_modulation', torch.ones(self.L, self.R, dtype=torch.float32, device=self.device))
        self.register_buffer('_lgh_last_curve_anchor', torch.tensor(0, dtype=torch.long, device=self.device))
        self._lgh_q_step = -1
        self._lgh_q_cache = {}
        self._lgh_q_scale = {}
        self._lgh_q_meta = {}
        self._lgh_use_int8 = bool(getattr(Config, 'RAM_INT8_INFER', True))
        
        self.rope = RotaryEmbedding(self.d_s2 * C, device=device)
        self.virtual_lab = VirtualLab(enabled=bool(Config.VIRTUAL_LAB_ENABLED))
        # Memory imprinting handled by strategic_gating
        
        # Preflight flag for performance
        self._preflight_ready = False
        self._preflight_steps = 0
        self.cfg_h_cycles = self.cfg_h_cycles
        self.cfg_l_cycles = self.cfg_l_cycles
        self.current_phase = 0
        self.register_buffer('step_counter', torch.zeros(1))
        self._current_engagement_rate = torch.tensor(1.0, device=self.device)
        self._layer_indices = torch.arange(self.L, device=self.device).view(-1, 1)
        self._block_indices = torch.arange(self.R, device=self.device).view(1, -1)
        self._last_cache_bits = None
        self._last_hit_mask = None
        self._last_hit_addrs = None
        self._last_hit_tables = None
        self._last_is_rule = None
        self._tps_pressure_cache = 0.0
        self._tps_pressure_step = -1
        self.importance_threshold_ema = 0.5
        self.mes_throttle_count = 0
        # All scalars are now consolidated in self.homeostasis.config_scalars
        # 6: hpc_error_ema, 7: hpc_last_error, 8: hpc_last_cycle_scale
        # 9: surprise_signal, 10: surprise_cycle_scale
        # 11: temporal_signal, 12: temporal_cycle_scale
        # 13: thermal_penalty_ema, 14: last_freq_ghz, 15: current_engagement_rate
        # 16-20: forward_stack_scalars, 21-23: mes_super_scalars
        self.homeostasis.config_scalars[6] = self.hpc_cfg.target_error
        self.homeostasis.config_scalars[8] = 1.0 # last_cycle_scale
        self.homeostasis.config_scalars[10] = 1.0 # surprise_cycle_scale
        self.homeostasis.config_scalars[12] = 1.0 # temporal_cycle_scale
        self.homeostasis.config_scalars[15] = 1.0 # engagement_rate


        # --- BIOLOGICAL FEATURE: Myelin Sheaths (Fast Pathway Insulation) ---
        # Myelin tracks which pathways are heavily used and should be "insulated" (penalized for energy)
        self.myelin_sheaths = nn.Parameter(torch.zeros(L, R, device=device))
        
        # --- HOMEOSTATIC OMEGA (Ω) REGULATOR ---
        # Ω ∈ [0, 1]: Controls the balance between Teacher imitation and Autonomous learning
        # Ω=0: Sponge (90% teacher, aggressive imprinting)
        # Ω=0.5: Student (50/50 blend)
        # Ω=1: Master (pure RRA, read-only memory)
        self.omega = 0.0  # Start as Sponge
        self.omega_momentum = self.runtime_cfg.omega_step  # How fast omega changes
        self.omega_history = []  # Track evolution
        self._params_dirty = True
        
        # --- Autonomous Intelligence: Genome Activation ---
        self.genome = Genome()
        self.metabolic_threshold = 0.0 # Safety Init
        self.update_phenotype_from_genome()
        # --------------------------------------------------
        
        # Suggested LR from genome (BDNF expression)
        # Initialized to Config value, updated by update_phenotype_from_genome()
        self.suggested_lr = Config.LEARNING_RATE
        
        # --- Consolidation State ---
        self.consolidation_interval = self.lifecycle_cfg.consolidate_every
        self.imprint_threshold = 50
        self.phenotype_update_interval = self.lifecycle_cfg.phenotype_update_every
        self._lifecycle_hooks = {'pre_forward': [], 'post_forward': []}
        self.register_lifecycle_hook('post_forward', self._lifecycle_phenotype_update)
        self.register_lifecycle_hook('post_forward', self._lifecycle_consolidate)
        
        # --- OPTIMIZED EXECUTION PATH ---
        # Use custom C++ kernels (cpp_loader) for maximum CPU performance.
        # torch.compile is disabled on Windows/CPU due to MSVC header issues.
        if cpp_loader is None:
            raise RuntimeError("cpp_loader extension is required; Python fallback paths are removed.")
        required_ops = [
            'rms_norm', 'parallel_scan', 'fused_rms_mean',
            'dcls_ram_addresses', 'dcls_ram_lookup', 'dcls_ram_lookup_int8', 'dcls_backward',
            'fused_lif_ram_lookup', 'batched_ademamix_update',
            'configure_hpc', 'configure_runtime',
            'forward_stack_io', 'mes_super_step_io',
            'survival_update_io', 'survival_mask_io', 'survival_losses_io'
        ]
        missing_ops = [op for op in required_ops if not hasattr(cpp_loader, op)]
        if missing_ops:
            raise RuntimeError(f"Missing required cpp_loader ops: {', '.join(missing_ops)}")
        available_ops = []
        for op in required_ops + ['quantized_matmul', 'ademamix_update', 'fused_cognitive_cycle', 'neural_cache_lookup_fast', 'geometric_manifold_forward_avx512']:
            if hasattr(cpp_loader, op):
                available_ops.append(op)
        event_mode_raw = self.exec_cfg.event_mode
        if event_mode_raw in ('off', 'dense', 'none'):
            event_mode_id = 0
        elif event_mode_raw in ('on', 'event', 'events'):
            event_mode_id = 1
        else:
            event_mode_id = 2
        cpp_loader.configure_runtime(
            self.hpc_cfg.temporal_folding,
            self.hpc_cfg.fold_alpha,
            int(event_mode_id),
            self.exec_cfg.event_density_threshold,
            self.exec_cfg.ttfs_enabled,
            self.exec_cfg.ttfs_slope_threshold
        )
        if hasattr(cpp_loader, 'configure_hpc'):
            cpp_loader.configure_hpc(
                float(self.hpc_cfg.target_error),
                float(self.hpc_cfg.error_ema_decay),
                float(self.hpc_cfg.halt_gain)
            )
        print(f">>> Apex CPU Optimizations (AVX2/FMA) ACTIVE")
        print(f">>> cpp_loader kernels available: {', '.join(available_ops)}")
        print(
            f">>> HPC Temporal Folding configured: "
            f"enabled={self.hpc_cfg.temporal_folding}, "
            f"alpha={self.hpc_cfg.fold_alpha:.3f}"
        )
        print(
            f">>> Event Runtime configured: mode={event_mode_raw}, "
            f"density_threshold={self.exec_cfg.event_density_threshold:.3f}, "
            f"ttfs={self.exec_cfg.ttfs_enabled}, "
            f"ttfs_slope={self.exec_cfg.ttfs_slope_threshold:.4f}"
        )
        
        # Eager mode is used - no torch.compile on CPU to avoid MSVC issues
        print(">>> HMI: Reasoning Core running in Eager Mode (cpp_loader accelerated).")
        if not self.cfg_cache_enabled:
            print(">>> NeuralCache reflex path disabled (reasoning-first mode).")
        if not self.exec_cfg.use_fused_cycle:
            print(">>> fused_cognitive_cycle disabled (using forward_stack_io path).")
        if self.cfg_lgh_enabled:
            if hasattr(cpp_loader, 'geometric_manifold_forward_avx512'):
                print(">>> LGH core enabled: geometric_manifold_forward_avx512 is available.")
            else:
                print(">>> LGH requested, but kernel is missing in current extension build; falling back to forward_stack_io.")
            if self._lgh_use_int8:
                print(">>> LGH int8 manifold mode enabled (RAM_INT8_INFER=true).")
        if not self.exec_cfg.use_forward_stack:
            raise RuntimeError("USE_FORWARD_STACK must be enabled; no Python reasoning fallback path exists.")
        print(">>> LGH-Manifold Consolidated Memory Active.")

    def _imprint_to_manifold(self, p, z, importance):
        """Placeholder for Manifold Imprinting (titans-style FAST learning)."""
        pass

    def _compute_importance_threshold(self, score):
        # Adaptive threshold based on EMA
        self.importance_threshold_ema = 0.95 * self.importance_threshold_ema + 0.05 * score.mean().item()
        return self.importance_threshold_ema

    def _init_manifold(self):
        self._lgh_shape3d = (self.L, self.R, self.lgh_cfg.morton_depth)
        self._lgh_morton = MortonBuffer(
            self._lgh_shape3d,
            device=self.device,
            align_multiple=self.lgh_cfg.align_multiple,
            temporal_bins=self.lgh_cfg.temporal_bins
        )
        self._lgh_morton.set_temporal_bins(self.lgh_cfg.temporal_bins)
        if self.cfg_lgh_enabled:
            n3 = int(self._lgh_morton.size)
            n = int(n3 * self.lgh_cfg.temporal_bins)
            m = int(self.d_s2 * self.C)
            self.lgh_manifold_morton = nn.Parameter(torch.randn(n, m, device=self.device) * 0.01)
            curve_len = min(self.lgh_cfg.curve_length, int(self._lgh_morton.size_original))
            curve_orig = self._lgh_morton.curve_segment_original(
                0,
                curve_len,
                wrap=self.lgh_cfg.curve_wrap,
                delta_t=0,
                temporal_bins=self.lgh_cfg.temporal_bins,
                fold_alpha=self.lgh_cfg.temporal_fold_alpha
            )
            self.register_buffer('_lgh_curve_indices', curve_orig.to(dtype=torch.long))
            self.register_buffer('_lgh_prefetch_curve_indices', curve_orig.to(dtype=torch.long))
            self.register_buffer('_lgh_mdna_mask', torch.ones(self.L, self.R, dtype=torch.float32, device=self.device))
            self.register_buffer('_lgh_synaptic_trace', torch.zeros(n, dtype=torch.float32, device=self.device))
            self.register_buffer('_lgh_morton_order', self._lgh_morton.order.clone().to(dtype=torch.long))
        else:
            self.lgh_manifold_morton = None
            self.register_buffer('_lgh_curve_indices', torch.empty(0, dtype=torch.long, device=self.device))
            self.register_buffer('_lgh_prefetch_curve_indices', torch.empty(0, dtype=torch.long, device=self.device))
            self.register_buffer('_lgh_mdna_mask', torch.empty(0, dtype=torch.float32, device=self.device))
            self.register_buffer('_lgh_synaptic_trace', torch.empty(0, dtype=torch.float32, device=self.device))
            self.register_buffer('_lgh_morton_order', torch.empty(0, dtype=torch.long, device=self.device))
    
    def _sync_stacked_params(self):
        """
        Cache and synchronize parameters for C++ forward_stack kernel.
        Returns a dict of stacked Tensors for all levels.
        """
        step = int(self.step_counter.item())
        if self._stacked_params is not None and not self._params_dirty:
            return self._stacked_params
        
        # Stack parameters from all levels
        delays = torch.stack([level.wsnn.delays for level in self.levels])       # [L, M, K]
        tables = torch.stack([level.wsnn.ram_tables for level in self.levels])   # [L, M, 2^K]
        conns = torch.stack([level.wsnn.connections for level in self.levels])   # [L, M, K]
        decays_raw = torch.stack([level.raw_decay.view(self.R, -1) for level in self.levels])  # [L, R, D*C]
        # Stability: recurrent decay must stay in (0, 1) for bounded state dynamics.
        decays = torch.sigmoid(decays_raw)
        decays = torch.nan_to_num(decays, nan=0.9, posinf=0.999, neginf=1e-4).clamp(1e-4, 0.999)
        halt_w = torch.stack([level.halt_head.weight.view(-1) for level in self.levels])  # [L, D*C]
        halt_b = torch.cat([level.halt_head.bias for level in self.levels])              # [L]
        
        self._stacked_params = {
            'delays': delays.contiguous(),
            'tables': tables.contiguous(),
            'conns': conns.contiguous(),
            'decays': decays.contiguous(),
            'halt_w': halt_w.contiguous(),
            'halt_b': halt_b.contiguous(),
        }
        self._stacked_params_step = step
        self._params_dirty = False
        return self._stacked_params

    def _cpp_io_call(self, op_name, x_input, state, params=None, scalars=None):
        params = [] if params is None else list(params)
        if isinstance(scalars, torch.Tensor):
            scalar_tensor = scalars.contiguous().float()
        elif scalars is None:
            scalar_tensor = torch.empty(0, dtype=torch.float32, device=x_input.device)
        else:
            scalar_tensor = torch.as_tensor(scalars, dtype=torch.float32, device=x_input.device).contiguous()
        io_tensors = [x_input, state, scalar_tensor]
        io_tensors.extend([p for p in params if isinstance(p, torch.Tensor)])
        return _cpp_try(
            op_name,
            x_input,
            state,
            params,
            scalar_tensor,
            tensors=tuple(io_tensors)
        )

    def _call_forward_stack(self, z_L, z_H, p_stack, dyn_threshold, l_cycles):
        """Call unified C++ forward stack kernel (IO signature only)."""
        io_params = [
            p_stack['delays'], p_stack['tables'], p_stack['conns'],
            p_stack['decays'], p_stack['halt_w'], p_stack['halt_b']
        ]
        self._forward_stack_scalars[0] = 0.0
        self._forward_stack_scalars[1] = self.cfg_lif_decay
        self._forward_stack_scalars[2] = self.cfg_lif_threshold
        self._forward_stack_scalars[3] = float(dyn_threshold)
        self._forward_stack_scalars[4] = float(max(1, int(l_cycles)))
        out_io = self._cpp_io_call(
            'forward_stack_io',
            z_L,
            z_H,
            params=io_params,
            scalars=self._forward_stack_scalars
        )
        if not isinstance(out_io, (list, tuple)) or len(out_io) != 3:
            raise RuntimeError("forward_stack_io returned unexpected output.")
        y_seq, h_next, p_halt = out_io
        return self._normalize_stack_outputs(y_seq, h_next, p_halt, z_L)

    def _normalize_stack_outputs(self, y_seq, h_next, p_halt, z_L_ref):
        """Normalize C++ forward_stack outputs to Python reasoning-core contracts."""
        B = z_L_ref.size(0)
        T = z_L_ref.size(1)
        if y_seq.dim() == 2 and y_seq.size(1) == (self.R * self.d_s2 * self.C):
            y_seq = y_seq.view(B, 1, self.R, self.d_s2, self.C)
        elif y_seq.dim() == 3 and y_seq.size(-1) == (self.R * self.d_s2 * self.C):
            y_seq = y_seq.view(B, y_seq.size(1), self.R, self.d_s2, self.C)
        if y_seq.dim() == 5 and y_seq.size(1) == 1 and T > 1:
            y_seq = y_seq.expand(-1, T, -1, -1, -1)

        if p_halt.dim() == 1:
            p_halt = p_halt.view(B, 1, 1)
        elif p_halt.dim() == 2:
            p_halt = p_halt.unsqueeze(-1)
        return y_seq, h_next, p_halt

    def _batched_mes_optimizer_step(self):
        """Apply MES optimizer updates across all levels using required C++ batched kernel."""
        with torch.no_grad():
            params_list = [l.wsnn.ram_tables for l in self.levels]
            grads_list = []
            for level in self.levels:
                grad = level.wsnn.ram_tables.grad
                if grad is None:
                    zero_buf = getattr(level, '_zero_grad_ram', None)
                    if zero_buf is None or zero_buf.shape != level.wsnn.ram_tables.shape:
                        zero_buf = torch.zeros_like(level.wsnn.ram_tables)
                        level.register_buffer('_zero_grad_ram', zero_buf)
                    grads_list.append(zero_buf)
                else:
                    grads_list.append(grad)
            m_fast_list = [l.m_fast for l in self.levels]
            m_slow_list = [l.m_slow for l in self.levels]
            v_list = [l.v_opt for l in self.levels]
            global_step = int(self.levels[0].opt_step)
            cpp_loader.batched_ademamix_update(
                params_list, grads_list, m_fast_list, m_slow_list, v_list,
                self.cfg.LEARNING_RATE, 0.9, 0.9999, 0.999,
                0.99,
                1e-8,
                global_step
            )
            for level in self.levels:
                level.opt_step += 1
                level.wsnn.ram_tables.grad = None

    def update_omega(self, train_loss, val_loss, force_delta=None):
        """
        Evolve Omega based on generalization performance.
        - Increase Ω when generalizing well (val_loss ≈ train_loss)
        - Decrease Ω when memorizing (val_loss >> train_loss)
        """
        if force_delta is not None:
            delta = force_delta
        else:
            generalization_gap = val_loss - train_loss
            
            # --- STRESS MODULATION (FKBP5) ---
            # High stress (FKBP5) makes it harder to become autonomous
            fkbp5 = self.genome.fkbp5 if hasattr(self, 'genome') else 0.5
            stress_damping = 1.0 / (1.0 + fkbp5)
            
            if generalization_gap < 0.1:
                # Excellent generalization - become more autonomous
                delta = self.omega_momentum * 2 * stress_damping
            elif generalization_gap < 0.3:
                # Good generalization - gradually increase autonomy
                delta = self.omega_momentum
            elif generalization_gap > 0.5:
                # Memorizing - fall back to teacher
                delta = -self.omega_momentum * 2
            else:
                # Neutral - small increase
                delta = self.omega_momentum * 0.5
        
        self.omega = max(0.0, min(1.0, self.omega + delta))
        self.omega_history.append(self.omega)
        return self.omega

    def set_phase(self, phase): self.current_phase = phase

    def update_phenotype_from_genome(self):
        """Map abstract genes to physical model hyperparameters."""
        if not hasattr(self, 'genome'):
            return

        # BDNF -> Learning Rate (Handled in Trainer usually, but can be scaled here)
        # CREB -> Memory Focus (Currently influence curiosity/stability weights)
        creb = self.genome.creb
        # Metabolism gamma is now handled internally by Genome based on CREB
        
        # DRD2 -> Gating Threshold (Confidence)
        drd2 = self.genome.drd2
        # Map 0.1-0.9 to 1.0-6.0 range for STRICT_CONFIDENCE
        self.confidence_multiplier = 1.0 + (drd2 * 5.0) # Higher DRD2 -> higher confidence multiplier
        
        # FKBP5 -> Target Sparsity Pressure (Metabolic Hunger) & Resonance Error Sensitivity
        fkbp5 = self.genome.fkbp5
        metabolic_efficiency = getattr(self.genome, 'metabolic_efficiency', 0.5)
        # Map FKBP5 to lambda_sparsity
        self.lambda_sparsity = (0.01 * fkbp5) / max(0.25, metabolic_efficiency)
        
        # --- COGNITIVE RESONANCE: FKBP5 controls error amplification ---
        # High FKBP5 = High stress = High sensitivity to prediction error
        # Input = (1-GABA)*Raw + FKBP5*Error
        self.fkbp5 = fkbp5
        
        # GABA -> Inhibition Control (Transparency Gate threshold modifier)
        gaba = self.genome.gaba
        # Higher GABA = more inhibitory = higher engagement threshold = more bypasses
        self.gaba_inhibition = gaba
        self.curve_trajectory_gene = float(getattr(self.genome, 'curve_trajectory', 0.5))
        self.mask_sparsity_bias = float(getattr(self.genome, 'mask_sparsity_bias', 0.5))
        self.metabolic_efficiency = float(metabolic_efficiency)
        self.wormhole_jump_bias = float(getattr(self.genome, 'wormhole_jump_bias', 0.1))
        self.focus_x = float(getattr(self.genome, 'focus_x', 0.5))
        self.focus_y = float(getattr(self.genome, 'focus_y', 0.5))
        self.focus_z = float(getattr(self.genome, 'focus_z', 0.5))
        self.focus_sharpness = float(getattr(self.genome, 'focus_sharpness', self.lgh_cfg.focus_sharpness))
        self.temporal_trace_bias = float(getattr(self.genome, 'temporal_trace_bias', 0.5))
        
        # BDNF: Learning Rate proxy
        bdnf_expression = self.genome.bdnf
        # Map BDNF [0.1, 2.0] to learning rate [1e-5, 1e-2]
        self.suggested_lr = 1e-5 + (bdnf_expression / 2.0) * (1e-2 - 1e-5)
        
        # Perfection Penalty: dynamic multiplier if loss is too low
        if getattr(self.genome, 'best_loss', 1.0) < 0.001:
             print(">>> METABOLIC HUNGER: Perfection Penalty Active (2x Sparsity Pressure)")
             self.lambda_sparsity *= 2.0
        
        # --- Metabolic Threshold (Efficiency Pressure) ---
        # As Omega rises (Autonomy), the threshold for neurons to fire increases
        self.metabolic_threshold = 0.05 * self.omega
        
        # --- Energy-Aware Learning Rate ---
        # If sparsity pressure is high (FKBP5), effectively "energy is low"
        # Reduce LR to prevent burnout/instability under high pressure
        energy_factor = self.metabolic_efficiency / (1.0 + self.lambda_sparsity * 10.0)
        self.suggested_lr = self.suggested_lr * energy_factor
        self._refresh_lgh_curve_from_genome()
        self._refresh_lgh_prefetch_curve()
        
        # --- BDNF-Scaled Omega Momentum ---
        # Omega changes faster when BDNF is high (higher plasticity)
        self.omega_momentum = self.runtime_cfg.omega_step * bdnf_expression
        
        print(f">>> Phenotype Updated (Gen {self.genome.generation}): CREB={creb:.2f} (Stab={self.genome.creb_gamma:.3f}), "
              f"DRD2={drd2:.2f} (Conf={self.confidence_multiplier:.2f}), "
              f"FKBP5={fkbp5:.2f} (Sparsity={self.lambda_sparsity:.6f}, Thresh={self.metabolic_threshold:.6f})")
        print(
            f">>> GABA={gaba:.2f} | mEff={self.metabolic_efficiency:.2f} | "
            f"Curve={self.curve_trajectory_gene:.2f} | MaskBias={self.mask_sparsity_bias:.2f} | "
            f"Wormhole={self.wormhole_jump_bias:.2f} | Focus=({self.focus_x:.2f},{self.focus_y:.2f},{self.focus_z:.2f}) "
            f"| TraceBias={self.temporal_trace_bias:.2f} | Energy Factor={energy_factor:.2f} "
            f"-> LR={self.suggested_lr:.2e} | Omega_mom={self.omega_momentum:.4f}"
        )

    def get_engagement_rate(self):
        """Returns current engagement rate for efficiency bonus calculation."""
        if hasattr(self._current_engagement_rate, 'item'):
            return self._current_engagement_rate.item()
        return self._current_engagement_rate

    def _cache(self):
        if not self.cfg_cache_enabled:
            return None
        return self.neural_cache

    def set_runtime_toggles(self, *, mes_enabled=None, cache_enabled=None, pruning_enabled=None, lgh_enabled=None):
        if mes_enabled is not None:
            enabled = bool(mes_enabled)
            self.mes_cfg.enabled = enabled
            self.cfg_mes_enabled = enabled
        if cache_enabled is not None:
            enabled = bool(cache_enabled)
            self.cfg_cache_enabled = enabled
        if pruning_enabled is not None:
            enabled = bool(pruning_enabled)
            self.cfg_pruning_enabled = enabled
        if lgh_enabled is not None:
            enabled = bool(lgh_enabled)
            self.cfg_lgh_enabled = enabled

    def _clear_cache_trace(self):
        self._last_cache_bits = None
        self._last_hit_mask = None
        self._last_hit_addrs = None
        self._last_hit_tables = None

    def _set_cache_trace(self, cache_bits, hit_mask, hit_addrs, hit_tables):
        self._last_cache_bits = cache_bits if bool(hit_mask.any().item()) else None
        self._last_hit_mask = hit_mask
        self._last_hit_addrs = hit_addrs
        self._last_hit_tables = hit_tables

    def _vl_add_scalar(self, tag, value, step=None):
        if (not self.virtual_lab.enabled) or (self.virtual_lab.writer is None):
            return
        use_step = int(self.step_counter) if step is None else int(step)
        self.virtual_lab.writer.add_scalar(tag, value, use_step)

    def register_lifecycle_hook(self, event_name, fn):
        if event_name not in self._lifecycle_hooks:
            raise ValueError(f"Unknown lifecycle event '{event_name}'.")
        self._lifecycle_hooks[event_name].append(fn)

    def _run_lifecycle_hooks(self, event_name, **context):
        hooks = self._lifecycle_hooks.get(event_name, [])
        for hook in hooks:
            hook(**context)

    def _lifecycle_phenotype_update(self, **context):
        step = int(self.step_counter.item())
        if step % self.phenotype_update_interval != 0:
            return
        self.update_phenotype_from_genome()

    def _lifecycle_consolidate(self, out=None, is_audit=False, **context):
        if out is None:
            return
        step = int(self.step_counter.item())
        if step % self.consolidation_interval != 0:
            return
        self._consolidate_memories(out.detach(), is_audit)

    def _get_tps_pressure(self):
        if not self.virtual_lab.enabled:
            return 0.0
        step = int(self.step_counter.item())
        if self._tps_pressure_step == step:
            return self._tps_pressure_cache
        self._tps_pressure_cache = float(self.virtual_lab.get_benchmarks().get('tps_pressure', 0.0))
        self._tps_pressure_step = step
        return self._tps_pressure_cache

    def _compute_audit_flag(self):
        step = int(self.step_counter.item())
        period = max(1, self.audit_cfg.period_steps)
        random_prob = max(0.0, min(1.0, self.audit_cfg.random_prob))
        periodic_trigger = self.training and ((step % period) == 0)
        random_trigger = self.training and (random.random() < random_prob)
        return bool(periodic_trigger or random_trigger)

    def _compute_importance_threshold(self, importance_score):
        if importance_score.numel() == 0:
            return 0.5
        mean_val = float(importance_score.mean().item())
        std_val = float(importance_score.std(unbiased=False).item()) if importance_score.numel() > 1 else 0.0
        std_factor = self.importance_cfg.std_factor
        target_threshold = mean_val + std_factor * std_val
        decay = self.importance_cfg.ema_decay
        decay = max(0.0, min(0.999, decay))
        self.importance_threshold_ema = decay * self.importance_threshold_ema + (1.0 - decay) * target_threshold
        return max(0.0, min(1.0, self.importance_threshold_ema))

    def _should_update_survival(self):
        if not self.training:
            return False
        update_every = max(1, self.survival_cfg.update_every)
        step = int(self.step_counter.item())
        return bool(step == 1 or (step % update_every) == 0)

    def _maybe_update_survival(self, H_next, H_prev, should_update=None):
        if not self.cfg_pruning_enabled:
            return
        if should_update is None:
            should_update = self._should_update_survival()
        if should_update:
            self.homeostasis.update_metabolism(H_next, H_prev=H_prev, bdnf_gamma=self.genome.creb_gamma)

    def _hpc_active(self):
        return bool(self.hpc_enabled and (self.L > 1) and (self.hpc_encoder is not None) and (self.hpc_decoder is not None))

    def _hpc_state_features(self, H_state):
        return H_state.mean(dim=2).reshape(H_state.size(0), self.L, self._hpc_dim)

    def _hpc_temporal_fold(self, h_curr, h_prev=None):
        if (h_prev is None) or (not self.hpc_cfg.temporal_folding):
            return h_curr
        alpha = self.hpc_cfg.fold_alpha
        alpha = max(0.0, min(1.0, alpha))
        return h_curr + alpha * (h_curr - h_prev)

    def _hpc_predict_lower(self, upper_states):
        B, Lm1, M = upper_states.shape
        pred = self.hpc_decoder(torch.tanh(self.hpc_encoder(upper_states.reshape(-1, M)))).reshape(B, Lm1, M)
        return pred * self.hpc_layer_gain.unsqueeze(0) + self.hpc_layer_bias.unsqueeze(0)

    def _hpc_error_terms(self, H_next, H_prev=None):
        if not self._hpc_active():
            zero_scalar = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            return zero_scalar, torch.zeros(self.L, dtype=torch.float32, device=self.device), None, None
        h_curr = self._hpc_state_features(H_next)
        h_prev_feat = self._hpc_state_features(H_prev) if H_prev is not None else None
        h_folded = self._hpc_temporal_fold(h_curr, h_prev_feat)
        lower_target = h_folded[:, :-1, :]
        upper_states = h_folded[:, 1:, :]
        pred_lower = self._hpc_predict_lower(upper_states)
        error = lower_target - pred_lower
        layer_error = error.pow(2).mean(dim=(0, 2))
        layer_error_full = torch.zeros(self.L, dtype=torch.float32, device=layer_error.device)
        layer_error_full[: self.L - 1] = layer_error
        global_error = layer_error.mean() if layer_error.numel() > 0 else torch.tensor(0.0, dtype=torch.float32, device=layer_error.device)
        return global_error, layer_error_full, lower_target, pred_lower

    def _hpc_update_error_stats(self, global_error, layer_error_full):
        """Update EMA tracking for HPC decisions."""
        decay = self.hpc_cfg.error_ema_decay
        decay = max(0.0, min(0.999, decay))
        
        with torch.no_grad():
            global_safe = torch.nan_to_num(global_error.detach(), nan=0.0, posinf=1e4, neginf=0.0)
            layer_safe = torch.nan_to_num(layer_error_full.detach(), nan=0.0, posinf=1e4, neginf=0.0)
            
            # Update C++ side EMA if available, otherwise Python
            if hasattr(cpp_loader, 'get_hpc_error_ema'):
                # Note: Currently C++ EMA is a placeholder, updating it here via Python's logic
                # for parity until the kernel is updated.
                self._hpc_error_ema.mul_(decay).add_(global_safe * (1.0 - decay))
                # Future: Core::g_hpc_error_ema = ... in C++
            else:
                self._hpc_error_ema.mul_(decay).add_(global_safe * (1.0 - decay))
            
            self._hpc_last_error.copy_(global_safe)
            self._hpc_layer_error_ema.mul_(decay).add_(layer_safe * (1.0 - decay))
        
        # Facade property update
        self.error_ema = float(self._hpc_error_ema.item())

    def _hpc_cycle_scale(self):
        if not self._hpc_active():
            self._hpc_last_cycle_scale.fill_(1.0)
            return 1.0
        target_error = max(1e-6, self.hpc_cfg.target_error)
        ema_error = float(self._hpc_error_ema.item())
        if not math.isfinite(ema_error):
            ema_error = target_error
        ratio = ema_error / target_error
        min_scale = self.hpc_cfg.cycle_min_scale
        max_scale = self.hpc_cfg.cycle_max_scale
        min_scale = max(0.1, min(min_scale, max_scale))
        max_scale = max(min_scale, max_scale)
        scale = max(min_scale, min(max_scale, ratio))
        self._hpc_last_cycle_scale.fill_(float(scale))
        return float(scale)

    def _dynamic_cycle_counts(self):
        scale = self._hpc_cycle_scale()
        h_cycles = max(1, int(round(self.cfg_h_cycles * scale)))
        l_cycles = max(1, int(round(self.cfg_l_cycles * scale)))
        h_cap = max(1, self.hpc_cfg.h_cycles_max)
        l_cap = max(1, self.hpc_cfg.l_cycles_max)
        return min(h_cycles, h_cap), min(l_cycles, l_cap)

    def _apply_surprise_cycle_control(self, h_cycles, l_cycles, surprise_signal, is_audit=False):
        if (not self.hpc_cfg.surprise_gate) or is_audit:
            self._last_surprise_cycle_scale.fill_(1.0)
            return h_cycles, l_cycles, False
        threshold = max(1e-6, self.hpc_cfg.surprise_threshold)
        score = float(surprise_signal.mean().item())
        if not math.isfinite(score):
            score = threshold
        score = max(0.0, score)
        min_scale = max(0.1, min(1.0, self.hpc_cfg.surprise_min_scale))
        if score >= threshold:
            scale = 1.0
        else:
            scale = min_scale + (1.0 - min_scale) * (score / threshold)
        if not math.isfinite(scale):
            scale = 1.0
        h_next = max(1, int(round(h_cycles * scale)))
        l_next = max(1, int(round(l_cycles * scale)))
        self._last_surprise_cycle_scale.fill_(float(scale))
        skip_reasoning = bool(
            self.hpc_cfg.surprise_skip_enabled
            and (scale <= self.hpc_cfg.surprise_skip_scale)
            and (not is_audit)
        )
        return h_next, l_next, skip_reasoning

    def _temporal_activity_signal(self, p):
        if (not self.hpc_cfg.temporal_gate_enabled) or p.dim() != 3 or p.size(1) <= 1:
            self._last_temporal_signal.fill_(1.0)
            return torch.ones((p.size(0), 1, 1), dtype=torch.float32, device=p.device)
        window = min(int(self.hpc_cfg.temporal_gate_window), int(p.size(1)))
        if window <= 1:
            self._last_temporal_signal.fill_(1.0)
            return torch.ones((p.size(0), 1, 1), dtype=torch.float32, device=p.device)
        p_tail = p[:, -window:, :]
        delta = (p_tail[:, 1:, :] - p_tail[:, :-1, :]).abs().mean(dim=(1, 2), keepdim=True)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=1.0, neginf=0.0)
        signal = torch.tanh(delta).to(dtype=torch.float32)
        signal = torch.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=0.0)
        self._last_temporal_signal.copy_(signal.mean().detach().to(self.device, dtype=torch.float32))
        return signal

    def _apply_temporal_cycle_control(self, h_cycles, l_cycles, temporal_signal, is_audit=False):
        if (not self.hpc_cfg.temporal_gate_enabled) or is_audit:
            self._last_temporal_cycle_scale.fill_(1.0)
            return h_cycles, l_cycles, False
        threshold = max(1e-6, self.hpc_cfg.temporal_gate_threshold)
        score = float(temporal_signal.mean().item())
        if not math.isfinite(score):
            score = threshold
        score = max(0.0, score)
        min_scale = max(0.1, min(1.0, self.hpc_cfg.temporal_gate_min_scale))
        if score >= threshold:
            scale = 1.0
        else:
            scale = min_scale + (1.0 - min_scale) * (score / threshold)
        if not math.isfinite(scale):
            scale = 1.0
        h_next = max(1, int(round(h_cycles * scale)))
        l_next = max(1, int(round(l_cycles * scale)))
        self._last_temporal_cycle_scale.fill_(float(scale))
        skip_reasoning = bool(
            self.hpc_cfg.temporal_gate_skip_enabled
            and (scale <= self.hpc_cfg.temporal_gate_skip_scale)
            and (not is_audit)
        )
        return h_next, l_next, skip_reasoning

    def _apply_efficiency_cycle_control(self, h_cycles, l_cycles, p_brain, gate, is_audit=False):
        if is_audit or (not isinstance(p_brain, torch.Tensor)) or p_brain.dim() != 3:
            return h_cycles, l_cycles
        # Easy-token proxy: low latent activity + low engaged gate density.
        latent_activity = float(torch.nan_to_num(p_brain[:, -1].abs().mean(), nan=0.0, posinf=1.0, neginf=0.0).item())
        gate_activity = 1.0
        if isinstance(gate, torch.Tensor) and gate.numel() > 0:
            gate_activity = float(torch.nan_to_num(gate.mean(), nan=1.0, posinf=1.0, neginf=1.0).item())
        activity_signal = 0.5 * math.tanh(latent_activity) + 0.5 * max(0.0, min(1.0, gate_activity))
        easy_threshold = 0.35
        min_scale = 0.45
        if activity_signal >= easy_threshold:
            scale = 1.0
        else:
            scale = min_scale + (1.0 - min_scale) * (activity_signal / easy_threshold)
        scale = max(min_scale, min(1.0, scale))
        h_next = max(1, int(round(h_cycles * scale)))
        l_next = max(1, int(round(l_cycles * scale)))
        return h_next, l_next

    def _hpc_refresh_from_states(self, H_next, H_prev=None):
        if not self._hpc_active():
            self._hpc_last_error.fill_(0.0)
            return self._hpc_last_error
        step = int(self.step_counter.item())
        monitor_every = max(1, self.hpc_cfg.monitor_every)
        if self.training and (step % monitor_every) != 0:
            return self._hpc_last_error
        with torch.no_grad():
            global_error, layer_error_full, _, _ = self._hpc_error_terms(H_next.detach(), H_prev.detach() if H_prev is not None else None)
            self._hpc_update_error_stats(global_error, layer_error_full)
        return self._hpc_last_error

    def _hpc_train_predictors(self, H_next, H_prev=None):
        if not (self._hpc_active() and hasattr(self, 'hpc_optimizer')):
            return 0.0
        self.hpc_optimizer.zero_grad(set_to_none=True)
        h_curr = self._hpc_state_features(H_next.detach())
        h_prev_feat = self._hpc_state_features(H_prev.detach()) if H_prev is not None else None
        h_folded = self._hpc_temporal_fold(h_curr, h_prev_feat)
        lower_target = h_folded[:, :-1, :]
        upper_states = h_folded[:, 1:, :]
        pred_lower = self._hpc_predict_lower(upper_states)
        hpc_loss = F.mse_loss(pred_lower, lower_target)
        hpc_loss.backward()
        self.hpc_optimizer.step()
        return float(hpc_loss.item())

    def _compute_dyn_halt_threshold(self, tps_pressure=None):
        if tps_pressure is None:
            tps_pressure = self._get_tps_pressure()
        fkbp5 = getattr(self, 'fkbp5', 0.0)
        gaba = getattr(self, 'gaba_inhibition', 0.0)
        dyn_threshold = self.cfg_halt_threshold * (1.0 + fkbp5) / (1.0 + gaba)
        dyn_threshold *= (1.0 - tps_pressure)
        if self._hpc_active():
            target_error = max(1e-6, self.hpc_cfg.target_error)
            ema_error = float(self._hpc_error_ema.item())
            if not math.isfinite(ema_error):
                ema_error = target_error
            ratio = ema_error / target_error
            ratio = max(0.5, min(2.0, ratio))
            halt_gain = self.hpc_cfg.halt_gain
            dyn_threshold *= (1.0 + halt_gain * (ratio - 1.0))
        return max(0.1, min(0.95, dyn_threshold))

    def _compute_engagement_rate(self):
        rates = [level._last_engagement_rate for level in self.levels if hasattr(level, '_last_engagement_rate')]
        if rates:
            return torch.stack(rates).mean()
        return torch.tensor(1.0, device=self.device)

    def _new_state(self, batch_size, clone=False):
        """Create or slice recurrent state with consistent [B, L, R, D, C] shape."""
        if batch_size <= self.max_batch_size:
            state = self.H_buffer[:batch_size]
        else:
            state = init_state(self.L, self.R, self.d_s2, self.C, device=self.device).unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            if not state.is_contiguous():
                state = state.contiguous()
        return state.clone() if clone else state

    def _prepare_state(self, H, batch_size):
        """Normalize incoming H to contiguous [B, L, R, D, C]."""
        if self._preflight_ready:
            return H if H.is_contiguous() else H.contiguous()

        if H is None:
            return self._new_state(batch_size)
        if H.dim() == 4:
            H = H.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        elif H.dim() == 5 and H.size(0) == 1 and batch_size > 1:
            H = H.expand(batch_size, -1, -1, -1, -1)
        if H.dim() != 5:
            raise ValueError(f"Expected H to have 5 dims after normalization, got shape={tuple(H.shape)}")
        if H.size(0) != batch_size:
            if H.size(0) == 1:
                H = H.expand(batch_size, -1, -1, -1, -1)
            else:
                raise ValueError(f"H batch mismatch: expected {batch_size}, got {H.size(0)}")
        if not H.is_contiguous() or any(s == 0 for s in H.stride()):
            H = H.contiguous()
        return H

    def _expand_brain_to_levels(self, p_brain):
        """[B, T, D*C] -> [B, T, R, D, C] without repeating shape logic."""
        B, T = p_brain.shape[:2]
        return p_brain.unsqueeze(2).expand(-1, -1, self.R, -1).reshape(B, T, self.R, self.d_s2, self.C)

    def _project_bits(self, bits):
        """Shared bit->latent->brain projection."""
        p = self.bit_to_latent(bits)
        p_brain = self.bridge_s1_to_s2(p)
        return p, p_brain

    def _bytes_to_bits(self, bytes_bt):
        """Convert byte tokens [B, T] to bit vectors [B, T, 8]."""
        # Centralized check and pre-allocation avoidance
        tokens = bytes_bt.to(device=self.device, dtype=torch.long)
        # byte_bit_shifts is already view(1,1,-1) or equivalent in __init__? 
        # Actually let's ensure it's correct here.
        return ((tokens.unsqueeze(-1) >> self.byte_bit_shifts) & 1).to(torch.float32)

    def _encode_s1_input(self, x):
        """
        Normalize external input to S1 latent representation `p`.
        Returns: p [B, T, d_s1*C], B, T
        """
        if self._preflight_ready:
            return self.bit_to_latent(x) if x.dim() == 3 and x.size(-1) == 8 else x, x.size(0), x.size(1)

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        x = x.to(self.device)

        if x.dim() == 2:
            if x.size(1) == self.d_s1 * self.C and torch.is_floating_point(x):
                p = x.unsqueeze(1).to(torch.float32)
                return p, p.size(0), p.size(1)
            if x.size(1) == 8:
                if torch.is_floating_point(x):
                    bits = x.unsqueeze(1).to(torch.float32)
                else:
                    is_binary = bool(((x == 0) | (x == 1)).all().item())
                    bits = x.unsqueeze(1).to(torch.float32) if is_binary else self._bytes_to_bits(x)
                p = self.bit_to_latent(bits)
                return p, p.size(0), p.size(1)
            bits = self._bytes_to_bits(x)
            p = self.bit_to_latent(bits)
            return p, p.size(0), p.size(1)

        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [B,T], [B,8], [B,T,8], or [B,T,{self.d_s1 * self.C}], got {tuple(x.shape)}")

        if x.size(-1) == self.d_s1 * self.C:
            p = x.to(torch.float32)
            return p, p.size(0), p.size(1)
        if x.size(-1) == 8:
            bits = x.to(torch.float32)
            p = self.bit_to_latent(bits)
            return p, p.size(0), p.size(1)

        raise ValueError(f"Unsupported last dimension for x: {x.size(-1)} (expected 8 or {self.d_s1 * self.C}).")

    def _run_lgh_cycle(self, p_brain, H, gate, B, T, h_cycles, l_cycles, dyn_threshold, p_s1=None):
        if not (self.cfg_lgh_enabled and hasattr(cpp_loader, 'geometric_manifold_forward_avx512')):
            return False, None, None, None

        manifold = self._lgh_morton_manifold()
        if manifold is None or manifold.numel() == 0:
            return False, None, None, None

        # Prep Curve Indices
        genome = getattr(self, 'genome', None)
        if genome is not None and (hasattr(genome, 'next_curve_anchor') or hasattr(genome, 'build_curve_chain')):
             self.update_phenotype_from_genome() # Refresh curves if genome-driven

        idx_c = self._lgh_curve_indices.contiguous()
        pidx_c = self._lgh_prefetch_curve_indices.contiguous()
        trace_c = self._lgh_synaptic_trace.contiguous()
        mdna_c = self._lgh_mdna_mask.contiguous()
        if gate.dim() == 4 and gate.size(0) == self.L and gate.size(1) == self.R:
            gate_cpp = gate.squeeze(-1).squeeze(-1) # [L, R, 1, 1] -> [L, R]
        elif gate.dim() == 4:
            gate_cpp = gate.mean(dim=(0, 1)) # [B, T, L, R] -> [L, R]
        elif gate.dim() == 2 and gate.size(0) == self.L and gate.size(1) == self.R:
            gate_cpp = gate
        elif gate.dim() == 2:
            gate_cpp = gate.reshape(self.L, self.R)
        else:
            # Handle scalars or other shapes securely
            val = gate.view(-1).mean() if gate.numel() > 0 else torch.tensor(1.0, device=gate.device)
            gate_cpp = val.expand(self.L, self.R)
        gate_c = gate_cpp.float().contiguous()
        p_brain_c = p_brain.float().contiguous()
        H_c = H.contiguous()

        step = int(self.step_counter.item()) if hasattr(self, 'step_counter') else 0
        
        if self._lgh_use_int8 and hasattr(cpp_loader, 'geometric_manifold_forward_avx512_int8'):
            q_manifold, q_scale = self._quantize_lgh_manifold(bits=8)
            z_lgh, H_next, halt_probs = cpp_loader.geometric_manifold_forward_avx512_int8(
                p_brain_c, H_c, gate_c, q_manifold, q_scale, idx_c, pidx_c, mdna_c, trace_c,
                step, int(h_cycles), int(l_cycles), float(dyn_threshold),
                int(self.lgh_cfg.prefetch_distance),
                float(self.exec_cfg.thermal_penalty if hasattr(self.exec_cfg, 'thermal_penalty') else 0.0), # fallback
                float(self.lgh_cfg.low_entropy_fold_threshold),
                int(self.lgh_cfg.wave_radius),
                float(self.lgh_cfg.wave_decay),
                float(self.lgh_cfg.trace_decay),
                float(self.lgh_cfg.trace_gain),
                int(self.lgh_cfg.temporal_bins)
            )
        elif hasattr(cpp_loader, 'pulse_gated_forward'):
            # Phase 2 & 3: Hardware-Native Pulse + Zero-Copy Workspace
            z_lgh, H_next, halt_probs = cpp_loader.pulse_gated_forward(
                p_brain_c, H_c, gate_c, mdna_c, idx_c, manifold, self._lgh_workspace,
                step, int(h_cycles), int(l_cycles), float(dyn_threshold),
                int(self.lgh_cfg.prefetch_distance),
                float(self.exec_cfg.thermal_penalty if hasattr(self.exec_cfg, 'thermal_penalty') else 0.0),
                float(self.lgh_cfg.low_entropy_fold_threshold),
                int(self.lgh_cfg.temporal_bins)
            )
        else:
            z_lgh, H_next, halt_probs = cpp_loader.geometric_manifold_forward_avx512(
                p_brain_c, H_c, gate_c, manifold, idx_c, pidx_c, mdna_c, trace_c,
                step, int(h_cycles), int(l_cycles), float(dyn_threshold),
                int(self.lgh_cfg.prefetch_distance),
                float(self.exec_cfg.thermal_penalty if hasattr(self.exec_cfg, 'thermal_penalty') else 0.0),
                float(self.lgh_cfg.low_entropy_fold_threshold),
                int(self.lgh_cfg.wave_radius),
                float(self.lgh_cfg.wave_decay),
                float(self.lgh_cfg.trace_decay),
                float(self.lgh_cfg.trace_gain),
                int(self.lgh_cfg.temporal_bins)
            )
        
        return True, z_lgh, H_next, halt_probs

    def _run_fused_cycle(self, p_brain, H, gate, B, T, h_cycles, l_cycles, dyn_threshold):
        if not (
            _cpp_has('fused_cognitive_cycle', p_brain, H, gate)
            and self.exec_cfg.use_fused_cycle
        ):
            return False, None, None, None

        p_stack = self._sync_stacked_params()
        h_out_passed = torch.empty(0)
        if not self.training:
            h_out_passed = self._new_state(B)
        gate_cpp = gate.reshape(self.L, self.R) if gate.dim() > 2 else gate
        p_brain_c = p_brain.float().contiguous()
        H_c = H.contiguous()
        gate_c = gate_cpp.float().contiguous()
        out_fused, H_next_fused, halt_probs_cycle = cpp_loader.fused_cognitive_cycle(
            p_brain_c, H_c, gate_c,
            p_stack['delays'], p_stack['tables'], p_stack['conns'],
            p_stack['decays'], p_stack['halt_w'], p_stack['halt_b'],
            int(h_cycles), int(l_cycles),
            self.cfg_lif_decay, self.cfg_lif_threshold, float(dyn_threshold),
            h_out_passed
        )
        z_L = out_fused.view(B, 1, self.R, self.d_s2, self.C).expand(-1, T, -1, -1, -1)
        return True, z_L, H_next_fused, halt_probs_cycle

    def _lgh_morton_manifold(self):
        if (not self.cfg_lgh_enabled) or (self.lgh_manifold_morton is None):
            return None
        return self.lgh_manifold_morton.contiguous()

    def _lgh_focus_point(self):
        gx = float(getattr(self, 'focus_x', 0.5))
        gy = float(getattr(self, 'focus_y', 0.5))
        gz = float(getattr(self, 'focus_z', 0.5))
        return (max(0.0, min(1.0, gx)), max(0.0, min(1.0, gy)), max(0.0, min(1.0, gz)))

    def _quantize_lgh_manifold(self, bits=8, focus_row=None, focus_strength=None, focus_sharpness=None):
        manifold = self._lgh_morton_manifold()
        if manifold is None or manifold.numel() == 0:
            return None, None
        bits = int(bits)
        if bits <= 4:
            bits = 4
        elif bits < 8:
            bits = 8
        else:
            bits = 8
        step = int(self.step_counter.item())
        focus_row_i = int(max(0, min(int(self._lgh_morton.size - 1), int(focus_row)))) if focus_row is not None else -1
        focus_strength_f = float(self.lgh_cfg.focus_strength if focus_strength is None else focus_strength)
        focus_strength_f = max(0.0, min(1.0, focus_strength_f))
        focus_sharp_f = float(self.lgh_cfg.focus_sharpness if focus_sharpness is None else focus_sharpness)
        focus_sharp_f = max(0.01, focus_sharp_f)
        phase = step % max(1, int(self.lgh_cfg.temporal_bins))
        key = (bits, focus_row_i, round(focus_strength_f, 3), round(focus_sharp_f, 3), int(phase))
        if self._lgh_q_step == step and key in self._lgh_q_cache and key in self._lgh_q_scale:
            return self._lgh_q_cache[key], self._lgh_q_scale[key]
        with torch.no_grad():
            qmax_base = 7.0 if bits <= 4 else 127.0
            qmax = torch.full((manifold.size(0),), qmax_base, dtype=torch.float32, device=manifold.device)
            bins = max(1, int(self.lgh_cfg.temporal_bins))
            n3 = max(1, int(self._lgh_morton.size))
            rows = torch.arange(manifold.size(0), device=manifold.device, dtype=torch.long)
            morton_rows = torch.div(rows, bins, rounding_mode='floor').clamp(0, max(0, n3 - 1))
            time_rows = torch.remainder(rows, bins).to(dtype=torch.float32)
            original_rows = self._lgh_morton.morton_to_original(morton_rows).clamp(0, max(0, self._lgh_morton.size_original - 1))
            space_dist = self._lgh_morton.focus_distance(original_rows, self._lgh_focus_point()).to(dtype=torch.float32)
            phase_f = float(phase)
            tdiff = torch.abs(time_rows - phase_f)
            tdist = torch.minimum(tdiff, float(bins) - tdiff) / max(1.0, float(bins))
            focal = torch.exp(-focus_sharp_f * (space_dist + 0.5 * tdist) * (space_dist + 0.5 * tdist))
            if focus_strength_f > 0.0:
                qmax = qmax + (focus_strength_f * focal * (127.0 - qmax))
            absmax = manifold.abs().amax(dim=1).clamp_min(1e-8)
            scale = absmax / qmax.clamp_min(1.0)
            q = torch.round(manifold / scale.unsqueeze(-1))
            q = torch.maximum(q, -qmax.unsqueeze(-1))
            q = torch.minimum(q, qmax.unsqueeze(-1))
            if bits <= 4 and focus_strength_f > 0.0:
                far_cut = 0.20 + (0.25 * (1.0 - focus_strength_f))
                far_mask = focal < float(far_cut)
                if far_mask.any():
                    q[far_mask] = 0.0
            q = q.to(torch.int8).contiguous()
        self._lgh_q_step = step
        self._lgh_q_cache[key] = q
        self._lgh_q_scale[key] = scale.contiguous()
        self._lgh_q_meta[key] = {'bits': bits, 'focus_row': focus_row_i, 'phase': int(phase)}
        return self._lgh_q_cache[key], self._lgh_q_scale[key]

    def _predict_curve_chain(self, p_s1):
        if (not self.cfg_lgh_enabled) or self._lgh_curve_indices.numel() == 0:
            return None
        if p_s1.dim() != 3:
            return None
        with torch.no_grad():
            s1_anchor = p_s1[:, -1, :].float()
            pred = torch.sigmoid(self.curve_index_head(s1_anchor)).mean()
            pred_start = int(round(float(pred.item()) * max(0, self._lgh_morton.size_original - 1)))
            self._lgh_last_curve_anchor.fill_(pred_start)
            if hasattr(self.genome, 'build_curve_chain'):
                chain = self.genome.build_curve_chain(
                    total_nodes=self._lgh_morton.size_original,
                    length=int(self._lgh_curve_indices.numel()),
                    hint=pred_start
                )
                # Convert Morton positions to original topology ids for kernel mapping via inverse_order.
                chain_morton = torch.as_tensor(chain, dtype=torch.long, device=self.device)
                dt = int(self.step_counter.item()) if hasattr(self, 'step_counter') else 0
                chain_morton = self._lgh_morton.temporal_fold_morton(
                    chain_morton,
                    delta_t=dt,
                    temporal_bins=self.lgh_cfg.temporal_bins,
                    fold_alpha=self.lgh_cfg.temporal_fold_alpha
                )
                return chain_morton
        return None

    def _predict_prefetch_curve_chain(self, p_s1):
        if (not self.cfg_lgh_enabled) or self._lgh_curve_indices.numel() == 0:
            return None
        if p_s1.dim() != 3:
            return None
        with torch.no_grad():
            s1_anchor = p_s1[:, -1, :].float()
            pred = torch.sigmoid(self.curve_index_head(s1_anchor)).mean()
            pred_start = int(round(float(pred.item()) * max(0, self._lgh_morton.size_original - 1)))
            prefetch_hint = pred_start + int(self.lgh_cfg.prefetch_distance)
            if hasattr(self.genome, 'build_curve_chain'):
                chain = self.genome.build_curve_chain(
                    total_nodes=self._lgh_morton.size_original,
                    length=int(self._lgh_curve_indices.numel()),
                    hint=prefetch_hint
                )
                chain_morton = torch.as_tensor(chain, dtype=torch.long, device=self.device)
                dt = int(self.step_counter.item()) if hasattr(self, 'step_counter') else 0
                chain_morton = self._lgh_morton.temporal_fold_morton(
                    chain_morton,
                    delta_t=dt + int(self.lgh_cfg.prefetch_distance),
                    temporal_bins=self.lgh_cfg.temporal_bins,
                    fold_alpha=self.lgh_cfg.temporal_fold_alpha
                )
                return chain_morton
        return None

    def _refresh_lgh_curve_from_genome(self, predicted_curve=None):
        if (not self.cfg_lgh_enabled) or self._lgh_curve_indices.numel() == 0:
            return
        if isinstance(predicted_curve, torch.Tensor) and predicted_curve.numel() == self._lgh_curve_indices.numel():
            self._lgh_curve_indices.copy_(predicted_curve.to(self._lgh_curve_indices.device, dtype=torch.long))
            return
        curve_gene = float(getattr(self, 'curve_trajectory_gene', 0.5))
        curve_gene = max(0.0, min(1.0, curve_gene))
        if hasattr(self.genome, 'next_curve_anchor'):
            start = int(self.genome.next_curve_anchor(self._lgh_morton.size_original))
        else:
            start = int(round(curve_gene * max(0, self._lgh_morton.size_original - 1)))
        focus_strength = float(self.lgh_cfg.focus_strength)
        if hasattr(self.genome, 'focus_sharpness'):
            focus_strength = max(0.0, min(1.0, 0.5 * self.lgh_cfg.focus_strength + 0.5 * float(getattr(self.genome, 'focus_sharpness', self.lgh_cfg.focus_strength))))
        focus_sharpness = float(self.lgh_cfg.focus_sharpness)
        if hasattr(self.genome, 'focus_sharpness'):
            focus_sharpness = max(0.01, float(getattr(self.genome, 'focus_sharpness', self.lgh_cfg.focus_sharpness)))
        new_curve = self._lgh_morton.curve_segment_morton(
            start,
            int(self._lgh_curve_indices.numel()),
            wrap=self.lgh_cfg.curve_wrap,
            focus_point=self._lgh_focus_point(),
            focus_strength=focus_strength,
            focus_sharpness=focus_sharpness,
            delta_t=int(self.step_counter.item()) if hasattr(self, 'step_counter') else 0,
            temporal_bins=self.lgh_cfg.temporal_bins,
            fold_alpha=self.lgh_cfg.temporal_fold_alpha
        )
        self._lgh_curve_indices.copy_(new_curve.to(self._lgh_curve_indices.device, dtype=torch.long))

    def _refresh_lgh_prefetch_curve(self, predicted_prefetch=None):
        if (not self.cfg_lgh_enabled) or self._lgh_prefetch_curve_indices.numel() == 0:
            return
        if isinstance(predicted_prefetch, torch.Tensor) and predicted_prefetch.numel() == self._lgh_prefetch_curve_indices.numel():
            self._lgh_prefetch_curve_indices.copy_(predicted_prefetch.to(self._lgh_prefetch_curve_indices.device, dtype=torch.long))
            return
        if self._lgh_curve_indices.numel() > 0:
            shifted = torch.roll(self._lgh_curve_indices, shifts=-1, dims=0)
            self._lgh_prefetch_curve_indices.copy_(shifted.to(self._lgh_prefetch_curve_indices.device, dtype=torch.long))

    def _refresh_lgh_mdna_mask(self, gate, modulation=None):
        if (not self.cfg_lgh_enabled) or self._lgh_mdna_mask.numel() == 0:
            return
        gate_2d = gate.reshape(self.L, self.R) if gate.dim() > 2 else gate
        gate_2d = torch.nan_to_num(gate_2d.float(), nan=0.0, posinf=1.0, neginf=0.0)
        if isinstance(modulation, torch.Tensor) and modulation.shape == gate_2d.shape:
            mod = torch.nan_to_num(modulation.float(), nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            self._lgh_mdna_modulation.copy_(mod)
            gate_2d = gate_2d * (0.5 + 0.5 * mod)
        min_keep = max(0.01, min(1.0, self.lgh_cfg.mask_min_keep))
        max_keep = max(min_keep, min(1.0, self.lgh_cfg.mask_max_keep))
        sparsity_bias = float(getattr(self, 'mask_sparsity_bias', 0.5))
        sparsity_bias = max(0.0, min(1.0, sparsity_bias))
        keep_ratio = max_keep - ((max_keep - min_keep) * sparsity_bias)
        keep_count = max(1, int(round(gate_2d.numel() * keep_ratio)))
        threshold = torch.topk(gate_2d.flatten(), keep_count).values.min()
        mdna = (gate_2d >= threshold).to(dtype=torch.float32)
        self._lgh_mdna_mask.copy_(mdna)

    def _update_thermal_penalty(self):
        freq_ghz = 0.0
        try:
            freq = psutil.cpu_freq()
            if freq is not None and getattr(freq, 'current', None) is not None:
                freq_ghz = max(0.0, float(freq.current) / 1000.0)
        except Exception:
            freq_ghz = 0.0
        self._lgh_last_freq_ghz.fill_(freq_ghz)

        target_ghz = max(0.1, float(self.lgh_cfg.thermal_freq_min_ghz))
        raw_penalty = 0.0 if freq_ghz <= 0.0 else max(0.0, (target_ghz - freq_ghz) / target_ghz)
        decay = max(0.0, min(0.999, float(self.lgh_cfg.thermal_ema_decay)))
        self._lgh_thermal_penalty_ema.mul_(decay).add_(float(raw_penalty) * (1.0 - decay))
        return float(self._lgh_thermal_penalty_ema.item())

    def get_thermal_penalty(self):
        return float(self._update_thermal_penalty())

    def _lgh_uncertainty(self, p_curve):
        if not isinstance(p_curve, torch.Tensor) or p_curve.dim() != 3:
            return 1.0
        with torch.no_grad():
            if p_curve.size(1) >= 2:
                d = (p_curve[:, -1, :] - p_curve[:, -2, :]).abs().mean()
            else:
                d = p_curve[:, -1, :].abs().mean()
            return float(torch.nan_to_num(d, nan=1.0, posinf=1.0, neginf=0.0).item())

    def _imprint_to_manifold(self, keys, values, importance):
        """Imprint patterns into the LGH-Manifold trajectory for O(1) density retrieval."""
        if self.lgh_manifold_morton is None or self._lgh_curve_indices.numel() == 0:
            return
        with torch.no_grad():
            indices = self._lgh_curve_indices.contiguous() # [S] Morton indices
            # Broadcast batch of important values to the global manifold
            # We use an EMA update weighted by importance.
            val_mean = values.mean(dim=0) # [D]
            imp_mean = importance.mean().item()
            lr = 0.05 * imp_mean
            # Vectorized scatter-add or indexing for imprinting along the path
            self.lgh_manifold_morton[indices] = (1.0 - lr) * self.lgh_manifold_morton[indices] + lr * val_mean.unsqueeze(0)



    def regulate_sensory_noise(self, val_loss):
        """Adjust sensory noise based on performance (Anti-Cheating)."""
        # Target loss floor: 0.5 (below this = memorization zone)
        target_floor = 0.5
        
        if val_loss < target_floor:
            # Too good? Make it harder.
            self.noise_scale = min(self.max_noise, self.noise_scale + 0.05)
            print(f">>> ANTI-CHEAT: Validation Loss {val_loss:.4f} < {target_floor}. Increasing Sensory Noise -> {self.noise_scale:.2f}")
        elif val_loss > 2.0:
            # Too hard? Make it easier.
            self.noise_scale = max(self.min_noise, self.noise_scale - 0.02)
            # print(f">>> SENSORY ADAPTATION: Reducing Noise -> {self.noise_scale:.2f}")
            
        return self.noise_scale

        # 0. Hot-path caching
        cfg_mes_enabled = self.cfg_mes_enabled
        cfg_cache_enabled = self.cfg_cache_enabled
        cfg_pruning_enabled = self.cfg_pruning_enabled
        L, R, C = self.L, self.R, self.C
        device = self.device
        current_phase = self.current_phase
        lambda_sparsity = self.lambda_sparsity
        cfg_param_cost_scale = self.cfg_param_cost_scale
        
        self.step_counter += 1
        self._run_lifecycle_hooks('pre_forward', x=x, H=H, gate=gate, mode=mode)
        
        # --- ADAPTIVE AUDIT TRIGGER ---
        is_audit = self._compute_audit_flag()
        
        # --- SYSTEM 1: Fast/Reflex Path (D_S1) ---
        p, B, T = self._encode_s1_input(x)
            
        # --- DYNAMIC SENSORY NOISE (Prevent Memorization) ---
        if self.training and self.noise_scale > 0:
            step_seed = int(self.step_counter.item())
            noise = (torch.sin(p * 100.0 + step_seed) * self.noise_scale).to(p.device)
            p = p + noise
        # ----------------------------------------------------
            
        H = self._prepare_state(H, B)
            
        # --- BIOLOGICAL GATING: Blend signals for stabilization ---
        genome_usage = self.genome.usage
        combined_scores = genome_usage.clone() if genome_usage is not None else torch.ones(L, R, device=device)
        if learning_brain is not None and hasattr(learning_brain, 'knowledge_map'):
            # Blend behavioral usage with learning contribution (Knowledge Map)
            combined_scores = 0.5 * combined_scores + 0.5 * learning_brain.knowledge_map
        
        # Add tie-breaking pseudo-noise to prevent mass-extinction of equal-score neurons.
        # Deterministic: based on level/region indices to ensure reproducibility.
        noise_vec = torch.linspace(1e-8, 1e-7, combined_scores.numel(), device=device).view_as(combined_scores)
        combined_scores = combined_scores + noise_vec


        # --- PERFORMANCE CONTROL: TPS Pressure ---
        tps_pressure = self._get_tps_pressure()
        thermal_penalty = self._update_thermal_penalty()
        
        if not cfg_pruning_enabled:
            gate = torch.ones(L, R, 1, 1, dtype=torch.float32, device=device)
        elif current_phase == 0:
            # Phase 0 (Warmup): Fixed Top-K for architecture stability
            k = max(1, int(combined_scores.numel() * Config.PHASE_0_KEEP_RATIO))
            threshold = torch.topk(combined_scores.flatten(), k).values.min()
            gate = (combined_scores >= threshold).float()[:, :, None, None]
        else:
            # Phase 1+ (Autonomous): Dynamic Metabolic + Performance Masking
            gate = self.genome.get_metabolic_mask(
                metabolic_pressure=lambda_sparsity,
                tps_pressure=tps_pressure
            )

        # L-Cycle: Cache Lookup (Instant Response)
        cache = self._cache()
        if current_phase > 1 and cfg_cache_enabled and cache is not None:
            p_last = p[:, -1]
            cache_val, hit_mask, max_sim, hit_addrs, hit_tables = cache.lookup(p_last)
            self._set_cache_trace(cache_val, hit_mask, hit_addrs, hit_tables)
            
            # --- SURPRISE-TRIGGERED AUDIT ---
            surprise_trigger = bool((max_sim < 0.85).any().item())
            
            # Full sequence bypass if everything is cached and high confidence
            if bool(hit_mask.all().item()) and (not is_audit) and (not surprise_trigger):
                out_full = cache_val.unsqueeze(1).expand(-1, T, -1)
                return out_full, H, 0.001, torch.zeros_like(gate) # Fixed fast-path cost
        else:
            self._clear_cache_trace()

        # --- SYSTEM 2: Slow/Reasoning/Cognitive Path (D_S2) ---
        # Bridge: S1 -> S2 (Information flow to reasoning layers)
        # --- COGNITIVE RESONANCE: Predictive Gating ---
        h_ctx = H.mean(dim=(1, 2)).reshape(B, -1)
        p_pred = self.surprise_head(h_ctx)
        error = p - p_pred.unsqueeze(1)
        surprise_error = torch.tanh(error.detach().abs().mean(dim=(1, 2), keepdim=True))
        surprise_gate_prob = torch.sigmoid(self.surprise_gate(h_ctx)).unsqueeze(-1)
        surprise_signal = (0.5 * surprise_error + 0.5 * surprise_gate_prob).clamp(0.0, 1.0)
        self._last_surprise_signal.copy_(surprise_signal.mean().detach().to(device, dtype=torch.float32))
        temporal_signal = self._temporal_activity_signal(p)
        gaba = getattr(self, 'gaba_inhibition', 0.5)
        fkbp5 = getattr(self, 'fkbp5', 0.5)
        # Resonance Logic:
        # - High GABA: Inhibits raw transport (Focus)
        # - High FKBP5: Amplifies prediction error (Stress)
        p = (1.0 - gaba) * p + fkbp5 * error
        
        p_brain = self.bridge_s1_to_s2(p) # [B, T, D_S2 * C]
        p_expanded = self._expand_brain_to_levels(p_brain)
        mode = 'parallel' if T > 1 else 'recurrent'
        
        z_H = H
        should_update_survival = self._should_update_survival()
        z_H_start = H.clone() if should_update_survival else None
        z_L = p_expanded
        
        

        # Contextualization is now handled implicitly by the LGH-Manifold traverser.

        # === DISTRIBUTED COGNITIVE CYCLE ===
        # Prefer fused kernel when enabled; otherwise use required forward_stack_io path.
        h_cycles_cfg, l_cycles_cfg = self._dynamic_cycle_counts()
        h_cycles_cfg, l_cycles_cfg, skip_surprise = self._apply_surprise_cycle_control(
            h_cycles_cfg, l_cycles_cfg, surprise_signal, is_audit=is_audit
        )
        h_cycles_cfg, l_cycles_cfg, skip_temporal = self._apply_temporal_cycle_control(
            h_cycles_cfg, l_cycles_cfg, temporal_signal, is_audit=is_audit
        )
        h_cycles_cfg, l_cycles_cfg = self._apply_efficiency_cycle_control(
            h_cycles_cfg, l_cycles_cfg, p_brain, gate, is_audit=is_audit
        )
        skip_reasoning = bool(skip_surprise and skip_temporal)
        dyn_threshold = self._compute_dyn_halt_threshold(tps_pressure=tps_pressure)
        h_cycles_used = float(max(1, h_cycles_cfg * l_cycles_cfg))
        if skip_reasoning:
            H_next = H
            halting_probability = torch.zeros(B, 1, 1, dtype=torch.float32, device=self.device)
            h_cycles_used = 0.0
        else:
            lgh_ok, z_L_lgh, H_next_lgh, halting_probability_lgh = self._run_lgh_cycle(
                p_brain, H, gate, B, T, h_cycles_cfg, l_cycles_cfg, dyn_threshold, p_s1=p
            )
            if lgh_ok:
                z_L = z_L_lgh
                H_next = H_next_lgh
                halting_probability = halting_probability_lgh
            else:
                fused_ok, z_L_fused, H_next_fused, halting_probability_fused = self._run_fused_cycle(
                    p_brain, H, gate, B, T, h_cycles_cfg, l_cycles_cfg, dyn_threshold
                )
            if (not lgh_ok) and fused_ok:
                z_L = z_L_fused
                H_next = H_next_fused
                halting_probability = halting_probability_fused
            elif not lgh_ok:
                cos_sin = self.rope(T, self.device)
                z_L, H_next, halting_probability = self._step_inner(
                    z_L, z_H, cos_sin, gate, mode, learning_brain, l_cycles=l_cycles_cfg, dyn_threshold=dyn_threshold
                )
                h_cycles_used = float(max(1, l_cycles_cfg))
                if z_L.dim() == 5 and z_L.size(1) == 1 and T > 1:
                    z_L = z_L.expand(-1, T, -1, -1, -1)
                if halting_probability.dim() > 2:
                    halting_probability = halting_probability.mean(dim=1)

        hpc_error = self._hpc_refresh_from_states(H_next, H)
        
        # --- BIOLOGICAL FEATURE: Apply Temporal Scaling (Multi-scale learning rates) ---
        # Matrix-based vectorized temporal scaling (No Loop)
        if learning_brain is not None and hasattr(learning_brain, 'apply_temporal_scaling_vec'):
            H_next = learning_brain.apply_temporal_scaling_vec(H_next, H)
        
        # --- BIOLOGICAL FEATURE: Manifold Memory Imprinting ---
        # Strategic Gating provides both Engagement (Transparency) and Importance (Imprinting).
        with torch.no_grad():
            p_flat = p.mean(dim=1).detach()
            # Engagement head not used here yet, but importance head drives imprinting.
            _, importance_score = self.strategic_gating(p_flat)
            importance_score = importance_score.squeeze(-1)
            importance_threshold = self._compute_importance_threshold(importance_score)
            should_store = (importance_score > importance_threshold).any()
            if should_store:
                self._imprint_to_manifold(p_flat, z_L.reshape(B, -1, self.R * self.d_s2 * self.C).mean(dim=1)[:, :self.d_s2 * self.C], importance_score)
        
        T_out = z_L.size(1)
        y_flat = z_L.reshape(B, T_out, self.R * self.d_s2 * self.C)
        y_flat = rms_norm(y_flat) # Stability before readout
        
        # --- BIOLOGICAL FEATURE: Myelin Update (Use-dependent insulation) ---
        if self.training and int(self.step_counter.item()) % 100 == 0:
            with torch.no_grad():
                self.myelin_sheaths.data.mul_(0.99).add_(H_next.view(B, self.L, self.R, -1).norm(dim=(0, 3)), alpha=0.01)
        
        out = self.readout(y_flat)
        
        # PRUNING 2.0: update survival usage on throttled cadence to reduce CPU stalls.
        self._maybe_update_survival(H_next, z_H_start, should_update=should_update_survival)
        
        cost_step = gate.sum() * self.cfg_param_cost_scale
        
        # Calculate aggregate engagement rate from all levels
        self._current_engagement_rate = self._compute_engagement_rate()
        
        # Log to Virtual Lab if enabled
        if self.virtual_lab.enabled:
            self.virtual_lab.log_step({
                'loss_task': torch.tensor(0.0),
                'cost_step': cost_step,
                'mask': gate,
                'halt': halting_probability.mean(), # Tracking the 'Mood'
                'audit': float(is_audit),
                'thinking_duration': h_cycles_used,
                'hpc_error': float(hpc_error.item()) if isinstance(hpc_error, torch.Tensor) else float(hpc_error),
                'hpc_cycle_scale': float(self._hpc_last_cycle_scale.item()),
                'hpc_surprise_signal': float(self._last_surprise_signal.item()),
                'hpc_surprise_scale': float(self._last_surprise_cycle_scale.item()),
                'hpc_temporal_signal': float(self._last_temporal_signal.item()),
                'hpc_temporal_scale': float(self._last_temporal_cycle_scale.item()),
                'engagement_rate': self._current_engagement_rate,
                'bypass_rate': 1.0 - self._current_engagement_rate,
                'thermal_penalty': float(thermal_penalty),
                'cpu_freq_ghz': float(self._lgh_last_freq_ghz.item()),
                't': T,
                'B': B
            })
            self._vl_add_scalar("TransparencyGate/engagement", self._current_engagement_rate)
            self._vl_add_scalar("TransparencyGate/omega", self.omega)
            self._vl_add_scalar("HPC/error_ema", self._hpc_error_ema)
            self._vl_add_scalar("HPC/cycle_scale", self._hpc_last_cycle_scale)
            self._vl_add_scalar("HPC/surprise_signal", self._last_surprise_signal)
            self._vl_add_scalar("HPC/surprise_scale", self._last_surprise_cycle_scale)
            self._vl_add_scalar("HPC/temporal_signal", self._last_temporal_signal)
            self._vl_add_scalar("HPC/temporal_scale", self._last_temporal_cycle_scale)
            
        # Update preflight status after stabilization
        if not self._preflight_ready:
            self._preflight_steps += 1
            if self._preflight_steps >= 10:
                self._preflight_ready = True
                print(">>> PERFORMANCE: Preflight Bypass Ready (Bypassing shape/contiguity checks).")

        # --- PERIODIC CONSOLIDATION (Shadow Brain) ---
        self._run_lifecycle_hooks(
            'post_forward',
            out=out,
            H_next=H_next,
            gate=gate,
            is_audit=is_audit
        )
        return out, H_next, cost_step, gate

    def _step_inner(self, z_L, z_H, _cos_sin, gate, mode, learning_brain, l_cycles=None, dyn_threshold=None):
        """
        Reasoning Core: Integrated C++ Stack Fusion (no Python fallback).
        """
        if not self.exec_cfg.use_forward_stack:
            raise RuntimeError("USE_FORWARD_STACK must be enabled; Python reasoning fallback is removed.")
        if not _cpp_has('forward_stack_io', z_L, z_H):
            raise RuntimeError("Missing required C++ op 'forward_stack_io'.")
        if not (
            (not torch.is_grad_enabled())
            or (
                self.mes_cfg.enabled
                and (not self.mes_cfg.global_backprop)
            )
        ):
            raise RuntimeError("forward_stack_io path is required; incompatible grad mode for this configuration.")
        p_stack = self._sync_stacked_params()
        if dyn_threshold is None:
            dyn_threshold = self._compute_dyn_halt_threshold()
        use_l_cycles = self.cfg_l_cycles if l_cycles is None else max(1, int(l_cycles))
        return self._call_forward_stack(z_L, z_H, p_stack, dyn_threshold, use_l_cycles)

    @staticmethod
    def _last_target_bits(target_bits):
        if target_bits.dim() == 3:
            return target_bits[:, -1:]
        if target_bits.dim() == 2:
            return target_bits.unsqueeze(1)
        raise ValueError(f"Unsupported target_bits shape: {tuple(target_bits.shape)}")

    def _mes_prepare_window(self, x, target_bits, max_window=32):
        start_t = max(0, x.size(1) - int(max_window))
        x_win = x[:, start_t:]
        target_win = target_bits[:, start_t:]
        p_latent, _, _ = self._encode_s1_input(x_win)
        p_brain = self.bridge_s1_to_s2(p_latent)
        target_latent, _, _ = self._encode_s1_input(target_win)
        target_brain = self.bridge_s1_to_s2(target_latent)
        return x_win, target_latent, p_brain, target_brain

    def _mes_super_kernel_step(self, p_brain, target_brain, H_inter, apply_update=True):
        if not self.mes_cfg.super_kernel:
            raise RuntimeError("MES_SUPER_KERNEL must be enabled; Python MES fallback is removed.")
        p_stack = self._sync_stacked_params()
        p_brain_c = p_brain.contiguous().float()
        target_brain_c = target_brain.contiguous().float()
        H_inter_c = H_inter.contiguous().float()
        self._mes_super_scalars[0] = self.runtime_cfg.lif_decay
        self._mes_super_scalars[1] = self.runtime_cfg.lif_threshold
        self._mes_super_scalars[2] = self.mes_cfg.local_l1
        out = self._cpp_io_call(
            'mes_super_step_io',
            p_brain_c,
            H_inter_c,
            params=[target_brain_c, p_stack['delays'], p_stack['tables'], p_stack['conns']],
            scalars=self._mes_super_scalars
        )
        if not isinstance(out, (list, tuple)) or len(out) != 2:
            raise RuntimeError("mes_super_step_io returned unexpected output.")
        loss_local, grad_tables = out
        if not isinstance(grad_tables, torch.Tensor) or grad_tables.dim() != 3 or grad_tables.size(0) != self.L:
            raise RuntimeError("mes_super_step_io returned invalid grad_tables tensor.")

        if apply_update:
            with torch.no_grad():
                for l, level in enumerate(self.levels):
                    level.wsnn.ram_tables.grad = grad_tables[l].contiguous()
            self._batched_mes_optimizer_step()
        return float(loss_local.item())

    def _mes_train_surprise_head(self, H_next, target_latent):
        self.surprise_optimizer.zero_grad(set_to_none=True)
        h_ctx = H_next.detach()[:, -1].mean(dim=1).reshape(H_next.size(0), -1)
        p_pred = self.surprise_head(h_ctx)
        target_specific = target_latent.detach()[:, -1, :]
        loss_surprise = F.mse_loss(p_pred, target_specific)
        gate_pred = torch.sigmoid(self.surprise_gate(h_ctx))
        surprise_target = torch.tanh((p_pred.detach() - target_specific).abs().mean(dim=-1, keepdim=True))
        loss_gate = F.binary_cross_entropy(gate_pred, surprise_target)
        total_surprise_loss = loss_surprise + self.hpc_cfg.surprise_loss_weight * loss_gate
        total_surprise_loss.backward()
        self.surprise_optimizer.step()
        return total_surprise_loss

    def _imprint_high_frequency_cache(self, cache):
        high_freq_mask = (cache.hit_frequency > self.imprint_threshold) & cache.valid
        if not bool(high_freq_mask.any().item()):
            return 0
        if not hasattr(self.levels[0].wsnn, 'ram_tables'):
            return 0
        indices = high_freq_mask.nonzero(as_tuple=False)
        t_idx = indices[:, 0]
        a_idx = indices[:, 1]
        ram_size = self.levels[0].wsnn.ram_tables.size(1)
        target_ram_addrs = a_idx % ram_size
        cache_bits = cache.values[t_idx, a_idx].float()
        self.levels[0].wsnn.ram_tables.data[0].scatter_add_(0, target_ram_addrs, 0.1 * cache_bits.mean(dim=1))
        return int(indices.size(0))

    def _apply_audit_dissonance(self, cache, s2_out, is_audit):
        if isinstance(is_audit, torch.Tensor):
            audit_flag = bool(is_audit.any().item())
        else:
            audit_flag = bool(is_audit)
        if not (audit_flag and self._last_cache_bits is not None):
            return 0
        s1_pred = (self._last_cache_bits > 0.0).float()
        s2_pred = (s2_out > 0.0).float()[:, -1]
        tps_pressure = self._get_tps_pressure()
        dissonance_weight = self.runtime_cfg.dissonance_penalty * (1.0 - 0.9 * tps_pressure)
        s1_conf = (self._last_cache_bits.abs()).mean(dim=-1)
        s2_conf = (s2_out.abs()).mean(dim=(-1, -2, -3))[:, -1]
        dissonance = (s1_pred != s2_pred).any(dim=-1)
        is_anxious = (s2_conf < (s1_conf + self.runtime_cfg.dissonance_confidence_threshold))
        valid_dissonance = dissonance & (~is_anxious)
        if not bool(valid_dissonance.any().item()):
            return 0

        # Relax reliability in genome
        self.genome.reliability[self._layer_indices, self._block_indices] *= (1.0 - dissonance_weight)
        self.genome.reliability[self._layer_indices, self._block_indices].clamp_(min=0.0)
        t_indices = self._last_hit_tables[valid_dissonance]
        a_indices = self._last_hit_addrs[valid_dissonance]
        cache.reliability[t_indices, a_indices] *= (1.0 - dissonance_weight)
        old_omega = self.omega
        self.omega = max(0.0, self.omega - 0.05)
        if self.virtual_lab.enabled:
            print(f">>> COGNITIVE DISSONANCE: Audit failed. Omega {old_omega:.3f} -> {self.omega:.3f} | Weight={dissonance_weight:.3f}")
        return int(valid_dissonance.sum().item())

    def _demyelinate_unreliable_cache(self, cache):
        unreliable_mask = (cache.reliability < 0.40) & cache.valid
        if not bool(unreliable_mask.any().item()):
            return 0
        cache.valid[unreliable_mask] = False
        indices = unreliable_mask.nonzero(as_tuple=False)
        layer_indices = indices[:, 0] % self.L
        block_indices = indices[:, 1] % self.R
        self.myelin_sheaths.data[layer_indices, block_indices] = 0.0
        if self.virtual_lab.enabled:
            print(f">>> DE-MYELINATION: {int(indices.size(0))} reflexes lost confidence. Stripping myelin...")
        return int(indices.size(0))

    def mes_step(self, x, target_bits, precomputed_H_next=None, dry_run=False):
        """
        Total Metabolic Engram Sculpting (MES): 
        Sole learning driver. No global backprop.
        """
        if not dry_run:
            self.mes_throttle_count += 1
        
        if not self.mes_cfg.enabled:
            return {}
        
        # Performance Throttling: Skip MES update occasionally to maintain high TPS
        if (not dry_run) and (self.mes_throttle_count % self.mes_cfg.skip_step != 0):
            return {'mes_loss': 0.0, 'throttled': True}

        B = x.size(0)
        
        # 1. READOUT UPDATE (The Objective Manifold)
        if not dry_run:
            self.readout_optimizer.zero_grad(set_to_none=True)
        
        # We need H_next to map to targets.
        if precomputed_H_next is None:
            with torch.no_grad():
                _, H_next, _, _ = self.forward(x, None)
        else:
            H_next = self._prepare_state(precomputed_H_next, B).detach()
        
        # [B, L, R, D_S2, C] -> [B, 1, R * D_S2 * C]
        H_avg = H_next.mean(dim=1)  # Average over layers
        H_flat = H_avg.reshape(B, 1, self.R * self.d_s2 * self.C)
        target_logits = self.readout(H_flat)
        
        # Use robust slicing to match the single-token outcome from forward().
        target_bits_last = self._last_target_bits(target_bits)
        # Ensure target_logits is [B, 1, 8] to match target_bits_last
        if target_logits.dim() == 2:
            target_logits = target_logits.unsqueeze(1)
            
        loss_readout = F.binary_cross_entropy_with_logits(target_logits, target_bits_last)

        # 2. INTERNAL MES (NoProp Target Descent)
        # Process a small window (32 tokens) to stay within 4GB RAM while keeping delays
        with torch.no_grad():
            x_win, target_latent, p_brain, target_brain = self._mes_prepare_window(
                x, target_bits, max_window=32
            )

        H_inter = self._new_state(B, clone=True)
        mes_loss_value = 0.0

        # Prefer C++ MES super-kernel path (single-call orchestration + table grads).
        local_loss_cpp = self._mes_super_kernel_step(
            p_brain, target_brain, H_inter, apply_update=(not dry_run)
        )
        # Super-kernel path only needs readout backward here.
        if not dry_run:
            loss_readout.backward()
        mes_loss_value = float(loss_readout.item() + local_loss_cpp)

        if not dry_run:
            self.readout_optimizer.step()
        
        # 3. BIT-TO-LATENT (The Sensory Manifold)
        if not dry_run:
            self.s1_optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():
                p_mes, _, _ = self._encode_s1_input(x_win)
                loss_s1 = F.mse_loss(p_mes, target_latent.detach())
                loss_s1.backward()
                self.s1_optimizer.step()

        # 4. SURPRISE HEAD TRAINING
        if not dry_run:
            self._mes_train_surprise_head(H_next, target_latent)
            hpc_loss = self._hpc_train_predictors(H_next)
        else:
            hpc_loss = 0.0
        hpc_weight = self.hpc_cfg.local_loss_weight
        mes_loss_value += hpc_weight * hpc_loss

        return {'mes_loss': mes_loss_value, 'hpc_loss': hpc_loss, 'dry_run': bool(dry_run)}

    @torch.no_grad()
    def _consolidate_memories(self, s2_out, is_audit):
        """
        THE SHADOW BRAIN (Vectorized):
        1. Imprints high-frequency reflexes into permanent weights.
        2. Slashes reliability of reflexes that fail the audit.
        3. Triggers de-myelination for unreliable pathways.
        """
        cache = self._cache()
        if cache is None or not hasattr(cache, 'hit_frequency'):
            return
        
        self._imprint_high_frequency_cache(cache)
                       
        # METABOLIC TAXATION (Homeostasis)
        # Every consolidation cycle, neurons that haven't fired lose a bit of energy
        # This prevents the 100M parameters from becoming "Stochastic Parrots"
        if self.genome.usage is not None:
            self.genome.usage.data *= (1.0 - self.runtime_cfg.metabolic_tax_rate)
        self._apply_audit_dissonance(cache, s2_out, is_audit)
        self._demyelinate_unreliable_cache(cache)

    @torch.no_grad()
    def get_state_dict_ram(self):
        """Returns a copy of the state_dict in RAM (Tensors only) to avoid serialization overhead."""
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    @torch.no_grad()
    def load_state_dict_ram(self, state_dict_ram):
        """Loads a state_dict directly from RAM tensors."""
        self.load_state_dict(state_dict_ram)
