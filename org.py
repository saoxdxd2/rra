import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from accelerator import get_accelerator

_ACCEL = get_accelerator()
cpp_loader = _ACCEL.loader
Config = cpp_loader

DEVICE = torch.device(getattr(Config, "DEVICE", "cpu"))
if DEVICE.type != "cpu":
    DEVICE = torch.device("cpu")


def init_state(
    L: int,
    R: int,
    D: int,
    C: int,
    device: Any = "cpu",
    scale: Optional[float] = None,
) -> torch.Tensor:
    dev = torch.device(device)
    if scale is None:
        scale = float(getattr(Config, "INIT_SCALE", 0.0))
    if float(scale) > 0.0:
        return (torch.randn(L, R, D, C, device=dev) * float(scale)).contiguous()
    return torch.zeros(L, R, D, C, device=dev, dtype=torch.float32).contiguous()


class VirtualLab:
    def __init__(self, enabled: bool = False, log_dir: str = "logs"):
        self.enabled = bool(enabled)
        self._log_dir = log_dir
        self.writer = None
        self._step_count = 0
        self._start = time.time()
        self._last = {}

    def enable(self):
        self.enabled = True
        if self.writer is None and SummaryWriter is not None:
            self.writer = SummaryWriter(self._log_dir)

    def disable(self):
        self.enabled = False

    def log_step(self, payload: Dict[str, Any]):
        if not isinstance(payload, dict):
            return
        self._step_count += 1
        self._last = payload

    def get_benchmarks(self) -> Dict[str, float]:
        elapsed = max(1e-6, time.time() - self._start)
        return {
            "tps_pressure": float(self._step_count) / elapsed,
        }


@dataclass
class _PolicyConfig:
    simd_penalty_weight: float = float(getattr(Config, "LGH_SIMD_CYCLE_PENALTY_WEIGHT", 0.15))
    simd_starvation_threshold: float = float(getattr(Config, "LGH_SIMD_STARVATION_THRESHOLD", 1200.0))


class Governor(nn.Module):
    GENE_DEFAULTS = {
        "bdnf": 0.5,
        "creb": 0.5,
        "drd2": 0.5,
        "fkbp5": 0.5,
    }

    def __init__(self, L: int, R: int, input_dim: int, hidden_dim: int = 64, device: Any = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.engagement_head = nn.Linear(hidden_dim, 1)
        self.importance_head = nn.Linear(hidden_dim, 1)

        self.register_buffer("usage", torch.zeros(L, R, dtype=torch.float32, device=self.device))
        self.register_buffer("reliability", torch.ones(L, R, dtype=torch.float32, device=self.device))
        self.register_buffer("config_scalars", torch.zeros(16, dtype=torch.float32, device="cpu"))
        self.generation = 0

        self.policy = SimpleNamespace(config=_PolicyConfig())
        for name, value in self.GENE_DEFAULTS.items():
            setattr(self, name, float(value))

        self.config_scalars[0] = float(getattr(Config, "HPC_ERROR_EMA_DECAY", 0.95))
        self.config_scalars[1] = 1.0
        self.config_scalars[2] = 1.0
        self.config_scalars[6] = 1.0 if bool(getattr(Config, "SURVIVAL_ENABLED", True)) else 0.0

    def forward(self, x: torch.Tensor):
        h = self.shared(x.float())
        engagement = torch.sigmoid(self.engagement_head(h))
        importance = torch.sigmoid(self.importance_head(h))
        return engagement, importance

    def update_metabolism(self, H_act: torch.Tensor):
        if not isinstance(H_act, torch.Tensor) or H_act.dim() != 5:
            return self.usage
        usage_now = H_act.abs().mean(dim=(0, 3, 4))
        self.usage.mul_(0.9).add_(usage_now.to(self.usage.dtype) * 0.1)
        return self.usage

    def get_expression(self, name: str):
        return float(getattr(self, name, 0.5))

    def get_all_expressions(self):
        return torch.tensor(
            [
                float(self.bdnf),
                float(self.creb),
                float(self.drd2),
                float(self.fkbp5),
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            dtype=torch.float32,
        )

    def invalidate_expression_cache(self):
        return None

    def evolutionary_step(self, model, metrics: Optional[Dict[str, Any]] = None):
        self.generation += 1
        if not isinstance(metrics, dict):
            return
        val_loss = float(metrics.get("val_loss", 1.0))
        delta = max(-0.02, min(0.02, 0.5 - val_loss))
        self.bdnf = max(0.0, min(1.0, self.bdnf + delta))
        self.fkbp5 = max(0.0, min(1.0, self.fkbp5 - delta))


class VNNILinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any = "cpu"):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.randn(out_features, in_features, device=device) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features, device=device)) if bias else None
        self.register_buffer("w_q", torch.empty(0, dtype=torch.int8, device=device))
        self.register_buffer("scale_w", torch.empty(0, dtype=torch.float32, device=device))
        self._is_quantized = False

    @torch.no_grad()
    def quantize(self):
        w = self.weight.detach()
        absmax = w.abs().amax(dim=1).clamp_min(1e-8)
        scale = (absmax / 127.0).contiguous()
        q = torch.round(w / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8).contiguous()
        self.w_q = q
        self.scale_w = scale
        self._is_quantized = True

    def forward(self, x: torch.Tensor):
        if self._is_quantized and self.w_q.numel() > 0 and self.scale_w.numel() > 0:
            return _ACCEL.call(
                "quantized_matmul",
                x.contiguous().float(),
                self.w_q,
                self.scale_w,
                self.bias if self.bias is not None else torch.empty(0, device=x.device),
                tensors=(x, self.w_q, self.scale_w),
            )
        return F.linear(x, self.weight, self.bias)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, out_dim: int, device: Any = "cpu"):
        super().__init__()
        self.w = nn.Linear(dim, out_dim, device=device)
        self.v = nn.Linear(dim, out_dim, device=device)

    def forward(self, x: torch.Tensor):
        return self.w(x) * torch.sigmoid(self.v(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, device: Any = "cpu"):
        super().__init__()
        self.dim = int(dim)
        self.device = torch.device(device)

    def forward(self, T: int, device: Optional[Any] = None):
        dev = self.device if device is None else torch.device(device)
        t = torch.arange(T, device=dev, dtype=torch.float32)
        freqs = torch.arange(self.dim, device=dev, dtype=torch.float32) / max(1.0, float(self.dim))
        phase = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cos(phase), torch.sin(phase)


class _TitansMemory:
    def reset_memory(self):
        return None


class CognitiveOrganism(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        L: Optional[int] = None,
        R: Optional[int] = None,
        d_s1: Optional[int] = None,
        d_s2: Optional[int] = None,
        D: Optional[int] = None,
        C: Optional[int] = None,
        vocab_size: int = 256,
        output_dim: int = 8,
        device: Any = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        cfg_L = int(getattr(Config, "L", 4))
        cfg_R = int(getattr(Config, "R", 8))
        cfg_C = int(getattr(Config, "C", 4))
        cfg_working_dim = int(getattr(Config, "WORKING_DIM", 64))
        cfg_d_s2 = int(max(1, cfg_working_dim // max(1, cfg_C)))

        if L is not None and int(L) != cfg_L:
            raise RuntimeError(f"CognitiveOrganism requires L={cfg_L} from C++ firmware, got L={int(L)}.")
        if R is not None and int(R) != cfg_R:
            raise RuntimeError(f"CognitiveOrganism requires R={cfg_R} from C++ firmware, got R={int(R)}.")
        if C is not None and int(C) != cfg_C:
            raise RuntimeError(f"CognitiveOrganism requires C={cfg_C} from C++ firmware, got C={int(C)}.")
        requested_D = D if D is not None else d_s2
        if requested_D is not None and int(requested_D) not in (cfg_working_dim, cfg_d_s2):
            raise RuntimeError(
                "CognitiveOrganism dimension mismatch. "
                f"Expected d_s2 in {{WORKING_DIM={cfg_working_dim}, WORKING_DIM/C={cfg_d_s2}}}, "
                f"got {int(requested_D)}."
            )

        self.L = cfg_L
        self.R = cfg_R
        self.C = cfg_C
        self.d_s2 = cfg_d_s2
        self.d_s1 = int(max(1, cfg_working_dim // 8) if d_s1 is None else d_s1)
        self.input_dim = int(input_dim)
        self.vocab_size = int(vocab_size)
        self.output_dim = 8
        self.current_phase = 0
        self.omega = 0.0
        self.suggested_lr = float(getattr(Config, "LEARNING_RATE", 1e-4))
        self.lambda_sparsity = float(getattr(Config, "LAMBDA_STABILITY", 0.01))
        self.mask_sparsity_bias = 0.0
        self.H_cycles = int(getattr(Config, "H_CYCLES", 1))
        self.L_cycles = int(getattr(Config, "L_CYCLES", 1))
        self.noise_scale = 1.0
        self._last_cache_bits = None
        self._params_dirty = True
        self._preflight_ready = False
        self._fwd_params = None
        self._fwd_scalars = None

        self.bit_to_latent = VNNILinear(8, self.d_s1 * self.C, bias=True, device=self.device)
        self.bridge_s1_to_s2 = VNNILinear(self.d_s1 * self.C, self.d_s2 * self.C, bias=False, device=self.device)
        self.readout = VNNILinear(self.R * self.d_s2 * self.C, self.output_dim, bias=True, device=self.device)
        self.surprise_head = nn.Linear(self.d_s1 * self.C, 1, device=self.device)
        self.surprise_gate = nn.Linear(self.d_s1 * self.C, 1, device=self.device)

        M = self.d_s2 * self.C
        self._stacked_delays = nn.Parameter(torch.zeros(self.L, M, 4, dtype=torch.float32, device=self.device))
        self._stacked_tables = nn.Parameter(torch.randn(self.L, M, 256, dtype=torch.float32, device=self.device) * 0.01)
        self.register_buffer("_stacked_conns", torch.zeros(self.L, M, 4, dtype=torch.long, device=self.device))
        self._stacked_decays = nn.Parameter(torch.zeros(self.L, self.R, self.d_s2 * self.C, dtype=torch.float32, device=self.device))
        self._stacked_halt_w = nn.Parameter(torch.randn(self.L, self.R, self.d_s2 * self.C, dtype=torch.float32, device=self.device) * 0.01)
        self._stacked_halt_b = nn.Parameter(torch.zeros(self.L, dtype=torch.float32, device=self.device))

        self.virtual_lab = VirtualLab(enabled=bool(getattr(Config, "VIRTUAL_LAB_ENABLED", False)))
        self.governor = Governor(self.L, self.R, input_dim=self.d_s2 * self.C, device=self.device)
        self.titans_memory = _TitansMemory()
        self.register_buffer("myelin_sheaths", torch.ones(self.L, self.R, dtype=torch.float32, device=self.device))
        self.register_buffer("step_counter", torch.zeros((), dtype=torch.long, device=self.device))

        if bool(getattr(Config, "LGH_ENABLED", False)):
            bins = int(max(1, getattr(Config, "LGH_TEMPORAL_BINS", 16)))
            n3 = int(max(1, self.L * self.R))
            self.lgh_manifold_morton = nn.Parameter(
                torch.randn(n3 * bins, self.d_s2 * self.C, dtype=torch.float32, device=self.device) * 0.01
            )
            curve_len = int(min(max(1, getattr(Config, "LGH_CURVE_LENGTH", 64)), n3 * bins))
            self.register_buffer("_lgh_curve_indices", torch.arange(curve_len, dtype=torch.long, device=self.device))
        else:
            self.lgh_manifold_morton = None
            self.register_buffer("_lgh_curve_indices", torch.empty(0, dtype=torch.long, device=self.device))

        self.register_buffer(
            "H_buffer",
            torch.zeros(
                max(1, int(getattr(Config, "BATCH_SIZE", 1))),
                self.L,
                self.R,
                self.d_s2,
                self.C,
                dtype=torch.float32,
                device=self.device,
            ),
        )
        self.update_phenotype()

    def _bytes_to_bits(self, x: torch.Tensor) -> torch.Tensor:
        bits = torch.arange(7, -1, -1, device=x.device, dtype=torch.long)
        return ((x.long().unsqueeze(-1) >> bits) & 1).to(torch.float32)

    def _encode_s1_input(self, x: torch.Tensor):
        if x.dim() == 2:
            x_bits = self._bytes_to_bits(x)
        elif x.dim() == 3 and x.size(-1) == 8:
            x_bits = x.to(torch.float32)
        else:
            raise ValueError(f"Unsupported input shape for _encode_s1_input: {tuple(x.shape)}")
        p = self.bit_to_latent(x_bits.reshape(-1, 8)).reshape(x_bits.size(0), x_bits.size(1), self.d_s1 * self.C)
        return p, x_bits.size(0), x_bits.size(1)

    def _prepare_state(self, H: Optional[torch.Tensor], batch_size: int):
        if H is None:
            return init_state(self.L, self.R, self.d_s2, self.C, device=self.device).unsqueeze(0).expand(batch_size, -1, -1, -1, -1).contiguous()
        if H.dim() == 4:
            return H.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).contiguous()
        if H.dim() == 5:
            if H.size(0) != batch_size:
                return H[:1].expand(batch_size, -1, -1, -1, -1).contiguous()
            return H.contiguous()
        raise ValueError(f"Unsupported H shape: {tuple(H.shape)}")

    def _pack_forward_params(self):
        if hasattr(self.bit_to_latent, "quantize"):
            self.bit_to_latent.quantize()
        if hasattr(self.readout, "quantize"):
            self.readout.quantize()

        def _vnni_parts(layer: VNNILinear):
            if layer._is_quantized and layer.w_q.numel() > 0 and layer.scale_w.numel() > 0:
                b = layer.bias if layer.bias is not None else torch.empty(0, device=self.device)
                return layer.w_q, layer.scale_w, b
            b = layer.bias if layer.bias is not None else torch.empty(0, device=self.device)
            return layer.weight, torch.empty(0, device=self.device), b

        bp_w, bp_s, bp_b = _vnni_parts(self.bit_to_latent)
        br_w = self.bridge_s1_to_s2.weight
        br_s = torch.empty(0, device=self.device)
        ro_w, ro_s, ro_b = _vnni_parts(self.readout)

        lgh_m = self.lgh_manifold_morton if self.lgh_manifold_morton is not None else torch.empty(0, device=self.device)
        lgh_idx = self._lgh_curve_indices if self._lgh_curve_indices.numel() > 0 else torch.empty(0, dtype=torch.long, device=self.device)

        self._fwd_params = [
            bp_w, bp_s, bp_b,
            self.surprise_head.weight, (self.surprise_head.bias if self.surprise_head.bias is not None else torch.empty(0, device=self.device)),
            self.surprise_gate.weight, (self.surprise_gate.bias if self.surprise_gate.bias is not None else torch.empty(0, device=self.device)),
            br_w, br_s,
            self._stacked_delays,
            self._stacked_tables,
            self._stacked_conns,
            torch.sigmoid(self._stacked_decays).clamp(1e-4, 0.999),
            self._stacked_halt_w.view(self.L, -1),
            self._stacked_halt_b.view(self.L),
            ro_w, ro_s, ro_b,
            lgh_m, lgh_idx, self.governor.usage,
        ]
        self._fwd_scalars = torch.tensor(
            [
                float(self.L), float(self.R), float(self.d_s1), float(self.d_s2), float(self.C),
                float(getattr(Config, "LIF_DECAY", 0.95)),
                float(getattr(Config, "LIF_THRESHOLD", 0.5)),
                float(getattr(Config, "HALT_THRESHOLD", 0.5)),
                float(self.lambda_sparsity),
                float(getattr(Config, "LGH_TRACE_GAIN", 0.2)),
                float(getattr(Config, "PARAM_COST_SCALE", 0.01)),
                float(getattr(Config, "PHASE_0_KEEP_RATIO", 0.5)),
                float(self.current_phase),
                0.8,
            ],
            dtype=torch.float32,
            device="cpu",
        )
        self._params_dirty = False

    def update_phenotype(self):
        self.suggested_lr = float(getattr(Config, "LEARNING_RATE", self.suggested_lr))
        self._pack_forward_params()

    def _compute_dyn_halt_threshold(self, tps_pressure: Optional[float] = None):
        base = float(getattr(Config, "HALT_THRESHOLD", 0.5))
        if tps_pressure is None:
            return base
        return float(max(0.05, min(0.99, base + 0.05 * float(tps_pressure))))

    def _dynamic_cycle_counts(self):
        return max(1, int(self.H_cycles)), max(1, int(self.L_cycles))

    def _lgh_manifold_recall(self, p_brain: torch.Tensor):
        if self.lgh_manifold_morton is None or self._lgh_curve_indices.numel() == 0:
            return torch.zeros_like(p_brain)
        if not _ACCEL.has("geometric_manifold_forward_avx512", p_brain):
            raise RuntimeError("Missing required C++ op 'geometric_manifold_forward_avx512'.")
        return _ACCEL.call(
            "geometric_manifold_forward_avx512",
            p_brain,
            self.lgh_manifold_morton,
            torch.empty(0, device=self.device),
            self._lgh_curve_indices,
            float(getattr(Config, "LGH_FOCUS_SHARPNESS", 2.0)),
            tensors=(p_brain, self.lgh_manifold_morton, self._lgh_curve_indices),
        )

    def _quantize_lgh_manifold(self, bits: int = 8, *args, **kwargs):
        if self.lgh_manifold_morton is None or self.lgh_manifold_morton.numel() == 0:
            return None, None
        qmax = 127.0 if int(bits) >= 8 else 7.0
        absmax = self.lgh_manifold_morton.abs().amax(dim=1).clamp_min(1e-8)
        scale = absmax / qmax
        q = torch.round(self.lgh_manifold_morton / scale.unsqueeze(-1)).clamp(-qmax, qmax).to(torch.int8).contiguous()
        return q, scale.contiguous()

    def _preflight_fast_path(self):
        x = torch.zeros(1, 4, 8, dtype=torch.float32, device=self.device)
        H = init_state(self.L, self.R, self.d_s2, self.C, device=self.device).unsqueeze(0)
        _ = self.forward(x, H)
        self._preflight_ready = True

    def get_thermal_penalty(self):
        return 0.0

    def _get_tps_pressure(self):
        return float(self.virtual_lab.get_benchmarks().get("tps_pressure", 0.0))

    def get_engagement_rate(self):
        return float(self.governor.usage.mean().item())

    def update_omega(self, train_loss: float, val_loss: float, force_delta: Optional[float] = None):
        if force_delta is not None:
            self.omega = max(0.0, min(1.0, float(self.omega) + float(force_delta)))
            return self.omega
        delta = max(-0.02, min(0.02, float(train_loss) - float(val_loss)))
        self.omega = max(0.0, min(1.0, float(self.omega) + delta))
        return self.omega

    def regulate_sensory_noise(self, val_loss: float):
        if float(val_loss) < 0.5:
            self.noise_scale = min(2.0, self.noise_scale + 0.05)
        else:
            self.noise_scale = max(0.1, self.noise_scale - 0.02)

    def mes_step(self, x, target_bits, precomputed_H_next=None, dry_run: bool = False):
        if dry_run:
            return {"mes_loss": 0.0, "hpc_loss": 0.0}
        x_bits = self._to_bits(x).contiguous().float()
        target_b = self._to_bits(target_bits).contiguous().float()
        B = x_bits.size(0)
        H_inter = self._prepare_state(precomputed_H_next, B).contiguous().float()

        p_lat = self.bit_to_latent(x_bits.reshape(-1, 8)).reshape(B, x_bits.size(1), self.d_s1 * self.C)
        t_lat = self.bit_to_latent(target_b.reshape(-1, 8)).reshape(B, target_b.size(1), self.d_s1 * self.C)
        p_brain = self.bridge_s1_to_s2(p_lat.reshape(-1, self.d_s1 * self.C)).reshape(B, x_bits.size(1), self.d_s2 * self.C).contiguous().float()
        t_brain = self.bridge_s1_to_s2(t_lat.reshape(-1, self.d_s1 * self.C)).reshape(B, target_b.size(1), self.d_s2 * self.C).contiguous().float()

        mes_scalars = torch.tensor(
            [
                float(getattr(Config, "LIF_DECAY", 0.95)),
                float(getattr(Config, "LIF_THRESHOLD", 0.5)),
                float(getattr(Config, "MES_LOCAL_L1", 0.0)),
            ],
            dtype=torch.float32,
            device="cpu",
        )
        out = _ACCEL.call(
            "mes_super_step_io",
            p_brain,
            H_inter,
            [t_brain, self._stacked_delays, self._stacked_tables, self._stacked_conns],
            mes_scalars,
            tensors=(p_brain, H_inter, t_brain, self._stacked_delays, self._stacked_tables, self._stacked_conns),
        )
        if not isinstance(out, (list, tuple)) or len(out) != 2:
            raise RuntimeError("mes_super_step_io returned unexpected output payload.")
        loss_local, grad_tables = out
        if not isinstance(loss_local, torch.Tensor):
            raise RuntimeError("mes_super_step_io did not return a tensor mes loss.")
        if not isinstance(grad_tables, torch.Tensor):
            raise RuntimeError("mes_super_step_io did not return a tensor table gradient.")

        mes_loss = float(loss_local.item())
        with torch.no_grad():
            if grad_tables.shape == self._stacked_tables.shape:
                step = float(getattr(Config, "MES_TABLE_LR", 1e-3))
                self._stacked_tables.add_(-step * grad_tables)
            else:
                raise RuntimeError(
                    "MES grad shape mismatch: "
                    f"grad_tables={tuple(grad_tables.shape)} expected={tuple(self._stacked_tables.shape)}"
                )
        self._params_dirty = True
        return {"mes_loss": mes_loss, "hpc_loss": 0.0}

    def _to_bits(self, x: torch.Tensor):
        if x.dim() == 2:
            return self._bytes_to_bits(x)
        if x.dim() == 3 and x.size(-1) == 8:
            return x.to(torch.float32)
        raise ValueError(f"Unsupported input shape for bits conversion: {tuple(x.shape)}")

    def forward(self, x, H=None, learning_brain=None, is_audit=False, return_all=False):
        x_bits = self._to_bits(x)
        B = x_bits.size(0)
        H_state = self._prepare_state(H, B)
        if self._params_dirty or self._fwd_params is None or self._fwd_scalars is None:
            self._pack_forward_params()
        scalars = self._fwd_scalars.clone()
        scalars[12] = float(self.current_phase)

        logits, H_next, cost, gate = _ACCEL.call(
            "unified_dispatch_io",
            x_bits.contiguous().float(),
            H_state.contiguous().float(),
            self._fwd_params,
            scalars,
            7,
            tensors=(x_bits, H_state),
        )
        self.step_counter.add_(int(x_bits.size(1)))
        if H_next.size(0) <= self.H_buffer.size(0):
            self.H_buffer[: H_next.size(0)].copy_(H_next)
        if return_all:
            return logits, H_next, cost, gate
        return logits, H_next, cost, gate


__all__ = [
    "_ACCEL",
    "cpp_loader",
    "Config",
    "DEVICE",
    "init_state",
    "VirtualLab",
    "Governor",
    "SwiGLU",
    "RotaryEmbedding",
    "VNNILinear",
    "CognitiveOrganism",
]
