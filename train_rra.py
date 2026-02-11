import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import time
import numpy as np
import psutil
from torch.utils.tensorboard import SummaryWriter
# from llama_cpp import Llama  # Removed for offline distillation
import traceback
import queue
import random

import logging
import datetime
import threading
import ctypes
import math
from types import SimpleNamespace

# --- JIT COMPILER CONFIG ---
if os.name == 'nt':
    try:
        import torch._inductor.config
        # Keep compile_threads=1 on Windows unless explicitly tuned by environment.
        # We'll set it to 1 by default on Windows to be safe, unless specifically overridden.
        torch._inductor.config.compile_threads = 1
        
        # Try to set cpp.openmp if it exists
        if hasattr(torch._inductor.config, 'cpp'):
            if hasattr(torch._inductor.config.cpp, 'openmp'):
                # If user installed OpenMP, we might want to set this to True.
                # But to avoid the 'omp.h' error if installation is partial, 
                # we'll keep it False by default and let user override via ENV if needed.
                torch._inductor.config.cpp.openmp = os.environ.get("USE_OPENMP", "False").lower() == "true"
    except Exception as e:
        pass

from organism import CognitiveOrganism, init_state, cpp_loader
from config import Config
from learning_brain import LearningBrain

# --- INTEGRATED OPTIMIZER SUITE ---
from optimizers import AdEMAMix, adaptive_gradient_clip

def create_optimizer(model, config):
    lr = getattr(model, 'suggested_lr', config.LEARNING_RATE)
    # Using consolidated AdEMAMix
    return AdEMAMix(
        model.parameters(), lr=lr,
        beta1_fast=getattr(config, 'ADEMAMIX_BETA1_FAST', 0.9),
        beta1_slow=getattr(config, 'ADEMAMIX_BETA1_SLOW', 0.9999),
        beta2=getattr(config, 'ADEMAMIX_BETA2', 0.999),
        weight_decay=getattr(config, 'WEIGHT_DECAY', 0.0)
    )

# --- LOGGING SETUP ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "training_debug.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- WINDOWS TERMINAL FIX ---
def disable_quickedit():
    """
    Disable QuickEdit mode in Windows Console to prevent training hangs 
    caused by clicking inside the terminal window.
    """
    if os.name == 'nt':
        try:
            kernel32 = ctypes.windll.kernel32
            # STD_INPUT_HANDLE = -10
            h = kernel32.GetStdHandle(-10)
            mode = ctypes.c_uint32()
            kernel32.GetConsoleMode(h, ctypes.byref(mode))
            # ENABLE_QUICK_EDIT_MODE = 0x0040, ENABLE_EXTENDED_FLAGS = 0x0080
            # To disable QuickEdit, we must disable 0x0040 and ensure 0x0080 is set
            mode.value &= ~0x0040
            mode.value |= 0x0080
            kernel32.SetConsoleMode(h, mode)
            logger.info(">>> Windows QuickEdit Mode Disabled")
        except Exception as e:
            logger.warning(f">>> Failed to disable QuickEdit: {e}")

def configure_runtime_threading():
    """
    Optional thread policy harmonization.
    Uses Config values; 0 means keep defaults.
    """
    runtime_cfg = SimpleNamespace(
        omp_threads=int(Config.CPP_OMP_THREADS),
        torch_threads=int(Config.TORCH_NUM_THREADS),
        interop_threads=int(Config.TORCH_INTEROP_THREADS),
    )
    omp_threads = runtime_cfg.omp_threads
    torch_threads = runtime_cfg.torch_threads
    interop_threads = runtime_cfg.interop_threads

    if omp_threads > 0:
        os.environ['OMP_NUM_THREADS'] = str(omp_threads)
        logger.info(f">>> Runtime Threading: OMP_NUM_THREADS={omp_threads}")

    if torch_threads > 0:
        try:
            torch.set_num_threads(torch_threads)
            logger.info(f">>> Runtime Threading: torch_num_threads={torch_threads}")
        except Exception as e:
            logger.warning(f">>> Failed to set torch_num_threads={torch_threads}: {e}")

    if interop_threads > 0:
        try:
            torch.set_num_interop_threads(interop_threads)
            logger.info(f">>> Runtime Threading: torch_interop_threads={interop_threads}")
        except Exception as e:
            logger.warning(f">>> Failed to set torch_interop_threads={interop_threads}: {e}")


def set_global_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f">>> Global Seed: {seed}")


def run_preflight_checks(model, device):
    """
    Lightweight startup smoke-check for deployment reliability.
    Runs one tiny forward pass and validates finite outputs.
    """
    if cpp_loader is None:
        raise RuntimeError("cpp_loader extension is required for this project.")
    required_ops = (
        'forward_stack_io', 'mes_super_step_io', 'survival_losses_io',
        'survival_mask_io', 'survival_update_io', 'quantized_matmul', 'ademamix_update'
    )
    missing = [op for op in required_ops if not hasattr(cpp_loader, op)]
    if missing:
        raise RuntimeError(f"Missing required cpp_loader ops: {', '.join(missing)}")

    if isinstance(device, torch.device) and device.type != 'cpu':
        raise RuntimeError(
            f"Unsupported runtime device '{device}'. Current production kernels require CPU tensors."
        )

    tiny_b, tiny_t = 2, 16
    xb = torch.randint(0, 2, (tiny_b, tiny_t, 8), device=device, dtype=torch.float32)
    H = init_state(model.L, model.R, model.d_s2, model.C, device=device)
    model.eval()
    with torch.no_grad():
        out, H_next, cost, gate = model(xb, H)

    tensors = {'out': out, 'H_next': H_next, 'cost': cost, 'gate': gate}
    bad = [name for name, t in tensors.items() if isinstance(t, torch.Tensor) and not torch.isfinite(t).all()]
    if bad:
        raise RuntimeError(f"Preflight produced non-finite tensors: {', '.join(bad)}")
    logger.info(">>> Preflight checks passed (C++ ops + finite forward).")

# --- WATCHDOG ---
class TrainingWatchdog(threading.Thread):
    def __init__(self, trainer, timeout_seconds=300):
        super().__init__(daemon=True)
        self.trainer = trainer
        self.timeout = timeout_seconds
        self.last_step = -1
        self.last_step_time = time.time()
        self.running = True

    def run(self):
        while self.running:
            current_step = self.trainer.global_step
            now = time.time()
            
            if current_step != self.last_step:
                self.last_step = current_step
                self.last_step_time = now
            elif now - self.last_step_time > self.timeout:
                logger.critical(f"\n\n>>> WATCHDOG WARNING: Training potentially STUCK at step {current_step}!")
                logger.critical(f">>> No progress for {int(now - self.last_step_time)}s. "
                                "Check if terminal is paused or LLM is unresponsive.")
                # Reset time to avoid flood, but keep warning
                self.last_step_time = now
            
            # Subtle visual pulse every 15 seconds during long steps
            if (int(now) % 15 == 0):
                # Using direct stdout write for a non-newline heartbeat
                import sys
                sys.stdout.write(".")
                sys.stdout.flush()
                
            time.sleep(5)

# --- SETTINGS ---
CHECKPOINT_DIR = "checkpoints"
DATA_PATH = "."
SEQ_LEN = Config.SEQ_LEN
BATCH_SIZE = Config.BATCH_SIZE
DEVICE = Config.DEVICE
# Optional model path configuration (kept env-driven for portability).
HF_GEMMA_PATH = os.environ.get("HF_GEMMA_PATH", "models/gemma-3-4b-it-q4_0.gguf")
GGUF_PATH = os.environ.get("GGUF_PATH", HF_GEMMA_PATH)

class ByteDataset(Dataset):
    _file_cache = {}
    _byte_bits_lut = None

    def __init__(
        self,
        file_path,
        seq_len=128,
        start_offset=0,
        end_offset=None,
        random_sampling=True,
        samples_per_epoch=None
    ):
        """
        Directly loads raw bytes from dataset.txt.
        Converts bytes to 8-bit vectors.
        """
        self.seq_len = seq_len
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file {file_path} not found.")
        
        if file_path not in ByteDataset._file_cache:
            with open(file_path, 'rb') as f:
                ByteDataset._file_cache[file_path] = f.read()
        self.data = ByteDataset._file_cache[file_path]
        
        self.total_bytes = len(self.data)
        self.start_offset = max(0, int(start_offset))
        self.end_offset = self.total_bytes if end_offset is None else min(self.total_bytes, int(end_offset))
        self.random_sampling = bool(random_sampling)
        self.window_count = self.end_offset - self.start_offset - self.seq_len
        if self.window_count <= 0:
            raise ValueError(
                f"Dataset span too small for seq_len={self.seq_len}. "
                f"Need at least seq_len+1 bytes in range [{self.start_offset}, {self.end_offset})."
            )
        default_samples = max(1, self.window_count // max(1, self.seq_len))
        if samples_per_epoch is None:
            self.sample_count = default_samples
        else:
            self.sample_count = max(1, min(int(samples_per_epoch), self.window_count))
        self.vocab_size = 256
        if ByteDataset._byte_bits_lut is None:
            values = torch.arange(256, dtype=torch.long)
            shifts = torch.arange(7, -1, -1, dtype=torch.long)
            ByteDataset._byte_bits_lut = ((values.unsqueeze(-1) >> shifts) & 1).to(torch.float32)
        self.byte_bits_lut = ByteDataset._byte_bits_lut
        logger.info(
            f">>> ByteDataset: Loaded {self.total_bytes} bytes from {file_path} "
            f"(range {self.start_offset}:{self.end_offset}, random={self.random_sampling}, "
            f"samples={self.sample_count})"
        )

    def __len__(self):
        return self.sample_count

    def __getitem__(self, idx):
        if self.random_sampling:
            start = random.randint(self.start_offset, self.end_offset - self.seq_len - 1)
        else:
            # Deterministic validation sampling to make metrics stable.
            stride = max(1, self.window_count // max(1, self.sample_count))
            start = self.start_offset + ((idx * stride) % self.window_count)
        chunk = self.data[start : start + self.seq_len + 1]
        
        b = torch.from_numpy(np.frombuffer(chunk, dtype=np.uint8).copy()).long()
        bits = self.byte_bits_lut[b]
        
        xb = bits[:-1] # [seq_len, 8]
        yb = bits[1:]  # [seq_len, 8]
        return xb, yb


def create_dataloaders(data_dir, batch_size, seq_len, val_split=0.1):
    data_cfg = SimpleNamespace(
        train_samples_per_epoch=Config.TRAIN_SAMPLES_PER_EPOCH,
        val_samples_per_epoch=Config.VAL_SAMPLES_PER_EPOCH,
    )
    # Check for direct text dataset first as requested
    text_file = os.path.join(data_dir, "dataset.txt")
    if os.path.exists(text_file):
        logger.info(f">>> Found direct text dataset: {text_file}")
        total_bytes = os.path.getsize(text_file)
        if total_bytes <= seq_len + 1:
            raise ValueError(
                f"dataset.txt too small ({total_bytes} bytes) for seq_len={seq_len}. "
                "Use a larger dataset or reduce seq_len."
            )
        min_non_overlap = 2 * (seq_len + 1)
        if total_bytes < min_non_overlap:
            raise ValueError(
                f"dataset.txt has {total_bytes} bytes, but non-overlapping train/val split requires at least "
                f"{min_non_overlap} bytes for seq_len={seq_len}."
            )

        val_split = float(max(0.01, min(0.5, val_split)))
        split_byte = int(total_bytes * (1.0 - val_split))
        split_byte = max(seq_len + 1, min(total_bytes - (seq_len + 1), split_byte))

        train_default_samples = max(1, (split_byte - seq_len) // max(1, seq_len))
        val_default_samples = max(1, (total_bytes - split_byte - seq_len) // max(1, seq_len))
        train_samples_cfg = data_cfg.train_samples_per_epoch
        val_samples_cfg = data_cfg.val_samples_per_epoch

        train_ds = ByteDataset(
            text_file,
            seq_len,
            start_offset=0,
            end_offset=split_byte,
            random_sampling=True,
            samples_per_epoch=train_default_samples if train_samples_cfg is None else int(train_samples_cfg)
        )
        val_ds = ByteDataset(
            text_file,
            seq_len,
            start_offset=split_byte,
            end_offset=total_bytes,
            random_sampling=False,
            samples_per_epoch=val_default_samples if val_samples_cfg is None else int(val_samples_cfg)
        )

        cpu_workers = os.cpu_count() or 1
        if os.name == 'nt':
            # dataset.txt is memory-backed and lightweight; worker process spawn on Windows
            # usually costs more than it saves and duplicates extension startup overhead.
            train_workers = 0
            val_workers = 0
        else:
            train_workers = min(4, max(0, cpu_workers - 1))
            val_workers = min(2, max(0, cpu_workers - 1))
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=train_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(train_workers > 0)
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=val_workers,
            persistent_workers=(val_workers > 0)
        )
        return train_dl, val_dl

    # Fallback to Distilled Data
    stream_file = os.path.join(data_dir, "distilled_data_stream.pt")
    if os.path.exists(stream_file):
        raise RuntimeError(
            f"Found '{stream_file}', but distilled stream loading is not implemented in this trainer. "
            "Place a 'dataset.txt' file in the data directory."
        )
    raise FileNotFoundError(
        f"No training source found in '{data_dir}'. Expected 'dataset.txt' or 'distilled_data_stream.pt'."
    )



class RRATrainer:
    def __init__(self, model, learning_brain=None, device='cpu', max_ram_gb=4.0):
        self.model = model
        self.learning_brain = learning_brain
        self.device = device
        self.max_ram_gb = max_ram_gb
        self.cfg = SimpleNamespace(
            dream_replay_batches=max(0, int(Config.DREAM_REPLAY_BATCHES)),
            batch_size=max(1, int(Config.BATCH_SIZE)),
            mes_enabled=bool(Config.MES_ENABLED),
            lambda_stability=float(Config.LAMBDA_STABILITY),
            lambda_energy=float(Config.LAMBDA_ENERGY),
            agc_clip_factor=float(Config.AGC_CLIP_FACTOR),
            coherence_weight=float(Config.COHERENCE_WEIGHT),
            dynamic_energy_scale=float(Config.DYNAMIC_ENERGY_SCALE),
            efficiency_bonus_cap=float(Config.EFFICIENCY_BONUS_CAP),
            reflex_dropout_rate=float(Config.REFLEX_DROPOUT_RATE),
            ram_init_scale=float(Config.RAM_INIT_SCALE),
            sleep_interval_steps=max(1, int(Config.SLEEP_INTERVAL_STEPS)),
            seq_len=int(Config.SEQ_LEN),
            ram_critical_threshold=float(Config.RAM_CRITICAL_THRESHOLD),
            ram_prune_fraction=float(Config.RAM_PRUNE_FRACTION),
        )
        
        # Use AdEMAMix optimizer with Omega-linked dual-momentum
        self.optimizer = create_optimizer(model, Config)
        initial_lr = self.optimizer.param_groups[0]['lr']
        optimizer_name = type(self.optimizer).__name__
        logger.info(f">>> {optimizer_name} Optimizer initialized with LR: {initial_lr:.2e} (from Genome BDNF)")
        
        self.metrics = {'hits': 0, 'misses': 0, 'imprints': 0}
        self.global_step = 0
        
        # Adaptive Curriculum Tracking
        self.streak_counter = 0
        self.best_loss = float('inf')
        self.current_epoch = 0
        self.start_batch_idx = 0
        
        # Time Tracking
        self.total_time = 0.0
        self.session_start = time.time()
        
        # Start Watchdog
        self.watchdog = TrainingWatchdog(self)
        self.watchdog.start()

        # Create Once, Reuse Forever: Persistent Hidden State Buffer
        self.register_h_buffer(self.cfg.batch_size)

    def register_h_buffer(self, bsz):
        """Pre-allocates the hidden state buffer to avoid redundant allocations."""
        self._h_buffer = init_state(
            self.model.L, self.model.R, self.model.d_s2, self.model.C, device=self.device
        ).unsqueeze(0).expand(bsz, -1, -1, -1, -1).contiguous()
        logger.info(f">>> RRATrainer: H-Buffer registered for batch_size={bsz}")

    def get_h_state(self, bsz):
        """Returns a correctly-sized H state from the buffer, re-allocating only if necessary."""
        if not hasattr(self, '_h_buffer') or self._h_buffer.size(0) < bsz:
            self.register_h_buffer(bsz)
        return self._h_buffer[:bsz]
    
    def _update_lr_from_genome(self):
        """Dynamically adjust learning rate based on genome expression."""
        new_lr = self.model.suggested_lr
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            param_group['lr'] = new_lr
            if abs(old_lr - new_lr) > 1e-6:
                logger.info(f">>> LR Updated: {old_lr:.2e} -> {new_lr:.2e} (Genome BDNF)")
        return new_lr

    def _optimizer_step(self):
        """Single optimizer step with optional omega-aware signature."""
        step_fn = getattr(self.optimizer, 'step', None)
        if step_fn is None:
            raise RuntimeError("Optimizer is missing a callable step() method.")
        if hasattr(step_fn, '__code__') and 'omega' in step_fn.__code__.co_varnames:
            step_fn(omega=self.model.omega)
        else:
            step_fn()

    def _cache(self):
        if not getattr(self.model, 'cache_enabled', False):
            return None
        return getattr(self.model, 'neural_cache', None)

    def _cache_write(self, keys, values, learning_rate=1.0, mask=None):
        cache = self._cache()
        if cache is None:
            return 0
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            if mask.numel() == 0 or (not bool(mask.any().item())):
                return 0
            keys = keys[mask]
            values = values[mask]
        if keys.numel() == 0 or values.numel() == 0:
            return 0
        cache.write(keys, values, learning_rate=float(learning_rate))
        return int(keys.size(0))

    def _tb_add_scalar(self, tag, value, step=None):
        if (not self.model.virtual_lab.enabled) or (self.model.virtual_lab.writer is None):
            return
        use_step = self.global_step if step is None else step
        self.model.virtual_lab.writer.add_scalar(tag, value, use_step)

    @staticmethod
    def _finite(x, nan=0.0, posinf=None, neginf=None):
        kwargs = {'nan': nan}
        if posinf is not None:
            kwargs['posinf'] = posinf
        if neginf is not None:
            kwargs['neginf'] = neginf
        return torch.nan_to_num(x, **kwargs)

    def _omega_state(self, omega=None):
        o = self.model.omega if omega is None else omega
        return "Sponge" if o < 0.3 else ("Student" if o < 0.7 else "Master")

    @staticmethod
    def _to_bits(x):
        return (x > 0.5).float()

    def _get_input(self, tokens):
        # Tokens are already bits [B, T, 8]
        device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device).split(':')[0]
        if tokens.device.type == device_type and tokens.dtype == torch.float32:
            return tokens
        return tokens.to(self.device, dtype=torch.float32)

    def _forward_with_state(self, xb, require_grad=True):
        H = self.get_h_state(xb.size(0))
        inp = self._get_input(xb)
        if require_grad:
            out, H_next, cost, gate = self.model(inp, H)
        else:
            with torch.no_grad():
                out, H_next, cost, gate = self.model(inp, H)
        return inp, H, out, H_next, cost, gate

    def _last_latent(self, inp_bits):
        return self.model.bit_to_latent(inp_bits[:, -1, :])

    def sanitize_model_parameters(self):
        """Replace non-finite model parameters with safe finite values."""
        fixed_count = 0
        with torch.no_grad():
            for _, param in self.model.named_parameters():
                if not torch.is_floating_point(param):
                    continue
                bad_mask = ~torch.isfinite(param)
                if bad_mask.any():
                    fixed_count += int(bad_mask.sum().item())
                    param.data = self._finite(param.data, nan=0.0, posinf=1.0, neginf=-1.0)
        if fixed_count > 0:
            logger.warning(f">>> Sanitized {fixed_count} non-finite parameter values in model state.")
        return fixed_count

    def _sanitize_gradients(self):
        """Replace non-finite gradients so optimizer steps stay numerically stable."""
        for param in self.model.parameters():
            if param.grad is None or not torch.is_floating_point(param.grad):
                continue
            if not torch.isfinite(param.grad).all():
                param.grad.data = self._finite(param.grad.data, nan=0.0, posinf=1.0, neginf=-1.0)

    def _step_phase_0_stability(self, xb, yb):
        self.optimizer.zero_grad()
        inp, H, out, H_next, cost, gate = self._forward_with_state(
            xb, require_grad=(not self.cfg.mes_enabled)
        )
        out = self._finite(out, nan=0.0, posinf=20.0, neginf=-20.0)
        H_next = self._finite(H_next, nan=0.0, posinf=1e4, neginf=-1e4)
        cost = self._finite(cost, nan=0.0, posinf=1e4, neginf=0.0)
        gate = self._finite(gate, nan=0.0, posinf=1.0, neginf=0.0)

        
        # Stability losses from SurvivalController
        l_stab, l_eng, l_coh = self.model.survival.calculate_losses(H_next, gate=gate, H_prev=H)
        l_stab = self._finite(l_stab, nan=0.0, posinf=1e4, neginf=0.0)
        l_eng = self._finite(l_eng, nan=0.0, posinf=1e4, neginf=0.0)
        
        logits = out
        targets = yb
        # Bit-BCE Loss via LearningBrain
        loss_task = self.learning_brain.calculate_task_loss(logits, targets)
        loss_task = self._finite(loss_task, nan=10.0, posinf=10.0, neginf=10.0)
        
        # Phase 0 only cares about Task + Stability
        total_loss = loss_task + (self.cfg.lambda_stability * l_stab) + (self.cfg.lambda_energy * l_eng)
        if not torch.isfinite(total_loss):
            logger.warning(f">>> Non-finite Phase-0 loss at global_step={self.global_step}. Skipping this batch.")
            self.optimizer.zero_grad(set_to_none=True)
            return None
        
        if self.cfg.mes_enabled:
            # Phase 0 MES: Use task targets directly for stability
            self.model.mes_step(xb, yb, precomputed_H_next=H_next)
        else:
            total_loss.backward()
            self._sanitize_gradients()
            # Apply AGC before optimizer step
            adaptive_gradient_clip(self.model, clip_factor=self.cfg.agc_clip_factor)
            self._optimizer_step()

        loss_value = float(total_loss.item())
        if not math.isfinite(loss_value):
            logger.warning(f">>> Non-finite Phase-0 scalar loss at global_step={self.global_step}. Skipping this batch.")
            return None
        return loss_value

    def _step_omega(self, xb, tb, dataloader, batch_idx=0, is_ground_truth=True):
        """
        Modified to handle both teacher distillation and direct ground-truth training.
        If tb contains probabilities, it distills. If it contains bits, it trains toward them.
        """
        B = xb.size(0)
        omega = self.model.omega
        self.optimizer.zero_grad()
        
        # 1. FORWARD PASS
        inp, H, out, H_next, cost, gate = self._forward_with_state(xb, require_grad=(not self.cfg.mes_enabled))
        out = self._finite(out, nan=0.0, posinf=20.0, neginf=-20.0)
        H_next = self._finite(H_next, nan=0.0, posinf=1e4, neginf=-1e4)
        cost = self._finite(cost, nan=0.0, posinf=1e4, neginf=0.0)
        gate = self._finite(gate, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 2. CALC TEACHER LOSS
        if is_ground_truth:
            yb = tb if tb.dtype == torch.float32 else tb.float()
            L_teacher = None # No teacher loss when training directly on ground-truth bits.
        else:
            yb = tb
            # Distillation Loss
            L_teacher = F.binary_cross_entropy_with_logits(out, tb)
        
        # 3. EXECUTE LEARNING STEP
        loss_info = {'L_task': float('nan'), 'L_teacher': float('nan')}
        if self.cfg.mes_enabled:
            # MES: Decentralized Descent
            mes_info = self.model.mes_step(xb, yb, precomputed_H_next=H_next)
            if self.model.virtual_lab.enabled:
                self._tb_add_scalar("MES/local_loss", mes_info.get('mes_loss', 0.0))
                self._tb_add_scalar("HPC/local_loss", mes_info.get('hpc_loss', 0.0))

            # Reuse current forward results to avoid a second full model forward in MES mode.
            with torch.no_grad():
                loss_task = self.learning_brain.calculate_task_loss(out, yb)
                l_stab, l_eng, l_coh = self.model.survival.calculate_losses(H_next, gate=gate, H_prev=H)
                l_myelin = self.model.survival.calculate_myelin_cost(self.model)
                l_teacher_eff = loss_task if L_teacher is None else L_teacher
                total_loss, loss_info = self.learning_brain.calculate_unified_loss(
                    L_task=loss_task,
                    L_teacher=l_teacher_eff,
                    L_stability=(l_stab + self.cfg.coherence_weight * l_coh + l_myelin),
                    omega=self.model.omega
                )
                total_loss = total_loss + (self.cfg.dynamic_energy_scale * (cost + l_eng))
                if hasattr(self.model, 'get_engagement_rate'):
                    engagement_rate = self.model.get_engagement_rate()
                    efficiency_bonus = min((1.0 - engagement_rate) * 0.1, self.cfg.efficiency_bonus_cap)
                    total_loss = total_loss - efficiency_bonus
            L_total_val = float(total_loss.item())
            loss_info['L_task'] = float(loss_info.get('L_task', float('nan')))
            loss_info['L_teacher'] = float(loss_info.get('L_teacher', float('nan')))
            if not math.isfinite(L_total_val):
                logger.warning(f">>> Non-finite Omega-mode loss at global_step={self.global_step}. Skipping this batch.")
                return None
        else:
            # Global Backprop: Centralized Orchestration
            # We use learning_step which handles backward() and optimizer.step()
            results = self.learning_brain.learning_step(
                self.model, inp, yb, H, self.optimizer, L_teacher=L_teacher, omega=self.model.omega
            )
            L_total_val = results['total_loss']
            H_next = results['H_next']
            loss_info['L_task'] = results.get('loss_task', float('nan'))
            if L_teacher is not None and hasattr(L_teacher, 'item'):
                loss_info['L_teacher'] = L_teacher.item()
            if not math.isfinite(L_total_val):
                logger.warning(f">>> Non-finite Omega-mode loss at global_step={self.global_step}. Skipping this batch.")
                return None
        
        # 6. OMEGA-MODULATED MEMORY
        with torch.no_grad():
            cache = self._cache()
            if cache is not None:
                # Project last bit vector to latent space for NeuralCache
                last_latent = self._last_latent(inp)
                if omega < 0.3:  # SPONGE: Aggressive write
                    self._cache_write(last_latent, out[:, -1, :], learning_rate=1.0)
                elif omega < 0.7:  # STUDENT: Balanced
                    self._cache_write(last_latent, out[:, -1, :], learning_rate=0.5)
                else:  # MASTER: Read-only (high confidence only)
                    probs = torch.sigmoid(out[:, -1, :]) # out is [B, 8] bits
                    # Bit-wise confidence (entropy check is harder for bits, using average confidence)
                    confidence = (probs - 0.5).abs().mean(dim=-1)
                    confident = confidence > 0.4 # High confidence threshold
                    self._cache_write(last_latent, out[:, -1, :], learning_rate=0.1, mask=confident)
        
        # 6. LOGGING
        if self.model.virtual_lab.enabled and batch_idx % 50 == 0:
            self._tb_add_scalar("Omega/value", omega)
            self._tb_add_scalar("Omega/L_task", loss_info['L_task'])
            self._tb_add_scalar("Omega/L_teacher", loss_info['L_teacher'])
        
        return float(L_total_val)

    def _step_phase_1_rra(self, xb, yb, batch_idx=0):
        B = xb.size(0)
        self.optimizer.zero_grad()
        
        reflex_dropout = torch.rand(1).item() < self.cfg.reflex_dropout_rate
        hit_mask = torch.zeros(B, dtype=torch.bool, device=self.device)
        
        cache = self._cache()
        if (not reflex_dropout and cache is not None):
            with torch.no_grad():
                # Cache lookup only needs the latest token context.
                xb_in = self._get_input(xb)
                last_latent = self.model.bit_to_latent(xb_in[:, -1, :])
                reflex_out, hit_mask = cache.lookup(last_latent)
            self.metrics['hits'] += hit_mask.sum().item()
            self.metrics['misses'] += (~hit_mask).sum().item()
        else:
            self.metrics['misses'] += B
 
        miss_indices = (~hit_mask).nonzero(as_tuple=False).flatten()
        if miss_indices.numel() == 0:
            return 0.0 
        
        xb_miss = xb[miss_indices]
        yb_miss = yb[miss_indices]
        
        inp_miss_bits, _, out_miss, H_next_miss, _, _ = self._forward_with_state(
            xb_miss, require_grad=(not self.cfg.mes_enabled)
        )
        logits = out_miss[:, -1, :]
        targets = yb_miss[:, -1] # [B, 8]
        loss_task = self.learning_brain.calculate_task_loss(logits, targets)
        
        if self.cfg.mes_enabled:
            self.model.mes_step(xb_miss, yb_miss, precomputed_H_next=H_next_miss)
        else:
            if loss_task.requires_grad:
                loss_task.backward()
                # Apply AGC before optimizer step
                adaptive_gradient_clip(self.model, clip_factor=self.cfg.agc_clip_factor)
                self._optimizer_step()
        
        with torch.no_grad():
            # Imprint using latent context
            last_latent_miss = self._last_latent(inp_miss_bits)
            self._imprint_knowledge(logits, targets, last_latent_miss, batch_idx=batch_idx, epoch_idx=self.current_epoch)

        return float(loss_task.item())

    def _imprint_knowledge(self, logits, targets, context_embeddings, batch_idx=0, epoch_idx=0):
        # Dynamic confidence threshold: stricter as training matures.
        dynamic_threshold = max(0.20, 0.45 - (epoch_idx * 0.01))
        
        probs = torch.sigmoid(logits)
        # Confidence logic for bits: how far from 0.5 are we on average?
        confidence = (probs - 0.5).abs().mean(dim=-1)
        confident_mask = confidence > dynamic_threshold
        preds = (probs > 0.5).float()
        correct_mask = (preds == targets).all(dim=-1)
        final_mask = confident_mask & correct_mask
        
        if batch_idx % 100 == 0:
            logger.info(
                f"Imprinting Debug [B:{batch_idx}]: correct={correct_mask.sum().item()}, "
                f"avg_conf={confidence.mean().item():.3f}, threshold={dynamic_threshold:.2f}"
            )
            
        if final_mask.any():
            # NeuralCache.write now handles normalization internally
            imprinted = self._cache_write(context_embeddings, logits, learning_rate=1.0, mask=final_mask)
            self.metrics['imprints'] += imprinted

    def _enter_sleep_cycle(self, dataloader, force=False):
        """
        SLEEP CONSOLIDATION (Dream Replay):
        Combats catastrophic forgetting by:
        1. Replaying bounded batches from the training distribution
        2. Pruning the NeuralCache to clear noise
        
        Triggered every 1000 steps or when confusion is very high.
        """
        logger.info(">>> ENTERING SLEEP CYCLE (Dream Consolidation)...")
        self.model.eval()
        
        if not force and self.model.omega < 0.1:
            logger.info(">>> Skipping Sleep Cycle (Brain too young/plastic)")
            return

        # 1. BRAIN SURGERY: active rewiring before replay.
        with torch.no_grad():
            self._perform_brain_surgery()

        # 2. DREAM REPLAY: bounded replay from valid train batches.
        dream_count = 0
        max_dream_batches = self.cfg.dream_replay_batches
        dataset = getattr(dataloader, 'dataset', None) if dataloader is not None else None
        if dataset is not None and max_dream_batches > 0:
            self.model.train()
            replay_batch = int(min(self.cfg.batch_size, 2))
            with torch.no_grad():
                for _ in range(max_dream_batches):
                    sample_indices = [random.randint(0, len(dataset) - 1) for _ in range(replay_batch)]
                    xb_list, _ = zip(*(dataset[i] for i in sample_indices))
                    xb = torch.stack(xb_list, dim=0).to(self.device)
                    inp, _, out, _, _, _ = self._forward_with_state(xb)
                    last_latent = self._last_latent(inp)
                    self._cache_write(last_latent, out[:, -1, :], learning_rate=0.05)
                    dream_count += 1
            self.model.eval()
        logger.info(f">>> Dream replay complete: {dream_count} batches")

        # 3. CACHE PRUNING: Remove bottom 30% of stale entries
        cache = self._cache()
        if cache is not None and hasattr(cache, 'prune_lru'):
            cache.prune_lru(fraction=0.3)
            logger.info(">>> Cache pruned (bottom 30% removed)")
        
        logger.info(">>> WAKING UP (Memories Consolidated)")
        self.model.train()

    def _perform_brain_surgery(self):
        """
        STRUCTURAL PLASTICITY: Rewire useless connections to useful inputs.
        Identifies "dead" RAM tables (low variance) and moves their connections
        to look at inputs that "hot" RAM tables are watching.
        """
        logger.info(">>> BRAIN SURGERY: Rewiring dead connections...")
        rewired_count = 0
        
        # Iterating through all VNN/RAM layers
        for layer in self.model.modules():
            # Check if it looks like a RAMTupleLayer
            if hasattr(layer, 'ram_tables') and hasattr(layer, 'connections'):
                wsnn = layer
                # wsnn.connections is [M, K] indices pointing to inputs
                # wsnn.ram_tables is [M, 2^K, D]
                
                with torch.no_grad():
                    # 1. Identify "Dead" RAM Rows (Low variance/usage)
                    # Variance across address/features tells us if the table stores diverse info.
                    ram_tables = wsnn.ram_tables
                    reduce_dims = tuple(range(1, ram_tables.dim()))
                    table_vars = ram_tables.var(dim=reduce_dims) if reduce_dims else ram_tables.var()
                    
                    # Find bottom 5% of useless RAM tables
                    threshold = torch.quantile(table_vars, 0.05)
                    dead_indices = (table_vars < threshold).nonzero(as_tuple=False).flatten()
                    
                    if dead_indices.numel() > 0:
                        # 2. Find "Hot" Input Indices to rewire TO
                        top_threshold = torch.quantile(table_vars, 0.8)
                        good_indices = (table_vars > top_threshold).nonzero(as_tuple=False).flatten()
                        
                        if good_indices.numel() > 0:
                            # Get the input indices that the GOOD tables are looking at
                            good_wiring = wsnn.connections[good_indices].flatten() # [Many]
                            
                            # 3. REWIRE
                            # For every dead row, pick K random connections from the "good wiring" pool
                            K = wsnn.connections.shape[1]
                            num_dead = dead_indices.numel()
                                
                            # Sample new connections from good_wiring
                            indices = torch.randint(0, good_wiring.size(0), (num_dead, K), device=self.device)
                            new_connections = good_wiring[indices]
                            wsnn.connections[dead_indices] = new_connections
                            
                            # 4. RESET weights (Fresh start)
                            # Re-initialize to small random values
                            wsnn.ram_tables.data[dead_indices] = (
                                torch.randn_like(wsnn.ram_tables.data[dead_indices]) * self.cfg.ram_init_scale
                            )
                            
                            rewired_count += int(num_dead)

        if rewired_count > 0:
            logger.info(f">>> Rewired {rewired_count} dead RAM tables to active input pathways.")
        else:
            logger.info(">>> No dead neurons found (Intelligence Density optimal).")

    def train_epoch(self, dataloader, epoch_idx, max_steps=None):
        self.model.train()
        total_loss = 0.0
        processed_batches = 0
        skipped_batches = 0
        steps_this_call = 0
        reached_step_limit = False
        last_batch_idx = self.start_batch_idx - 1
        self.metrics = {'hits': 0, 'misses': 0, 'imprints': 0}
        self.sanitize_model_parameters()
        self._update_lr_from_genome()

        for batch_idx, (xb, tb) in enumerate(dataloader):
            if batch_idx < self.start_batch_idx:
                continue
            if max_steps is not None and steps_this_call >= max_steps:
                reached_step_limit = True
                break
            
            xb = xb.to(self.device)
            tb = tb.to(self.device)
            steps_this_call += 1
            last_batch_idx = batch_idx
            
            # Ground truth yb for task loss (reconstructed from tb if no separate labels)
            # Actually, yb in _step_omega was bit-level targets.
            # If tb is the teacher's prediction for the NEXT bit vector, it serves as the target.
            # But the student also has a task loss against actual text.
            # Reconstructing byte-level target from bit-level prediction (hard target):
            yb = self._to_bits(tb)
            
            # --- FIX: Reset memory states for new shuffled batch ---
            if hasattr(self.model, 'titans_memory'):
                self.model.titans_memory.reset_memory()
            # -------------------------------------------------------
            
            # --- SMART PHASE TRANSITIONS ---
            self.global_step = epoch_idx * len(dataloader) + batch_idx
            step = self.global_step
            
            if self.model.current_phase == 0:
                loss = self._step_phase_0_stability(xb, yb)
                # Auto-switch to Omega mode after 100 TOTAL stability warmup steps
                if self.global_step >= 100 and self.model.omega == 0.0:
                    logger.info(">>> OMEGA MODE ACTIVATED: Switching to Unified Training")
                    # Gradual ramp-up: Start at 0.01, let it evolve via self.model.update_omega()
                    self.model.omega = 0.01
                    self.model.current_phase = 1

            else:
                # OMEGA MODE: Unified training equation (Phase 1+)
                loss = self._step_omega(xb, tb, dataloader, batch_idx=batch_idx, is_ground_truth=True)
            if loss is None or not math.isfinite(loss):
                skipped_batches += 1
                continue
                
            total_loss += loss
            processed_batches += 1
            if batch_idx % 50 == 0: self._check_ram_health()
            
            # Periodic TensorBoard Logging for Hit Rate
            if self.model.virtual_lab.enabled and batch_idx % 10 == 0:
                self._tb_add_scalar("VirtualLab/hit_rate", self._get_hit_rate())
                self._tb_add_scalar("VirtualLab/loss_total", loss)

                # --- Monitor Biological Variables ---
                self._tb_add_scalar("Genome/BDNF", self.model.genome.bdnf)
                self._tb_add_scalar("Genome/CREB", self.model.genome.creb)
                self._tb_add_scalar("Genome/DRD2", self.model.genome.drd2)
                self._tb_add_scalar("Genome/FKBP5", self.model.genome.fkbp5)

            # Adaptive Logging: Log at least 5 times per epoch, or every step if tiny
            log_freq = max(1, len(dataloader) // 5)
            if batch_idx % log_freq == 0:
                elapsed = self.total_time + (time.time() - self.session_start)
                elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
                omega_state = self._omega_state(self.model.omega)
                logger.info(f"Epoch {epoch_idx} | Step {batch_idx} | "
                            f"Omega={self.model.omega:.2f} ({omega_state}) | Loss: {loss:.4f} | "
                            f"Hit Rate: {self._get_hit_rate():.1%} | Time: {elapsed_str}")
            
            # Periodic Auto-Save (Every 100 steps)
            if batch_idx > 0 and batch_idx % 100 == 0:
                print(f"\n>>> Auto-Saving Checkpoint (Step {step})...")
                self.save_checkpoint("checkpoints/latest.pt", batch_idx=batch_idx)
            
            # --- SLEEP CONSOLIDATION: Triggered every N global steps ---
            # Combats catastrophic forgetting via dream replay and cache pruning
            # User Manual: Force sleep after 200 steps to stabilize Master state
            if step == 200:
                self._enter_sleep_cycle(dataloader, force=True)
            elif step > 0 and step % self.cfg.sleep_interval_steps == 0 and self.model.omega > 0:
                self._enter_sleep_cycle(dataloader)

        # Resume from the next batch if we intentionally stopped early.
        if reached_step_limit:
            self.start_batch_idx = last_batch_idx + 1
        else:
            self.start_batch_idx = 0
        if skipped_batches > 0:
            logger.warning(f">>> Skipped {skipped_batches} batches due to non-finite losses in epoch {epoch_idx}.")
        
        # Calculate epoch loss
        epoch_loss = total_loss / max(1, processed_batches)
        
        # --- Autonomous Curriculum & Evolution Logic ---
        # 3. Meta-Evolutionary Loop (Moved to validate)
        # -----------------------------------------------

        return epoch_loss, steps_this_call, reached_step_limit

    def validate(self, dataloader, epoch_loss):
        # --- VALIDATION LOSS: Detect Memorization ---
        val_loss = 0.0
        val_count = 0
        self.model.eval()
        with torch.no_grad():
            for val_idx, (xb, tb) in enumerate(dataloader):
                xb, tb = xb.to(self.device), tb.to(self.device)
                yb = tb
                inp, _, out, _, _, _ = self._forward_with_state(xb)
                # Bit-BCE for validation
                val_loss += self.learning_brain.calculate_task_loss(out, yb).item()
                val_count += 1
        
        self.model.train()
        val_loss = val_loss / max(1, val_count) if val_count > 0 else epoch_loss
        
        # Log validation loss to VirtualLab
        if self.model.virtual_lab.enabled:
            self.model.virtual_lab.log_step({
                'val_loss': val_loss,
                'loss_task': torch.tensor(epoch_loss),
                't': self.cfg.seq_len,
            })
            self._tb_add_scalar("VirtualLab/val_loss", val_loss)
            self._tb_add_scalar("VirtualLab/generalization_gap", val_loss - epoch_loss)
            
            # Log warning if memorizing
            if val_loss - epoch_loss > 0.5:
                logger.warning(f"WARNING MEMORIZATION DETECTED: train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}, gap={val_loss-epoch_loss:.4f}")
        
        # --- OMEGA EVOLUTION: Based on Generalization Gap ---
        # REMOVED if self.model.omega > 0 check to allow immediate evolution
        old_omega = self.model.omega
        new_omega = self.model.update_omega(epoch_loss, val_loss)
        omega_state = self._omega_state(new_omega)
        logger.info(f">>> OMEGA EVOLUTION: {old_omega:.3f} -> {new_omega:.3f} ({omega_state})")
        
        if self.model.virtual_lab.enabled:
             self._tb_add_scalar("Omega/epoch_value", new_omega, step=self.current_epoch)

        # Meta-Evolutionary Loop
        self.model.genome.evolutionary_step(self.model, {'val_loss': val_loss})
        
        # --- ANTI-CHEATING: Dynamic Sensory Noise ---
        # Adjust input noise based on validation performance
        if hasattr(self.model, 'regulate_sensory_noise'):
             self.model.regulate_sensory_noise(val_loss)

        self._update_lr_from_genome()
             
        return val_loss
        # -----------------------------------------------

    def _check_ram_health(self):
        cache = self._cache()
        if cache is None:
            return
        usage_gb = cache.memory_usage_gb()
        if usage_gb > (self.max_ram_gb * self.cfg.ram_critical_threshold):
            cache.prune_lru(fraction=self.cfg.ram_prune_fraction)

    def _get_hit_rate(self):
        total = self.metrics['hits'] + self.metrics['misses']
        return self.metrics['hits'] / total if total > 0 else 0.0

    def save_checkpoint(self, path="checkpoints/latest.pt", batch_idx=0):
        # safe_save using temp file + rename to avoid corruption
        temp_path = path + ".tmp"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'current_phase': self.model.current_phase,
                'omega': self.model.omega, # FIX: Save Omega state
                'global_step': self.global_step,
                'epoch': self.current_epoch,
                'batch_idx': batch_idx,
                'total_time': self.total_time + (time.time() - self.session_start),
                'rng_state': torch.get_rng_state()
            }, temp_path)
            
            # Atomic rename if possible
            if os.path.exists(path):
                os.replace(temp_path, path)
            else:
                os.rename(temp_path, path)
            
            # Run cleanup after successful save
            self.cleanup_old_checkpoints(os.path.dirname(path))
                
        except Exception as e:
            logger.error(f"FAILED TO SAVE CHECKPOINT to {path}: {e}")
            # Try emergency backup to current directory
            try:
                emergency_path = f"emergency_ckpt_step_{self.global_step}.pt"
                torch.save(self.model.state_dict(), emergency_path)
                logger.warning(f"Saved emergency checkpoint to {emergency_path}")
            except:
                logger.error("Emergency save also failed. Disk likely full.")

    def cleanup_old_checkpoints(self, checkpoint_dir, keep_best=True):
        """
        Deletes all epoch-specific checkpoints except the most recent one.
        Always keeps 'best_model.pt', 'genome_best.pt', and 'latest.pt'.
        """
        try:
            files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            protected_files = {'latest.pt'}
            if keep_best:
                protected_files.update({'best_model.pt', 'genome_best.pt'})
            
            # Identify epoch files: rra_epoch_X.pt
            epoch_files = []
            for f in files:
                if f in protected_files:
                    continue
                if f.startswith('rra_epoch_'):
                    try:
                        epoch_num = int(f.split('_')[-1].replace('.pt', ''))
                        epoch_files.append((epoch_num, f))
                    except ValueError:
                        pass
            
            # Sort by epoch number (ascending)
            epoch_files.sort(key=lambda x: x[0])
            
            # Determine which files to delete (keep only the last one)
            if len(epoch_files) > 1:
                files_to_delete = epoch_files[:-1] # All except the last one
                
                for _, fname in files_to_delete:
                    full_path = os.path.join(checkpoint_dir, fname)
                    try:
                        os.remove(full_path)
                        logger.info(f"Cleanup: Deleted old checkpoint {fname}")
                    except OSError as e:
                        logger.warning(f"Cleanup: Failed to delete {fname}: {e}")
                        
        except Exception as e:
            logger.warning(f"Checkpoint cleanup failed: {e}")

    def load_checkpoint(self, path):
        if not os.path.exists(path): return False
        try:
            ckpt = torch.load(path, map_location=self.device)
            
            # Sanitization: Remove quantization keys (w_q, scale_w) if they exist
            # These are transient buffers that cause "Unexpected key" errors if model is fresh
            state_dict = ckpt['model_state_dict']
            keys_to_remove = [k for k in state_dict.keys() if 'w_q' in k or 'scale_w' in k]
            if keys_to_remove:
                logger.info(f"Checkpoint Sanitizer: Removing {len(keys_to_remove)} quantization keys (w_q/scale_w)")
                for k in keys_to_remove:
                    del state_dict[k]
            
            self.model.load_state_dict(state_dict, strict=False) # tolerate missing legacy keys
            
            # Try to restore optimizer state - may fail if model architecture changed
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except ValueError as opt_err:
                logger.warning(f">>> Optimizer state incompatible (architecture changed): {opt_err}")
                logger.warning(">>> Continuing with fresh optimizer state. Previous momentum will be lost.")
            self.model.current_phase = ckpt.get('current_phase', 0)
            self.model.omega = ckpt.get('omega', 0.0) # FIX: Load Omega state
            self.global_step = ckpt.get('global_step', 0)
            self.current_epoch = ckpt.get('epoch', 0)
            self.start_batch_idx = ckpt.get('batch_idx', 0)
            self.total_time = ckpt.get('total_time', 0.0)
            self.session_start = time.time()
            if 'rng_state' in ckpt:
                torch.set_rng_state(ckpt['rng_state'].cpu())
            return True
        except RuntimeError as e:
            print(f"\n>>> WARNING: Checkpoint Incompatible (Architecture Changed). Error: {e}")
            print(">>> Starting with fresh weights for the new architecture.")
            return False

def parse_args():
    parser = argparse.ArgumentParser(description="Train the RRA model.")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of training batches to run in this invocation."
    )
    parser.add_argument(
        "--fresh_start",
        action="store_true",
        help="Ignore existing checkpoints and start from freshly initialized model weights."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=Config.SEED,
        help="Global random seed."
    )
    parser.add_argument(
        "--preflight_only",
        action="store_true",
        help="Run startup preflight checks and exit."
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning(f">>> Ignoring unknown CLI arguments: {unknown}")
    if args.max_steps is not None and args.max_steps <= 0:
        logger.warning(">>> --max_steps must be > 0. Ignoring provided value.")
        args.max_steps = None
    return args


def main():
    args = parse_args()
    disable_quickedit()
    configure_runtime_threading()
    set_global_seed(args.seed)
    logger.info(">>> INITIALIZING UNIFIED RRA PRODUCTION PIPELINE")
    logger.info(f">>> Device: {DEVICE} | Batch Size: {Config.BATCH_SIZE} | Seq Len: {Config.SEQ_LEN}")
    if args.max_steps is not None:
        logger.info(f">>> Run constraint active: max_steps={args.max_steps}")
    if Config.STRICT_CPU_ONLY and isinstance(DEVICE, torch.device) and DEVICE.type != 'cpu':
        raise RuntimeError(
            f"strict_cpu_only is enabled but resolved device is '{DEVICE}'. Check conf/config.yaml runtime.device."
        )
    
    # 1. Dataset Initialization
    train_loader, val_loader = create_dataloaders(DATA_PATH, Config.BATCH_SIZE, Config.SEQ_LEN)
    logger.info(f">>> Loaded dataset from {DATA_PATH} | Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")
    
    # 2. Model Initialization
    # input_dim: Bytes (dataset.vocab_size) -> d_s1
    model = CognitiveOrganism(
        input_dim=Config.D_S1 * Config.C, 
        L=Config.L, 
        R=Config.R, 
        d_s1=Config.D_S1, 
        d_s2=Config.D_S2,
        vocab_size=train_loader.dataset.vocab_size,
        output_dim=train_loader.dataset.vocab_size,
        device=DEVICE
    )
    
    # 3. High-Impact Optimizations
    # torch.compile DISABLED: Avoiding C++/OMP overhead on Windows.
    # The current focus is on stability and foundation learning in Phase 0.
    logger.info(">>> JIT COMPILER DISABLED (Eager Mode active for stability)")

    # Optional VirtualLab benchmarking/logging
    if Config.VIRTUAL_LAB_ENABLED:
        model.virtual_lab.enable()
        logger.info(">>> VirtualLab Initialized. Logging to logs/virtual_lab")
    else:
        model.virtual_lab.disable()
        logger.info(">>> VirtualLab disabled (production profile).")
    
    # --- CRITICAL FIX: Initialize LearningBrain ---
    learning_brain = LearningBrain(
        L=Config.L, R=Config.R, D=Config.D_S2, C=Config.C, device=DEVICE
    )
    
    trainer = RRATrainer(model, learning_brain=learning_brain, device=DEVICE)
    if Config.PREFLIGHT_ENABLED or args.preflight_only:
        run_preflight_checks(model, DEVICE)
        if args.preflight_only:
            logger.info(">>> --preflight_only completed successfully. Exiting.")
            return
    
    # 3. Load Checkpoint
    latest = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if args.fresh_start:
        logger.info(">>> Fresh start requested. Skipping checkpoint load.")
    elif trainer.load_checkpoint(latest):
        logger.info(f">>> Resumed from Phase {model.current_phase}")
    else:
        logger.info(">>> No checkpoint found. Starting from Phase 0 (Stability).")
    trainer.sanitize_model_parameters()

    # 4. Training Loop
    start_epoch = trainer.current_epoch
    steps_remaining = args.max_steps
    try:
        for epoch in range(start_epoch, Config.EPOCHS):
            trainer.current_epoch = epoch
            logger.info(f"\n>>> Starting Epoch {epoch}")
            avg_loss, steps_done, hit_step_limit = trainer.train_epoch(
                train_loader,
                epoch,
                max_steps=steps_remaining
            )
            if steps_remaining is not None:
                steps_remaining -= steps_done
            if hit_step_limit or (steps_remaining is not None and steps_remaining <= 0):
                logger.info(">>> Reached --max_steps limit. Saving checkpoint and exiting without full validation.")
                trainer.current_epoch = epoch
                trainer.save_checkpoint(latest, batch_idx=trainer.start_batch_idx)
                break

            val_loss = trainer.validate(val_loader, avg_loss)
            logger.info(f">>> Epoch {epoch} Complete | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # End of epoch: next resume starts at next epoch
            trainer.current_epoch = epoch + 1
            trainer.start_batch_idx = 0
            
            # Periodic Checkpoints
            trainer.save_checkpoint(os.path.join(CHECKPOINT_DIR, f"rra_epoch_{epoch}.pt"), batch_idx=0)
            trainer.save_checkpoint(latest, batch_idx=0)
            
            # Output High-Resolution Benchmarks
            benchmarks = model.virtual_lab.get_benchmarks()
            logger.info(f"\n>>> Epoch {epoch} Benchmarks: {benchmarks}")
    except KeyboardInterrupt:
        logger.info("\n\n>>> SAFE QUIT DETECTED (Ctrl+C). Saving and Exiting...")
        trainer.save_checkpoint(latest, batch_idx=trainer.start_batch_idx)
        logger.info(">>> Checkpoint Saved. Goodbye.")
    finally:
        if hasattr(trainer, 'watchdog'):
            trainer.watchdog.running = False
            trainer.watchdog.join(timeout=1.0)
        if hasattr(model, 'virtual_lab') and getattr(model.virtual_lab, 'writer', None) is not None:
            model.virtual_lab.writer.flush()
            model.virtual_lab.writer.close()
        
    logger.info("\n>>> Training Cycle Complete.")

if __name__ == "__main__":
    main()
