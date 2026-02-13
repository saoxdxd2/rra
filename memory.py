import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Optional, Tuple
from accelerator import get_accelerator

ACCEL = get_accelerator()
cpp_opt = ACCEL.loader
APEX_AVAILABLE = cpp_opt is not None
APEX_HAS_CACHE_LOOKUP = ACCEL.has('neural_cache_lookup_fast')
if APEX_AVAILABLE:
    print(">>> Apex CPU Optimizations (AVX2/FMA) ACTIVE")
else:
    print(">>> Apex CPU Optimizations not found. NeuralCache is unavailable (fail-fast mode).")

class LSHReflex(nn.Module):
    def __init__(self, input_dim, hash_bits=16, num_tables=4, device='cpu'):
        super().__init__()
        self.hash_bits = hash_bits
        self.num_tables = num_tables
        dev = torch.device(device)
        self.register_buffer(
            'planes',
            (torch.randn(num_tables, input_dim, hash_bits, device=dev) / (input_dim ** 0.5)).contiguous()
        )
        self.register_buffer(
            'bit_powers',
            (2 ** torch.arange(hash_bits, dtype=torch.long, device=dev)).view(1, 1, -1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects input into hash space using precomputed planes."""
        # Create Once, Reuse Forever: Planes and bit_powers are buffers
        projection = torch.einsum('bd,tdh->bth', x, self.planes)
        bits = (projection > 0).to(dtype=torch.long)
        return (bits * self.bit_powers).sum(dim=-1)

class NeuralCache(nn.Module):
    def __init__(self, input_dim, output_dim, hash_bits=18, num_tables=4, device='cpu'):
        super().__init__()
        if not APEX_HAS_CACHE_LOOKUP:
            raise RuntimeError("NeuralCache requires C++ op 'neural_cache_lookup_fast' (no Python fallback).")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_tables = num_tables
        self.ram_size = 2 ** hash_bits
        self.device = torch.device(device)
        
        # Aligned memory layout for SIMD + BF16 Compression
        self.register_buffer('keys', torch.zeros(num_tables, self.ram_size, input_dim, dtype=torch.bfloat16, device=self.device).contiguous())
        self.register_buffer('values', torch.zeros(num_tables, self.ram_size, output_dim, dtype=torch.bfloat16, device=self.device).contiguous())
        self.register_buffer('valid', torch.zeros(num_tables, self.ram_size, dtype=torch.bool, device=self.device).contiguous())
        self.register_buffer('last_access', torch.zeros(num_tables, self.ram_size, device=self.device).contiguous())
        self.register_buffer('hit_frequency', torch.zeros(num_tables, self.ram_size, dtype=torch.int32, device=self.device).contiguous())
        self.register_buffer('reliability', torch.ones(num_tables, self.ram_size, dtype=torch.float32, device=self.device).contiguous())
        self.register_buffer('table_indices', torch.arange(num_tables, dtype=torch.long, device=self.device).view(1, -1))
        self.register_buffer('batch_index_cache', torch.empty(0, dtype=torch.long, device=self.device))
        
        # Ensure memory is shared across multiprocess workers
        if self.device.type == 'cpu':
            self.keys.share_memory_()
            self.values.share_memory_()
            self.valid.share_memory_()
        
        self.hasher = LSHReflex(input_dim, hash_bits, num_tables, device=self.device)
        self.hit_count = 0
        self.miss_count = 0
        self.imprint_count = 0
        self.write_min_confidence = 0.15
        self.write_min_surprise = 0.20
        self.max_write_fraction = 0.50
        self.low_value_decay = 0.995
        self.low_value_min_reliability = 0.10
        self.low_value_decay_every = 32
        self._writes_since_decay = 0

    def _coerce_score(self, score: Optional[torch.Tensor], batch_size: int, name: str) -> Optional[torch.Tensor]:
        if score is None:
            return None
        if not isinstance(score, torch.Tensor):
            score = torch.as_tensor(score, dtype=torch.float32, device=self.device)
        score = score.to(self.device, dtype=torch.float32).flatten()
        if score.numel() != batch_size:
            raise ValueError(f"{name} must have batch size {batch_size}, got {score.numel()}.")
        return torch.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def _build_write_mask(
        self,
        batch_size: int,
        mask: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        surprise: Optional[torch.Tensor] = None,
        min_confidence: Optional[float] = None,
        min_surprise: Optional[float] = None,
        max_write_fraction: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        write_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            mask = mask.to(self.device, dtype=torch.bool).flatten()
            if mask.numel() != batch_size:
                raise ValueError(f"mask must have batch size {batch_size}, got {mask.numel()}.")
            write_mask &= mask

        confidence_s = self._coerce_score(confidence, batch_size, "confidence")
        surprise_s = self._coerce_score(surprise, batch_size, "surprise")
        if confidence_s is not None or surprise_s is not None:
            quality_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            if confidence_s is not None:
                conf_thr = self.write_min_confidence if min_confidence is None else float(min_confidence)
                quality_mask |= confidence_s >= conf_thr
            if surprise_s is not None:
                surprise_thr = self.write_min_surprise if min_surprise is None else float(min_surprise)
                quality_mask |= surprise_s >= surprise_thr
            write_mask &= quality_mask

        if not bool(write_mask.any().item()):
            return write_mask, confidence_s, surprise_s

        keep_fraction = self.max_write_fraction if max_write_fraction is None else float(max_write_fraction)
        keep_fraction = max(0.0, min(1.0, keep_fraction))
        if keep_fraction < 1.0:
            selected = write_mask.nonzero(as_tuple=False).squeeze(-1)
            keep_count = max(1, int(round(batch_size * keep_fraction)))
            if selected.numel() > keep_count:
                score = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
                if confidence_s is not None:
                    score += confidence_s
                if surprise_s is not None:
                    score += surprise_s
                if confidence_s is None and surprise_s is None:
                    score += 1.0
                top_sel = torch.topk(score[selected], k=keep_count, largest=True).indices
                kept_rows = selected[top_sel]
                new_mask = torch.zeros_like(write_mask)
                new_mask[kept_rows] = True
                write_mask &= new_mask
        return write_mask, confidence_s, surprise_s

    def _update_lookup_stats(self, hit_mask: torch.Tensor, batch_size: int) -> None:
        hits = int(hit_mask.sum().item())
        self.hit_count += hits
        self.miss_count += (batch_size - hits)

    def _touch_hits(self, tables: torch.Tensor, addrs: torch.Tensor, now: float) -> None:
        if tables.numel() == 0:
            return
        ones = torch.ones_like(tables, dtype=self.hit_frequency.dtype)
        self.hit_frequency.index_put_((tables, addrs), ones, accumulate=True)
        self.last_access[tables, addrs] = now

    def _batch_indices(self, batch_size: int) -> torch.Tensor:
        if self.batch_index_cache.numel() < batch_size:
            self.batch_index_cache = torch.arange(batch_size, dtype=torch.long, device=self.device)
        return self.batch_index_cache[:batch_size]

    def lookup(self, x: torch.Tensor, key_similarity_threshold: float = 0.3, use_avx512: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a fast lookup in the neural cache using LSH and SIMD if available."""
        B = x.size(0)
        x_normalized = F.normalize(x.float(), dim=1).contiguous()
        
        values, hit_mask, addresses_out = ACCEL.call(
            'neural_cache_lookup_fast',
            x_normalized.detach(),
            self.keys,
            self.values,
            self.hasher.planes,
            self.valid,
            key_similarity_threshold,
            use_avx512,
            tensors=(x_normalized, self.keys, self.values, self.hasher.planes, self.valid)
        )
        
        if not hit_mask.any():
            self.miss_count += B
            return values.to(torch.float32), hit_mask, torch.zeros(B, device=self.device), addresses_out[:, 0], torch.zeros(B, dtype=torch.long, device=self.device)

        # Vectorized similarity for hit rows only
        hit_indices = hit_mask.nonzero().squeeze(-1)
        hit_subset = x_normalized[hit_indices]
        addr_subset = addresses_out[hit_indices]
        keys_subset = self.keys[:, addr_subset] # [Tables, Hits, Dim]
        sims = (hit_subset.unsqueeze(0) * keys_subset.to(hit_subset.dtype)).sum(dim=-1) # [Tables, Hits]
        now = time.time()
        reliability_subset = self.reliability[:, addr_subset].to(sims.dtype)
        age_subset = (now - self.last_access[:, addr_subset]).clamp_min(0.0)
        recency_subset = (1.0 / (1.0 + age_subset)).to(sims.dtype)
        blended_score = (0.70 * sims) + (0.20 * reliability_subset) + (0.10 * recency_subset)
        _, best_table_subset = blended_score.max(dim=0)
        max_sim_subset = sims.gather(0, best_table_subset.unsqueeze(0)).squeeze(0)
        
        hit_addrs = addresses_out[:, 0].clone()
        hit_tables = torch.zeros(B, dtype=torch.long, device=self.device)
        max_sim = torch.zeros(B, device=self.device)
        
        hit_addrs[hit_indices] = addr_subset[range(len(hit_indices)), best_table_subset]
        hit_tables[hit_indices] = best_table_subset
        max_sim[hit_indices] = max_sim_subset

        self._update_lookup_stats(hit_mask, B)
        self._touch_hits(hit_tables[hit_mask], hit_addrs[hit_mask], now)
                
        return values.to(torch.float32), hit_mask, max_sim, hit_addrs, hit_tables

    def write(
        self,
        x,
        y,
        learning_rate=1.0,
        mask: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        surprise: Optional[torch.Tensor] = None,
        min_confidence: Optional[float] = None,
        min_surprise: Optional[float] = None,
        max_write_fraction: Optional[float] = None,
        decay_low_value: bool = True,
    ):
        if not hasattr(self, 'imprint_count'): self.imprint_count = 0
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        B = x.size(0)
        if B <= 0:
            return 0
        if y.size(0) != B:
            raise ValueError(f"NeuralCache.write batch mismatch: keys={B}, values={y.size(0)}")
        write_mask, conf_score, surprise_score = self._build_write_mask(
            B,
            mask=mask,
            confidence=confidence,
            surprise=surprise,
            min_confidence=min_confidence,
            min_surprise=min_surprise,
            max_write_fraction=max_write_fraction,
        )
        if not bool(write_mask.any().item()):
            return 0
        rows = write_mask.nonzero(as_tuple=False).squeeze(-1)
        x = x[rows]
        y = y[rows]
        B = x.size(0)
        x = x.to(torch.float32)
        addresses = self.hasher(x)
        x_normalized = F.normalize(x.detach(), dim=1).contiguous()
        y_detached = y.detach().contiguous()
        lr = float(max(0.0, min(1.0, learning_rate)))
        conf_rows = (
            torch.full((B,), 0.5, dtype=torch.float32, device=self.device)
            if conf_score is None else conf_score[rows]
        )
        
        t = 0
        target_indices = addresses[:, t]
        old_keys = self.keys[t, target_indices].to(torch.float32)
        old_vals = self.values[t, target_indices].to(torch.float32)
        new_keys = (1.0 - lr) * old_keys + lr * x_normalized
        new_vals = (1.0 - lr) * old_vals + lr * y_detached
        self.keys[t, target_indices] = F.normalize(new_keys, dim=-1).to(torch.bfloat16)
        self.values[t, target_indices] = new_vals.to(torch.bfloat16)
        self.valid[t, target_indices] = True
        self.last_access[t, target_indices] = time.time()
        self.hit_frequency[t, target_indices] = 0
        old_rel = self.reliability[t, target_indices]
        self.reliability[t, target_indices] = torch.clamp((1.0 - lr) * old_rel + lr * conf_rows, 0.0, 1.0)
        self.imprint_count += B
        self._writes_since_decay += B
        if decay_low_value and self._writes_since_decay >= 32:
            self._writes_since_decay = 0
            self.decay_low_value()
        return int(B)

    @torch.no_grad()
    def decay_low_value(self, base_decay: Optional[float] = None, min_reliability: Optional[float] = None, stale_seconds: float = 120.0) -> int:
        if self.valid.numel() == 0:
            return 0
        if not bool(self.valid.any().item()):
            return 0
        decay = self.low_value_decay if base_decay is None else float(base_decay)
        rel_floor = self.low_value_min_reliability if min_reliability is None else float(min_reliability)
        now = time.time()
        ages = (now - self.last_access).clamp_min(0.0)
        stale_scale = (ages / max(1.0, float(stale_seconds))).clamp(0.0, 1.0)
        low_freq = self.hit_frequency <= 1
        decay_tensor = torch.full_like(self.reliability, decay)
        decay_tensor[low_freq] = torch.clamp(decay - (0.08 * stale_scale[low_freq]), min=0.80, max=decay)
        self.reliability[self.valid] *= decay_tensor[self.valid]
        drop_mask = self.valid & (self.reliability < rel_floor)
        dropped = int(drop_mask.sum().item())
        if dropped > 0:
            self.valid[drop_mask] = False
        return dropped

    def memory_usage_gb(self):
        return (self.keys.element_size() * self.keys.nelement() + 
                self.values.element_size() * self.values.nelement()) / 1e9

    def prune_lru(self, fraction=0.1):
        num_to_prune = int(self.ram_size * self.num_tables * fraction)
        valid_count = int(self.valid.sum().item())
        if valid_count == 0:
            return
        num_to_prune = min(num_to_prune, valid_count)
        self.decay_low_value()
        ages = time.time() - self.last_access
        ages[~self.valid] = 0
        reliability_penalty = (1.0 - self.reliability).clamp(0.0, 1.0)
        scores = ages * (1.0 + reliability_penalty)
        _, indices = torch.topk(scores.view(-1), k=num_to_prune)
        self.valid.view(-1)[indices] = False

