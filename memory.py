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
        # Avoid repeated arange/indexing checks
        if self.batch_index_cache.numel() < batch_size:
            self.batch_index_cache = torch.arange(batch_size, dtype=torch.long, device=self.device)
        return self.batch_index_cache[:batch_size]

    def lookup(self, x: torch.Tensor, key_similarity_threshold: float = 0.3, use_avx512: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a fast lookup in the neural cache using LSH and SIMD if available."""
        B = x.size(0)
        x_fp32 = x.float()
        addresses = self.hasher(x_fp32)
        x_normalized = F.normalize(x_fp32, dim=1).contiguous()
        
        values, hit_mask = ACCEL.call(
            'neural_cache_lookup_fast',
            x_normalized.detach(),
            self.keys,
            self.values,
            addresses,
            self.valid,
            key_similarity_threshold,
            use_avx512,
            tensors=(x_normalized, self.keys, self.values, addresses, self.valid)
        )
        
        # Only perform manual similarity check if there's a hit, or use a faster way
        # Actually, best_table is only used for touch_hits. If we have multiple hits,
        # we can just pick the first hit table to save Python time.
        if not hit_mask.any():
            self.miss_count += B
            return values.to(torch.float32), hit_mask, torch.zeros(B, device=self.device), addresses[:, 0], torch.zeros(B, dtype=torch.long, device=self.device)

        # Vectorized similarity for hit rows only
        hit_indices = hit_mask.nonzero().squeeze(-1)
        hit_subset = x_normalized[hit_indices]
        addr_subset = addresses[hit_indices]
        keys_subset = self.keys[:, addr_subset] # [Tables, Hits, Dim]
        # keys_subset is [Tables, Hits, Dim], hit_subset is [Hits, Dim]
        sims = (hit_subset.unsqueeze(0) * keys_subset.to(hit_subset.dtype)).sum(dim=-1) # [Tables, Hits]
        now = time.time()
        reliability_subset = self.reliability[:, addr_subset].to(sims.dtype)
        age_subset = (now - self.last_access[:, addr_subset]).clamp_min(0.0)
        recency_subset = (1.0 / (1.0 + age_subset)).to(sims.dtype)
        blended_score = (0.70 * sims) + (0.20 * reliability_subset) + (0.10 * recency_subset)
        _, best_table_subset = blended_score.max(dim=0)
        max_sim_subset = sims.gather(0, best_table_subset.unsqueeze(0)).squeeze(0)
        
        hit_addrs = addresses[:, 0].clone()
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
        x = x.to(torch.float32) # Ensure FP32 for hasher/norm
        addresses = self.hasher(x)
        x_normalized = F.normalize(x.detach(), dim=1).contiguous()
        y_detached = y.detach().contiguous()
        lr = float(max(0.0, min(1.0, learning_rate)))
        conf_rows = (
            torch.full((B,), 0.5, dtype=torch.float32, device=self.device)
            if conf_score is None else conf_score[rows]
        )
        
        if self.imprint_count % 1000 == 0:
            pass # Silent
            # logger.debug(f"Neural Cache: {self.imprint_count} samples imprinted.")
        
        # Always write to first table (LRU-like overwrite)
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
        if decay_low_value and self._writes_since_decay >= int(self.low_value_decay_every):
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
        decay = max(0.90, min(0.9999, decay))
        rel_floor = self.low_value_min_reliability if min_reliability is None else float(min_reliability)
        rel_floor = max(0.0, min(1.0, rel_floor))
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
        if num_to_prune <= 0:
            return
        valid_count = int(self.valid.sum().item())
        if valid_count == 0:
            return
        num_to_prune = min(num_to_prune, valid_count)
        self.decay_low_value()
        # Vectorized age-based pruning
        ages = time.time() - self.last_access
        ages[~self.valid] = 0 # Don't prune already invalid
        reliability_penalty = (1.0 - self.reliability).clamp(0.0, 1.0)
        scores = ages * (1.0 + reliability_penalty)
        _, indices = torch.topk(scores.view(-1), k=num_to_prune)
        self.valid.view(-1)[indices] = False

class EpisodicMemory(nn.Module):
    def __init__(
        self,
        dim,
        capacity=1000,
        device='cpu',
        dense_mode=False,
        key_latent_dim=None,
        value_latent_dim=None,
        capacity_multiplier=1.0
    ):
        super().__init__()
        self.dim = dim
        self.base_capacity = int(max(1, capacity))
        self.capacity_multiplier = float(max(1.0, capacity_multiplier))
        self.capacity = int(max(1, round(self.base_capacity * self.capacity_multiplier)))
        self.device = torch.device(device)
        self.dense_mode = bool(dense_mode)
        self.key_latent_dim = int(max(1, min(dim, key_latent_dim if key_latent_dim is not None else dim)))
        self.value_latent_dim = int(max(1, min(dim, value_latent_dim if value_latent_dim is not None else self.key_latent_dim)))
        self.storage_dtype = torch.bfloat16 if self.dense_mode else torch.float32

        if self.dense_mode:
            key_proj = self._orthogonal_projection(dim, self.key_latent_dim, self.device)
            value_proj = self._orthogonal_projection(dim, self.value_latent_dim, self.device)
            self.register_buffer('key_proj', key_proj)
            self.register_buffer('value_proj', value_proj)
            self.register_buffer('value_deproj', value_proj.t().contiguous())
            self.register_buffer('keys', torch.zeros(self.capacity, self.key_latent_dim, dtype=self.storage_dtype, device=self.device))
            self.register_buffer('values', torch.zeros(self.capacity, self.value_latent_dim, dtype=self.storage_dtype, device=self.device))
        else:
            self.register_buffer('key_proj', torch.empty(0, 0, dtype=torch.float32, device=self.device))
            self.register_buffer('value_proj', torch.empty(0, 0, dtype=torch.float32, device=self.device))
            self.register_buffer('value_deproj', torch.empty(0, 0, dtype=torch.float32, device=self.device))
            self.register_buffer('keys', torch.zeros(self.capacity, dim, dtype=self.storage_dtype, device=self.device))
            self.register_buffer('values', torch.zeros(self.capacity, dim, dtype=self.storage_dtype, device=self.device))

        self.register_buffer('age', torch.zeros(self.capacity, device=self.device))
        self.register_buffer('_index_cache', torch.empty(0, dtype=torch.long, device=self.device))
        self.ptr = 0
        self.is_full = False

    @property
    def count(self):
        return self.capacity if self.is_full else self.ptr

    def _recent_indices(self, scan_size: int) -> torch.Tensor:
        if scan_size <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        if not self.is_full:
            start = max(0, self.ptr - scan_size)
            return torch.arange(start, self.ptr, dtype=torch.long, device=self.device)
        if self._index_cache.numel() < scan_size:
            self._index_cache = torch.arange(scan_size, dtype=torch.long, device=self.device)
        offset = self._index_cache[:scan_size]
        return (self.ptr - scan_size + offset) % self.capacity

    @staticmethod
    def _orthogonal_projection(in_dim: int, out_dim: int, device: torch.device) -> torch.Tensor:
        if out_dim >= in_dim:
            return torch.eye(in_dim, dtype=torch.float32, device=device)
        # Reduced QR gives orthonormal columns for stable latent compression.
        mat = torch.randn(in_dim, out_dim, dtype=torch.float32, device=device)
        q, _ = torch.linalg.qr(mat, mode='reduced')
        return q[:, :out_dim].contiguous()

    def memory_usage_gb(self) -> float:
        total_bytes = (
            self.keys.element_size() * self.keys.nelement()
            + self.values.element_size() * self.values.nelement()
        )
        return float(total_bytes) / 1e9

    def _compress_key(self, k: torch.Tensor) -> torch.Tensor:
        if not self.dense_mode:
            return k
        k_lat = torch.matmul(k, self.key_proj)
        return F.normalize(k_lat, dim=-1)

    def _compress_value(self, v: torch.Tensor) -> torch.Tensor:
        if not self.dense_mode:
            return v
        return torch.matmul(v, self.value_proj)

    def _decompress_key(self, k_store: torch.Tensor) -> torch.Tensor:
        if not self.dense_mode:
            return k_store
        key = torch.matmul(k_store, self.key_proj.t())
        return F.normalize(key, dim=-1)

    def _decompress_value(self, v_store: torch.Tensor) -> torch.Tensor:
        if not self.dense_mode:
            return v_store
        return torch.matmul(v_store, self.value_deproj)

    def write(self, k, v):
        # Norm and detach
        k = F.normalize(k.detach().to(self.device, dtype=torch.float32), dim=-1)
        v = v.detach().to(self.device, dtype=torch.float32)
        
        # Handle batch or single
        if k.dim() == 1: k = k.unsqueeze(0)
        if v.dim() == 1: v = v.unsqueeze(0)
        
        batch_size = k.size(0)
        if batch_size == 0:
            return
        if v.size(-1) != self.dim or k.size(-1) != self.dim:
            raise ValueError(
                f"EpisodicMemory expects key/value last dim={self.dim}, got key={k.size(-1)}, value={v.size(-1)}."
            )

        k_store = self._compress_key(k).to(dtype=self.keys.dtype)
        v_store = self._compress_value(v).to(dtype=self.values.dtype)

        # Keep only latest items when caller provides more than capacity.
        if batch_size >= self.capacity:
            k_store = k_store[-self.capacity:]
            v_store = v_store[-self.capacity:]
            batch_size = self.capacity
            self.keys.copy_(k_store)
            self.values.copy_(v_store)
            self.age.zero_()
            self.ptr = 0
            self.is_full = True
            return
        
        # Calculate how many slots are available before we wrap
        remaining = self.capacity - self.ptr
        if batch_size <= remaining:
            self.keys[self.ptr : self.ptr + batch_size] = k_store
            self.values[self.ptr : self.ptr + batch_size] = v_store
            self.age[self.ptr : self.ptr + batch_size] = 0
            self.ptr = (self.ptr + batch_size) % self.capacity
        else:
            # First part (to the end)
            self.keys[self.ptr :] = k_store[:remaining]
            self.values[self.ptr :] = v_store[:remaining]
            self.age[self.ptr :] = 0
            # Second part (wrap around)
            wrapped = batch_size - remaining
            self.keys[:wrapped] = k_store[remaining:]
            self.values[:wrapped] = v_store[remaining:]
            self.age[:wrapped] = 0
            self.ptr = wrapped
            self.is_full = True
        
        if self.ptr == 0 and batch_size > 0:
            self.is_full = True
        
    def read(self, query, top_k=1, max_scan=None):
        query = query.to(self.device, dtype=torch.float32)
        if self.ptr == 0 and not self.is_full:
            bsz = query.size(0)
            return (
                torch.zeros(bsz, 0, self.dim, device=self.device),
                torch.zeros(bsz, 0, self.dim, device=self.device),
                torch.zeros(bsz, 0, device=self.device),
            )
        if top_k <= 0:
            bsz = query.size(0)
            return (
                torch.zeros(bsz, 0, self.dim, device=self.device),
                torch.zeros(bsz, 0, self.dim, device=self.device),
                torch.zeros(bsz, 0, device=self.device),
            )
            
        # Cosine similarity
        query_norm = F.normalize(query, dim=-1)
        query_eval = self._compress_key(query_norm) if self.dense_mode else query_norm
        active_count = self.count
        if max_scan is None:
            scan_size = active_count
        else:
            scan_size = max(1, min(int(max_scan), active_count))

        if scan_size < active_count:
            recent_idx = self._recent_indices(scan_size)
            valid_keys = self.keys[recent_idx]
            valid_values = self.values[recent_idx]
        else:
            valid_keys = self.keys if self.is_full else self.keys[:self.ptr]
            valid_values = self.values if self.is_full else self.values[:self.ptr]
        
        valid_keys_f = valid_keys.to(torch.float32)
        sim = torch.mm(query_eval, valid_keys_f.t())
        k = min(int(top_k), valid_keys.size(0))
        scores, indices = torch.topk(sim, k=k, dim=-1) # [B, k]
        
        # Return distinct top-k memories (Keys=Context, Values=Targets)
        # indices is [B, k]
        # values[indices] -> [B, k, D]
        
        retrieved_keys_store = valid_keys_f[indices]
        retrieved_values_store = valid_values.to(torch.float32)[indices]
        retrieved_keys = self._decompress_key(retrieved_keys_store)
        retrieved_values = self._decompress_value(retrieved_values_store)
        retrieved_scores = scores
        
        return retrieved_keys, retrieved_values, retrieved_scores


class HybridEpisodicMemory(nn.Module):
    """
    Two-tier episodic memory:
    - Hot bank: small/full-dim for fast recent access
    - Cold bank: larger optional dense-compressed storage
    """

    def __init__(
        self,
        dim,
        hot_capacity=256,
        cold_capacity=1000,
        device='cpu',
        cold_dense_mode=True,
        key_latent_dim=64,
        value_latent_dim=64,
        cold_capacity_multiplier=1.0,
        fallback_threshold=0.35,
        hot_top_k=2,
        cold_top_k=2,
    ):
        super().__init__()
        self.dim = int(dim)
        self.device = torch.device(device)
        self.hot_top_k = max(1, int(hot_top_k))
        self.cold_top_k = max(1, int(cold_top_k))
        self.fallback_threshold = float(fallback_threshold)

        self.hot = EpisodicMemory(
            dim=self.dim,
            capacity=max(1, int(hot_capacity)),
            device=self.device,
            dense_mode=False,
            capacity_multiplier=1.0,
        )
        self.cold = EpisodicMemory(
            dim=self.dim,
            capacity=max(1, int(cold_capacity)),
            device=self.device,
            dense_mode=bool(cold_dense_mode),
            key_latent_dim=int(key_latent_dim),
            value_latent_dim=int(value_latent_dim),
            capacity_multiplier=float(cold_capacity_multiplier),
        )
        self.register_buffer('_overwrite_index_cache', torch.empty(0, dtype=torch.long, device=self.device))

    @property
    def count(self):
        return int(self.hot.count + self.cold.count)

    @property
    def capacity(self):
        return int(self.hot.capacity + self.cold.capacity)

    def memory_usage_gb(self) -> float:
        return float(self.hot.memory_usage_gb() + self.cold.memory_usage_gb())

    def _overwrite_indices(self, batch_size: int) -> torch.Tensor:
        if batch_size <= 0 or self.hot.count <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        cap = int(self.hot.capacity)
        ptr = int(self.hot.ptr)

        if batch_size >= cap:
            if self.hot.is_full:
                return torch.arange(cap, dtype=torch.long, device=self.device)
            return torch.arange(self.hot.count, dtype=torch.long, device=self.device)

        if not self.hot.is_full:
            remaining = cap - ptr
            if batch_size <= remaining:
                return torch.empty(0, dtype=torch.long, device=self.device)
            wrapped = batch_size - remaining
            existing = min(int(self.hot.count), int(wrapped))
            return torch.arange(existing, dtype=torch.long, device=self.device)

        # Full ring: next writes overwrite from ptr onward.
        if self._overwrite_index_cache.numel() < batch_size:
            self._overwrite_index_cache = torch.arange(batch_size, dtype=torch.long, device=self.device)
        offset = self._overwrite_index_cache[:batch_size]
        return (ptr + offset) % cap

    def write(self, k, v):
        if isinstance(k, torch.Tensor) and k.dim() == 1:
            k = k.unsqueeze(0)
        if isinstance(v, torch.Tensor) and v.dim() == 1:
            v = v.unsqueeze(0)
        if not isinstance(k, torch.Tensor):
            k = torch.as_tensor(k, dtype=torch.float32, device=self.device)
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v, dtype=torch.float32, device=self.device)
        if k.numel() == 0 or v.numel() == 0:
            return
        if k.size(0) != v.size(0):
            raise ValueError(f"HybridEpisodicMemory write batch mismatch: keys={k.size(0)} values={v.size(0)}")

        k = k.to(self.device, dtype=torch.float32)
        v = v.to(self.device, dtype=torch.float32)
        batch_size = int(k.size(0))

        # If incoming batch is larger than hot capacity, preserve overflow in cold.
        hot_cap = int(self.hot.capacity)
        if batch_size > hot_cap:
            spill = batch_size - hot_cap
            self.cold.write(k[:spill], v[:spill])
            k = k[spill:]
            v = v[spill:]
            batch_size = int(k.size(0))

        # Demote entries that are about to be evicted from hot -> cold.
        evict_idx = self._overwrite_indices(batch_size)
        if evict_idx.numel() > 0:
            hot_keys_store = self.hot.keys[evict_idx].to(torch.float32)
            hot_vals_store = self.hot.values[evict_idx].to(torch.float32)
            hot_keys = self.hot._decompress_key(hot_keys_store)
            hot_vals = self.hot._decompress_value(hot_vals_store)
            self.cold.write(hot_keys, hot_vals)

        self.hot.write(k, v)

    def _merge_topk(self, hot_k, hot_v, hot_s, cold_k, cold_v, cold_s, top_k):
        if hot_s.numel() == 0 and cold_s.numel() == 0:
            bsz = hot_k.size(0)
            return (
                torch.zeros(bsz, 0, self.dim, device=self.device),
                torch.zeros(bsz, 0, self.dim, device=self.device),
                torch.zeros(bsz, 0, device=self.device),
            )
        if cold_s.numel() == 0:
            k_take = min(int(top_k), hot_s.size(1))
            return hot_k[:, :k_take], hot_v[:, :k_take], hot_s[:, :k_take]
        if hot_s.numel() == 0:
            k_take = min(int(top_k), cold_s.size(1))
            return cold_k[:, :k_take], cold_v[:, :k_take], cold_s[:, :k_take]

        all_keys = torch.cat([hot_k, cold_k], dim=1)
        all_vals = torch.cat([hot_v, cold_v], dim=1)
        all_scores = torch.cat([hot_s, cold_s], dim=1)
        k_take = min(int(top_k), all_scores.size(1))
        scores, idx = torch.topk(all_scores, k=k_take, dim=-1)
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, self.dim)
        keys = torch.gather(all_keys, 1, gather_idx)
        vals = torch.gather(all_vals, 1, gather_idx)
        return keys, vals, scores

    def read(self, query, top_k=1, max_scan=None):
        if not isinstance(query, torch.Tensor):
            query = torch.as_tensor(query, dtype=torch.float32, device=self.device)
        query = query.to(self.device, dtype=torch.float32)
        bsz = query.size(0)
        if top_k <= 0:
            return (
                torch.zeros(bsz, 0, self.dim, device=self.device),
                torch.zeros(bsz, 0, self.dim, device=self.device),
                torch.zeros(bsz, 0, device=self.device),
            )
        if self.hot.count == 0 and self.cold.count == 0:
            return (
                torch.zeros(bsz, 0, self.dim, device=self.device),
                torch.zeros(bsz, 0, self.dim, device=self.device),
                torch.zeros(bsz, 0, device=self.device),
            )

        hot_scan = None if max_scan is None else min(int(max_scan), int(self.hot.count))
        hot_k_req = min(self.hot_top_k, int(top_k))
        hot_keys, hot_vals, hot_scores = self.hot.read(query, top_k=hot_k_req, max_scan=hot_scan)

        # If no cold bank content, return hot-only top-k.
        if self.cold.count == 0:
            k_take = min(int(top_k), hot_scores.size(1))
            return hot_keys[:, :k_take], hot_vals[:, :k_take], hot_scores[:, :k_take]

        if hot_scores.size(1) == 0:
            need_cold = torch.ones(bsz, dtype=torch.bool, device=self.device)
        else:
            need_cold = hot_scores[:, 0] < self.fallback_threshold

        if not bool(need_cold.any().item()):
            k_take = min(int(top_k), hot_scores.size(1))
            return hot_keys[:, :k_take], hot_vals[:, :k_take], hot_scores[:, :k_take]

        cold_k_req = min(self.cold_top_k, int(top_k))
        if bool(need_cold.all().item()):
            cold_keys, cold_vals, cold_scores = self.cold.read(query, top_k=cold_k_req, max_scan=max_scan)
            return self._merge_topk(hot_keys, hot_vals, hot_scores, cold_keys, cold_vals, cold_scores, top_k)

        # Query cold only for low-confidence rows, then scatter back.
        subset_query = query[need_cold]
        ck_sub, cv_sub, cs_sub = self.cold.read(subset_query, top_k=cold_k_req, max_scan=max_scan)
        cold_width = int(cs_sub.size(1))
        if cold_width == 0:
            k_take = min(int(top_k), hot_scores.size(1))
            return hot_keys[:, :k_take], hot_vals[:, :k_take], hot_scores[:, :k_take]

        cold_keys = torch.zeros(bsz, cold_width, self.dim, dtype=torch.float32, device=self.device)
        cold_vals = torch.zeros(bsz, cold_width, self.dim, dtype=torch.float32, device=self.device)
        cold_scores = torch.full((bsz, cold_width), -1e9, dtype=torch.float32, device=self.device)
        cold_keys[need_cold] = ck_sub
        cold_vals[need_cold] = cv_sub
        cold_scores[need_cold] = cs_sub
        return self._merge_topk(hot_keys, hot_vals, hot_scores, cold_keys, cold_vals, cold_scores, top_k)

class MemoryGovernor(nn.Module):
    def __init__(self, dim, device='cpu'):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        return self.predictor(x)
