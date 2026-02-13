import torch


def _split_by_3bits(v: torch.Tensor) -> torch.Tensor:
    v = v.to(dtype=torch.long)
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8)) & 0x0300F00F
    v = (v | (v << 4)) & 0x030C30C3
    v = (v | (v << 2)) & 0x09249249
    return v


def morton3d_code(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return _split_by_3bits(x) | (_split_by_3bits(y) << 1) | (_split_by_3bits(z) << 2)


def morton4d_code(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, t: torch.Tensor, bits: int = 16) -> torch.Tensor:
    """
    Generic 4D Morton encoder for spatiotemporal indexing.
    Interleaves bits as x0,y0,z0,t0,x1,y1,z1,t1,...
    """
    x = x.to(dtype=torch.long)
    y = y.to(dtype=torch.long)
    z = z.to(dtype=torch.long)
    t = t.to(dtype=torch.long)
    code = torch.zeros_like(x, dtype=torch.long)
    for b in range(max(1, int(bits))):
        shift = 4 * b
        code |= ((x >> b) & 1) << shift
        code |= ((y >> b) & 1) << (shift + 1)
        code |= ((z >> b) & 1) << (shift + 2)
        code |= ((t >> b) & 1) << (shift + 3)
    return code


class MortonBuffer:
    """
    Morton/Z-order index buffer for mapping 3D topology to linear memory.
    """

    def __init__(self, shape3d, device="cpu", align_multiple=1, temporal_bins=16):
        if len(shape3d) != 3:
            raise ValueError(f"MortonBuffer expects shape=(X,Y,Z), got {shape3d}")
        self.shape3d = tuple(max(1, int(s)) for s in shape3d)
        self.device = torch.device(device)
        self.align_multiple = max(1, int(align_multiple))

        x = torch.arange(self.shape3d[0], device=self.device, dtype=torch.long)
        y = torch.arange(self.shape3d[1], device=self.device, dtype=torch.long)
        z = torch.arange(self.shape3d[2], device=self.device, dtype=torch.long)
        gx, gy, gz = torch.meshgrid(x, y, z, indexing="ij")
        self._coords_original = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1).to(torch.float32)
        norm = torch.tensor(
            [max(1, self.shape3d[0] - 1), max(1, self.shape3d[1] - 1), max(1, self.shape3d[2] - 1)],
            dtype=torch.float32,
            device=self.device
        )
        self._coords_original_norm = self._coords_original / norm
        self.temporal_bins = max(1, int(temporal_bins))
        codes = morton3d_code(gx.reshape(-1), gy.reshape(-1), gz.reshape(-1))

        self.order = torch.argsort(codes)
        self.size_original = int(self.order.numel())
        if self.align_multiple > 1 and (self.size_original % self.align_multiple) != 0:
            pad = self.align_multiple - (self.size_original % self.align_multiple)
            pad_src = self.order[torch.arange(pad, device=self.device) % self.size_original]
            self.order = torch.cat([self.order, pad_src], dim=0)
        self.size = int(self.order.numel())

        self.inverse_order = torch.full((self.size,), self.size - 1, dtype=torch.long, device=self.device)
        first_pos = torch.full((self.size_original,), -1, dtype=torch.long, device=self.device)
        for pos, orig in enumerate(self.order.tolist()):
            if orig < self.size_original and first_pos[orig].item() < 0:
                first_pos[orig] = int(pos)
        self.inverse_order[:self.size_original] = torch.where(first_pos >= 0, first_pos, torch.zeros_like(first_pos))
        if self.size > self.size_original:
            tail = torch.arange(self.size_original, self.size, device=self.device, dtype=torch.long)
            self.inverse_order[self.size_original:] = tail
        self._build_spatiotemporal_atlas(self.temporal_bins)

    def morton_to_original(self, morton_indices: torch.Tensor) -> torch.Tensor:
        morton_indices = morton_indices.to(dtype=torch.long, device=self.device)
        morton_indices = morton_indices.clamp(0, max(0, self.size - 1))
        return self.order[morton_indices]

    def original_to_morton(self, original_indices: torch.Tensor) -> torch.Tensor:
        original_indices = original_indices.to(dtype=torch.long, device=self.device)
        original_indices = original_indices.clamp(0, max(0, self.size_original - 1))
        return self.inverse_order[original_indices]

    def reorder_rows(self, rows: torch.Tensor) -> torch.Tensor:
        if rows.size(0) != self.size:
            raise ValueError(f"Expected first dim={self.size}, got {rows.size(0)}")
        return rows[self.order]

    def restore_rows(self, rows_morton: torch.Tensor) -> torch.Tensor:
        if rows_morton.size(0) != self.size:
            raise ValueError(f"Expected first dim={self.size}, got {rows_morton.size(0)}")
        return rows_morton[self.inverse_order]

    def flatten(self, tensor3d: torch.Tensor) -> torch.Tensor:
        if tensor3d.dim() < 3:
            raise ValueError("flatten expects tensor with at least 3 dims.")
        if tuple(tensor3d.shape[:3]) != self.shape3d:
            raise ValueError(f"Expected leading shape {self.shape3d}, got {tuple(tensor3d.shape[:3])}")
        flat = tensor3d.reshape(self.size, *tensor3d.shape[3:])
        return flat[self.order]

    def unflatten(self, rows_morton: torch.Tensor) -> torch.Tensor:
        if rows_morton.size(0) != self.size:
            raise ValueError(f"Expected first dim={self.size}, got {rows_morton.size(0)}")
        rows = rows_morton[self.inverse_order]
        return rows.reshape(*self.shape3d, *rows_morton.shape[1:])

    def curve_segment(self, start: int, length: int, wrap=True) -> torch.Tensor:
        n = self.size_original
        if n <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        length = max(1, min(int(length), n))
        start = int(start)
        if wrap:
            idx = (torch.arange(length, device=self.device, dtype=torch.long) + start) % n
        else:
            lo = max(0, min(n - 1, start))
            hi = min(n, lo + length)
            idx = torch.arange(lo, hi, device=self.device, dtype=torch.long)
            if idx.numel() < length:
                pad = torch.full((length - idx.numel(),), hi - 1, device=self.device, dtype=torch.long)
                idx = torch.cat([idx, pad], dim=0)
        return idx

    def _focus_vector(self, focus_point):
        if focus_point is None:
            return None
        if isinstance(focus_point, torch.Tensor):
            fp = focus_point.to(device=self.device, dtype=torch.float32).flatten()
        elif isinstance(focus_point, (list, tuple)):
            fp = torch.tensor(list(focus_point), dtype=torch.float32, device=self.device).flatten()
        else:
            v = float(focus_point)
            fp = torch.tensor([v, v, v], dtype=torch.float32, device=self.device)
        if fp.numel() == 1:
            fp = fp.repeat(3)
        if fp.numel() < 3:
            fp = torch.cat([fp, fp.new_full((3 - fp.numel(),), 0.5)], dim=0)
        return fp[:3].clamp(0.0, 1.0)

    def focus_distance(self, original_indices: torch.Tensor, focus_point) -> torch.Tensor:
        focus = self._focus_vector(focus_point)
        if focus is None:
            return torch.zeros_like(original_indices, dtype=torch.float32, device=self.device)
        original_indices = original_indices.to(dtype=torch.long, device=self.device).clamp(0, max(0, self.size_original - 1))
        coords = self._coords_original_norm[original_indices]
        dist = torch.linalg.norm(coords - focus.view(1, 3), dim=-1)
        return dist

    def _build_spatiotemporal_atlas(self, temporal_bins: int):
        bins = max(1, int(temporal_bins))
        self.temporal_bins = bins
        morton_rows = torch.arange(self.size, device=self.device, dtype=torch.long)
        row_major_rows = morton_rows.repeat_interleave(bins)
        row_major_t = torch.arange(bins, device=self.device, dtype=torch.long).repeat(self.size)
        orig_idx = self.morton_to_original(row_major_rows).clamp(0, max(0, self.size_original - 1))
        coords = self._coords_original[orig_idx].to(dtype=torch.long)
        codes4d = morton4d_code(coords[:, 0], coords[:, 1], coords[:, 2], row_major_t)
        order4d = torch.argsort(codes4d)
        inverse4d = torch.empty_like(order4d)
        inverse4d[order4d] = torch.arange(order4d.numel(), device=self.device, dtype=torch.long)

        self.atlas_size = int(row_major_rows.numel())
        self.atlas_row_to_morton = row_major_rows
        self.atlas_row_to_time = row_major_t
        self.atlas_order4d = order4d
        self.atlas_inverse4d = inverse4d
        self.atlas_linear_row_major = (row_major_rows * bins) + row_major_t
        self.atlas_linear_4d = self.atlas_linear_row_major[order4d]

    def set_temporal_bins(self, bins: int):
        self._build_spatiotemporal_atlas(max(1, int(bins)))

    def get_morton_coords(self, morton_index: int, t: int = 0) -> torch.Tensor:
        """Returns normalized [X, Y, Z, T] coordinates for a Morton index."""
        orig_idx = self.morton_to_original(torch.tensor([morton_index], device=self.device)).item()
        coords_3d = self._coords_original_norm[int(orig_idx)]
        t_norm = torch.tensor([float(t) / max(1, self.temporal_bins)], device=self.device)
        return torch.cat([coords_3d, t_norm])

    def get_path_momentum(self, start_idx: int, end_idx: int, t_start: int, t_end: int) -> torch.Tensor:
        """Calculates the 4D momentum vector between two points in Morton-Time space."""
        c1 = self.get_morton_coords(start_idx, t_start)
        c2 = self.get_morton_coords(end_idx, t_end)
        return c2 - c1

    def get_path_momentum_vec(self, chain_indices: torch.Tensor, t_start: int) -> torch.Tensor:
        """Calculates the aggregate 4D momentum vector for a sequence of points."""
        if chain_indices.numel() < 2:
            return torch.zeros(4, device=self.device)
        
        # Look at the start and end of the chain for momentum
        c_start = self.get_morton_coords(int(chain_indices[0].item()), t_start)
        c_end = self.get_morton_coords(int(chain_indices[-1].item()), t_start + chain_indices.numel())
        return c_end - c_start

    def spatiotemporal_row(self, morton_rows: torch.Tensor, delta_t: int, temporal_bins: int = None) -> torch.Tensor:
        if temporal_bins is None:
            temporal_bins = self.temporal_bins
        bins = max(1, int(temporal_bins))
        rows = morton_rows.to(dtype=torch.long, device=self.device).clamp(0, max(0, self.size - 1))
        t = int(delta_t) % bins
        return rows * bins + t

    def temporal_fold_morton(
        self,
        morton_indices: torch.Tensor,
        delta_t: int,
        temporal_bins: int = None,
        fold_alpha: float = 0.25
    ) -> torch.Tensor:
        """
        Fold a 4D temporal phase into 3D Morton rows using rotating-mask semantics.
        """
        if temporal_bins is None:
            temporal_bins = self.temporal_bins
        bins = max(1, int(temporal_bins))
        idx = morton_indices.to(dtype=torch.long, device=self.device).clamp(0, max(0, self.size_original - 1))
        if bins <= 1 or self.size_original <= 1:
            return idx
        dt = int(delta_t) % bins
        amp = max(0.0, min(1.0, float(fold_alpha)))
        span = max(1, int(round((self.size_original - 1) * amp)))
        shift = (dt * span) % self.size_original
        return (idx + shift) % self.size_original

    def encode4d_from_original(
        self,
        original_indices: torch.Tensor,
        delta_t: int,
        temporal_bins: int = None
    ) -> torch.Tensor:
        """
        Expands 3D topology indices with a temporal coordinate into a 4D Morton code.
        """
        if temporal_bins is None:
            temporal_bins = self.temporal_bins
        bins = max(1, int(temporal_bins))
        original_indices = original_indices.to(dtype=torch.long, device=self.device).clamp(0, max(0, self.size_original - 1))
        coords = self._coords_original[original_indices].to(dtype=torch.long)
        t = torch.full((original_indices.numel(),), int(delta_t) % bins, dtype=torch.long, device=self.device)
        return morton4d_code(coords[:, 0], coords[:, 1], coords[:, 2], t)

    def curve_segment_original(
        self,
        start: int,
        length: int,
        wrap=True,
        focus_point=None,
        focus_strength: float = 0.0,
        focus_sharpness: float = 1.0,
        delta_t: int = 0,
        temporal_bins: int = None,
        fold_alpha: float = 0.25
    ) -> torch.Tensor:
        """
        Returns a curve segment encoded in original (topology) row indices.
        Kernel can map them back to Morton rows using inverse_order.
        """
        morton_idx = self.curve_segment(start, length, wrap=wrap)
        morton_idx = self.temporal_fold_morton(
            morton_idx,
            delta_t=delta_t,
            temporal_bins=temporal_bins,
            fold_alpha=fold_alpha
        )
        original = self.morton_to_original(morton_idx)
        fs = max(0.0, min(1.0, float(focus_strength)))
        if fs > 0.0 and focus_point is not None and original.numel() > 1:
            sharp = max(0.01, float(focus_sharpness))
            rank = torch.arange(original.numel(), device=self.device, dtype=torch.float32)
            dist = self.focus_distance(original, focus_point)
            focal = torch.exp(-sharp * dist)
            score = rank - (fs * focal * float(original.numel()))
            order = torch.argsort(score)
            original = original[order]
        return original

    def curve_segment_morton(self, start: int, length: int, **kwargs) -> torch.Tensor:
        """
        Returns a curve segment encoded in Morton row indices.
        Used for O(1) Manifold addressing without kernel-side inverse mapping.
        """
        original = self.curve_segment_original(start=start, length=length, **kwargs)
        return self.original_to_morton(original)
