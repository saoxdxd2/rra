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


class MortonBuffer:
    """
    Morton/Z-order index buffer for mapping 3D topology to linear memory.
    """

    def __init__(self, shape3d, device="cpu"):
        if len(shape3d) != 3:
            raise ValueError(f"MortonBuffer expects shape=(X,Y,Z), got {shape3d}")
        self.shape3d = tuple(max(1, int(s)) for s in shape3d)
        self.device = torch.device(device)

        x = torch.arange(self.shape3d[0], device=self.device, dtype=torch.long)
        y = torch.arange(self.shape3d[1], device=self.device, dtype=torch.long)
        z = torch.arange(self.shape3d[2], device=self.device, dtype=torch.long)
        gx, gy, gz = torch.meshgrid(x, y, z, indexing="ij")
        codes = morton3d_code(gx.reshape(-1), gy.reshape(-1), gz.reshape(-1))

        self.order = torch.argsort(codes)
        self.inverse_order = torch.empty_like(self.order)
        self.inverse_order[self.order] = torch.arange(self.order.numel(), device=self.device)
        self.size = int(self.order.numel())

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
        n = self.size
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
