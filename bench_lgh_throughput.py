import torch
import accelerator
import time
import argparse

def benchmark_lgh(L=64, R=2, M=256, bits=8):
    device = torch.device('cpu')
    print(f"Benchmarking LGH Kernel | L={L} R={R} M={M} bits={bits}")
    
    B = 32
    T = 64
    N = L * R * 128 # Manifold size (Depth=128)
    S = 64 # Curve length
    
    p_brain = torch.randn(B, T, M, device=device)
    
    if bits == 8:
        manifold = torch.randint(-127, 127, (N, M), dtype=torch.int8, device=device)
        scales = torch.rand(N, device=device)
    else:
        manifold = torch.randn(N, M, device=device)
        scales = torch.empty(0)
        
    indices = torch.randint(0, N, (S,), dtype=torch.long, device=device)
    focus = 2.0
    
    loader = accelerator.get_cpp_loader()
    
    # Warmup
    for _ in range(10):
        _ = loader.geometric_manifold_forward_avx512(p_brain, manifold, scales, indices, focus)
        
    # C++ Benchmark
    start = time.time()
    iters = 100
    for _ in range(iters):
        _ = loader.geometric_manifold_forward_avx512(p_brain, manifold, scales, indices, focus)
    end = time.time()
    cpp_time = (end - start) / iters
    print(f"C++ Kernel: {cpp_time*1000:.3f} ms/iter")
    
    # Python Benchmark (Simulated)
    # We simulate what the fallback does: indexing + mean
    start = time.time()
    for _ in range(iters):
        idx = indices.long()
        if bits == 8:
            raw = manifold[idx].float() * scales[idx].unsqueeze(-1)
        else:
            raw = manifold[idx]
        recall = raw.mean(dim=0).expand_as(p_brain) * focus
    end = time.time()
    py_time = (end - start) / iters
    print(f"Python Fallback: {py_time*1000:.3f} ms/iter")
    
    speedup = py_time / cpp_time
    print(f"Speedup: {speedup:.2f}x")
    
if __name__ == "__main__":
    benchmark_lgh()
