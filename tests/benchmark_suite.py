import torch
import torch.nn as nn
import time
import numpy as np
import accelerator
from organism import CognitiveOrganism
import math

def benchmark_kernel(name, fn, args, iterations=100, warmup=10):
    # Warmup
    for _ in range(warmup):
        fn(*args)
    
    # Measure
    start = time.perf_counter()
    for _ in range(iterations):
        fn(*args)
    end = time.perf_counter()
    
    avg_time = (end - start) / iterations
    return avg_time

def run_benchmarks():
    accel = accelerator.get_accelerator()
    cpp = accel.loader
    device = torch.device('cpu')
    
    print("="*60)
    print("RRA PERFORMANCE BENCHMARK SUITE")
    print("="*60)
    
    # 1. Calculate NIS Sparsity Benchmark
    # Represents the cost of computing metabolic efficiency groups
    group_scores = torch.randn(1024, 1024, device=device)
    t_avg = benchmark_kernel("calculate_nis_sparsity", cpp.calculate_nis_sparsity, (group_scores, 0.5))
    print(f"calculate_nis_sparsity (1024x1024): {t_avg*1000:.3f} ms")
    
    # 2. Neural Cache Lookup Fast Benchmark
    # Represents System 1 reflex path latency
    B, D_in, D_out = 64, 256, 8
    query = torch.randn(B, D_in, device=device)
    keys = torch.randn(4, 1024, D_in, device=device)
    values = torch.randn(4, 1024, D_out, device=device)
    planes = torch.randn(4, D_in, 8, device=device)
    valid = torch.ones(4, 1024, device=device).bool()
    
    t_avg = benchmark_kernel("neural_cache_lookup_fast", cpp.neural_cache_lookup_fast, (query, keys, values, planes, valid, 0.8, True))
    print(f"neural_cache_lookup_fast (B=64, 4 tables): {t_avg*1000:.3f} ms")
    print(f"   Throughput: {B/t_avg:.1f} lookups/sec")
    
    # 3. AdEMAMix Update Benchmark
    # Represents the cost of the metabolic optimizer loop
    param = torch.randn(1024, 1024, device=device)
    grad = torch.randn(1024, 1024, device=device)
    mf = torch.zeros_like(param)
    ms = torch.zeros_like(param)
    v = torch.zeros_like(param)
    
    t_avg = benchmark_kernel("ademamix_update", cpp.batched_ademamix_update, (
        [param], [grad], [mf], [ms], [v],
        0.001, 0.9, 0.999, 0.99, 5.0, 1e-8, 0.01, 0.01, 0
    ))
    print(f"batched_ademamix_update (1M params): {t_avg*1000:.3f} ms")
    
    # 4. End-to-End Organism Forward Pass
    # The ultimate test of the reasoning core
    L, R, C = 4, 8, 4
    model = CognitiveOrganism(input_dim=8, L=L, R=R, device=device)
    x = torch.randn(32, 16, 8, device=device) # B=32, T=16
    
    t_avg = benchmark_kernel("organism_forward", model.forward, (x, None))
    print(f"organism_forward (B=32, T=16, L=4, R=8): {t_avg*1000:.3f} ms")
    
    tokens_per_sec = (32 * 16) / t_avg
    print(f"   Throughput: {tokens_per_sec:.1f} tokens/sec")
    print("="*60)

if __name__ == '__main__':
    run_benchmarks()
