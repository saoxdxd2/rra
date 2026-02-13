import torch
import unittest
import accelerator
cpp_loader = accelerator.get_cpp_loader()
import time
import math

class TestMemorySystem(unittest.TestCase):
    def setUp(self):
        self.B = 4
        self.D = 64
        self.RAM = 1024
        self.TBL = 4
        self.BITS = 8
        self.O = 128
        
        self.device = torch.device('cpu')
        torch.manual_seed(42)

    def test_lsh_lookup_integrity(self):
        # Setup mock LSH components
        query = torch.randn(self.B, self.D)
        keys = torch.randn(self.TBL, self.RAM, self.D)
        values = torch.randn(self.TBL, self.RAM, self.O)
        planes = torch.randn(self.TBL, self.BITS, self.D)
        valid = torch.ones(self.TBL, self.RAM, dtype=torch.bool)
        
        # Insert a specific key-value pair to verify "hit"
        target_b = 0
        target_tbl = 0
        target_q = query[target_b] / (query[target_b].norm() + 1e-8)
        query[target_b] = target_q # Normalize to ensure high cosine similarity
        
        # Calculate hash for query[0] on table 0
        hash_val = 0
        for bit in range(self.BITS):
            dot = torch.dot(target_q, planes[target_tbl, bit, :])
            if dot > 0:
                hash_val |= (1 << bit)
        
        target_slot = hash_val % self.RAM
        
        # Make the key at target_slot identical to query
        keys[target_tbl, target_slot] = target_q
        target_val = torch.ones(self.O) * 7.0
        values[target_tbl, target_slot] = target_val
        
        threshold = 0.9
        
        # Call fast lookup
        results = cpp_loader.neural_cache_lookup_fast(
            query, keys, values, planes, valid, threshold, True
        )
        
        out, hit_mask, hit_addrs = results
        
        self.assertTrue(hit_mask[target_b].item())
        torch.testing.assert_close(out[target_b], target_val)
        self.assertEqual(hit_addrs[target_b, target_tbl].item(), target_slot)

    def test_performance_benchmark(self):
        # Large scale test
        B_large = 64
        RAM_large = 16384
        
        query = torch.randn(B_large, self.D)
        keys = torch.randn(self.TBL, RAM_large, self.D)
        values = torch.randn(self.TBL, RAM_large, self.O)
        planes = torch.randn(self.TBL, self.BITS, self.D)
        valid = torch.ones(self.TBL, RAM_large, dtype=torch.bool)
        
        # Warmup
        for _ in range(5):
            _ = cpp_loader.neural_cache_lookup_fast(
                query, keys, values, planes, valid, 0.5, True
            )
            
        start = time.time()
        iters = 20
        for _ in range(iters):
            _ = cpp_loader.neural_cache_lookup_fast(
                query, keys, values, planes, valid, 0.5, True
            )
        end = time.time()
        print(f"\nLSH Lookup Latency (B={B_large}, RAM={RAM_large}): {(end-start)/iters*1000:.2f}ms")

if __name__ == '__main__':
    unittest.main()
