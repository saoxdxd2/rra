import torch
import unittest
import accelerator
import time

# Force heavy reload or just trust the system?
# We assume the environment handles rebuilds if setups are correct.
# If not, we might need to trigger it.

class TestLGHManifoldAVX(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.B, self.T, self.M = 4, 16, 256
        self.N = 1024 # Manifold size
        self.S = 32 # Curve length
        
    def test_kernel_existence(self):
        loader = accelerator.get_cpp_loader()
        self.assertTrue(hasattr(loader, 'geometric_manifold_forward_avx512'), "Kernel not found in cpp_loader!")

    def test_forward_recall(self):
        loader = accelerator.get_cpp_loader()
        if not hasattr(loader, 'geometric_manifold_forward_avx512'):
            return

        p_brain = torch.randn(self.B, self.T, self.M, device=self.device)
        manifold = torch.randn(self.N, self.M, device=self.device)
        
        # Random indices
        indices = torch.randint(0, self.N, (self.S,), dtype=torch.long, device=self.device)
        
        # Test 1: Float32 Path
        scales = torch.empty(0, device=self.device) # Empty for float
        focus = 2.0
        
        start = time.time()
        recall = loader.geometric_manifold_forward_avx512(p_brain, manifold, scales, indices, focus)
        end = time.time()
        
        # Verify shape (Optimized: returns [1, 1, M] for broadcasting)
        self.assertEqual(recall.shape, (1, 1, self.M))
        
        # Verify values (should be non-zero and related to manifold)
        self.assertNotEqual(recall.abs().max().item(), 0.0)
        
        # Manual check
        expected = manifold[indices].mean(dim=0) * focus
        # With small float errors, should be close
        # Note: Kernel does averaging? Yes, I implemented normalization by S.
        
        # My kernel implementation logic:
        # for q in queries: accumulator += manifold[indices]; output = accumulator * (1/S * focus)
        # So it broadcasts the mean of the curve to all queries?
        # Re-check kernel: 
        # Yes, "Copy accumulator to output (recall)". 
        # And "Accumulate manifold curve" is done once per query?
        # Actually I put the accumulation INSIDE the query loop, which is redundant if it doesn't depend on query.
        # But that's fine for correctness.
        
        diff = (recall[0,0] - expected).abs().max()
        print(f"Max Diff FP32: {diff.item()}")
        self.assertTrue(diff < 1e-4)

    def test_int8_path(self):
        loader = accelerator.get_cpp_loader()
        if not hasattr(loader, 'geometric_manifold_forward_avx512'):
            return
            
        p_brain = torch.randn(self.B, self.T, self.M, device=self.device)
        # Mock int8 manifold
        manifold = torch.randint(-127, 127, (self.N, self.M), dtype=torch.int8, device=self.device)
        scales = torch.rand(self.N, device=self.device)
        indices = torch.randint(0, self.N, (self.S,), dtype=torch.long, device=self.device)
        focus = 1.0
        
        recall = loader.geometric_manifold_forward_avx512(p_brain, manifold, scales, indices, focus)
        
        self.assertEqual(recall.shape, (1, 1, self.M))
        self.assertNotEqual(recall.abs().max().item(), 0.0)
        print("INT8 Kernel Test Passed")

if __name__ == '__main__':
    unittest.main()
