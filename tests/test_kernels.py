import torch
import unittest
import accelerator
cpp_loader = accelerator.get_cpp_loader()
import math

class TestCognitiveKernels(unittest.TestCase):
    def setUp(self):
        self.B, self.T = 2, 4
        self.L, self.R = 3, 8
        self.D, self.C = 16, 4
        self.M = self.D * self.C
        
        self.device = torch.device('cpu')
        torch.manual_seed(42)

    def test_calculate_nis_sparsity(self):
        # Create a tensor with 50% zeros
        x = torch.ones(10, 10)
        x[:5, :] = 0.0
        
        sparsity = cpp_loader.calculate_nis_sparsity(x, 1e-5)
        self.assertAlmostEqual(sparsity, 0.5, places=5)
        
        # Test with custom threshold
        x = torch.zeros(10, 10)
        x[0, 0] = 0.05
        # With threshold 0.1, it should be 1.0 (all "zeros")
        self.assertEqual(cpp_loader.calculate_nis_sparsity(x, 0.1), 1.0)
        # With threshold 0.01, it should be 0.99
        self.assertAlmostEqual(cpp_loader.calculate_nis_sparsity(x, 0.01), 0.99)

    def test_ademamix_update_restored(self):
        # Verify ademamix_update can be called
        p = torch.randn(10, device=self.device)
        g = torch.randn(10, device=self.device)
        mf = torch.zeros(10, device=self.device)
        ms = torch.zeros(10, device=self.device)
        v = torch.ones(10, device=self.device)
        
        # Should not crash
        cpp_loader.ademamix_update(p, g, mf, ms, v, 0.001, 0.9, 0.999, 0.99, 1.0, 1e-8, 0.0, 0.0, 1)
        self.assertTrue(torch.isfinite(p).all())

    def test_fused_cognitive_step_integrity(self):
        # Minimal input for fused_cognitive_step
        x = torch.randn(self.B, self.T, self.M, device=self.device)
        H = torch.randn(self.B, self.L, self.R, self.D, self.C, device=self.device)
        mask = torch.ones(self.L, self.R, device=self.device)
        
        # Params: delays, tables, conns, decays, hw, hb
        params = [
            torch.zeros(self.L, self.R, 4, device=self.device), # delays
            torch.randn(self.L, self.R, 64, device=self.device), # tables
            torch.zeros(self.L, self.R, 4, dtype=torch.long, device=self.device), # conns
            torch.ones(self.L, self.R, self.M, device=self.device) * 0.9, # decays
            torch.randn(self.M, self.M, device=self.device), # hw
            torch.randn(self.M, device=self.device) # hb
        ]
        
        h_out = H.clone()
        y_out = torch.zeros(self.B, self.T, self.R, self.M, device=self.device)
        
        # Call kernel
        # h_cycles=1, l_cycles=1, decay=0.9, threshold=0.5, halt=0.1, tax=0.0
        results = cpp_loader.fused_cognitive_step(
            x, H, mask,
            params[0], params[1], params[2],
            params[3], params[4], params[5],
            1, 1, 0.9, 0.5, 0.1, h_out, y_out, 0.0
        )
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].shape, y_out.shape)
        self.assertEqual(results[1].shape, H.shape)
        self.assertTrue(torch.isfinite(results[0]).all())
        self.assertTrue(torch.isfinite(results[1]).all())

if __name__ == '__main__':
    unittest.main()
