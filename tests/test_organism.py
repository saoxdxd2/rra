import torch
import unittest
import accelerator
from organism import CognitiveOrganism
import math

class TestCognitiveOrganism(unittest.TestCase):
    def setUp(self):
        self.B = 2
        self.L = 2
        self.R = 4
        self.input_dim = 8
        # Use small dimensions for fast testing
        self.device = torch.device('cpu')
        torch.manual_seed(42)

    def test_organism_initialization(self):
        model = CognitiveOrganism(
            input_dim=self.input_dim,
            L=self.L,
            R=self.R,
            device=self.device
        )
        self.assertTrue(isinstance(model, torch.nn.Module))

    def test_organism_forward_basic(self):
        model = CognitiveOrganism(
            input_dim=self.input_dim,
            L=self.L,
            R=self.R,
            device=self.device
        )
        
        # Test input sequence [B, T, D]
        T = 4
        x = torch.randn(self.B, T, self.input_dim, device=self.device)
        
        # Run forward pass
        out, H_next, cost, gate = model.forward(x, None)
        
        # Verify output shapes
        self.assertEqual(out.dim(), 3)
        self.assertEqual(out.shape[0], self.B)
        self.assertEqual(H_next.dim(), 5) # [B, L, R, D, C]
        self.assertEqual(H_next.shape[0], self.B)
        self.assertEqual(H_next.shape[1], self.L)
        
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(H_next).all())

if __name__ == '__main__':
    unittest.main()
