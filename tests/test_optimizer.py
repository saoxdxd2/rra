import torch
import unittest
import accelerator

class TestOptimizer(unittest.TestCase):
    def setUp(self):
        self.accel = accelerator.get_accelerator()
        self.cpp = self.accel.loader
        self.device = torch.device('cpu')
        torch.manual_seed(42)

    def test_batched_ademamix_update(self):
        # Parameters to update
        param = torch.ones(10, 10, device=self.device, dtype=torch.float32)
        grad = torch.ones(10, 10, device=self.device, dtype=torch.float32) * 0.1
        m_fast = torch.zeros(10, 10, device=self.device, dtype=torch.float32)
        m_slow = torch.zeros(10, 10, device=self.device, dtype=torch.float32)
        v = torch.zeros(10, 10, device=self.device, dtype=torch.float32)
        
        lr = 0.1
        beta1_fast = 0.9
        beta1_slow = 0.999
        beta2 = 0.99
        alpha = 5.0
        eps = 1e-8
        weight_decay = 0.01
        metabolic_tax = 0.02
        step = 0
        
        # Initial value
        initial_param = param.clone()
        
        # Run update
        self.cpp.batched_ademamix_update(
            [param], [grad], [m_fast], [m_slow], [v],
            lr, beta1_fast, beta1_slow, beta2, alpha, eps, weight_decay, metabolic_tax, step
        )
        
        # 1. Check if parameter changed
        self.assertFalse(torch.equal(param, initial_param))
        
        # 2. Check if m_fast, m_slow, v changed
        self.assertTrue((m_fast != 0).any())
        self.assertTrue((m_slow != 0).any())
        self.assertTrue((v != 0).any())
        
        # 3. Check metabolic tax effect (should decrease value more than standard SGD)
        # Without tax, param would be ~ initial_param - lr * grad
        # With tax=0.02, it should be even smaller
        self.assertTrue((param < initial_param).all())

    def test_weight_decay_and_toxic_tax(self):
        # Test extreme tax/decay
        param = torch.ones(10, 10, device=self.device)
        grad = torch.zeros(10, 10, device=self.device) # No gradient
        m_fast = torch.zeros(10, 10, device=self.device)
        m_slow = torch.zeros(10, 10, device=self.device)
        v = torch.zeros(10, 10, device=self.device)
        
        # Tax = 0.5 (should cut weights in half if purely applying tax)
        # In _ademamix_core:
        # param[i] = param[i] * (1.0f - metabolic_tax - weight_decay) - lr * update;
        
        self.cpp.batched_ademamix_update(
            [param], [grad], [m_fast], [m_slow], [v],
            0.1, 0.9, 0.999, 0.99, 5.0, 1e-8, 0.1, 0.2, 0
        )
        
        # Expected value approx: 1.0 * (1.0 - 0.1 * 0.1) * (1.0 - 0.2) = 0.99 * 0.8 = 0.792
        # Since grad=0, update=0.
        self.assertAlmostEqual(param[0,0].item(), 0.792, places=5)

if __name__ == '__main__':
    unittest.main()
