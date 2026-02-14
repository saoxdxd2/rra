import unittest

import accelerator
import torch

from org import CognitiveOrganism, Config


class TestPhenotypeABI(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        torch.manual_seed(7)

    def test_gene_mapping_updates_lambda_sparsity(self):
        model = CognitiveOrganism(input_dim=8, device=self.device)

        model.governor.bdnf = 0.10
        model.governor.creb = 0.40
        model.governor.drd2 = 0.40
        model.governor.fkbp5 = 0.95
        model.update_phenotype()
        stress_lambda = float(model.lambda_sparsity)

        model.governor.bdnf = 0.95
        model.governor.creb = 0.70
        model.governor.drd2 = 0.70
        model.governor.fkbp5 = 0.10
        model.update_phenotype()
        recovery_lambda = float(model.lambda_sparsity)

        self.assertGreater(stress_lambda, recovery_lambda)

    def test_forward_scalar_abi_is_strict(self):
        model = CognitiveOrganism(input_dim=8, device=self.device)
        model.update_phenotype()
        scalars = model._build_forward_scalars()

        self.assertEqual(int(scalars.numel()), int(Config.FWD_SCALARS_MIN))
        self.assertAlmostEqual(float(scalars[-1].item()), float(Config.PHENOTYPE_ABI_VERSION), places=6)

    def test_dynamic_focus_kernel_changes_recall(self):
        if not bool(getattr(Config, "LGH_ENABLED", False)):
            self.skipTest("LGH is disabled in firmware config.")

        loader = accelerator.get_cpp_loader()
        p = torch.randn(2, 8, 64, dtype=torch.float32, device=self.device)
        manifold = torch.randn(512, 64, dtype=torch.float32, device=self.device)
        indices = torch.arange(96, dtype=torch.long, device=self.device)
        trace_a = torch.zeros(512, dtype=torch.float32, device=self.device)
        trace_b = torch.zeros(512, dtype=torch.float32, device=self.device)

        recall_a = loader.geometric_manifold_recall_dynamic_avx512(
            p, manifold, torch.empty(0), indices, trace_a,
            0.9, 0.8, 0.7, 1.2, 3.0, 0, 0.0
        )
        recall_b = loader.geometric_manifold_recall_dynamic_avx512(
            p, manifold, torch.empty(0), indices, trace_b,
            -0.9, -0.8, -0.7, 0.2, 0.5, 17, 0.0
        )

        delta = float((recall_a - recall_b).abs().mean().item())
        self.assertGreater(delta, 1e-6)


if __name__ == "__main__":
    unittest.main()
