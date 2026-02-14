"""
=============================================================================
  Organism Benchmark & Complexity Profiler
  ----------------------------------------
  Benchmarks every major function in org.py:
    - Execution Speed (avg/min/max ms)
    - Estimated TPS (Tokens Per Second)
    - Cyclomatic Complexity (AST-based, automatic)
    - Memory delta (RSS or CUDA)
=============================================================================
"""
import unittest
import torch
import torch.nn as nn
import time
import ast
import inspect
import sys
import os
import psutil
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np

# Adjust path to import org
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# Mock C++ Backend (used ONLY when real extension is unavailable/incomplete)
# =============================================================================

# Now safe to import org
try:
    import org as organism
    from org import CognitiveOrganism, Governor, VirtualLab, Config, init_state
except ImportError as e:
    print(f"FATAL: Could not import org: {e}")
    sys.exit(1)




# =============================================================================
# 1. Cyclomatic Complexity Analyzer (AST-based, zero dependencies)
# =============================================================================

class ComplexityVisitor(ast.NodeVisitor):
    """Counts decision points to compute McCabe Cyclomatic Complexity."""
    def __init__(self):
        self.complexity = 1  # Base path

    def visit_If(self, node):       self.complexity += 1; self.generic_visit(node)
    def visit_For(self, node):      self.complexity += 1; self.generic_visit(node)
    def visit_While(self, node):    self.complexity += 1; self.generic_visit(node)
    def visit_With(self, node):     self.complexity += 1; self.generic_visit(node)
    def visit_Try(self, node):      self.complexity += 1; self.generic_visit(node)
    def visit_ExceptHandler(self, node): self.complexity += 1; self.generic_visit(node)
    def visit_BoolOp(self, node):
        self.complexity += len(node.values) - 1
        self.generic_visit(node)


def get_cyclomatic_complexity(func):
    """Return integer CC or '?' on failure."""
    try:
        if hasattr(func, '__wrapped__'):
            func = func.__wrapped__
        src = inspect.getsource(func)
        lines = src.split('\n')
        indent = 0
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                break
        src = '\n'.join(line[indent:] for line in lines)
        tree = ast.parse(src)
        v = ComplexityVisitor()
        v.visit(tree)
        return v.complexity
    except Exception:
        return "?"


# =============================================================================
# 2. Benchmarking Framework
# =============================================================================

@dataclass
class BenchmarkResult:
    name: str
    complexity: Any
    avg_time_ms: float
    std_dev_ms: float
    min_time_ms: float
    max_time_ms: float
    tps_est: float
    memory_delta_mb: float


class BenchmarkSuite:
    def __init__(self, warmup=3, iterations=20):
        self.warmup = warmup
        self.iterations = iterations
        self.results: List[BenchmarkResult] = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Benchmark Device: {self.device}")

    def measure(self, name, func, *args, tps_factor=1.0, **kwargs):
        # --- Complexity ---
        complexity = "?"
        try:
            complexity = get_cyclomatic_complexity(func)
            if complexity == "?" and hasattr(func, '__func__'):
                complexity = get_cyclomatic_complexity(func.__func__)
        except Exception:
            pass

        # --- Memory Baseline ---
        proc = psutil.Process(os.getpid())
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            mem_before = proc.memory_info().rss / 1024 / 1024

        # --- Warmup ---
        try:
            for _ in range(self.warmup):
                func(*args, **kwargs)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
        except Exception as e:
            print(f"  SKIP  {name}: {e}")
            return None

        # --- Timing ---
        times = []
        for _ in range(self.iterations):
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            func(*args, **kwargs)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        if self.device.type == 'cuda':
            mem_after = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            mem_after = proc.memory_info().rss / 1024 / 1024

        avg = np.mean(times)
        res = BenchmarkResult(
            name=name,
            complexity=complexity,
            avg_time_ms=avg,
            std_dev_ms=np.std(times),
            min_time_ms=np.min(times),
            max_time_ms=np.max(times),
            tps_est=(1000.0 / avg) * tps_factor if avg > 0 else 0,
            memory_delta_mb=mem_after - mem_before,
        )
        self.results.append(res)
        return res

    def print_report(self):
        hdr = (
            f"{'FUNCTION':<45} | {'CC':<4} | {'AVG ms':<9} | {'STD ms':<9} "
            f"| {'MIN ms':<9} | {'MAX ms':<9} | {'TPS':<11} | {'MEM MB':<9}"
        )
        sep = "-" * len(hdr)
        print(f"\n{'=' * len(hdr)}")
        print(hdr)
        print(sep)
        for r in self.results:
            print(
                f"{r.name:<45} | {str(r.complexity):<4} | {r.avg_time_ms:<9.3f} "
                f"| {r.std_dev_ms:<9.3f} | {r.min_time_ms:<9.3f} | {r.max_time_ms:<9.3f} "
                f"| {r.tps_est:<11.1f} | {r.memory_delta_mb:<9.2f}"
            )
        print(f"{'=' * len(hdr)}\n")


# =============================================================================
# 3. Organism Profiler
# =============================================================================

class OrganismProfiler:
    def __init__(self):
        self.suite = BenchmarkSuite()
        self.device = self.suite.device

        # Use BIOS dimensions for consistency with C++ kernel
        self.B, self.T = 1, 16 # Keep batch small for speed
        self.L, self.R = Config.L, Config.R
        self.D, self.C = Config.WORKING_DIM, Config.C
        
        # Override constants for benchmark if needed (already set in Config)

        print("Instantiating CognitiveOrganism...")
        self.organism = CognitiveOrganism(
            input_dim=8,  # bit-level
            L=self.L, R=self.R, D=self.D, C=self.C,
            device=self.device,
        )
        self.organism.eval()
        self.gov = self.organism.governor
        print("Organism ready.\n")

    # -----------------------------------------------------------------
    def run(self):
        B, T, D, C = self.B, self.T, self.D, self.C
        dev = self.device
        org = self.organism
        suite = self.suite

        print(f">>> Benchmark Suite  [B={B}  T={T}  Device={dev}]\n")

        # ---- 1. Governor.forward ----
        x_gov = torch.randn(B, self.gov.shared[0].in_features, device=dev)
        suite.measure("Governor.forward", self.gov.forward, x_gov, tps_factor=B)

        # ---- 2. Governor.update_metabolism ----
        H_act = torch.randn(B, self.L, self.R, D, C, device=dev)
        suite.measure("Governor.update_metabolism", self.gov.update_metabolism, H_act)

        # ---- 3. Governor.get_expression ----
        suite.measure("Governor.get_expression('bdnf')", self.gov.get_expression, 'bdnf')

        # ---- 4. Governor.evolutionary_step ----
        suite.measure(
            "Governor.evolutionary_step",
            self.gov.evolutionary_step, org, {'val_loss': 1.0, 'thermal_penalty': 0.0}
        )

        # ---- 5. SwiGLU ----
        swiglu = organism.SwiGLU(dim=64, out_dim=64).to(dev)
        suite.measure("SwiGLU.forward", swiglu.forward, torch.randn(B, 64, device=dev), tps_factor=B)

        # ---- 6. RotaryEmbedding ----
        rope = organism.RotaryEmbedding(dim=32, device=dev)
        suite.measure("RotaryEmbedding.forward(T=32)", rope.forward, 32, dev)

        # ---- 7. VNNILinear ----
        vnni = organism.VNNILinear(64, 64, device=dev)
        vnni.eval()
        suite.measure("VNNILinear.forward", vnni.forward, torch.randn(B, 64, device=dev), tps_factor=B)

        # ---- 8. VNNILinear.quantize ----
        suite.measure("VNNILinear.quantize", vnni.quantize)

        # ---- 9. _encode_s1_input ----
        x_bits = torch.randint(0, 2, (B, T, 8), dtype=torch.float32, device=dev)
        suite.measure("Organism._encode_s1_input", org._encode_s1_input, x_bits, tps_factor=B * T)

        # ---- 10. update_phenotype ----
        suite.measure("Organism.update_phenotype", org.update_phenotype)

        # ---- 11. _compute_dyn_halt_threshold ----
        suite.measure("Organism._compute_dyn_halt_threshold", org._compute_dyn_halt_threshold)

        # ---- 12. _dynamic_cycle_counts ----
        suite.measure("Organism._dynamic_cycle_counts", org._dynamic_cycle_counts)

        # ---- 13. Full forward (Inference) ----
        H0 = init_state(self.L, self.R, D, C, device=dev).unsqueeze(0).expand(B, -1, -1, -1, -1).contiguous()
        suite.measure("Organism.forward (Inference)", org.forward, x_bits, H0, tps_factor=B * T)

        # ---- 14. MES Step ----
        org.train()
        tgt = torch.randint(0, 2, (B, T, 8), dtype=torch.float32, device=dev)
        suite.measure("Organism.mes_step (Training)", org.mes_step, x_bits, tgt, tps_factor=B * T)
        org.eval()

        # ---- 15. LGH Manifold Recall ----
        if getattr(org, 'lgh_manifold_morton', None) is not None:
            p_brain = torch.randn(B, T, D * C, device=dev)
            suite.measure("Organism._lgh_manifold_recall", org._lgh_manifold_recall, p_brain, tps_factor=B * T)

        # ---- 16. _quantize_lgh_manifold ----
        if getattr(org, 'lgh_manifold_morton', None) is not None:
             suite.measure("Organism._quantize_lgh_manifold", org._quantize_lgh_manifold, 8)

        # ---- 17. NIS Fast Path (CMD_FULL_FORWARD) ----
        if dev.type == 'cpu':
             # Force preflight for benchmark
             org._preflight_ready = True
             org.eval()
             org.training = False
             # Ensure params are packed
             org._pack_forward_params()
             
             x_fwd = torch.randn(B, T, 8, device=dev)
             H_fwd = torch.randn(B, self.L, self.R, D, C, device=dev)
             
             # Verify dispatch availability
             try:
                 acc = organism._ACCEL
                 if acc.has('unified_dispatch_io'):
                      suite.measure("Organism.forward (NIS Fast Path)", org.forward, x_fwd, H_fwd, tps_factor=B*T)
                 else:
                      print(">>> SKIP: unified_dispatch_io not available for NIS benchmark.")
             except Exception as e:
                 print(f">>> SKIP NIS Benchmark: {e}")

        # ---- 18. Missing Components Coverage ----
        # S1 Encoding
        suite.measure("Organism._encode_s1_input", org._encode_s1_input, torch.randn(B, T, 8, device=dev), tps_factor=B*T)
        
        # State Prep
        suite.measure("Organism._prepare_state", org._prepare_state, torch.randn(B, self.L, self.R, D, C, device=dev), B)
        
        # Cycle Controls
        suite.measure("Organism._dynamic_cycle_counts", org._dynamic_cycle_counts)
        suite.measure("Organism._compute_dyn_halt_threshold", org._compute_dyn_halt_threshold, 0.5)
        
        # Bridge
        p_s1 = torch.randn(B, T, org.d_s1*C, device=dev)
        suite.measure("Organism.bridge_s1_to_s2", org.bridge_s1_to_s2, p_s1, tps_factor=B*T)

        suite.print_report()

def print_complexity_report():
    print("\n" + "="*80)
    print("CYCLOMATIC COMPLEXITY AUDIT (Static Analysis)")
    print("="*80)
    print(f"{'Method/Function':<50} | {'CC':<5} | {'Status':<10}")
    print("-" * 75)
    
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'org.py')
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    try:
        tree = ast.parse(source)
    except Exception as e:
        print(f"Error parsing org.py: {e}")
        return

    methods = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            complexity = 1
            for child in ast.walk(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.Assert, 
                                    ast.ExceptHandler, ast.With, ast.Try)):
                    complexity += 1
                elif isinstance(child, ast.BoolOp):
                    complexity += len(child.values) - 1
            
            # Context
            name = node.name
            if isinstance(node.parent, ast.ClassDef) if hasattr(node, 'parent') else False:
                 name = f"{node.parent.name}.{name}"
            
            methods.append((name, complexity))
            
    # Sort by complexity desc
    methods.sort(key=lambda x: x[1], reverse=True)
    
    for name, score in methods:
        status = "🟢"
        if score > 10: status = "🟡" 
        if score > 20: status = "🔴"
        if score > 50: status = "⛔"
        print(f"{name:<50} | {score:<5} | {status:<10}")
    print("="*80 + "\n")


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == '__main__':
    # Parse args for --complexity
    if '--complexity' in sys.argv:
        print_complexity_report()
        sys.argv.remove('--complexity')

    # Run benchmarks if not explicitly disabled
    if '--no-bench' not in sys.argv:
        profiler = OrganismProfiler()
        profiler.run()
    else:
        sys.argv.remove('--no-bench')
        unittest.main() # Run any unittests if defined (class BenchmarkSuite doesn't inherit TestCase correctly?)

