import random
import torch
import os
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple

# ==============================================================================
# CONFIGURATION & DATA STRUCTURES
# ==============================================================================

@dataclass
class MutationConfig:
    """Hyperparameters for the evolutionary process."""
    base_rate: float = 0.1
    stress_multiplier: float = 0.3
    max_intensity: float = 0.2
    thermal_penalty_weight: float = 0.25
    
class EventLogger:
    """Basic observability implementation."""
    def emit(self, event_name: str, payload: Dict[str, Any]):
        # Enterprise TODO: Hook this into structured logging (ELK/Splunk)
        # For now, deterministic stdout is the contract.
        log_entry = json.dumps({"event": event_name, **payload}, sort_keys=True)
        print(f"[GENOME_EVENT] {log_entry}")

# ==============================================================================
# COMPONENT 1: GENE STORAGE (Immutable-ish Data)
# ==============================================================================

class BrainGene:
    """
    Represents a single gene. 
    ENFORCES: Mutations must go through specific paths with logging.
    """
    def __init__(self, name, baseline, min_val=0.0, max_val=1.0, rng=None):
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self._rng = rng if rng else random.Random()
        
        # Protected State
        self._allele_1 = self._clamp(baseline + self._rng.uniform(-0.1, 0.1))
        self._allele_2 = self._clamp(baseline + self._rng.uniform(-0.1, 0.1))
        self._methylation = 1.0 

    def _clamp(self, val):
        return max(self.min_val, min(self.max_val, val))

    @property
    def expression(self):
        """Phenotype = Average of Alleles * Methylation"""
        base = (self._allele_1 + self._allele_2) / 2.0
        return base * self._methylation

    # Controlled Mutation Interface
    def apply_mutation(self, target: str, delta: float) -> float:
        """
        Apply a delta to a specific target (allele_1, allele_2, methylation).
        Returns the actual applied change (after clamping).
        """
        old_val = 0.0
        new_val = 0.0
        
        if target == 'allele_1':
            old_val = self._allele_1
            self._allele_1 = self._clamp(self._allele_1 + delta)
            new_val = self._allele_1
        elif target == 'allele_2':
            old_val = self._allele_2
            self._allele_2 = self._clamp(self._allele_2 + delta)
            new_val = self._allele_2
        elif target == 'methylation':
            old_val = self._methylation
            self._methylation = max(0.0, min(1.0, self._methylation + delta))
            new_val = self._methylation
        else:
            raise ValueError(f"Unknown mutation target: {target}")
            
        return new_val - old_val

    def state_dict(self):
        return {
            'allele_1': self._allele_1,
            'allele_2': self._allele_2,
            'methylation': self._methylation
        }

    def load_state_dict(self, state):
        self._allele_1 = state['allele_1']
        self._allele_2 = state['allele_2']
        self._methylation = state['methylation']


class GenomeState:
    """
    Holds the Data (Genes, RNG state, Metrics).
    Separated from Logic.
    """
    GENE_ROLES = {
        "learning_rate": "bdnf",
        "memory_plasticity": "creb",
        "sparsity_pressure": "fkbp5",
        "engagement_threshold": "drd2",
        "inhibition": "gaba",
        "curve_trajectory": "curve_trajectory",
        "mask_sparsity": "mask_sparsity_bias",
        "thermal_efficiency": "metabolic_efficiency",
    }

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Core State
        self.generation = 0
        self.health = 1.0
        self.best_loss = float('inf')
        
        # Genes
        self.genes = {
            'bdnf': BrainGene('BDNF', baseline=0.5, min_val=0.1, max_val=2.0, rng=self.rng),
            'creb': BrainGene('CREB', baseline=0.5, min_val=0.0, max_val=1.0, rng=self.rng),
            'drd2': BrainGene('DRD2', baseline=0.5, min_val=0.1, max_val=0.9, rng=self.rng),
            'fkbp5': BrainGene('FKBP5', baseline=0.3, min_val=0.1, max_val=5.0, rng=self.rng),
            'gaba': BrainGene('GABA', baseline=0.5, min_val=0.1, max_val=1.0, rng=self.rng),
            'curve_trajectory': BrainGene('CURVE_TRAJECTORY', baseline=0.5, min_val=0.0, max_val=1.0, rng=self.rng),
            'mask_sparsity_bias': BrainGene('MASK_SPARSITY_BIAS', baseline=0.5, min_val=0.05, max_val=0.95, rng=self.rng),
            'metabolic_efficiency': BrainGene('METABOLIC_EFFICIENCY', baseline=0.5, min_val=0.1, max_val=1.5, rng=self.rng),
        }
        
        # Static Linkage Map (Could be dynamic in future)
        self.linkage = {
            'bdnf': {'fkbp5': -0.5},
            'fkbp5': {'drd2': 0.3, 'gaba': 0.6},
            'creb': {'bdnf': 0.2},
            'gaba': {'drd2': 0.4},
            'metabolic_efficiency': {'fkbp5': -0.4, 'mask_sparsity_bias': 0.2},
            'curve_trajectory': {'mask_sparsity_bias': 0.1},
        }

    def state_dict(self):
        return {
            "generation": self.generation,
            "health": self.health,
            "best_loss": self.best_loss,
            "rng_state": self.rng.getstate(),
            "genes": {k: v.state_dict() for k, v in self.genes.items()}
        }

    def load_state_dict(self, state):
        self.generation = state["generation"]
        self.health = state["health"]
        self.best_loss = state["best_loss"]
        if "rng_state" in state:
            self.rng.setstate(state["rng_state"])
        for k, v in state["genes"].items():
            if k in self.genes:
                self.genes[k].load_state_dict(v)

# ==============================================================================
# COMPONENT 2: POLICY (Logic / Decision Making)
# ==============================================================================

class EvolutionPolicy:
    """
    Pure logic component. Decides WHAT to do, doesn't actually DO it.
    """
    def __init__(self, config: MutationConfig):
        self.config = config

    def analyze_stress(self, metrics: Dict[str, float]) -> str:
        """Categorize failure mode."""
        val_loss = metrics.get('val_loss', 1.0)
        cost_step = metrics.get('cost_step', 0.0)
        thermal_penalty = metrics.get('thermal_penalty', 0.0)
        
        if val_loss > 1.5: return "TASK_FAILURE"
        if thermal_penalty > 0.25: return "THERMAL_FAILURE"
        # Enterprise tuning: Lower threshold for energy failure to force efficiency early?
        if cost_step > 0.5: return "ENERGY_FAILURE" 
        return "STAGNATION"

    def propose_mutation(self, state: GenomeState, stress_dir: str) -> Tuple[str, float]:
        """
        Decides on a primary target and direction.
        Returns (gene_name, direction_sign).
        """
        if stress_dir == "TASK_FAILURE":
            return 'bdnf', 1.0
        elif stress_dir == "THERMAL_FAILURE":
            return 'metabolic_efficiency', 1.0
        elif stress_dir == "ENERGY_FAILURE":
            return 'fkbp5', 1.0
        else:
            # Random exploration
            target = state.rng.choice(list(state.genes.keys()))
            direction = state.rng.choice([-1.0, 1.0])
            return target, direction

    def calculate_intensity(self, state: GenomeState) -> float:
        return self.config.max_intensity * state.health

# ==============================================================================
# COMPONENT 3: ENGINE (Execution / Orchestration)
# ==============================================================================

class Genome: # Renamed from GenomeEngine to maintain compat with organism.py
    """
    The Orchestrator. 
    Integrates State, Policy, and external Model interaction.
    """
    def __init__(self, seed: Optional[int] = None):
        self.state = GenomeState(seed)
        self.policy = EvolutionPolicy(MutationConfig())
        self.logger = EventLogger()
        
        # Execution State
        self.patience_counter = 0
        self.patience_threshold = 5 
        self._best_state_dict_ram = None
        self.best_checkpoint_path = "checkpoints/genome_best.pt"

    # --- Facade Properties for Compatibility ---
    @property
    def generation(self): return self.state.generation
    @property
    def best_loss(self): return self.state.best_loss
    @property
    def genes(self): return self.state.genes
    @property
    def bdnf(self): return self.get_expression('bdnf')
    @property
    def creb(self): return self.get_expression('creb')
    @property
    def drd2(self): return self.get_expression('drd2')
    @property
    def fkbp5(self): return self.get_expression('fkbp5')
    @property
    def gaba(self): return self.get_expression('gaba')
    @property
    def curve_trajectory(self): return self.get_expression('curve_trajectory')
    @property
    def mask_sparsity_bias(self): return self.get_expression('mask_sparsity_bias')
    @property
    def metabolic_efficiency(self): return self.get_expression('metabolic_efficiency')

    def get_expression(self, name):
        if name in self.state.genes:
            expr = self.state.genes[name].expression
            if name == 'fkbp5': return min(expr, 3.0) # Legacy cap
            return expr
        return 0.5
        
    def get_by_role(self, role):
        if role in self.state.GENE_ROLES:
            return self.get_expression(self.state.GENE_ROLES[role])
        return 0.5

    # --- Execution Logic ---
    
    def _execute_mutation(self, stress_dir: str):
        """
        Executes a mutation transaction on the state.
        """
        target_name, direction = self.policy.propose_mutation(self.state, stress_dir)
        intensity = self.policy.calculate_intensity(self.state)
        
        # 1. Primary
        if target_name in self.state.genes:
            gene = self.state.genes[target_name]
            delta = self.state.rng.gauss(direction * intensity, intensity * 0.5)
            actual_delta = gene.apply_mutation('allele_1', delta)
            
            self.logger.emit("mutation_primary", {
                "gene": target_name,
                "target": "allele_1",
                "delta": actual_delta,
                "reason": stress_dir
            })
            
            # 2. Linkage
            if target_name in self.state.linkage:
                for linked_name, coeff in self.state.linkage[target_name].items():
                    if linked_name in self.state.genes:
                        lg = self.state.genes[linked_name]
                        nudge = delta * coeff
                        actual_nudge = lg.apply_mutation('allele_2', nudge)
                        
                        self.logger.emit("mutation_linked", {
                            "primary": target_name,
                            "linked": linked_name,
                            "delta": actual_nudge
                        })

        # 3. Drift (All others)
        drift_rate = self.policy.config.base_rate * 0.5
        drift_intensity = intensity * 0.5
        
        for name, gene in self.state.genes.items():
            if name != target_name:
                if self.state.rng.random() < drift_rate:
                    d = self.state.rng.gauss(0, drift_intensity)
                    gene.apply_mutation('allele_1', d)
                if self.state.rng.random() < drift_rate:
                    d = self.state.rng.gauss(0, drift_intensity)
                    gene.apply_mutation('allele_2', d)

        self.state.generation += 1

    def evolutionary_step(self, model, performance_metrics):
        """
        Main tick method called by Trainer.
        """
        assert 0.0 <= self.state.health <= 1.5, "Health invariant violated"
        
        val_loss = performance_metrics.get('val_loss', 100.0)
        thermal_penalty = max(0.0, float(performance_metrics.get('thermal_penalty', 0.0)))
        effective_loss = val_loss + self.policy.config.thermal_penalty_weight * thermal_penalty
        
        # 1. Improvement Branch
        if effective_loss < self.state.best_loss:
            self.state.best_loss = effective_loss
            self.state.health = min(1.0, self.state.health * 1.1)
            
            # Checkpoint
            if hasattr(model, 'get_state_dict_ram'):
                self._best_state_dict_ram = model.get_state_dict_ram()
            else:
                torch.save(model.state_dict(), self.best_checkpoint_path)
            
            self.patience_counter = 0
            self.logger.emit("improvement", {"loss": val_loss, "effective_loss": effective_loss, "thermal_penalty": thermal_penalty, "health": self.state.health})
            print(f">>> GENOME: Improvement! Gen {self.state.generation} | Loss: {val_loss:.4f} | Effective: {effective_loss:.4f} | Thermal: {thermal_penalty:.4f}")
            return True
            
        # 2. Stagnation Branch
        self.patience_counter += 1
        self.state.health = max(0.1, self.state.health * 0.95)
        
        active_patience = 5 if self.state.health < 0.5 else 10
        
        if self.patience_counter >= active_patience:
            stress_dir = self.policy.analyze_stress(performance_metrics)
            print(f">>> GENOME: Repurposing... ({stress_dir})")
            
            # Rollback
            try:
                if self._best_state_dict_ram is not None and hasattr(model, 'load_state_dict_ram'):
                    model.load_state_dict_ram(self._best_state_dict_ram)
                elif os.path.exists(self.best_checkpoint_path):
                    state = torch.load(self.best_checkpoint_path)
                    model.load_state_dict(state)
            except Exception as e:
                self.logger.emit("rollback_error", {"error": str(e)})
                print(f"[GENOME] Rollback Failed: {e}")

            # Mutate
            self._execute_mutation(stress_dir)
            
            # Update Phenotype
            if hasattr(model, 'update_phenotype_from_genome'):
                model.update_phenotype_from_genome()
            
            self.patience_counter = 0
            return True
            
        return False
        
    def state_dict(self):
        # We only save State. Policy is config (static or recreated).
        # Engine state (patience) is transient or should be saved? 
        # Enterprise: Save everything.
        d = self.state.state_dict()
        d['engine'] = {
            'patience_counter': self.patience_counter,
            'best_checkpoint_path': self.best_checkpoint_path
        }
        return d

    def load_state_dict(self, state):
        if 'engine' in state:
            self.patience_counter = state['engine'].get('patience_counter', 0)
            self.best_checkpoint_path = state['engine'].get('best_checkpoint_path', "checkpoints/genome_best.pt")
        self.state.load_state_dict(state)
