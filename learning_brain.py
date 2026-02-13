import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

from typing import Optional, Dict, Any, Tuple

def _masked_ema_update(current: torch.Tensor, new_value: torch.Tensor, gamma: float, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Performs an Exponential Moving Average update, optionally with a mask."""
    if mask is None:
        return (1.0 - gamma) * current + gamma * new_value

    keep = mask.to(dtype=current.dtype, device=current.device)
    updated = (1.0 - gamma) * current + gamma * new_value
    return keep * updated + (1.0 - keep) * current

class LearningBrain(nn.Module):
    def __init__(self, L, R, D, C, device='cpu'):
        super().__init__()
        # Ensure D is passed or used from Config if needed
        self.L, self.R, self.D, self.C = L, R, D, C
        self.device = device
        
        self._step = 0
        self._warmup_steps = float(Config.WARM_UP_STEPS)
        
        # Multi-scale update rates
        self.update_rates = nn.Parameter(torch.linspace(1.0, 0.1, L, device=device))
        
        self.knowledge_map = torch.zeros(L, R, device=device)
        self.knowledge_gamma = Config.KNOWLEDGE_GAMMA

        self.importance_every = Config.IMPORTANCE_EVERY
        self.importance_sample_ratio = Config.IMPORTANCE_RATIO
        self.gate_update_every = Config.GATE_UPDATE_EVERY
        
    def calculate_task_loss(self, out: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates Binary Cross Entropy loss for the task, handling shape mismatches."""
        if out.shape != targets.shape:
            # Safe reshaping for mismatched dimensions
            out = out.reshape(-1, out.size(-1))
            targets = targets.reshape(-1, targets.size(-1))
        return F.binary_cross_entropy_with_logits(out, targets)

    def calculate_teacher_loss(self, student_logits: torch.Tensor, teacher_probs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Distillation loss using cached bit projection matrix for efficiency."""
        B_T, D = student_logits.shape
        V = teacher_probs.shape[-1]
        
        # Create Once, Reuse Forever: Cache the bit projection matrix
        if not hasattr(self, '_bit_proj_matrix') or self._bit_proj_matrix.device != student_logits.device:
            bytes_range = torch.arange(V, device=student_logits.device)
            bits = torch.arange(7, -1, -1, device=student_logits.device)
            # Create the matrix once and register it as an internal buffer if appropriate, 
            # though here we just cache it on the instance for simplicity and speed.
            self._bit_proj_matrix = bytes_range.unsqueeze(-1).bitwise_and(2**bits).ne(0).float()
            
        teacher_bit_probs = torch.matmul(teacher_probs.view(-1, V), self._bit_proj_matrix)
        
        return F.binary_cross_entropy_with_logits(
            student_logits, 
            teacher_bit_probs,
            reduction='mean'
        )

    def calculate_unified_loss(self, L_task: torch.Tensor, L_teacher: torch.Tensor, L_stability: torch.Tensor, omega: float, thermal_penalty: float = 0.0) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Combines task, teacher, and stability losses into a single modulated objective."""
        lambda_stability = Config.SURVIVAL_WEIGHT * (1.0 - omega * 0.8)
        
        # Phase 4: Thermal Feedback - If throttled, increase sensitivity to stability/energy
        if thermal_penalty > 0:
            lambda_stability *= (1.0 + thermal_penalty)

        L_total = (1.0 - omega) * L_teacher + omega * L_task + lambda_stability * L_stability
        
        return L_total, {
            'L_task': L_task.item() if hasattr(L_task, 'item') else L_task,
            'L_teacher': L_teacher.item() if hasattr(L_teacher, 'item') else L_teacher,
            'L_stability': L_stability.item() if hasattr(L_stability, 'item') else L_stability,
            'omega': omega,
            'lambda_stability': lambda_stability,
            'thermal_penalty': thermal_penalty,
        }

    def get_usefulness_mask(self, threshold=0.1):
        return (self.knowledge_map > threshold).float()

    def update_knowledge_map(self, contribution, mask=None):
        self.knowledge_map = _masked_ema_update(
            self.knowledge_map,
            contribution,
            float(self.knowledge_gamma),
            mask=mask
        )

    def apply_temporal_scaling(self, H_new, H_old, level_idx):
        rate = self.update_rates[level_idx]
        return (1.0 - rate) * H_old + rate * H_new

    def forward_step(self, model, inp, targets, H, L_teacher=None, thermal_penalty: float = 0.0):
        out, H_next, cost_step, gate = model(inp, H, learning_brain=self)
        
        loss_task = self.calculate_task_loss(out, targets)
        
        if L_teacher is None:
            L_teacher = loss_task
        
        confusion_ratio = torch.clamp(loss_task.detach(), 0.0, Config.CONFUSION_NORM) / Config.CONFUSION_NORM
        warmup_factor = min(1.0, self._step / self._warmup_steps)
        self._step += 1
        
        dynamic_energy_weight = Config.DYNAMIC_ENERGY_SCALE * (1.0 - confusion_ratio) * warmup_factor
        
        # Phase 4: Doubled energy loss during thermal throttling
        if thermal_penalty > 0:
            dynamic_energy_weight *= (1.0 + thermal_penalty)

        loss_stability, loss_energy, loss_coherence = model.metabolism.calculate_losses(H_next, gate=gate, H_prev=H)
        loss_myelin = model.metabolism.calculate_myelin_cost(model)
        
        omega = getattr(model, 'omega', 0.0)
        combined_stability = loss_stability + Config.COHERENCE_WEIGHT * loss_coherence + loss_myelin
        
        total_loss, loss_info = self.calculate_unified_loss(
            L_task=loss_task, 
            L_teacher=L_teacher, 
            L_stability=combined_stability, 
            omega=omega,
            thermal_penalty=thermal_penalty
        )
        
        total_loss = total_loss + (dynamic_energy_weight * (cost_step + loss_energy))
        
        if hasattr(model, 'get_engagement_rate'):
            engagement_rate = model.get_engagement_rate()
            efficiency_bonus = min((1.0 - engagement_rate) * 0.1, Config.EFFICIENCY_BONUS_CAP)
            total_loss = total_loss - efficiency_bonus
        
        if hasattr(model, 'virtual_lab') and model.virtual_lab.enabled:
            model.virtual_lab.log_step({
                'loss_task': loss_task,
                'loss_energy': loss_energy,
                'cost_step': cost_step,
                'mask': gate,
                't': inp.size(1) if inp.dim() > 1 else 1
            })

        if hasattr(model, '_last_cache_bits') and model._last_cache_bits is not None:
            s1_pred = (model._last_cache_bits > 0.5).float()
            s2_pred = (out[:, -1] > 0).float()
            dissonance = (s1_pred != s2_pred).any(dim=1).float().mean()
            if dissonance > 0:
                loss_task = loss_task * (1.0 + dissonance)

        metrics = {
            'loss_task': loss_task,
            'total_loss': total_loss,
            'cost_step': cost_step,
            'loss_stability': loss_stability,
            'loss_energy': loss_energy,
            'loss_coherence': loss_coherence,
            'loss_myelin': loss_myelin,
            'gate': gate
        }
        
        return total_loss, H_next, metrics

    def learning_step(self, model, inp, targets, H, optimizer, L_teacher=None, loss_contribution_tracker=None, gate_optimizer=None, omega=None, thermal_penalty: float = 0.0):
        step = int(self._step)
        importance_needed = (loss_contribution_tracker is not None and (step % int(self.importance_every) == 0))
        hook_handle = None
        self.last_H_grad = None

        if importance_needed:
            if not H.requires_grad:
                H = H.detach().requires_grad_(True)
            def save_grad_hook(grad):
                self.last_H_grad = grad
            hook_handle = H.register_hook(save_grad_hook)

        optimizer.zero_grad()
        total_loss, H_next, metrics = self.forward_step(model, inp, targets, H, L_teacher=L_teacher, thermal_penalty=thermal_penalty)
        if not torch.isfinite(total_loss):
            optimizer.zero_grad(set_to_none=True)
            return {'loss_task': float('nan'), 'total_loss': float('nan'), 'H_next': H_next.detach()}
        
        total_loss.backward()
        
        if importance_needed:
            if self.last_H_grad is not None:
                importance = (self.last_H_grad.abs() * H.abs()).mean(dim=(0, 3, 4))
                loss_contribution_tracker.update(importance)
                self.update_knowledge_map(importance)
            if hook_handle:
                hook_handle.remove()

        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step(omega=omega) if hasattr(optimizer.step, '__code__') and 'omega' in optimizer.step.__code__.co_varnames else optimizer.step()
        
        return {
            'loss_task': metrics['loss_task'].item(),
            'total_loss': metrics['total_loss'].item(),
            'H_next': H_next.detach()
        }

class LossContributionTracker:
    def __init__(self, L, R, gamma=0.01, device='cpu'):
        self.contribution = torch.zeros(L, R, device=device)
        self.gamma = gamma
    def update(self, new_grads, mask=None):
        self.contribution = _masked_ema_update(self.contribution, new_grads, float(self.gamma), mask=mask)
