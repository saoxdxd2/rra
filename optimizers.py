import torch
import torch.optim as optim
from accelerator import get_accelerator

ACCEL = get_accelerator()
cpp_loader = ACCEL.loader

# --- INTEGRATED OPTIMIZER SUITE ---
def _require_cpp_ademamix(*tensors):
    if cpp_loader is None:
        raise RuntimeError("AdEMAMix requires the cpp_loader extension (no Python fallback).")
    if not ACCEL.has('ademamix_update'):
        raise RuntimeError("AdEMAMix requires C++ op 'ademamix_update'.")
    if not ACCEL.ready('ademamix_update', *tensors):
        raise RuntimeError("AdEMAMix requires CPU tensors for C++ execution.")

def adaptive_gradient_clip(model, clip_factor=0.1, eps=1e-3):
    """
    Clips gradients based on the ratio of parameter norm to gradient norm.
    Similar to block-wise AGC.
    """
    for p in model.parameters():
        if p.grad is None: continue
        p_norm = p.detach().data.norm(2)
        g_norm = p.grad.detach().data.norm(2)
        max_g = p_norm * clip_factor + eps
        if g_norm > max_g:
            p.grad.detach().data.mul_(max_g / (g_norm + eps))

class AdEMAMix(optim.Optimizer):
    """
    AdEMAMix Optimizer with C++ Kernel Acceleration.
    Features:
    - Dual momentum (Fast/Slow lines)
    - Omega blending (alpha in paper)
    - C++ acceleration via cpp_loader_optimized
    """
    def __init__(self, params, lr=1e-4, beta1_fast=0.9, beta1_slow=0.9999, beta2=0.999, eps=1e-8, weight_decay=0.0, alpha=5.0):
        # Note: alpha=5.0 is default "Mix" factor. 
        # In train_rra.py, 'omega' was passed to step(), overriding alpha.
        # We store alpha in defaults for non-step-specified usage.
        defaults = dict(lr=lr, beta1_fast=beta1_fast, beta1_slow=beta1_slow, beta2=beta2, eps=eps, weight_decay=weight_decay, alpha=alpha)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, omega=None):
        """
        Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
            omega (float, optional): Dynamic mixing factor (alpha). If None, uses group['alpha'].
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Determine alpha/omega for this step
            current_alpha = omega if omega is not None else group['alpha']
            
            for p in group['params']:
                if p.grad is None: continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdEMAMix does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m_fast'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['m_slow'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m_fast = state['m_fast']
                m_slow = state['m_slow']
                v = state['v']
                state['step'] += 1
                
                lr = group['lr']
                beta1_fast = group['beta1_fast']
                beta1_slow = group['beta1_slow']
                beta2 = group['beta2']
                eps = group['eps']
                wd = group['weight_decay']
                 
                _require_cpp_ademamix(p, grad, m_fast, m_slow, v)
                if not (p.is_contiguous() and m_fast.is_contiguous() and m_slow.is_contiguous() and v.is_contiguous()):
                    raise RuntimeError("AdEMAMix expects contiguous parameter/state tensors.")
                grad_c = grad.contiguous()
                ACCEL.call(
                    'ademamix_update',
                    p, grad_c, m_fast, m_slow, v,
                    lr, beta1_fast, beta1_slow, beta2,
                    current_alpha, eps, wd, state['step'],
                    tensors=(p, grad_c, m_fast, m_slow, v)
                )

        return loss
