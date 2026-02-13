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
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return
    
    # Vectorized norm calculation (much faster than Python loop)
    p_norms = torch._foreach_norm(params, 2)
    g_norms = torch._foreach_norm([p.grad for p in params], 2)
    
    for p, p_norm, g_norm in zip(params, p_norms, g_norms):
        max_g = p_norm * clip_factor + eps
        if g_norm > max_g:
            p.grad.detach().data.mul_(max_g / (g_norm + eps))

def sign_gradients(model):
    """
    Reduces gradients to signs immediately to save cache bandwidth.
    Aligns with int8 plan for ram_tables.
    """
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach().copy_(p.grad.sign())

class AdEMAMix(optim.Optimizer):
    """
    AdEMAMix Optimizer with C++ Kernel Acceleration.
    Features:
    - Dual momentum (Fast/Slow lines)
    - Omega blending (alpha in paper)
    - C++ acceleration via cpp_loader_optimized
    """
    def __init__(self, params, lr=1e-4, beta1_fast=0.9, beta1_slow=0.9999, beta2=0.999, eps=1e-8, weight_decay=0.0, alpha=5.0, sign_sgd=False):
        # Note: alpha=5.0 is default "Mix" factor. 
        # In train_rra.py, 'omega' was passed to step(), overriding alpha.
        # We store alpha in defaults for non-step-specified usage.
        defaults = dict(lr=lr, beta1_fast=beta1_fast, beta1_slow=beta1_slow, beta2=beta2, eps=eps, weight_decay=weight_decay, alpha=alpha, sign_sgd=sign_sgd)
        super().__init__(params, defaults)
        self.sign_sgd = sign_sgd

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
            current_alpha = float(omega if omega is not None else group['alpha'])
            lr, bf, bs, b2, eps, wd = group['lr'], group['beta1_fast'], group['beta1_slow'], group['beta2'], group['eps'], group['weight_decay']
            
            p_b, g_b, mf_b, ms_b, v_b = [], [], [], [], []
            last_step = 0
            
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.detach()
                if group.get('sign_sgd', False): grad = grad.sign()
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m_fast'] = torch.zeros_like(p)
                    state['m_slow'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                p_b.append(p)
                g_b.append(grad.contiguous())
                mf_b.append(state['m_fast'])
                ms_b.append(state['m_slow'])
                v_b.append(state['v'])
                state['step'] += 1
                last_step = state['step']

            if p_b:
                # Command 6 = CMD_GROUP_ADEMAMIX (or use batched_ademamix_update directly)
                # But here we use batched_ademamix_update as before, just with metabolic_tax
                metabolic_tax = float(getattr(group, 'metabolic_tax', 0.0001))
                ACCEL.call('batched_ademamix_update', p_b, g_b, mf_b, ms_b, v_b, lr, bf, bs, b2, current_alpha, eps, wd, metabolic_tax, last_step, tensors=tuple(p_b + g_b + mf_b + ms_b + v_b))

        return loss
