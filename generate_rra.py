import torch
import os
from org import CognitiveOrganism, init_state, Config

# Constants
VOCAB_SIZE = 256 # Byte-level
DEVICE = Config.DEVICE

def load_master_checkpoint(model, path="checkpoints/latest.pt"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: '{path}'. Strict loading is enabled; no fallback checkpoint path will be used."
        )
        
    print(f">>> Loading Checkpoint: {path}")
    try:
        ckpt = torch.load(path, map_location=DEVICE)
        if 'model_state_dict' not in ckpt:
            raise RuntimeError("Checkpoint missing required key 'model_state_dict'.")
        
        # Strict checkpoint compatibility: no key sanitization fallback.
        state_dict = ckpt['model_state_dict']
        keys_to_remove = [k for k in state_dict.keys() if 'w_q' in k or 'scale_w' in k]
        if keys_to_remove:
            preview = ", ".join(keys_to_remove[:8])
            extra = "" if len(keys_to_remove) <= 8 else f" (+{len(keys_to_remove) - 8} more)"
            raise RuntimeError(
                "Checkpoint contains deprecated quantization keys; strict inference loading rejects fallback sanitization. "
                f"keys={preview}{extra}"
            )

        model_state = model.state_dict()
        missing_keys = [k for k in model_state.keys() if k not in state_dict]
        unexpected_keys = [k for k in state_dict.keys() if k not in model_state]
        if missing_keys or unexpected_keys:
            missing_preview = ", ".join(missing_keys[:8]) if missing_keys else "none"
            unexpected_preview = ", ".join(unexpected_keys[:8]) if unexpected_keys else "none"
            missing_extra = "" if len(missing_keys) <= 8 else f" (+{len(missing_keys) - 8} more)"
            unexpected_extra = "" if len(unexpected_keys) <= 8 else f" (+{len(unexpected_keys) - 8} more)"
            raise RuntimeError(
                "Checkpoint key mismatch for inference load. "
                f"missing={len(missing_keys)} [{missing_preview}{missing_extra}] "
                f"unexpected={len(unexpected_keys)} [{unexpected_preview}{unexpected_extra}]"
            )
        model.load_state_dict(state_dict, strict=True)
        model.current_phase = int(ckpt.get('current_phase', getattr(model, 'current_phase', 0)))
        model.omega = float(ckpt.get('omega', getattr(model, 'omega', 0.0)))
        model.eval()
        print(f">>> Loaded Successfully. Phase: {model.current_phase} | Omega: {model.omega:.2f}")
        return True
    except Exception as e:
        msg = f"Failed to load checkpoint '{path}': {type(e).__name__}: {e}"
        print(f">>> CHECKPOINT LOAD ERROR: {msg}")
        raise RuntimeError(msg) from e

def bytes_to_bits_tensor(bytes_tensor):
    """
    Converts [B, T] byte tensor to [B, T, 8] float bit tensor.
    """
    bit_shifts = torch.arange(7, -1, -1, device=bytes_tensor.device, dtype=torch.long)
    return ((bytes_tensor.long().unsqueeze(-1) >> bit_shifts) & 1).to(torch.float32)

def sample_byte(logits, temperature=1.0):
    """
    Samples a byte from [1, 1, 8] bit logits.
    """
    # logits shape: [1, 1, 8]
    temp = max(float(temperature), 1e-6)
    probs = torch.sigmoid(logits[0, 0] / temp)
    
    # Bernoulli sampling for each bit
    bits = torch.bernoulli(probs).to(torch.long)
    
    # Reconstruct byte from [MSB..LSB]
    weights = (2 ** torch.arange(7, -1, -1, device=bits.device, dtype=torch.long))
    return int((bits * weights).sum().item())

def generate(model, prompt_text="The concept of", max_len=512, temperature=0.7):
    print(f"\n>>> Generating with Temperature {temperature}...")
    print(f"Prompt: '{prompt_text}'")
    
    # Encode
    input_bytes = list(prompt_text.encode('utf-8'))
    input_indices = torch.tensor(input_bytes, dtype=torch.long, device=DEVICE).unsqueeze(0) # [1, T]
    
    # Convert to bits
    bits_input = bytes_to_bits_tensor(input_indices) # [1, T, 8]
    
    # Init State
    H = init_state(Config.L, Config.R, Config.WORKING_DIM, Config.C, device=DEVICE, scale=Config.INIT_SCALE)
    
    # Prefill (Process prompt)
    with torch.no_grad():
        # Feed all but last token bits to warm up state
        if bits_input.size(1) > 1:
            _, H, _, _ = model(bits_input[:, :-1], H)
            
        next_bits = bits_input[:, -1:] # [1, 1, 8]
        
        generated = input_bytes[:]
        
        print(">>> Stream: ", end='', flush=True)
        print(prompt_text, end='', flush=True)
        
        for _ in range(max_len):
            # Forward
            logits, H, _, _ = model(next_bits, H) # [1, 1, 8]
            
            # Sample next byte
            token_idx = sample_byte(logits, temperature)
            
            # Append
            generated.append(token_idx)
            
            # Decode & Print
            try:
                char = bytes([token_idx]).decode('utf-8')
                print(char, end='', flush=True)
            except Exception as e:
                print(
                    f"[DECODE_ERROR byte={token_idx} idx={len(generated)-1} type={type(e).__name__}: {e}]",
                    end='',
                    flush=True
                )
                
            # Prepare bits for next step
            next_indices = torch.tensor([[token_idx]], dtype=torch.long, device=DEVICE)
            next_bits = bytes_to_bits_tensor(next_indices)
            
            if token_idx == 0: # Null byte termination
                break
                
    try:
        return bytes(generated).decode('utf-8')
    except UnicodeDecodeError as e:
        msg = (
            "Final UTF-8 decode failed for generated byte stream. "
            f"len={len(generated)} error={e}"
        )
        print(f">>> GENERATION DECODE ERROR: {msg}")
        raise RuntimeError(msg) from e

if __name__ == "__main__":
    print(">>> RRA MASTER GENERATION SCRIPT")
    print(f">>> Device: {DEVICE}")
    
    model = CognitiveOrganism(
        input_dim=(Config.WORKING_DIM // 8) * Config.C, 
        vocab_size=VOCAB_SIZE,
        output_dim=8,
        device=DEVICE
    )
    
    load_master_checkpoint(model)
    output = generate(model, "The history of science is", max_len=600, temperature=0.6)
    print("\n\n>>> GENERATION COMPLETE")
