import torch
import torch.nn.functional as F
import os
import time
from organism import CognitiveOrganism, init_state, Config

# Constants
VOCAB_SIZE = 256 # Byte-level
DEVICE = Config.DEVICE

def load_master_checkpoint(model, path="checkpoints/latest.pt"):
    if not os.path.exists(path):
        print(f"Checkpoint {path} not found. Trying 'checkpoints/latest.pt'...")
        path = "checkpoints/latest.pt"
    
    if not os.path.exists(path):
        print("No checkpoint found!")
        return False
        
    print(f">>> Loading Checkpoint: {path}")
    try:
        ckpt = torch.load(path, map_location=DEVICE)
        
        # Sanitization (Same as train_rra.py just in case)
        state_dict = ckpt['model_state_dict']
        keys_to_remove = [k for k in state_dict.keys() if 'w_q' in k or 'scale_w' in k]
        if keys_to_remove:
            print(f"Sanitizer: Removed {len(keys_to_remove)} quantization keys.")
            for k in keys_to_remove:
                del state_dict[k]
                
        model.load_state_dict(state_dict, strict=False)
        model.current_phase = int(ckpt.get('current_phase', getattr(model, 'current_phase', 0)))
        model.omega = float(ckpt.get('omega', getattr(model, 'omega', 0.0)))
        model.eval()
        print(f">>> Loaded Successfully. Phase: {model.current_phase} | Omega: {model.omega:.2f}")
        return True
    except Exception as e:
        print(f"Failed to load: {e}")
        return False

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
            except:
                # Handle potential multi-byte characters partially printed
                pass 
                
            # Prepare bits for next step
            next_indices = torch.tensor([[token_idx]], dtype=torch.long, device=DEVICE)
            next_bits = bytes_to_bits_tensor(next_indices)
            
            if token_idx == 0: # Null byte termination
                break
                
    # Use replace to handle potential decoding errors at the end
    return bytes(generated).decode('utf-8', errors='replace')

if __name__ == "__main__":
    print(">>> RRA MASTER GENERATION SCRIPT")
    print(f">>> Device: {DEVICE}")
    
    model = CognitiveOrganism(
        input_dim=(Config.WORKING_DIM // 8) * Config.C, 
        L=Config.L, 
        R=Config.R, 
        d_s1=(Config.WORKING_DIM // 8), 
        d_s2=Config.WORKING_DIM,
        vocab_size=VOCAB_SIZE,
        output_dim=VOCAB_SIZE,
        device=DEVICE
    )
    
    if load_master_checkpoint(model):
        output = generate(model, "The history of science is", max_len=600, temperature=0.6)
        print("\n\n>>> GENERATION COMPLETE")
