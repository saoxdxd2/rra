import torch
import time
from organism import CognitiveOrganism, init_state
from config import Config

def verify_one_manifold():
    print(">>> Starting One-Manifold Verification...")
    device = 'cpu'
    
    # Mock config
    Config.LGH_ENABLED = True
    Config.LGH_REPLACE_FORWARD_STACK = True
    
    model = CognitiveOrganism(input_dim=8, L=Config.L, R=Config.R).to(device)
    model.eval()
    
    B, T = 1, 16
    x = torch.randn(B, T, 8)
    H = init_state(model.L, model.R, model.d_s2, model.C, device=device)
    
    print(">>> Running forward pass...")
    start_time = time.time()
    with torch.no_grad():
        out, H_next, cost, gate = model(x, H)
    end_time = time.time()
    
    print(f">>> Forward pass complete. Time: {end_time - start_time:.4f}s")
    print(f">>> Output shape: {out.shape}")
    print(f">>> H_next shape: {H_next.shape}")
    
    # Check if imprinting is called (we need training mode)
    print(">>> Verification of Imprinting (Training Mode)...")
    model.train()
    # We need a longer sequence or multiple steps to see imprinting effects if they are probabilistic
    # But for now, we just check if it doesn't crash.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for i in range(5):
        optimizer.zero_grad()
        out, H_next, cost, gate = model(x, H)
        loss = out.abs().mean()
        loss.backward()
        optimizer.step()
        print(f"    Step {i+1} complete.")
    
    print(">>> One-Manifold Verification SUCCESSFUL.")

if __name__ == "__main__":
    try:
        verify_one_manifold()
    except Exception as e:
        import traceback
        traceback.print_exc()
        exit(1)
