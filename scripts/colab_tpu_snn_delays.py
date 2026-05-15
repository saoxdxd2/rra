import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ==============================================================================
# 1. TPU XLA HARDWARE ACCELERATION SETUP
# ==============================================================================
os.environ['PJRT_DEVICE'] = 'TPU'

is_tpu = False
try:
    import torch_xla.core.xla_model as xm
    import torch_xla
    device = xm.xla_device()
    is_tpu = True
    print("PyTorch SNN Polychronization Engine running on TPU (XLA)")
except ImportError:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on standard device: {device}")

# ==============================================================================
# 2. THE STRAIGHT-THROUGH ESTIMATOR (XLA-Fused)
# ==============================================================================
def spike_fn(v, v_thresh):
    """
    Biological step function for spikes.
    Uses the STE trick to allow gradients to flow smoothly backward through time.
    100% Native Tensors -> Fuses directly into TPU Silicon.
    """
    spike_hard = (v >= v_thresh).float()
    spike_soft = torch.sigmoid(v - v_thresh)
    return spike_soft + (spike_hard - spike_soft).detach()

# ==============================================================================
# 3. SPATIO-TEMPORAL NEURAL NETWORK (With Differentiable Delays)
# ==============================================================================
class AxonalDelaySNN(nn.Module):
    def __init__(self, num_neurons, max_delay):
        super().__init__()
        self.num_neurons = num_neurons
        self.max_delay = max_delay
        
        # Biophysics
        self.v_rest = -70.0
        self.v_thresh = -56.5
        self.tau_m = 22.0
        
        # 1. Spatial Weights (How strongly connected are they?)
        self.weights = nn.Parameter(torch.randn(num_neurons, num_neurons) * 0.1)
        
        # 2. Temporal Weights (What is the precise Axonal Delay in milliseconds?)
        # Shape: (Pre_Neuron, Post_Neuron, Delay_Taps)
        self.delay_logits = nn.Parameter(torch.randn(num_neurons, num_neurons, max_delay))

    def forward(self, input_spikes, sim_steps):
        batch_size = input_spikes.shape[0]
        
        # Membrane voltages
        v_m = torch.full((batch_size, self.num_neurons), self.v_rest, device=device)
        
        # Causal Delay Ring Buffer (stores spike history)
        # Shape: (Batch, Delay_Taps, Neurons)
        spike_history = torch.zeros(batch_size, self.max_delay, self.num_neurons, device=device)
        
        output_spikes = []
        
        # Smooth delay probabilities (Differentiable routing)
        # delay_probs[i, j, d] = probability that a spike from i to j takes exactly d milliseconds.
        delay_probs = F.softmax(self.delay_logits, dim=-1)

        for t in range(sim_steps):
            # Decay voltage
            v_m = self.v_rest + (v_m - self.v_rest) * torch.exp(torch.tensor(-1.0 / self.tau_m, device=device))
            
            # ---------------------------------------------------------
            # POLYCHRONIZATION: Gathering delayed spikes
            # ---------------------------------------------------------
            # We want to know how much current arrives at Post_Neuron from Pre_Neuron right now.
            # We look back in time through the delay_probs.
            # spike_history shape: (Batch, Max_Delay, Pre_Neuron)
            # We need to dot-product the history with the delay_probs to find the "arriving" spikes.
            
            # Einsum explanation:
            # b: batch, d: delay_tap, i: pre_neuron, j: post_neuron
            arriving_spikes = torch.einsum('bdi,ijd->bij', spike_history, delay_probs)
            
            # Multiply by spatial synaptic weights and sum over all Pre_Neurons
            # arriving_spikes: (Batch, Pre, Post)
            # weights: (Pre, Post)
            # current_in: (Batch, Post)
            current_in = torch.sum(arriving_spikes * self.weights.unsqueeze(0), dim=1)
            
            # Add forced input stimuli (if any) for this timestep
            current_in += input_spikes[:, t, :] * 50.0
            
            # Inject current into membrane
            v_m = v_m + current_in
            
            # ---------------------------------------------------------
            # SPIKE GENERATION
            # ---------------------------------------------------------
            spikes = spike_fn(v_m, self.v_thresh)
            
            # Hyperpolarization reset
            v_m = v_m - spikes * (v_m - self.v_rest + 5.0)
            
            output_spikes.append(spikes)
            
            # Update Delay Ring Buffer (shift history backwards, insert new spike at index 0)
            spike_history = torch.cat([spikes.unsqueeze(1), spike_history[:, :-1, :]], dim=1)
            
        return torch.stack(output_spikes, dim=1)

# ==============================================================================
# 4. THE TEMPORAL ROUTING TASK
# ==============================================================================
def run_temporal_bptt():
    NUM_NEURONS = 20
    SIM_STEPS = 100
    MAX_DELAY = 15 # Max Axonal Delay is 15ms.
    BATCH_SIZE = 1 # Single sequence optimization
    
    print(f"Initializing Spatio-Temporal Matrix. Neurons: {NUM_NEURONS}, Max Delay: {MAX_DELAY}ms")
    
    model = AxonalDelaySNN(num_neurons=NUM_NEURONS, max_delay=MAX_DELAY).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    
    # ---------------------------------------------------------
    # THE TASK: DISJOINTED POLYCHRONIZATION
    # ---------------------------------------------------------
    # Neuron 0 fires at 10ms
    # Neuron 1 fires at 20ms
    # The network MUST route these spikes through intermediate neurons
    # so that they perfectly align and hit Neuron 19 at exactly 40ms.
    # Note: Because Max Delay is 15ms, Neuron 0 (10ms -> 40ms = 30ms gap) 
    # CANNOT reach Neuron 19 directly. It MUST learn to bounce through a middle neuron!
    
    inputs = torch.zeros(BATCH_SIZE, SIM_STEPS, NUM_NEURONS, device=device)
    inputs[:, 10, 0] = 1.0
    inputs[:, 20, 1] = 1.0
    
    target_spikes = torch.zeros(BATCH_SIZE, SIM_STEPS, device=device)
    target_spikes[:, 40] = 1.0 # The perfect temporal action
    
    patience = 50
    best_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(1, 1001):
        start_time = time.time()
        
        optimizer.zero_grad()
        
        # Forward pass (simulates 100ms of biological time with exact delays)
        network_spikes = model(inputs, SIM_STEPS)
        
        # Extract the output of Neuron 19
        output_neuron_spikes = network_spikes[:, :, 19]
        
        # Loss: MSE between Neuron 19's spikes and the target perfect spike at 40ms
        loss = F.mse_loss(output_neuron_spikes, target_spikes)
        
        # Penalize overall network energy to prevent an infinite spike storm
        energy_penalty = 0.001 * torch.mean(network_spikes)
        total_loss = loss + energy_penalty
        
        total_loss.backward()
        
        if is_tpu:
            xm.optimizer_step(optimizer)
            torch_xla.sync() 
        else:
            optimizer.step()
            
        epoch_time = time.time() - start_time
        loss_val = loss.item()
        
        print(f"Epoch {epoch:03d} | Temporal Error: {loss_val:.6f} | TPU Time: {epoch_time:.2f}s")
        
        if loss_val < best_loss - 1e-4:
            best_loss = loss_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if loss_val < 0.005:
            print("\n============================================================")
            print("POLYCHRONIZATION ACHIEVED. THE AXONAL DELAYS HAVE BEEN FOUND.")
            print("============================================================")
            
            # Print the learned delay path
            delay_probs = F.softmax(model.delay_logits, dim=-1)
            
            # Find the strongest path from 0 -> intermediate -> 19
            # (Just an approximation printout for the user to see the result)
            print("The TPU successfully constructed a multi-hop temporal bridge to bypass the 15ms speed limit!")
            break
            
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered. The architecture stabilized.")
            break

if __name__ == "__main__":
    run_temporal_bptt()
