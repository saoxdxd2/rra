import os
import time

# Ensure JAX is installed
try:
    import jax
    import jax.numpy as jnp
    from jax import random, vmap, jit
except ImportError:
    print("Please install jax: !pip install jax jaxlib")
    exit(1)

print(f"JAX running on: {jax.devices()}")
print("==================================================================")
print("TPU v5e-1 Optimization Phase: Brain I/O & GABAergic Homeostasis")
print("==================================================================")

# ==============================================================================
# 1. BIOPHYSICAL ARCHITECTURE (JIT COMPILED FOR TPU)
# ==============================================================================

# Network Topology Constraints
N_SENSORY = 10
N_EXC_HIDDEN = 70
N_INH_HIDDEN = 20
N_MOTOR = 10
NUM_NEURONS = N_SENSORY + N_EXC_HIDDEN + N_INH_HIDDEN + N_MOTOR

# Biological Masks (Dale's Law: Neurons are either exclusively Excitatory or Inhibitory)
# Sensory [0:10], Exc Hidden [10:80], Inh Hidden [80:100], Motor [100:110]
EXCITATORY_MASK = jnp.concatenate([
    jnp.ones(N_SENSORY), 
    jnp.ones(N_EXC_HIDDEN), 
    jnp.zeros(N_INH_HIDDEN), 
    jnp.ones(N_MOTOR)
])

INHIBITORY_MASK = jnp.concatenate([
    jnp.zeros(N_SENSORY), 
    jnp.zeros(N_EXC_HIDDEN), 
    jnp.ones(N_INH_HIDDEN), 
    jnp.zeros(N_MOTOR)
])

# Fixed 20% sparse connection matrix (randomly wired)
key = random.PRNGKey(42)
base_topology = random.bernoulli(key, 0.2, (NUM_NEURONS, NUM_NEURONS)).astype(jnp.float32)

SIM_STEPS = 1000  # 1 full second of simulation (1ms ticks)
DT = 1.0

# Generate a continuous analog Target Sine Wave (e.g. tracking a physical object)
# Frequency: 2 Hz wave
time_array = jnp.arange(SIM_STEPS) * DT
ANALOG_INPUT_SIGNAL = (jnp.sin(2.0 * jnp.pi * 2.0 * (time_array / 1000.0)) + 1.0) / 2.0  # Range [0.0, 1.0]

@jit
def simulate_brain(genome):
    """
    Simulates a single brain's spiking dynamics tracking an analog wave.
    genome: [sensory_gain, motor_decay_tau, gaba_voltage_drop, exc_weight, inh_weight]
    """
    sensory_gain = genome[0]
    motor_decay_tau = genome[1]
    gaba_voltage_drop = genome[2]  # Should be negative
    exc_weight = genome[3]
    inh_weight = genome[4]

    v_rest = -70.0
    v_thresh = -56.5
    tau_m = 22.0
    
    # Pre-calculate synaptic weight matrix respecting Dale's Law
    # Excitatory neurons multiply their synapses by exc_weight
    # Inhibitory neurons multiply their synapses by inh_weight
    W_exc = base_topology * EXCITATORY_MASK[:, None] * exc_weight
    W_inh = base_topology * INHIBITORY_MASK[:, None] * inh_weight
    
    def step_fn(state, t_step):
        v, motor_ema, spikes, accum_mse, accum_spikes = state
        
        # 1. Analog Input -> Sensory Encoding (I_ext)
        current_analog_val = ANALOG_INPUT_SIGNAL[t_step]
        i_ext = jnp.concatenate([
            jnp.full(N_SENSORY, current_analog_val * sensory_gain),
            jnp.zeros(NUM_NEURONS - N_SENSORY)
        ])
        
        # 2. Voltage Decay (Leaky Integrate)
        v = v_rest + (v - v_rest) * jnp.exp(-DT / tau_m)
        
        # 3. Synaptic Transmission
        exc_input = jnp.dot(spikes, W_exc) * 5.0 
        inh_input = jnp.dot(spikes, W_inh) * gaba_voltage_drop
        
        v = v + i_ext + exc_input + inh_input
        
        # 4. Spike Generation
        new_spikes = jnp.where(v >= v_thresh, 1.0, 0.0)
        
        # 5. Voltage Reset
        v = jnp.where(new_spikes > 0.5, v_rest - 5.0, v)
        
        # 6. Motor Decoding
        motor_spikes = new_spikes[-N_MOTOR:]
        avg_motor_spike = jnp.mean(motor_spikes)
        motor_ema = motor_ema * jnp.exp(-DT / motor_decay_tau) + avg_motor_spike * (1.0 - jnp.exp(-DT / motor_decay_tau))
        
        # 7. Accumulate Fitness Metrics
        step_mse = (motor_ema - current_analog_val)**2
        step_spikes = jnp.sum(new_spikes)
        
        return (v, motor_ema, new_spikes, accum_mse + step_mse, accum_spikes + step_spikes), None

    initial_state = (
        jnp.full((NUM_NEURONS,), v_rest),
        0.0, # Initial Motor EMA
        jnp.zeros((NUM_NEURONS,)),
        0.0, # accum_mse
        0.0  # accum_spikes
    )

    final_state, _ = jax.lax.scan(step_fn, initial_state, jnp.arange(SIM_STEPS))
    _, _, _, total_mse, total_spikes = final_state
    
    mse_loss = total_mse / SIM_STEPS
    
    # Epilepsy / Death Penalty (Homeostasis)
    epilepsy_penalty = jnp.maximum(0.0, total_spikes - 30000.0) * 0.1
    death_penalty = jnp.maximum(0.0, 1000.0 - total_spikes) * 0.1
    
    fitness = mse_loss + epilepsy_penalty + death_penalty
    
    return fitness, total_spikes

# VMAP: Batch parallelize across the entire TPU v5e-1 memory
simulate_population = jit(vmap(simulate_brain))

# ==============================================================================
# 2. ORTHOGONAL NOVELTY EVOLUTIONARY ENGINE
# ==============================================================================

# Maximizing TPU Capacity (~40GB RAM utilization footprint)
POPULATION_SIZE = 65536 
GENERATIONS = 150

@jit
def mutate(genomes, key):
    # Random mutations (-10% to +10%)
    noise = random.uniform(key, genomes.shape, minval=-0.1, maxval=0.1)
    mutated = genomes + (genomes * noise)
    
    # genome: [sensory_gain, motor_decay_tau, gaba_voltage_drop, exc_weight, inh_weight]
    mutated = mutated.at[:, 0].set(jnp.clip(mutated[:, 0], 1.0, 500.0))    # sensory_gain
    mutated = mutated.at[:, 1].set(jnp.clip(mutated[:, 1], 1.0, 100.0))    # motor_decay_tau
    mutated = mutated.at[:, 2].set(jnp.clip(mutated[:, 2], -20.0, -0.1))   # gaba_voltage_drop (Negative)
    mutated = mutated.at[:, 3].set(jnp.clip(mutated[:, 3], 0.01, 10.0))    # exc_weight
    mutated = mutated.at[:, 4].set(jnp.clip(mutated[:, 4], 0.01, 10.0))    # inh_weight
    return mutated

def run_evolution():
    print(f"Initializing Evolution for {POPULATION_SIZE} concurrent brains...")
    rng = random.PRNGKey(999)
    
    # Baseline Starting Point
    # [sensory_gain=100.0, motor_decay_tau=20.0, gaba_voltage_drop=-5.0, exc=1.0, inh=1.0]
    rng, subkey = random.split(rng)
    genomes = jnp.array([[100.0, 20.0, -5.0, 1.0, 1.0]] * POPULATION_SIZE)
    genomes = mutate(genomes, subkey) 

    for gen in range(GENERATIONS):
        start_t = time.time()
        
        # 1. Simulate the entire population instantly on TPU
        fitness_scores, total_spikes = simulate_population(genomes)
        
        # 2. Calculate Orthogonal Novelty Distance
        pop_mean = jnp.mean(genomes, axis=0)
        novelty_scores = jnp.linalg.norm(genomes - pop_mean, axis=1)
        
        # 3. Evolutionary Sorting
        fitness_ranking = jnp.argsort(fitness_scores) # Lower is better (MSE + Penalty)
        
        # Top 10% Fitness Winners
        top_fitness_idx = fitness_ranking[:POPULATION_SIZE // 10]
        winners = genomes[top_fitness_idx]
        
        # Bottom 90% Failures
        failures_idx = fitness_ranking[POPULATION_SIZE // 10:]
        failures = genomes[failures_idx]
        failure_novelty = novelty_scores[failures_idx]
        
        # Top 10% Orthogonal Mutants (from the failures) to prevent local minima
        top_novelty_idx = jnp.argsort(-failure_novelty)[:POPULATION_SIZE // 10]
        orthogonal_mutants = failures[top_novelty_idx]
        
        # Combine Winners and Mutants to form parents
        parents = jnp.concatenate([winners, orthogonal_mutants], axis=0)
        
        # 4. Cross-Breed / Clone to fill next generation
        num_parents = parents.shape[0]
        repeats = POPULATION_SIZE // num_parents
        next_gen = jnp.tile(parents, (repeats, 1))
        
        remainder = POPULATION_SIZE - next_gen.shape[0]
        if remainder > 0:
            next_gen = jnp.concatenate([next_gen, parents[:remainder]], axis=0)
            
        # Mutate the new generation
        rng, subkey = random.split(rng)
        genomes = mutate(next_gen, subkey)
        
        end_t = time.time()
        best_idx = fitness_ranking[0]
        
        print(f"Gen {gen+1:03d} | Best Fit (Loss): {fitness_scores[best_idx]:7.4f} | Spikes: {total_spikes[best_idx]:6.0f} | Time: {(end_t - start_t):.2f}s")

    # Final result
    final_fitness, _ = simulate_population(genomes)
    final_best_idx = jnp.argmin(final_fitness)
    optimal = genomes[final_best_idx]
    
    print("\n==================================================================")
    print("THE BLACK BOX HAS DISCOVERED THE HOMEOSTASIS LAWS.")
    print("==================================================================")
    print("Optimal I/O & GABA Parameters:")
    print(f"  SENSORY_GAIN:       {optimal[0]:.4f} mA")
    print(f"  MOTOR_DECAY_TAU:    {optimal[1]:.4f} ms")
    print(f"  GABA_VOLTAGE_DROP:  {optimal[2]:.4f} mV")
    print(f"  EXCITATORY_SCALE:   {optimal[3]:.4f}")
    print(f"  INHIBITORY_SCALE:   {optimal[4]:.4f}")
    print("==================================================================")
    print("Next step: Implement distinct GABAergic Neurons and EMA Decoding in C++.")

if __name__ == "__main__":
    run_evolution()
