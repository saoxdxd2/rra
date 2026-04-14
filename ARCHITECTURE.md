# THE $D^D$ TERNARY MANIFOLD: MASTER ARCHITECTURE

## Vision: The Silicon Cerebellum
Traditional AI (Transformers) is modeled after the neocortex — deep, recursive, computationally heavy. This engine is modeled after the **Cerebellum** — dense, fast, and optimized for high-speed sensory-motor mapping using **Functional Locality** and **Structural Sparsity**.

---

## Phase I: The Execution Physics (Compute & Hardware)
- **True Ternary Synapse (2-Bit):** Synapses exist only in `{+1, 0, -1}`. A single `uint32_t` packs the state (2 bits) and target index (30 bits). No floats in the hot-loop.
- **Zero-Multiplication Branchless Kernel:** `sao_tick_physics` uses `input += s_nb * (is_ex - is_in)` — pure shift-and-subtract AVX2/AVX-512 auto-vectorized. No FMAs, no branch divergence.
- **Latent Weight Isolation:** High-precision float weights are isolated in `SynapseWeightPage` / Q4 nibble store. The physics hot-loop cannot see them.
- **Q4 TurboQuant Compression:** NREM latent weights stored as 4-bit nibbles (16-entry codebook, `[-3,3]`). Footprint: `N×64×4B → N×32B` (8x reduction). 32K neurons: **8MB → 1MB** (fully L2-cache resident).
- **Soft Reset LIF:** `v = mem - threshold` on spike — preserves excess membrane energy, eliminating high-frequency temporal precision loss from hard resets.

---

## Phase II: The Shape-Shifting Topology — Temporal DAG Geometry
- **Temporal MSB Morton:** Time (`W`) lives in the top 16 bits of the 64-bit Morton key. Spatial `(X,Y,Z)` in the lower 48 bits via hardware `PDEP`. Result: neurons chronologically clustered in RAM.
- **Dual-Path Routing:**
    - **Dense Local Spatial Band:** ~64 nearest post-sort neighbors via 64-bit SIMD masks. Fast, approximate.
    - **Paged Wormhole Pool:** Long-range learned associations via linked `SynapsePage` pool. Exact topology surviving across Barycenter reshuffles.
- **NREM Topology Compiler:** Barycenter Sort physically migrates neurons in RAM to form a **Banded Matrix**. Active queues preserved via `old_to_new` index remapping — no amnesia.
- **Organic Neurogenesis:** `organic_growth()` capped at 32,768 neurons. $K_{min}=4$ axonal sprouting prevents manifold fragmentation.

---

## Phase III: The Polymorphic Sensorium (Modality)
- **ZeroCopy Dataset Streaming:** `CreateFileMapping` kernel-bypass pipes real SSD bytes directly — no VirtualEnvironment placeholder.
- **GNF Embedding:** Each of the 256 byte values has a learned 4D coordinate in `ByteField`. Morton keys are recomputed from these evolving coordinates. GNF activation computes `exp2(-POPCNT(byte_morton XOR neuron_morton) / tau)` — a continuous soft proximity metric.

---

## Phase IV: The Learning Paradigm — Temporal Morton Predictive Coding

### The Full Gradient Stack
| Signal | Scope | Role |
|---|---|---|
| Morton Hamming Credit (delayed) | Per-neuron | **Sole learning gradient** |
| Eligibility Trace (src × tgt) | Per-synapse | Temporal credit carrier |
| STE 3x boost | Dead-zone weights | Ternary escape amplifier |
| Stagnation Guard | Global | Structural reset |
| Energy Clamp | Per-neuron | Anti-seizure gating |

### Temporal Delay-Line (True Next-Token Prediction)
Credit is computed at tick `t` matching byte `B_t`, but **injected at tick `t+1`** when `B_{t+1}` arrives. Neurons are rewarded for having predicted the incoming byte — not for matching the current one.
- **Complexity:** O(N) per tick. No O(V×N²) enumeration of possible next tokens.
- Morton geometry collapses the vocabulary search to a single `POPCNT` against the actual arrived byte.

### 3-Factor True E-Prop
`delta = LR × Dopamine × source_eligibility_trace × target_eligibility_trace`
Both local-mask and paged-pool synapses use the identical formula.

### Straight-Through Estimator (STE)
Weights in the ternary dead-zone `[-0.1, +0.1]` with a nonzero gradient receive a **3x boost** to escape faster. Prevents dead gradients where latent floats drift but ternary bits never flip.

### No Gas Diffusion
`diffuse_dopamine()` is a **no-op stub**. Gas diffusion corrupts Morton credit with topology-dependent noise. Dopamine decay (`d *= 0.95f`) is inlined as a NREM epilogue.

### Structural Homeostasis Guard (not learning)
`engine_compute_endogenous_reward()` performs defensive operations only:
- **Stagnation wipe:** zero spikes → clear all accumulated dopamine
- **Per-neuron energy clamp:** chronically over-firing neurons get individual dopamine reduced

### Smooth Homeostatic Plasticity
Log-scaled PD threshold adaptation replaces the hard binary discontinuity:
`damped_error = sign(err) × log1p(|err| × 20)`
Prevents runaway seizure cascades while still aggressively rescuing dead neurons.

### Q4 TurboQuant + STE NREM Pass
1. Decode nibble → float
2. Apply `delta = LR × Dopamine × src_elig × tgt_elig`
3. If in dead-zone and gradient nonzero → apply 3x STE boost
4. Re-encode float → nearest codebook nibble
5. Update ternary bitmasks
6. `d *= 0.95f` (decay epilogue)

---

## Phase V: The System Engineering Stack
- **Kernel-Bypass I/O:** `CreateFileMapping` zero-copy SSD streaming.
- **Distributed Continents:** The 1D RAM manifold is designed to shard across machines, with sparse wormhole spikes crossing the network fabric.
