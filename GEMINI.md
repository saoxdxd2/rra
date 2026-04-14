# S4M: $D^D$ Ternary Manifold - Operational Guidelines

## Foundation: The Silicon Cerebellum
Self-compiling Ternary Manifold. All contributions MUST adhere to the hardware-accelerator efficiency constraints below.

### 1. The Ternary Mandate
- **Weights are NOT floats in the hot-loop.** States: Excitatory (+1), Inhibitory (-1), Dead (0).
- **Bit-Packing:** `uint32_t` — Bits 0-29: Index, Bits 30-31: State.
- **No FMAs:** `sao_tick_physics` uses `input += s_nb * (is_ex - is_in)` only.
- **Latent Physics:** Floats isolated to `SynapseWeightPage` + Q4 nibble store. Hot-loop cannot see them.

### 2. Physical RAM Locality — Temporal DAG Compiler
- **Topology is Dynamic:** `TopologyCompiler` physically reorganizes RAM via **Barycenter Sort**.
- **The Forwarding Table:** Any neuron move MUST do an $O(E)$ synapse index sweep. Active queues (`active_nodes_read_`, `active_nodes_write_`) MUST be remapped via `old_to_new` — never cleared.
- **Temporal MSB Morton:** Time (`W`) = top 16 bits. Space (`X,Y,Z`) = lower 48 bits. Chronological RAM layout.
- **Dual-Path Routing:**
    - **Dense:** 64 backward neighbors via bit-masks (spatial shortcut post-sort).
    - **Sparse:** Long-range wormholes via paged Synapse Pool (exact learned topology).

### 3. Credit Assignment — Temporal Morton Predictive Coding
- **No Backpropagation.** Replaced by **True Eligibility Propagation (E-Prop)**.
- **True 3-Factor Rule:** `delta = LR × Dopamine × source_eligibility × target_eligibility`. Both local-mask and paged-pool use the identical formula. Local mask MUST use source eligibility trace, NOT binary spike.
- **Geometric Credit (Temporal Delay-Line):** `gnf_activate(dopamine_out)` computes `credit_i = exp2(-POPCNT(byte_morton XOR neuron_morton_i) / tau)`. Credit from tick `t` is **injected at tick `t+1`** — neurons rewarded for predicting the NEXT byte, not matching the current one. O(N) per tick. No O(V×N²).
- **Straight-Through Estimator (STE):** Weights in dead-zone `[-0.1, +0.1]` with nonzero gradient receive a **3x boost** to escape ternary boundary faster.
- **No Gas Diffusion:** `diffuse_dopamine()` is a no-op stub. NEVER re-implement spatial diffusion — it corrupts Morton credit with topology-dependent noise. Dopamine decay (`d *= 0.95f`) is inlined at end of `consolidate_learning`.
- **Structural Guard Only:** `engine_compute_endogenous_reward()` performs stagnation wipe + per-neuron energy clamp. Does NOT inject global scalar reward.

### 4. Memory & I/O
- **Kernel-Bypass:** `CreateFileMapping` zero-copy streaming. `ZeroCopyDataset` MUST connect to actual data — not a `VirtualEnvironment` mock.
- **Q4 TurboQuant:** Latent weights stored as 4-bit nibbles (16-entry codebook). 8x memory reduction. For 32K neurons: 8MB → 1MB (L2-cache resident).
- **Aligned Access:** `AlignedAllocator<T, 32>` for all SIMD tensors.

### 5. Growth & Stability
- **Axonal Sprouting ($K_{min}$):** Minimum 4 connections per neuron.
- **Safety Counters:** All linked-list traversals MUST have safety counters (max 1000 iterations).
- **Organic Growth Cap:** 32,768 neurons max.
- **Soft LIF Reset:** `v = mem - threshold` on spike. Preserves excess energy. Hard reset to 0.0 is forbidden.
- **Smooth Homeostasis:** Log-scaled PD: `damped_error = sign(err) × log1p(|err| × 20)`. No hardcoded 0.05f step.

---

## Troubleshooting Checklist
- **Initialization Hangs:** Check `pool_syn_offset` init (`0xFFFFFFFF`) and `force_wire` safety counters.
- **Homeostatic Dead-Lock:** Verify log-scaled PD homeostasis in `sao_core.hpp`. No hardcoded `0.05f` binary threshold.
- **Pipeline Stalls:** `sao_tick_physics` inner loop must be branchless (`(ex >> j) & 1ULL`). Any `if/else` breaks AVX2 vectorization.
- **Memory Mapping Errors:** Verify dataset path and process permissions.
- **Topology Amnesia:** `active_nodes_read_` must be remapped via `old_to_new`, never cleared.
- **Scalar Credit Bottleneck:** `gnf_activate()` must be called with valid `dopamine_out` buffer and real `neuron_mortons[]` from `node.spatial_id`. Credit must be injected from the **previous tick's buffer** (delay-line), not the current tick.
- **Gas Diffusion Regression:** `diffuse_dopamine()` MUST remain a no-op stub. Spatial reward spreading corrupts Morton geometric credit.
- **Dead Ternary Gradients:** Verify STE is applied in `consolidate_learning` for dead-zone weights `(-0.1, +0.1)`.
