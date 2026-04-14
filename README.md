# S4M: The $D^D$ Ternary Manifold (Silicon Cerebellum)

**S4M** is a high-performance, self-compiling neural engine built in C++20. It implements a biologically-grounded alternative to the Transformer architecture, designed to execute purely in ultra-fast **L1/L2/L3 CPU Cache** (~45MB footprint) using commodity CPU hardware. It features true zero-multiplication Ternary routing, BPTT-mirrored causal temporal geometry, and Morton-grounded geometric credit assignment — making it simultaneously an E-Prop, Forward-Forward, Predictive Coding, and RTRL-approximate system.

---

## The Mathematical Bedrock: Surpassing $O(N^2)$

The primary bottleneck of modern AI (Transformers) is the **Self-Attention Mechanism** with $O(N^2)$ complexity. S4M bypasses this entirely using **Spatial Hashing** and **Banded Matrix Physicality**, achieving near **$O(1)$** routing efficiency.

### 1. The Morton Z-Order Curve — Temporal Hierarchy (MSB Time)
Neurons are assigned a 64-bit **Morton Key** with a **hierarchical temporal structure**:
- **Bits 48–63 (MSB):** Time/Momentum dimension (`W`) — strictly sequential, no hash whitening.
- **Bits 0–47 (LSB):** 3D interleaved spatial coordinates `(X, Y, Z)` via hardware `PDEP`.

This physically arrays the entire manifold chronologically in RAM. Neurons at the same temporal depth are spatially clustered — replicating BPTT's unrolled temporal DAG natively in CPU cache-line sequential memory.

### 2. From Dot-Product to Hamming Distance ($O(N^2) \rightarrow O(1)$)
Traditional Attention uses a Softmax(QK^T) operation. S4M replaces this with **Hamming Distance Routing**:
- `distance = __popcnt64(byte_morton XOR neuron_morton)`
- A CPU performs XOR+Popcount in a single clock cycle.
- The RAM is Morton-sorted, so relevant neighbors are physically adjacent in L1 cache.

### 3. The Barycenter Evolution (Functional Gravity)
During the NREM sleep cycle, the `TopologyCompiler` measures functional gravity between neurons and physically migrates them in RAM (`std::sort`) so that co-activating neurons literally **live together** in the same L1/L2 cache lines — forming a **Banded Matrix**.

---

## Learning Paradigm: Morton Predictive Coding

S4M avoids the global scalar reward bottleneck of classical RL (DQN) by using the GNF Hamming distance as a **per-neuron credit signal**:

```
credit_i = exp2(-POPCNT(byte_morton XOR neuron_morton_i) / tau)
```

For each streamed input byte, neurons geometrically closest to that byte in 4D Morton space receive the strongest reward. This provides LLM-grade high-dimensional feedback bandwidth at $O(N)$ cost with **zero extra compute** — the Hamming distances are already calculated during routing.

This simultaneously implements:
- **Predictive Coding** — residual surprise = Hamming distance from predicted input
- **Forward-Forward Algorithm** — local goodness = Hamming proximity
- **DDQN Decoupling** — the Morton geometry acts as the independent target evaluator
- **RTRL Approximation** — eligibility traces carry temporal gradients forward online

---

## Technical Stack & Optimizations

### 1. Hardware-Accelerated Math
- **Branchless SIMD Kernels:** AVX2/AVX-512 auto-vectorized `sao_tick_physics` using shift-and-subtract `(is_ex - is_in)` — zero branch divergence.
- **Morton Encoding:** Hardware `PDEP` intrinsic for $O(1)$ 4D coordinate bit-interleaving.
- **Sparse Routing:** $O(1)$ attention via `__popcnt64` Hamming Distance.

### 2. Data & Visualization
- **Kernel-Bypass I/O:** Zero-copy dataset streaming via Windows `CreateFileMapping` (mmap).
- **Real-time Telemetry:** Zero-copy **Manifold IPC** system for monitoring the 4D embedding space.

---

## Core Architectural Pillars

### 1. True Ternary Execution (Zero-Multiplication Kernel)
- Synapses: `uint32_t` — 2-bit ternary state + 30-bit target index. No float in the hot-loop.
- Physics loop: branchless `input += s_nb * (is_ex - is_in)`.
- NREM STDP masks: branchless `state_.local_ex_masks[i] |= (is_ex << j)`.

### 2. Strict Causal DAG — BPTT Arrow-of-Time
- Physics kernel window: `neighbor_idx = i - j` (strictly backward-looking, 0 to -64).
- Topology rebuild: paged synapses remapped to local masks using `bit = t_idx - i` (target > source).
- Active queues preserved across topology reshuffles via `old_to_new` remapping.

### 3. Morton Geometric Credit Assignment
- `gnf_activate()` accepts `float* dopamine_out` — writes `exp2(-delta * inv_tau)` per neuron.
- `train.cpp` builds `neuron_mortons[]` from `node.spatial_id`, calls `gnf_activate()` per batch byte, injects normalized credit via `add_reward()` before `consolidate_learning()`.

### 4. Graceful Homeostasis & Organic Growth
- Dead neurons: `adaptive_thresholds -= 0.05f` progressively until they reactivate.
- Network grows via `organic_growth()`, capped at 32,768 neurons.
- Minimum connectivity: `axonal_sprouting()` enforces $K_{min} = 4$ connections per neuron.

---

## Technical Performance Targets
| Metric | Target | Status |
|---|---|---|
| **RAM Usage** | ~45 MB | L1/L2/L3 Cache-resident |
| **Physics Tick** | ~14µs | AVX2 branchless |
| **I/O Throughput** | NVMe Wire Speed | Kernel-Bypass mmap |
| **Credit Bandwidth** | Per-Neuron | Morton Hamming per byte |

---

## Building and Running

### Requirements
- **OS:** Windows 10/11 (x64)
- **Compiler:** MSVC 2022 (C++20)
- **Hardware:** AVX2/AVX-512 Support, NVMe SSD

### Build Commands
```powershell
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

### Run
```powershell
.\build\Release\launcher.exe --dataset dataset.txt
```

---

## File Map
- `include/sao_core.hpp`: Branchless causal Ternary Physics Kernel.
- `include/byte_field.hpp`: Morton encoding API + `gnf_activate` with `dopamine_out`.
- `src/byte_field.cpp`: GNF activation + geometric credit assignment.
- `src/nis_engine.cpp`: NREM Topology Compiler, causal STDP, Barycenter Sort.
- `train.cpp`: Kernel-Bypass data loader, Morton credit wiring, autonomous training loop.
- `include/synapse.hpp`: 2-bit Ternary Synapse packing + `SynapseWeightPage`.
- `ARCHITECTURE.md`: The exhaustive 5-phase Master Architecture blueprint.

## License
Private / Internal use.
