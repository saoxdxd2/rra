# RRA Cognitive Engine: Unified Architecture Summary

## 1. The Vision
The RRA (Recurrent Resonating Architecture) project aims to build an alternative to auto-regressive Transformers by using highly efficient, biologically inspired Spiking Neural Networks (SNNs). The goal is to create a brain-like cognitive core that can operate in real-time, execute continuous learning, and process complex streams of data (both physical sensors and symbolic text) directly on local hardware (CPU/C++). 

To find the optimal biophysics, we ran massive blackbox evolutionary algorithms on Google TPUs to "discover" the fundamental hyperparameters (genomes) and learning laws of this artificial brain.

## 2. The Twin Engines
As the RRA project evolved, we discovered that continuous physical embodiment (driving a car) and discrete symbolic reasoning (natural language processing) require two slightly different operational paradigms of the same underlying SNN. 

Today, both paradigms live side-by-side in our native C++ `brain_tree` production environment.

### A. The Embodiment Engine (RL & Motor Control)
* **Components**: `CorticalTissue`, `EventDrivenSynapse`, `UniversalPlasticity`
* **History**: Documented in `docs/1.md` and `docs/2.md`.
* **How it works**: 
  - Simulates spikes asynchronously using an event-driven priority queue (microsecond precision).
  - Uses the TPU-discovered 369-parameter **Universal Plasticity MLP**.
  - Learns *online* via global Dopamine waves, modifying synaptic weights based on temporal eligibility traces (`trace_v_pre`, `trace_v_post`, `trace_ca`).
* **Successes**: Evolved to successfully navigate a virtual car around a track. The "Contrary Scalar Selection" allowed the agent to explore deep weight landscapes without getting trapped in local minima.

### B. The Language Engine (NLP & Symbolic Processing)
* **Components**: `ReservoirEngine`
* **History**: Documented in `docs/3.md`.
* **How it works**:
  - We hit a hard ceiling trying to use online Hebbian learning for exact text recovery (letters are too precise and interference causes catastrophic forgetting).
  - **The Breakthrough (Reservoir Computing)**: We took the SNN genome discovered by the TPU and **froze** the synaptic weights. We treat the spiking brain as a high-dimensional nonlinear "liquid."
  - We feed letters in using a **Gaussian population code** (which maps coordinates onto sensory neurons). 
  - We read the total spikes of all 128 neurons, generate 256 quadratic features, and train a simple Linear Readout layer (`R` matrix) via batch gradient descent.
* **Successes**: Reached 23/23 (100%) perfect text generation on the target string (`HELLO WORLD THIS IS RRA`) directly on the TPU, proving that the frozen SNN acts as an incredibly powerful feature extractor for text.

## 3. The Current State of Production
In `nn/brain_tree/brain_tree.hpp` and `.cpp`, we now have the complete cognitive toolbox:

1. When we need an agent to physically interact with an environment and learn from delayed rewards, we instantiate `CorticalTissue` and let the Dopamine MLP rewire the brain.
2. When we need to process symbolic tokens, text, or perform high-fidelity inference, we instantiate `ReservoirEngine`, feed the text into the frozen topological SNN, and use the trained linear readout.

Both systems are natively compiled, C++ optimized, and entirely independent of Python/PyTorch for inference.
