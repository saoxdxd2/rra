# RRA Neural Network Framework

## Project Overview
This project is a custom, high-performance C++ neural network and machine learning framework (`rra_native`). It focuses on novel architectural approximations, bypassing traditional quadratic token-to-token attention matrices. Key architectural components include:
- **AETHER Field Operator**: An implicit field update over a Morton-ordered recursive manifold, utilizing spectral heat propagation and local transport.
- **Dual-Vector CAFE-NIS Engine**: A high-throughput hybrid engine processing streams of data.
- **Hardware Optimization**: Heavily relies on CPU-bound optimizations, specifically AVX-512 SIMD intrinsics (e.g., `__m512`) and OpenMP, to achieve high throughput without relying on GPUs. 
- **Memory Mapped Datasets**: Uses OS-level memory mapping for extremely fast data ingestion during training.

## Building and Dependencies
The project uses CMake as its build system.

**Dependencies:**
- C++20 compatible compiler (supports AVX-512 and OpenMP)
- CMake 3.20+
- OpenCV
- OpenMP

**Standard Build Commands:**
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```
*Note: The CMake configuration automatically targets `-march=native` or `/arch:AVX512` based on the compiler to unlock host CPU optimizations.*

## Key Executables
The build process generates several executables for different stages of the model lifecycle:

- **`launcher`** (from `train.cpp`): The primary training engine ("Hybrid Sweep"). It uses a memory-mapped `ThreadedMultiFileDataset` to ingest large amounts of text data.
  *Usage:* `./launcher --dataset <path_to_dataset.txt_or_directory>`
- **`generate`** (from `generate.cpp`): The inference and text generation interface. It utilizes high-throughput TMUL ingest for the context prompt, followed by byte-by-byte autoregressive generation.
  *Usage:* `./generate <model.bin> "<prompt>" [length]`
- **`evaluate`**: Evaluation binary for assessing model accuracy and loss metrics.
- **`manifold_ui`**: Likely a visualization tool for inspecting the internal state or field topology.

## Development Conventions
- **Namespaces:** Code is generally structured under the `s4m` and `s4m::core` namespaces.
- **Performance First:** The codebase avoids standard abstraction overhead in hot paths, favoring explicit SIMD vectorization and lock-free or memory-mapped I/O patterns.
- **Directory Structure:**
  - `nn/`: Contains the neural network modules, layers, losses, and custom architectures (AETHER, CAFE, MoE, Transformer).
  - `include/` & `src/`: Core engine utilities, tokenizer, and math operations.
