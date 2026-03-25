
# High-Performance k-NN Implementation with CUDA

This project presents a series of high-performance implementations of the **k-Nearest Neighbors (k-NN)** algorithm, optimized for NVIDIA GPUs using the CUDA SIMT architecture. The study explores various optimization strategies to overcome the O(M*N) complexity of sequential processing.

## Authors
* **Simone Pipola**
* **Ulisse Steccanella**

## Optimization Strategies & Dual-Path Approach
Based on our experimental results, the project implements a **Dual-Path strategy** to maintain peak performance across different values of K:

### 1. Low-K Path: Insertion Sort (`knn_shared_insertion.cu`)
* **Mechanism**: Each thread identifies its local neighbors using **Insertion Sort** in registers.
* **Collaboration**: Threads within a block merge their lists using a **logarithmic Tree Reduction** in Shared Memory (SMEM).
* **Performance**: This is the fastest method for small K, achieving **0.0034s** for 100k points.
* **Constraint**: Limited by SMEM capacity as K increases (e.g., K=23 requires approx 48 KB per block).

### 2. High-K Path: Global Sort with CUB (`knn_cub_segmented.cu`)
* **Mechanism**: A 2D grid configuration where each thread computes a single distance.
* **Sorting**: Uses the **CUB Library** (`DeviceSegmentedSort`) for efficient global sorting.

### 3. Memory & Architectural Optimizations
* **Tiling & SMEM (`knn_tiled_smem.cu`)**: Thread blocks process data "tiles," pre-fetching query and reference vectors into SMEM to transform redundant DRAM accesses into high-speed local lookups.
* **Coalesced Access - SoA (`knn_cub_soa.cu`)**: Data is reorganized into a **Structure of Arrays (SoA)** format. This ensures that adjacent threads in a warp access contiguous memory addresses, maximizing throughput.

## Experimental Results (Time in Seconds)
Tests conducted with **1,000,000 reference points** (unless specified):

| Optimization Stage | K=1 | K=23 | K=1000 |
| :--- | :--- | :--- | :--- |
| **Insertion Sort** (100k ref) | **0.003402s** | 0.056672s | *N/A* |
| **Tiling & SMEM** | 0.097352s | 0.106997s | 0.104846s |
| **Coalesced Access (SoA)** | 0.098254s | 0.107324s | 0.105828s |
| **Baseline Optimized** | 0.095657s | 0.104509s | 0.104685s |

## Requirements
* NVIDIA GPU (tested on architecture with 40 SMs)
* CUDA Toolkit
* [CUB Library](https://nvlabs.github.io/cub/)
