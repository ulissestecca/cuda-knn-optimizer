# cuda-knn-optimizer
A high-performance k-Nearest Neighbors (k-NN) implementation optimized for NVIDIA GPUs using CUDA. This project features a Dual-Path strategy, utilizing shared memory tiling, coalesced memory access (SoA), and a hybrid sorting approach (Insertion Sort vs. CUB Merge Sort) to achieve massive parallel speedup over sequential algorithms.
