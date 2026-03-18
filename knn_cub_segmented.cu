#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>
#include <vector>

// CUB Library
#include <cub/cub.cuh>

// Error control macro
#define CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// Calculate distances between all Query and Reference points
__global__ void compute_distances_kernel(float* ref_points, int* ref_labels, float* query_points, float* out_dists, int* out_labels, int n_ref, int n_query, int dim) {
    // Reference Points index (Coalescing)
    int r_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Query Points index
    int q_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (q_idx >= n_query || r_idx >= n_ref) return;

    // Linearized index
    size_t base_idx = q_idx * n_ref + r_idx;

    float distance_sq = 0.0f;
        
    // Euclidean distance calculation
    for(int d = 0; d < dim; d++) {
        float diff = query_points[q_idx * dim + d] - ref_points[r_idx * dim + d];
        distance_sq += diff * diff;
    }

    // Save distance and original label
    out_dists[base_idx]  = distance_sq;
    out_labels[base_idx] = ref_labels[r_idx];
}

// Voting kernel on sorted labels
__global__ void voting_kernel(int* sorted_labels, int* final_preds, int n_ref, int n_query, int K, int N_COLOURS) {
    
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_idx >= n_query) return;

    // Vote counting for the first K neighbors
    int counts[50]; 
    for(int i = 0; i < N_COLOURS; i++) counts[i] = 0;

    int segment_start = q_idx * n_ref;

    for (int i = 0; i < K; i++) {
        // Read i-th neighbor label
        int label = sorted_labels[segment_start + i];
        if(label < N_COLOURS) {
            counts[label]++;
        }
    }

    // Search for the winning class
    int max_votes = -1;
    int winner = 0;
    
    for (int i = 0; i < N_COLOURS; i++) {
        if (counts[i] > max_votes) {
            max_votes = counts[i];
            winner = i;
        }
    }

    final_preds[q_idx] = winner;
}

double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

int main() {
    // Data Loading
    printf("--- Loading Data ---\n");
    FILE *f = fopen("knn_dataset.txt", "r");
    if (!f) { 
        printf("ERROR: knn_dataset.txt not found!\n"); 
        return -1; 
    }    
    
    int N_REF, N_QUERY, DIM, K, N_COLOURS;
    if(fscanf(f, "%d %d %d %d %d", &N_REF, &N_QUERY, &DIM, &K, &N_COLOURS) != 5) return -1;
    printf("Dataset: %d Ref, %d Queries, Dim=%d, Classes=%d, K=%d\n", N_REF, N_QUERY, DIM, N_COLOURS, K);

    // Allocation dimensions
    size_t size_ref      = N_REF * DIM * sizeof(float);
    size_t size_ref_lbl  = N_REF * sizeof(int);
    size_t size_query    = N_QUERY * DIM * sizeof(float);
    size_t size_pred     = N_QUERY * sizeof(int);

    // Host Allocation
    float *h_ref    = (float*)malloc(size_ref);
    int   *h_labels = (int*)malloc(size_ref_lbl);
    float *h_query  = (float*)malloc(size_query);
    int   *h_pred   = (int*)malloc(size_pred);

    // Reference Reading
    for (int i = 0; i < N_REF; i++) {
        for(int d = 0; d < DIM; d++) fscanf(f, "%f", &h_ref[i*DIM + d]);
        fscanf(f, "%d", &h_labels[i]);
    }
    // Query Reading
    for (int i = 0; i < N_QUERY; i++) {
        for(int d = 0; d < DIM; d++) fscanf(f, "%f", &h_query[i*DIM + d]);
    }
    fclose(f);
    
    // GPU Execution
    printf("\n--- GPU Execution ---\n");

    // Device pointers
    float *d_ref, *d_query;
    int   *d_ref_labels, *d_pred;

    // GPU Input/Output buffer allocation
    cudaMalloc(&d_ref, size_ref);
    cudaMalloc(&d_ref_labels, size_ref_lbl);
    cudaMalloc(&d_query, size_query);
    cudaMalloc(&d_pred, size_pred);

    // Host to Device copy
    cudaMemcpy(d_ref, h_ref, size_ref, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref_labels, h_labels, size_ref_lbl, cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, h_query, size_query, cudaMemcpyHostToDevice);

    // Intermediate buffers for CUB sorting
    // Distance-label pair arrays for each query-ref
    size_t total_pairs = (size_t)N_QUERY * N_REF;
    float *d_dist_in, *d_dist_out;
    int   *d_lbl_in,  *d_lbl_out;
    
    // Intermediate buffer allocation
    CHECK(cudaMalloc(&d_dist_in,  total_pairs * sizeof(float))); 
    CHECK(cudaMalloc(&d_dist_out, total_pairs * sizeof(float))); 
    CHECK(cudaMalloc(&d_lbl_in,   total_pairs * sizeof(int)));   
    CHECK(cudaMalloc(&d_lbl_out,  total_pairs * sizeof(int)));   

    // 2D Grid configuration
    dim3 blockDim2D(16, 16);
    dim3 gridDim2D(
        (N_REF   + blockDim2D.x - 1) / blockDim2D.x,
        (N_QUERY + blockDim2D.y - 1) / blockDim2D.y
    );

    printf("Grid Config: Grid(%d, %d), Block(%d, %d)\n", 
           gridDim2D.x, gridDim2D.y, blockDim2D.x, blockDim2D.y);

    // Distance Calculation
    double iStart = cpuSecond();
    
    compute_distances_kernel<<<gridDim2D, blockDim2D>>>(
        d_ref, d_ref_labels, d_query, 
        d_dist_in, d_lbl_in, 
        N_REF, N_QUERY, DIM
    );
    cudaDeviceSynchronize();

    // CUB Segmented Sorting
    
    // Offset creation
    std::vector<int> h_offsets(N_QUERY + 1);
    for (int i = 0; i <= N_QUERY; i++) h_offsets[i] = i * N_REF;
    
    int *d_offsets;
    cudaMalloc(&d_offsets, sizeof(int) * (N_QUERY + 1));
    cudaMemcpy(d_offsets, h_offsets.data(), sizeof(int) * (N_QUERY + 1), cudaMemcpyHostToDevice);

    // CUB Preparation
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // Request workspace size
    cub::DeviceSegmentedSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_dist_in, d_dist_out,   // Keys (Distances)
        d_lbl_in, d_lbl_out,     // Values (Labels)
        total_pairs,             
        N_QUERY,                 
        d_offsets, d_offsets + 1
    );

    // Workspace allocation
    CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Sorting execution
    cub::DeviceSegmentedSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_dist_in, d_dist_out,
        d_lbl_in, d_lbl_out,
        total_pairs, N_QUERY,
        d_offsets, d_offsets + 1
    );
    cudaDeviceSynchronize();

    // Voting
    int threadsPerBlock = 256;
    int blocksPerGrid = (N_QUERY + threadsPerBlock - 1) / threadsPerBlock;
    voting_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_lbl_out, d_pred, 
        N_REF, N_QUERY, K, N_COLOURS
    );
    cudaDeviceSynchronize();
    
    double iElaps = cpuSecond() - iStart;
    printf("Total GPU time (Compute + Sort + Vote): %f seconds\n", iElaps);

    // Result Recovery
    cudaMemcpy(h_pred, d_pred, size_pred, cudaMemcpyDeviceToHost);

    // Print first results
    printf("First 10 Predictions: ");
    for(int i=0; i<10 && i<N_QUERY; i++){
        printf("%d ", h_pred[i]);
    }
    printf("\n");

    // Saving to file
    printf("Saving predictions to 'predictions.txt'...\n");
    FILE *f_out = fopen("predictions.txt", "w");
    if (f_out) {
        fprintf(f_out, "%d %d\n", N_QUERY, DIM); 
        for(int i = 0; i < N_QUERY; i++) {
            fprintf(f_out, "%d\n", h_pred[i]);
        }
        fclose(f_out);
        printf("Saved successfully.\n");
    } else {
        printf("Error opening predictions.txt for writing.\n");
    }

    // Memory cleanup
    cudaFree(d_ref); cudaFree(d_query); cudaFree(d_ref_labels); cudaFree(d_pred);
    cudaFree(d_dist_in); cudaFree(d_dist_out); cudaFree(d_lbl_in); cudaFree(d_lbl_out);
    cudaFree(d_offsets); cudaFree(d_temp_storage);
    
    free(h_ref); free(h_query); free(h_labels); free(h_pred);

    return 0;
}
