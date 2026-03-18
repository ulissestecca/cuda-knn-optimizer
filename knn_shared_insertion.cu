#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>
#include <float.h>

/* * KNN Implementation: 1 block per query.
 * Top-K reduction in shared memory to avoid excessive global writes.
 */

#define CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

#define MAX_K 32 

// Merge two sorted lists, keeping only the top K elements
__device__ void merge_topk(float* keys_a, int* vals_a, const float* keys_b, const int* vals_b, int K) {
    float temp_k[MAX_K];
    int   temp_v[MAX_K];

    int i = 0, j = 0, pos = 0;
    
    while(pos < K) {
        float val_a = (i < K) ? keys_a[i] : FLT_MAX;
        float val_b = (j < K) ? keys_b[j] : FLT_MAX;

        if (val_a <= val_b) {
            temp_k[pos] = val_a;
            temp_v[pos] = vals_a[i];
            i++;
        } else {
            temp_k[pos] = val_b;
            temp_v[pos] = vals_b[j];
            j++;
        }
        pos++;
    }

    for(int x = 0; x < K; x++) {
        keys_a[x] = temp_k[x];
        vals_a[x] = temp_v[x];
    }
}

__global__ void knn_block_parallel_kernel(const float* __restrict__ ref_points, const int* __restrict__ ref_labels, const float* __restrict__ query_points, float* out_dists, int* out_labels, int n_ref, int n_query, int dim, int K) 
{
    int q_idx = blockIdx.x; 
    if (q_idx >= n_query) return;

    int tid = threadIdx.x;

    // Buffers for the query and for the subsequent reduction
    extern __shared__ float s_query[]; 
    
    // Cooperative query loading
    for (int d = tid; d < dim; d += blockDim.x) {
        s_query[d] = query_points[q_idx * dim + d];
    }
    __syncthreads();

    // Local top-K initialization for each thread
    float my_dists[MAX_K];
    int   my_lbls[MAX_K];

    for(int i=0; i<K; i++) {
        my_dists[i] = FLT_MAX;
        my_lbls[i]  = -1;
    }

    // Grid-stride loop over reference points
    for(int r_idx = tid; r_idx < n_ref; r_idx += blockDim.x) {
        
        float dist_sq = 0.0f;
        for(int d = 0; d < dim; d++) {
            float diff = s_query[d] - ref_points[r_idx * dim + d];
            dist_sq += diff * diff;
        }

        // Ordered insertion if the distance is valid
        if (dist_sq < my_dists[K-1]) {
            int current_lbl = ref_labels[r_idx];
            int i = K - 1;
            while (i > 0 && dist_sq < my_dists[i-1]) {
                my_dists[i] = my_dists[i-1];
                my_lbls[i]  = my_lbls[i-1];
                i--;
            }
            my_dists[i] = dist_sq;
            my_lbls[i]  = current_lbl;
        }
    }
    
    // Shared memory pointers setup for the merge tree
    float* s_lists_dists = (float*)&s_query[dim]; 
    int* s_lists_lbls  = (int*)&s_lists_dists[blockDim.x * K];

    // Move local results to shared memory
    for(int i=0; i<K; i++) {
        s_lists_dists[tid * K + i] = my_dists[i];
        s_lists_lbls[tid * K + i]  = my_lbls[i];
    }
    __syncthreads();

    // Tree reduction to merge top-K results from all threads in the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            merge_topk(
                &s_lists_dists[tid * K], &s_lists_lbls[tid * K], 
                &s_lists_dists[(tid + stride) * K], &s_lists_lbls[(tid + stride) * K], 
                K
            );
        }
        __syncthreads();
    }

    // Thread 0 writes the final query result
    if (tid == 0) {
        for(int i=0; i<K; i++) {
            out_dists[q_idx * K + i]  = s_lists_dists[i];
            out_labels[q_idx * K + i] = s_lists_lbls[i];
        }
    }
}

__global__ void voting_kernel(int* knn_labels, int* final_preds, int n_query, int K, int N_COLOURS) {
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_idx >= n_query) return;

    // Move counts to shared memory if N_COLOURS increases
    int counts[50]; 
    for(int i = 0; i < N_COLOURS; i++) counts[i] = 0;

    int segment_start = q_idx * K;
    for (int i = 0; i < K; i++) {
        int label = knn_labels[segment_start + i];
        if(label < N_COLOURS && label >= 0) counts[label]++;
    }

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
    FILE *f = fopen("knn_dataset.txt", "r");
    if (!f) { perror("Dataset open"); return -1; }    
    
    int N_REF, N_QUERY, DIM, K, N_COLOURS;
    if(fscanf(f, "%d %d %d %d %d", &N_REF, &N_QUERY, &DIM, &K, &N_COLOURS) != 5) return -1;

    if(K > MAX_K) {
        fprintf(stderr, "K out of bounds. Max: %d\n", MAX_K);
        return -1;
    }

    float *h_ref    = (float*)malloc(N_REF * DIM * sizeof(float));
    int   *h_labels = (int*)malloc(N_REF * sizeof(int));
    float *h_query  = (float*)malloc(N_QUERY * DIM * sizeof(float));
    int   *h_pred   = (int*)malloc(N_QUERY * sizeof(int));

    for (int i = 0; i < N_REF; i++) {
        for(int d = 0; d < DIM; d++) fscanf(f, "%f", &h_ref[i*DIM + d]);
        fscanf(f, "%d", &h_labels[i]);
    }
    for (int i = 0; i < N_QUERY; i++) {
        for(int d = 0; d < DIM; d++) fscanf(f, "%f", &h_query[i*DIM + d]);
    }
    fclose(f);
    
    float *d_ref, *d_query, *d_dist_out;
    int   *d_ref_labels, *d_pred, *d_lbl_out;

    CHECK(cudaMalloc(&d_ref, N_REF * DIM * sizeof(float)));
    CHECK(cudaMalloc(&d_ref_labels, N_REF * sizeof(int)));
    CHECK(cudaMalloc(&d_query, N_QUERY * DIM * sizeof(float)));
    CHECK(cudaMalloc(&d_pred, N_QUERY * sizeof(int)));
    CHECK(cudaMalloc(&d_dist_out, N_QUERY * K * sizeof(float))); 
    CHECK(cudaMalloc(&d_lbl_out,  N_QUERY * K * sizeof(int)));   

    CHECK(cudaMemcpy(d_ref, h_ref, N_REF * DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ref_labels, h_labels, N_REF * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_query, h_query, N_QUERY * DIM * sizeof(float), cudaMemcpyHostToDevice));

    dim3 gridDim(N_QUERY, 1, 1);
    dim3 blockDim(256, 1, 1);
    
    // smem size: query vector + distance buffer + label buffer
    size_t shared_mem_size = (DIM * sizeof(float)) + (blockDim.x * K * sizeof(float)) + (blockDim.x * K * sizeof(int));

    double iStart = cpuSecond();
    
    knn_block_parallel_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        d_ref, d_ref_labels, d_query, 
        d_dist_out, d_lbl_out, 
        N_REF, N_QUERY, DIM, K
    );
    CHECK(cudaDeviceSynchronize());

    int vote_blocks = (N_QUERY + 256 - 1) / 256;
    voting_kernel<<<vote_blocks, 256>>>(d_lbl_out, d_pred, N_QUERY, K, N_COLOURS);
    CHECK(cudaDeviceSynchronize());
    
    double iElaps = cpuSecond() - iStart;
    printf("GPU Time: %f s\n", iElaps);

    CHECK(cudaMemcpy(h_pred, d_pred, N_QUERY * sizeof(int), cudaMemcpyDeviceToHost));

    FILE *f_out = fopen("predictions.txt", "w");
    if (f_out) {
        fprintf(f_out, "%d %d\n", N_QUERY, DIM); 
        for(int i = 0; i < N_QUERY; i++) fprintf(f_out, "%d\n", h_pred[i]);
        fclose(f_out);
    }

    cudaFree(d_ref); cudaFree(d_query); cudaFree(d_ref_labels); cudaFree(d_pred);
    cudaFree(d_dist_out); cudaFree(d_lbl_out);
    free(h_ref); free(h_query); free(h_labels); free(h_pred);

    return 0;
}
