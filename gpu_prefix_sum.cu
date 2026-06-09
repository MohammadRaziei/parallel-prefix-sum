#include "gpu_prefix_sum.cuh"
#include <cstdio>
#include <cuda_runtime.h>

// --- 1. Define the Error Handling Macro ---
static void handleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s in %s at line %d\n", 
                cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (handleError(err, __FILE__, __LINE__))

__global__ void blellochKernel(float *ac)
{
    extern __shared__ float sdata[]; // allocated on invocation
    float lastElement = 0;
    const int n = blockDim.x;
    const int idx = blockIdx.x * n + threadIdx.x;
    sdata[threadIdx.x] = ac[idx];
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1)
    { // build sum in place up the tree
        __syncthreads();
        if (threadIdx.x < d)
        {
            const int ai = offset * (2 * threadIdx.x + 1) - 1;
            const int bi = offset * (2 * threadIdx.x + 2) - 1;
            sdata[bi] += sdata[ai];
        }
        offset <<= 1;
    }
    if (threadIdx.x == 0)
    {
        lastElement = sdata[n - 1];
        sdata[n - 1] = 0; // clear the last element
    }
    for (int d = 1; d < n; d <<= 1)
    { // build scan
        offset >>= 1;
        __syncthreads();
        if (threadIdx.x < d)
        {
            const int ai = offset * (2 * threadIdx.x + 1) - 1;
            const int bi = offset * (2 * threadIdx.x + 2) - 1;
            float tmp = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] += tmp;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
        ac[idx + n - 1] = lastElement;
    else
        ac[idx - 1] = sdata[threadIdx.x];
}

__device__ __inline__ unsigned int bit_reverse(unsigned int x, int n) {
    return __brev(x) >> (32 - n);
}


__global__ void bitReverseScanKernel(float ac[]){
    extern __shared__ float sdata[];// allocated on invocation
    float lastElement; 
    const int n = blockDim.x;
    const int idx = blockIdx.x * n + 1;
    sdata[threadIdx.x] = ac[idx]; // global memory coalescing
    for (unsigned int s = n >> 1; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
    }
    __syncthreads();
    if (threadIdx.x == n - 1) {  
        lastElement = sdata[0];
        sdata[0] = 0;  // clear the last element
    }
    for (unsigned int s = 1; s < n; s <<= 1) {
        __syncthreads();
        if (threadIdx.x < s) {
            float tmp = sdata[threadIdx.x + s];
            sdata[threadIdx.x + s] = sdata[threadIdx.x];
            sdata[threadIdx.x] += tmp;
        }
    }
    __syncthreads();
    if(threadIdx.x == n - 1)
        ac[idx + n - 1] = lastElement;
    else 
        ac[idx-1] = sdata[threadIdx.x];
}

// Global kernel: Each thread reverses one element of the array
__global__ void bitReverseKernel(unsigned int* d_out, unsigned int* d_in, int n, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        d_out[idx] = bit_reverse(d_in[idx], n);
    }
}




void gpuPrefixSum(float* h_data, int n) {
    float* d_data;
    size_t size = n * sizeof(float);
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    blellochKernel<<<1, n>>>(d_data);

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

void gpuBitReverse(unsigned int* h_out, const unsigned int* h_in, int n, int size) {
    unsigned int *d_in, *d_out;
    size_t bytes = size * sizeof(unsigned int);

    // Allocation
    HANDLE_ERROR(cudaMalloc(&d_in, bytes));
    HANDLE_ERROR(cudaMalloc(&d_out, bytes));

    // Data Transfer to GPU
    HANDLE_ERROR(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Kernel Launch
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    bitReverseKernel<<<gridSize, blockSize>>>(d_out, d_in, n, size);

    // Check for launch errors (e.g. invalid dimensions)
    HANDLE_ERROR(cudaGetLastError());

    // Wait for completion and check for execution errors
    HANDLE_ERROR(cudaDeviceSynchronize());

    // Data Transfer back to CPU
    HANDLE_ERROR(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // Cleanup
    HANDLE_ERROR(cudaFree(d_in));
    HANDLE_ERROR(cudaFree(d_out));
}