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

__device__ __forceinline__ unsigned int bit_reverse(unsigned int x, int m_bits) {
    return __brev(x) >> (32 - m_bits);
}

__device__ __forceinline__ int fast_log2(unsigned int x) {
    return 31 - __clz(x);
}

__device__ __forceinline__ unsigned int bit_reverse_pow2(unsigned int x, int n_pow2) {
    return __brev(x) >> (1 + __clz(n_pow2));
}

// Global kernel: Each thread reverses one element of the array
__global__ void bitReverseKernel(unsigned int* d_out, unsigned int* d_in, int m_bits, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        d_out[idx] = bit_reverse(d_in[idx], m_bits);
    }
}

// Global kernel: Each thread reverses one element of the array
__global__ void bitReversePow2Kernel(unsigned int* d_out, unsigned int* d_in, int n_pow2, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        d_out[idx] = bit_reverse_pow2(d_in[idx], n_pow2);
    }
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
    int blockSize = 1024; // Assuming size is a power of 2 and <= 1024
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


void gpuBitReversePow2(unsigned int* h_out, const unsigned int* h_in, int n, int size) {
    unsigned int *d_in, *d_out;
    size_t bytes = size * sizeof(unsigned int);

    // Allocation
    HANDLE_ERROR(cudaMalloc(&d_in, bytes));
    HANDLE_ERROR(cudaMalloc(&d_out, bytes));

    // Data Transfer to GPU
    HANDLE_ERROR(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Kernel Launch
    int blockSize = 1024; // Assuming size is a power of 2 and <= 1024
    int gridSize = (size + blockSize - 1) / blockSize;
    bitReversePow2Kernel<<<gridSize, blockSize>>>(d_out, d_in, n, size);

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


__global__ void bitReverseScanKernel(float ac[]){
    extern __shared__ float sdata[];// allocated on invocation
    float lastElement; 
    const int n = blockDim.x;
    const int idx = blockIdx.x * n + (n - 1 - bit_reverse_pow2(threadIdx.x, n));

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





void gpuPrefixSum(float* h_data, int n) {
    float* d_data;
    size_t size = n * sizeof(float);
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    blellochKernel<<<1, n>>>(d_data);

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}
