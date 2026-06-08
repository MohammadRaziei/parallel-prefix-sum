#include "gpu_prefix_sum.cuh"

__global__ void blellochKernel(float* data, int n) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int offset = 1;

    int ai = 2 * tid;
    int bi = 2 * tid + 1;

    temp[ai] = data[ai];
    temp[bi] = data[bi];

    // Up-Sweep
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int i = offset * (2 * tid + 1) - 1;
            int j = offset * (2 * tid + 2) - 1;
            temp[j] += temp[i];
        }
        offset *= 2;
    }

    if (tid == 0) temp[n - 1] = 0;

    // Down-Sweep
    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int i = offset * (2 * tid + 1) - 1;
            int j = offset * (2 * tid + 2) - 1;
            float t = temp[i];
            temp[i] = temp[j];
            temp[j] += t;
        }
    }
    __syncthreads();

    data[ai] = temp[ai];
    data[bi] = temp[bi];
}

void gpuPrefixSum(float* h_data, int n) {
    float* d_data;
    size_t size = n * sizeof(float);
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    blellochKernel<<<1, n / 2, size>>>(d_data, n);

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}