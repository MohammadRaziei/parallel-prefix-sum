#include "gpu_prefix_sum.cuh"
#include <cstdio>
#include <cuda_runtime.h>

// --- 1. Define the Error Handling Macro ---
static void handleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s in %s:%d\n", 
                cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (handleError(err, __FILE__, __LINE__))


struct GpuTimer{
	cudaEvent_t start_, stop_;
	GpuTimer(){
		cudaEventCreate(&start_);
		cudaEventCreate(&stop_);
	}
	~GpuTimer(){
		cudaEventDestroy(start_);
		cudaEventDestroy(stop_);
	}
	void start(){
		cudaEventRecord(start_, 0);
	}
	void stop(){
		cudaEventRecord(stop_, 0);
	}
	float elapsedMs(){
		float elapsed;
		cudaEventSynchronize(stop_);
		cudaEventElapsedTime(&elapsed, start_, stop_);
		return elapsed;
	}
} gpuTimer;

__device__ __forceinline__ unsigned int bit_reverse(unsigned int x, int m_bits) {
    return __brev(x) >> (32 - m_bits);
}

__device__ __forceinline__ int fast_log2(unsigned int x) {
    return 31 - __clz(x);
}

__device__ __forceinline__ unsigned int bit_reverse_pow2(unsigned int x, int n_pow2) {
    return __brev(x) >> (1 + __clz(n_pow2));
}

__global__ void bitReverseKernel(unsigned int* d_out, unsigned int* d_in, int m_bits, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) d_out[idx] = bit_reverse(d_in[idx], m_bits);
}

__global__ void bitReversePow2Kernel(unsigned int* d_out, unsigned int* d_in, int n_pow2, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) d_out[idx] = bit_reverse_pow2(d_in[idx], n_pow2);
}


template <typename KernelFunc>
void gpuBitReverseRunner(KernelFunc kernel, unsigned int* h_out, const unsigned int* h_in, int param, int size) {
    unsigned int *d_in, *d_out;
    size_t bytes = size * sizeof(unsigned int);

    // 1. Allocation
    HANDLE_ERROR(cudaMalloc(&d_in, bytes));
    HANDLE_ERROR(cudaMalloc(&d_out, bytes));

    // 2. Host to Device
    HANDLE_ERROR(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // 3. Launch Logic
    int blockSize = (size < 1024) ? ((size + 31) / 32) * 32 : 1024;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    kernel<<<gridSize, blockSize>>>(d_out, d_in, param, size);

    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    // 4. Device to Host
    HANDLE_ERROR(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // 5. Cleanup
    HANDLE_ERROR(cudaFree(d_in));
    HANDLE_ERROR(cudaFree(d_out));
}

void gpuBitReverse(unsigned int* h_out, const unsigned int* h_in, int n, int size) {
    gpuBitReverseRunner(bitReverseKernel, h_out, h_in, n, size);
}

void gpuBitReversePow2(unsigned int* h_out, const unsigned int* h_in, int n, int size) {
    gpuBitReverseRunner(bitReversePow2Kernel, h_out, h_in, n, size);
}


template <typename T, typename KernelFunc>
void gpuScanRunner(KernelFunc kernel, T h_data[], int size, float* kernel_time_ms = nullptr, int shift = 0, float shared_mem_vs_block_size = 1.f) {
    T* d_data;
    const size_t n_bytes = size * sizeof(T);

    // 1. Allocation
    HANDLE_ERROR(cudaMalloc(&d_data, n_bytes));

    // 2. Data Transfer (Host to Device)
    HANDLE_ERROR(cudaMemcpy(d_data, h_data, n_bytes, cudaMemcpyHostToDevice));

    // 3. Launch Configuration
    // For single-block algorithms (like Vanilla Blelloch)
    size >>= shift;
    const int blockSize = (size < 1024) ? ((size + 31) / 32) * 32 : 1024;
    const int gridSize = ((size + blockSize - 1) / blockSize);

    // 4. Kernel Launch
    // We pass the kernel function and the dynamic shared memory size
    if (kernel_time_ms) gpuTimer.start();
    kernel<<<gridSize, blockSize, static_cast<int>(shared_mem_vs_block_size * blockSize * sizeof(T))>>>(d_data);
    if (kernel_time_ms) {
        gpuTimer.stop();
        *kernel_time_ms = gpuTimer.elapsedMs();
    }

    // 5. Error Check and Sync
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    // 6. Data Transfer (Device to Host)
    HANDLE_ERROR(cudaMemcpy(h_data, d_data, n_bytes, cudaMemcpyDeviceToHost));

    // 7. Cleanup
    HANDLE_ERROR(cudaFree(d_data));
}



template <typename T>
__global__ void vannilaKernel(T *ac) {
    extern __shared__ unsigned char shared_memory[]; // allocated on invocation
    T* sdata = reinterpret_cast<T*>(shared_memory);
    T lastElement = 0;
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
            T tmp = sdata[ai];
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

template <typename T>
void gpuVanillaPrefixSum(T h_data[], int n, float* kernel_time_ms) {
    gpuScanRunner(vannilaKernel<T>, h_data, n, kernel_time_ms);
}

template void gpuVanillaPrefixSum<int>(int[], int, float*);
template void gpuVanillaPrefixSum<float>(float[], int, float*);
template void gpuVanillaPrefixSum<unsigned int>(unsigned int[], int, float*);



#define LOG_NUM_BANKS 5
#define INSERT_PADDING(n) ((n) + ((n) >> LOG_NUM_BANKS))

#define LOG_NUM_BANKS 5
#define INSERT_PADDING(n) ((n) + ((n) >> LOG_NUM_BANKS))

template <typename T>
__global__ void paddedVanillaKernel(T *ac) {
    extern __shared__ unsigned char shared_memory[];
    T* sdata = reinterpret_cast<T*>(shared_memory);
    
    T lastElement = 0;
    const int n = blockDim.x;
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * n + tid;

    sdata[INSERT_PADDING(tid)] = ac[idx];

    int offset = 1;
    // Upsweep (Reduction Phase)
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            
            sdata[INSERT_PADDING(bi)] += sdata[INSERT_PADDING(ai)];
        }
        offset <<= 1;
    }

    if (tid == 0) {
        lastElement = sdata[INSERT_PADDING(n - 1)];
        sdata[INSERT_PADDING(n - 1)] = 0; // Clear root
    }

    // Downsweep Phase
    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;

            int pad_ai = INSERT_PADDING(ai);
            int pad_bi = INSERT_PADDING(bi);

            T tmp = sdata[pad_ai];
            sdata[pad_ai] = sdata[pad_bi];
            sdata[pad_bi] += tmp;
        }
    }

    __syncthreads();

    if (tid == 0)
        ac[idx + n - 1] = lastElement;
    else
        ac[idx - 1] = sdata[INSERT_PADDING(tid)];
}



template <typename T>
void gpuPaddedVanillaPrefixSum(T h_data[], int n, float* kernel_time_ms) {
    gpuScanRunner(vannilaKernel<T>, h_data, n, kernel_time_ms, 0, 1.032f);
}

template void gpuPaddedVanillaPrefixSum<int>(int[], int, float*);
template void gpuPaddedVanillaPrefixSum<float>(float[], int, float*);
template void gpuPaddedVanillaPrefixSum<unsigned int>(unsigned int[], int, float*);



template <typename T>
__global__ void bitReverseSimpleKernel(T ac[]){
    extern __shared__ unsigned char shared_memory[]; // allocated on invocation
    T* sdata = reinterpret_cast<T*>(shared_memory);
    T lastElement;
    const int n = blockDim.x;
    const int idx = n - 1 - bit_reverse_pow2(threadIdx.x, n);
    const int offsetIdx = blockIdx.x * n;
    sdata[idx] = ac[offsetIdx + threadIdx.x]; 
    for (unsigned int s = n >> 1; s > 0; s >>= 1) {
        __syncthreads();
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
    }
    if (threadIdx.x == 0) {
        lastElement = sdata[0];
        sdata[0] = 0;  // clear the last element
    }
    for (unsigned int s = 1; s < n; s <<= 1) {
        __syncthreads();
        if (threadIdx.x < s) {
            T tmp = sdata[threadIdx.x + s];
            sdata[threadIdx.x + s] = sdata[threadIdx.x];
            sdata[threadIdx.x] += tmp;
        }
    }
    __syncthreads();
    if(threadIdx.x == 0)
        ac[offsetIdx + n - 1] = lastElement;
    else 
        ac[offsetIdx + threadIdx.x - 1] = sdata[idx];
}


template <typename T>
void gpuBitReversePrefixSumSimple(T* h_data, int n, float* kernel_time_ms) {
    gpuScanRunner(bitReverseSimpleKernel<T>, h_data, n, kernel_time_ms);
}
template void gpuBitReversePrefixSumSimple<int>(int[], int, float*);
template void gpuBitReversePrefixSumSimple<float>(float[], int, float*);
template void gpuBitReversePrefixSumSimple<unsigned int>(unsigned int[], int, float*);


template <typename T>
__global__ void bitReverseWarpKernel(T ac[]){
    extern __shared__ unsigned char shared_memory[]; // allocated on invocation
    T* sdata = reinterpret_cast<T*>(shared_memory);
    T lastElement;
    const int n = blockDim.x;
    const int tid = threadIdx.x;
    const int idx = n - 1 - bit_reverse_pow2(tid, n);
    const int offsetIdx = blockIdx.x * n;
    sdata[idx] = ac[offsetIdx + tid];
    unsigned int s;
    T tmp;
    for (s = n >> 1; s >= 32; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
    }
    if (tid < 32) {
        // Cache in register
        if (tid < 16) sdata[tid] += sdata[tid + 16]; __syncwarp();
        if (tid < 8)  sdata[tid] += sdata[tid + 8];  __syncwarp();
        if (tid < 4)  sdata[tid] += sdata[tid + 4];  __syncwarp();
        if (tid < 2)  sdata[tid] += sdata[tid + 2];  __syncwarp();
        if (tid < 1)  sdata[tid] += sdata[tid + 1];  __syncwarp();
    }
    if (tid == 0) {
        lastElement = sdata[0];
        sdata[0] = 0;  // clear the last element
    }
    if (tid < 32) {
        if (tid < 1) { tmp = sdata[tid + 1]; sdata[tid + 1] = sdata[tid]; sdata[tid] += tmp; } __syncwarp();
        if (tid < 2) { tmp = sdata[tid + 2]; sdata[tid + 2] = sdata[tid]; sdata[tid] += tmp; } __syncwarp();
        if (tid < 4) { tmp = sdata[tid + 4]; sdata[tid + 4] = sdata[tid]; sdata[tid] += tmp; } __syncwarp();
        if (tid < 8) { tmp = sdata[tid + 8]; sdata[tid + 8] = sdata[tid]; sdata[tid] += tmp; } __syncwarp();
        if (tid < 16) { tmp = sdata[tid + 16]; sdata[tid + 16] = sdata[tid]; sdata[tid] += tmp; } __syncwarp();
    }
    for (s = 32; s < n; s <<= 1) {
        if (tid < s) {
            tmp = sdata[tid + s];
            sdata[tid + s] = sdata[tid];
            sdata[tid] += tmp;
        }
        __syncthreads();
    }
    if(tid == 0)
        ac[offsetIdx + n - 1] = lastElement;
    else 
        ac[offsetIdx + tid - 1] = sdata[idx];
}

template <typename T>
void gpuBitReversePrefixSumWarp(T* h_data, int n, float* kernel_time_ms) {
    gpuScanRunner(bitReverseWarpKernel<T>, h_data, n, kernel_time_ms);
}
template void gpuBitReversePrefixSumWarp<int>(int[], int, float*);
template void gpuBitReversePrefixSumWarp<float>(float[], int, float*);
template void gpuBitReversePrefixSumWarp<unsigned int>(unsigned int[], int, float*);


template <typename T>
__global__ void bitReverseShuffleKernel(T ac[]){
    extern __shared__ unsigned char shared_memory[]; // allocated on invocation
    T* sdata = reinterpret_cast<T*>(shared_memory);
    T lastElement;
    const int n = blockDim.x;
    const int tid = threadIdx.x;
    const int idx = n - 1 - bit_reverse_pow2(tid, n);
    const int offsetIdx = blockIdx.x * n;
    sdata[idx] = ac[offsetIdx + tid];
    unsigned int s;
    T v, remote, r_up, r_down;
    for (s = n >> 1; s >= 32; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
    }
    if (tid < 32) {
        v = sdata[tid];

        // Warp Upsweep
        remote = __shfl_down_sync(0xffffffff, v, 16); if (tid < 16) v += remote;
        remote = __shfl_down_sync(0xffffffff, v, 8);  if (tid < 8)  v += remote;
        remote = __shfl_down_sync(0xffffffff, v, 4);  if (tid < 4)  v += remote;
        remote = __shfl_down_sync(0xffffffff, v, 2);  if (tid < 2)  v += remote;
        remote = __shfl_down_sync(0xffffffff, v, 1);  if (tid < 1)  v += remote;

        // Clear root and save total sum
        if (tid == 0) {
            lastElement = v;
            v = 0;
        }

        // Warp Downsweep (Manual Unroll)
        // Stride 1
        r_up = __shfl_up_sync(0xffffffff, v, 1); r_down = __shfl_down_sync(0xffffffff, v, 1);
        if (tid < 1) v += r_down; else if (tid < 2) v = r_up;

        // Stride 2
        r_up = __shfl_up_sync(0xffffffff, v, 2); r_down = __shfl_down_sync(0xffffffff, v, 2);
        if (tid < 2) v += r_down; else if (tid < 4) v = r_up;

        // Stride 4
        r_up = __shfl_up_sync(0xffffffff, v, 4); r_down = __shfl_down_sync(0xffffffff, v, 4);
        if (tid < 4) v += r_down; else if (tid < 8) v = r_up;

        // Stride 8
        r_up = __shfl_up_sync(0xffffffff, v, 8); r_down = __shfl_down_sync(0xffffffff, v, 8);
        if (tid < 8) v += r_down; else if (tid < 16) v = r_up;

        // Stride 16
        r_up = __shfl_up_sync(0xffffffff, v, 16); r_down = __shfl_down_sync(0xffffffff, v, 16);
        if (tid < 16) v += r_down; else if (tid < 32) v = r_up;

        sdata[tid] = v;
    }
    for (s = 32; s < n; s <<= 1) {
        if (tid < s) {
            v = sdata[tid + s];
            sdata[tid + s] = sdata[tid];
            sdata[tid] += v;
        }
        __syncthreads();
    }
    if(tid == 0)
        ac[offsetIdx + n - 1] = lastElement;
    else 
        ac[offsetIdx + tid - 1] = sdata[idx];
}

template <typename T>
void gpuBitReversePrefixSumShuffle(T* h_data, int n, float* kernel_time_ms) {
    gpuScanRunner(bitReverseShuffleKernel<T>, h_data, n, kernel_time_ms);
}
template void gpuBitReversePrefixSumShuffle<int>(int[], int, float*);
template void gpuBitReversePrefixSumShuffle<float>(float[], int, float*);
template void gpuBitReversePrefixSumShuffle<unsigned int>(unsigned int[], int, float*);

template <typename T>
__global__ void bitReverseShuffleTwiceKernel(T ac[]){
    extern __shared__ unsigned char shared_memory[]; // allocated on invocation
    T* sdata = reinterpret_cast<T*>(shared_memory);
    T lastElement;
    const int n = blockDim.x << 1;
    const int tid = threadIdx.x;
    const int idx = n - 1 - bit_reverse_pow2(tid, n); // always odd
    const int offsetIdx = blockIdx.x * n;
    sdata[idx] = ac[offsetIdx + tid];
    sdata[idx - 1] = ac[offsetIdx + tid + blockDim.x];
    T v, remote, r_up, r_down;
    unsigned int s;
    for (s = n >> 1; s >= 32; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
    }
    if (tid < 32) {
        v = sdata[tid];

        // Warp Upsweep
        remote = __shfl_down_sync(0xffffffff, v, 16); if (tid < 16) v += remote;
        remote = __shfl_down_sync(0xffffffff, v, 8);  if (tid < 8)  v += remote;
        remote = __shfl_down_sync(0xffffffff, v, 4);  if (tid < 4)  v += remote;
        remote = __shfl_down_sync(0xffffffff, v, 2);  if (tid < 2)  v += remote;
        remote = __shfl_down_sync(0xffffffff, v, 1);  if (tid < 1)  v += remote;

        // Clear root and save total sum
        if (tid == 0) {
            lastElement = v;
            v = 0;
        }

        // Warp Downsweep (Manual Unroll)
        // Stride 1
        r_up = __shfl_up_sync(0xffffffff, v, 1); r_down = __shfl_down_sync(0xffffffff, v, 1);
        if (tid < 1) v += r_down; else if (tid < 2) v = r_up;

        // Stride 2
        r_up = __shfl_up_sync(0xffffffff, v, 2); r_down = __shfl_down_sync(0xffffffff, v, 2);
        if (tid < 2) v += r_down; else if (tid < 4) v = r_up;

        // Stride 4
        r_up = __shfl_up_sync(0xffffffff, v, 4); r_down = __shfl_down_sync(0xffffffff, v, 4);
        if (tid < 4) v += r_down; else if (tid < 8) v = r_up;

        // Stride 8
        r_up = __shfl_up_sync(0xffffffff, v, 8); r_down = __shfl_down_sync(0xffffffff, v, 8);
        if (tid < 8) v += r_down; else if (tid < 16) v = r_up;

        // Stride 16
        r_up = __shfl_up_sync(0xffffffff, v, 16); r_down = __shfl_down_sync(0xffffffff, v, 16);
        if (tid < 16) v += r_down; else if (tid < 32) v = r_up;

        sdata[tid] = v;
    }
    for (s = 32; s < n; s <<= 1) {
        if (tid < s) {
            v = sdata[tid + s];
            sdata[tid + s] = sdata[tid];
            sdata[tid] += v;
        }
        __syncthreads();
    }
    if(tid == 0) {
        ac[offsetIdx + n - 1] = lastElement;
    }
    else {
        ac[offsetIdx + tid - 1] = sdata[idx];
    }
    ac[offsetIdx + tid + blockDim.x - 1] = sdata[idx - 1];
}

template <typename T>
void gpuBitReversePrefixSumShuffleTwice(T* h_data, int n, float* kernel_time_ms) {
    gpuScanRunner(bitReverseShuffleTwiceKernel<T>, h_data, n, kernel_time_ms, 1, 2.f);
}
template void gpuBitReversePrefixSumShuffleTwice<int>(int[], int, float*);
template void gpuBitReversePrefixSumShuffleTwice<float>(float[], int, float*);
template void gpuBitReversePrefixSumShuffleTwice<unsigned int>(unsigned int[], int, float*);


template <typename T>
__global__ void bitReverseVectorizeKernel(T ac[]){
    extern __shared__ unsigned char shared_memory[]; 
    T* sdata = reinterpret_cast<T*>(shared_memory);
    T lastElement;
    
    const int n = blockDim.x << 1;
    const int tid = threadIdx.x;
    const int offsetIdx = blockIdx.x * n;

    // --- 1. Vectorized Load (The Magic) ---
    T val0, val1;
    if constexpr (sizeof(T) == 4) {
        // Load two adjacent 4-byte elements in one 8-byte transaction
        uint2 raw = reinterpret_cast<uint2*>(ac + offsetIdx)[tid];
        val0 = *(reinterpret_cast<T*>(&raw.x));
        val1 = *(reinterpret_cast<T*>(&raw.y));
    } else {
        val0 = ac[offsetIdx + tid << 1];
        val1 = ac[offsetIdx + (tid << 1) + 1];
    }

    // --- 2. Map to Shared Memory using Bit-Reverse ---
    // ac[i] must go to sdata[n - 1 - bit_reverse(i, n)] to keep logic same
    const int idx0 = n - 1 - bit_reverse_pow2(tid << 1, n);
    const int idx1 = n - 1 - bit_reverse_pow2((tid << 1) + 1, n);
    
    sdata[idx0] = val0;
    sdata[idx1] = val1;

    // --- 3. Parallel Scan Logic (Same as before) ---
    T v, remote, r_up, r_down;
    unsigned int s;
    for (s = n >> 1; s >= 32; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
    }

    if (tid < 32) {
        v = sdata[tid];
        // Warp Upsweep
        remote = __shfl_down_sync(0xffffffff, v, 16); if (tid < 16) v += remote;
        remote = __shfl_down_sync(0xffffffff, v, 8);  if (tid < 8)  v += remote;
        remote = __shfl_down_sync(0xffffffff, v, 4);  if (tid < 4)  v += remote;
        remote = __shfl_down_sync(0xffffffff, v, 2);  if (tid < 2)  v += remote;
        remote = __shfl_down_sync(0xffffffff, v, 1);  if (tid < 1)  v += remote;

        if (tid == 0) { lastElement = v; v = 0; }

        // Warp Downsweep
        r_up = __shfl_up_sync(0xffffffff, v, 1); r_down = __shfl_down_sync(0xffffffff, v, 1);
        if (tid < 1) v += r_down; else if (tid < 2) v = r_up;
        r_up = __shfl_up_sync(0xffffffff, v, 2); r_down = __shfl_down_sync(0xffffffff, v, 2);
        if (tid < 2) v += r_down; else if (tid < 4) v = r_up;
        r_up = __shfl_up_sync(0xffffffff, v, 4); r_down = __shfl_down_sync(0xffffffff, v, 4);
        if (tid < 4) v += r_down; else if (tid < 8) v = r_up;
        r_up = __shfl_up_sync(0xffffffff, v, 8); r_down = __shfl_down_sync(0xffffffff, v, 8);
        if (tid < 8) v += r_down; else if (tid < 16) v = r_up;
        r_up = __shfl_up_sync(0xffffffff, v, 16); r_down = __shfl_down_sync(0xffffffff, v, 16);
        if (tid < 16) v += r_down; else if (tid < 32) v = r_up;

        sdata[tid] = v;
    }

    for (s = 32; s < n; s <<= 1) {
        __syncthreads(); // Synchronize before each stage
        if (tid < s) {
            v = sdata[tid + s];
            sdata[tid + s] = sdata[tid];
            sdata[tid] += v;
        }
    }
    __syncthreads();

    // --- 4. Vectorized Store (Optional but consistent) ---
    // For the store, we use your original mapping to ensure correctness
    if(tid == 0) {
        ac[offsetIdx + n - 1] = lastElement;
    }
    else {
        ac[offsetIdx + (tid << 1) - 1] = sdata[idx0]; // Use original thread-to-idx mapping
    }
    // ac[offsetIdx + tid << 1] = sdata[idx1];
}

template <typename T>
void gpuBitReversePrefixSumVectorize(T* h_data, int n, float* kernel_time_ms) {
    gpuScanRunner(bitReverseVectorizeKernel<T>, h_data, n, kernel_time_ms, 1, 2.f);
}
template void gpuBitReversePrefixSumVectorize<int>(int[], int, float*);
template void gpuBitReversePrefixSumVectorize<float>(float[], int, float*);
template void gpuBitReversePrefixSumVectorize<unsigned int>(unsigned int[], int, float*);


/**
 * Swizzle function: XORs the index with its higher bits.
 * This effectively rotates the bank assignment for every 32-element stride,
 * ensuring that power-of-2 strides (common in tree-based scans) 
 * hit different shared memory banks.
 */
__device__ __forceinline__ int swz(int idx) {
    return idx ^ (idx >> 5);
}

template <typename T>
__global__ void swizzledKernel(T *ac) {
    extern __shared__ unsigned char shared_memory[];
    T* sdata = reinterpret_cast<T*>(shared_memory);
    
    T lastElement = 0;
    const int n = blockDim.x;
    const int tid = threadIdx.x;
    const int globalIdx = blockIdx.x * n + tid;

    // 1. Standard Load (Non-Vectorized)
    // Every thread loads exactly one element into swizzled shared memory
    sdata[swz(tid)] = ac[globalIdx];

    int offset = 1;

    // 2. Upsweep (Reduction Phase)
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            // Calculate logical indices
            const int ai = offset * (2 * tid + 1) - 1;
            const int bi = offset * (2 * tid + 2) - 1;
            
            // Access shared memory using Swizzled indices
            sdata[swz(bi)] += sdata[swz(ai)];
        }
        offset <<= 1;
    }

    // 3. Root handling (Exclusive Scan logic)
    if (tid == 0) {
        lastElement = sdata[swz(n - 1)];
        sdata[swz(n - 1)] = 0; // Clear the last element
    }

    // 4. Downsweep Phase
    for (int d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            const int ai = offset * (2 * tid + 1) - 1;
            const int bi = offset * (2 * tid + 2) - 1;

            int swz_ai = swz(ai);
            int swz_bi = swz(bi);

            T tmp = sdata[swz_ai];
            sdata[swz_ai] = sdata[swz_bi];
            sdata[swz_bi] += tmp;
        }
    }

    __syncthreads();

    // 5. Store Back to Global Memory
    // Maintaining your original shifting logic
    if (tid == 0) {
        ac[globalIdx + n - 1] = lastElement;
    } else {
        ac[globalIdx - 1] = sdata[swz(tid)];
    }
}

template <typename T>
void gpuSwizzledPrefixSum(T* h_data, int n, float* kernel_time_ms) {
    gpuScanRunner(swizzledKernel<T>, h_data, n, kernel_time_ms);
}
template void gpuSwizzledPrefixSum<int>(int[], int, float*);
template void gpuSwizzledPrefixSum<float>(float[], int, float*);
template void gpuSwizzledPrefixSum<unsigned int>(unsigned int[], int, float*);




template <typename T>
__global__ void bitReverseShuffleOptimizedKernel(T ac[]){
    extern __shared__ unsigned char shared_memory[];
    T* sdata = reinterpret_cast<T*>(shared_memory);
    T lastElement;
    const int n = blockDim.x;
    const int tid = threadIdx.x;
    const int offsetIdx = blockIdx.x * n;

    // Use fast hardware bit-reverse
    const int rev_idx = n - 1 - bit_reverse_pow2(tid, n);

    // --- INITIAL LOAD (Fixed 32-way bank conflict) ---
    sdata[swz(rev_idx)] = ac[offsetIdx + tid];

    __syncthreads();

    T v, remote, r_up, r_down;
    unsigned int s;

    // --- UPSWEEP TREE ---
    for (s = n >> 1; s >= 32; s >>= 1) {
        if (tid < s) {
            // We use swz() here because the data was stored with swz()
            // This remains Bank-Conflict Free!
            sdata[swz(tid)] += sdata[swz(tid + s)];
        }
        __syncthreads();
    }

    if (tid < 32) {
        v = sdata[swz(tid)];

        // --- WARP SHUFFLE (Register level, no bank conflicts) ---
        remote = __shfl_down_sync(0xffffffff, v, 16); if (tid < 16) v += remote;
        remote = __shfl_down_sync(0xffffffff, v, 8);  if (tid < 8)  v += remote;
        remote = __shfl_down_sync(0xffffffff, v, 4);  if (tid < 4)  v += remote;
        remote = __shfl_down_sync(0xffffffff, v, 2);  if (tid < 2)  v += remote;
        remote = __shfl_down_sync(0xffffffff, v, 1);  if (tid < 1)  v += remote;

        if (tid == 0) { lastElement = v; v = 0; }

        r_up = __shfl_up_sync(0xffffffff, v, 1); r_down = __shfl_down_sync(0xffffffff, v, 1);
        if (tid < 1) v += r_down; else if (tid < 2) v = r_up;
        r_up = __shfl_up_sync(0xffffffff, v, 2); r_down = __shfl_down_sync(0xffffffff, v, 2);
        if (tid < 2) v += r_down; else if (tid < 4) v = r_up;
        r_up = __shfl_up_sync(0xffffffff, v, 4); r_down = __shfl_down_sync(0xffffffff, v, 4);
        if (tid < 4) v += r_down; else if (tid < 8) v = r_up;
        r_up = __shfl_up_sync(0xffffffff, v, 8); r_down = __shfl_down_sync(0xffffffff, v, 8);
        if (tid < 8) v += r_down; else if (tid < 16) v = r_up;
        r_up = __shfl_up_sync(0xffffffff, v, 16); r_down = __shfl_down_sync(0xffffffff, v, 16);
        if (tid < 16) v += r_down; else if (tid < 32) v = r_up;

        sdata[swz(tid)] = v;
    }

    // --- DOWNSWEEP TREE ---
    for (s = 32; s < n; s <<= 1) {
        __syncthreads();
        if (tid < s) {
            v = sdata[swz(tid + s)];
            sdata[swz(tid + s)] = sdata[swz(tid)];
            sdata[swz(tid)] += v;
        }
    }
    
    __syncthreads();

    // --- FINAL STORE (Fixed 32-way bank conflict) ---
    if(tid == 0) {
        ac[offsetIdx + n - 1] = lastElement;
    } else {
        ac[offsetIdx + tid - 1] = sdata[swz(rev_idx)];
    }
}

template <typename T>
void gpuBitReverseSwizzledPrefixSum(T* h_data, int n, float* kernel_time_ms) {
    gpuScanRunner(bitReverseShuffleOptimizedKernel<T>, h_data, n, kernel_time_ms);
}
template void gpuBitReverseSwizzledPrefixSum<int>(int[], int, float*);
template void gpuBitReverseSwizzledPrefixSum<float>(float[], int, float*);
template void gpuBitReverseSwizzledPrefixSum<unsigned int>(unsigned int[], int, float*);
