#ifndef GPU_PREFIX_SUM_CUH
#define GPU_PREFIX_SUM_CUH


void gpuBitReverse(unsigned int* h_out, const unsigned int* h_in, int n, int size);
void gpuBitReversePow2(unsigned int* h_out, const unsigned int* h_in, int n, int size);

template <typename T> void gpuVanillaPrefixSum(T h_data[], int n, float* kernel_time_ms = nullptr);
template <typename T> void gpuPaddedVanillaPrefixSum(T h_data[], int n, float* kernel_time_ms = nullptr);
template <typename T> void gpuBitReversePrefixSumSimple(T h_data[], int n, float* kernel_time_ms = nullptr);
template <typename T> void gpuBitReversePrefixSumWarp(T h_data[], int n, float* kernel_time_ms = nullptr);
template <typename T> void gpuBitReversePrefixSumShuffle(T h_data[], int n, float* kernel_time_ms = nullptr);
template <typename T> void gpuBitReversePrefixSumShuffleTwice(T h_data[], int n, float* kernel_time_ms = nullptr);
template <typename T> void gpuBitReversePrefixSumVectorize(T h_data[], int n, float* kernel_time_ms = nullptr);
template <typename T> void gpuSwizzledPrefixSum(T h_data[], int n, float* kernel_time_ms = nullptr);
template <typename T> void gpuBitReverseSwizzledPrefixSum(T* h_data, int n, float* kernel_time_ms = nullptr);

#endif