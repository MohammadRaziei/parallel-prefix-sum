#ifndef GPU_PREFIX_SUM_CUH
#define GPU_PREFIX_SUM_CUH


void gpuBitReverse(unsigned int* h_out, const unsigned int* h_in, int n, int size);
void gpuBitReversePow2(unsigned int* h_out, const unsigned int* h_in, int n, int size);

template <typename T> void gpuVanillaPrefixSum(T* h_data, int n);
template <typename T> void gpuBitReversePrefixSum(T* h_data, int n);

#endif