#ifndef GPU_PREFIX_SUM_CUH
#define GPU_PREFIX_SUM_CUH

void gpuPrefixSum(float* h_data, int n);
void gpuBitReverse(unsigned int* h_out, const unsigned int* h_in, int n, int size);

#endif