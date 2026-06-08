#ifndef GPU_PREFIX_SUM_CUH
#define GPU_PREFIX_SUM_CUH

#include <cuda_runtime.h>

void gpuPrefixSum(float* h_data, int n);

#endif