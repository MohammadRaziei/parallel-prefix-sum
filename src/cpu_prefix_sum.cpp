#include "cpu_prefix_sum.h"

template <typename T>
void cpuPrefixSum(T* data, int n) {
    if (n <= 0) return;
    for (int i = 1; i < n; i++) {
        data[i] += data[i - 1];
    }
}
template void cpuPrefixSum<int>(int* h_data, int n);
template void cpuPrefixSum<float>(float* h_data, int n);
template void cpuPrefixSum<unsigned int>(unsigned int* h_data, int n);
