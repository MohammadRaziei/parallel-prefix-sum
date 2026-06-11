#include "cpu_prefix_sum.h"

void cpuPrefixSum(float* data, int n) {
    if (n <= 0) return;
    for (int i = 1; i < n; i++) {
        data[i] += data[i - 1];
    }
}