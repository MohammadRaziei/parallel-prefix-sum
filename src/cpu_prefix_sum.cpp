#include "cpu_prefix_sum.h"

void cpuPrefixSum(float* data, int n) {
    if (n <= 0) return;
    
    float prev = data[0];
    data[0] = 0; // Exclusive scan
    for (int i = 1; i < n; i++) {
        float current = data[i];
        data[i] = data[i - 1] + prev;
        prev = current;
    }
}