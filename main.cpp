#include <iostream>
#include <vector>
#include <cmath>
#include "prefix_sum.h"

// Reference CPU implementation for validation
void cpuExclusivePrefixSum(const float* input, float* output, int n) {
    output[0] = 0;
    for (int i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

int main() {
    const int N = 512; // Must be power of 2 for this simple kernel
    std::vector<float> h_input(N);
    std::vector<float> h_gpu_data(N);
    std::vector<float> h_cpu_ref(N);

    // Fill input with some values (e.g., 1.0, 2.0, 3.0...)
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i + 1);
    }

    // Copy to the buffer we will send to GPU
    h_gpu_data = h_input;

    // Perform GPU Prefix Sum
    runGpuPrefixSum(h_gpu_data.data(), N);

    // Perform CPU Prefix Sum for comparison
    cpuExclusivePrefixSum(h_input.data(), h_cpu_ref.data(), N);

    // Verification
    bool match = true;
    for (int i = 0; i < N; i++) {
        if (std::abs(h_gpu_data[i] - h_cpu_ref[i]) > 1e-4) {
            std::cout << "Mismatch at [" << i << "]: GPU=" << h_gpu_data[i] 
                      << ", CPU=" << h_cpu_ref[i] << std::endl;
            match = false;
            break;
        }
    }

    if (match) {
        std::cout << "SUCCESS: GPU Prefix Sum matches CPU reference!" << std::endl;
        std::cout << "First 5 results: ";
        for(int i=0; i<5; ++i) std::cout << h_gpu_data[i] << " ";
        std::cout << "..." << std::endl;
    }

    return 0;
}