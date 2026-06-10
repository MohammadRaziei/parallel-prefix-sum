#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include "prefix_sum.h" // Assumes cpuPrefixSum and gpuPrefixSum are defined here


// --- MAE Calculation Utility ---
double calculateMAE(const float* arr1, const float* arr2, int n) {
    double totalError = 0.0;
    for (int i = 0; i < n; ++i) {
        totalError += std::abs(arr1[i] - arr2[i]);
    }
    return totalError / n;
}

int main() {
    const int N = 1024; 
    std::vector<float> h_input(N);
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i % 10 + 1); 
    }

    std::vector<float> cpu_buffer = h_input;
    std::vector<float> gpu_buffer = h_input;

    std::cout << "Starting Prefix Sum comparison (N = " << N << ")..." << std::endl;

    cpuPrefixSum(cpu_buffer.data(), N);
    gpuPrefixSum(gpu_buffer.data(), N);

    double mae = calculateMAE(cpu_buffer.data(), gpu_buffer.data(), N);

    std::cout << "\n================ RESULT ANALYSIS ================" << std::endl;
    std::cout << "Mean Absolute Error (MAE): " << std::scientific << mae << std::fixed << std::endl;
    
    if (mae < 1e-5) {
        std::cout << "Status: SUCCESS (GPU result is accurate)" << std::endl;
    } else {
        std::cout << "Status: FAILURE (High error detected)" << std::endl;
    }
    std::cout << "=================================================" << std::endl;

    return 0;
}