#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include "prefix_sum.h" // Assumes cpuPrefixSum and gpuPrefixSum are defined here



/**
 * @brief Prints an array to the console.
 * 
 * @tparam T The data type of the array elements.
 * @param arr Pointer to the array on the Host (CPU).
 * @param size Total number of elements in the array.
 * @param n_max Maximum number of elements to print. If -1, prints all.
 */
template <typename T>
void printArray(const T* arr, int size, int n_max = 16) {
    if (!arr) {
        std::cout << "Array is NULL" << std::endl;
        return;
    }

    if (size <= 0) {
        std::cout << "[]" << std::endl;
        return;
    }

    // Determine the actual number of elements to print
    int count = (n_max == -1 || n_max > size) ? size : n_max;

    std::cout << "[";
    for (int i = 0; i < count; ++i) {
        std::cout << arr[i];
        if (i < count - 1) {
            std::cout << ", ";
        }
    }

    // If truncated, show ellipsis and total size
    if (count < size) {
        std::cout << ", ... (total size: " << size << ")";
    }
    std::cout << "]" << std::endl;
}


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
        h_input[i] = static_cast<float>(i); 
    }

    std::vector<float> cpu_buffer = h_input;
    std::vector<float> gpu_buffer = h_input;

    std::cout << "Starting Prefix Sum comparison (N = " << N << ")..." << std::endl;
    printArray(h_input.data(), N);

    cpuPrefixSum(cpu_buffer.data(), N);
    printArray(cpu_buffer.data(), N);

    gpuBitReversePrefixSum(gpu_buffer.data(), N);
    printArray(gpu_buffer.data(), N);


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