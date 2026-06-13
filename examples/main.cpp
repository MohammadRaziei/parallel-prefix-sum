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


int main2() {
    constexpr int n = 1024;
    
    unsigned int bit_revese[n];
    for (int i = 0; i < n; i++) bit_revese[i] = i;
    gpuBitReversePow2(bit_revese, bit_revese, n, n);
    std::cout << "Analysis of Memory Bank Access for n=1024 (10 bits)" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Analyze each Warp (32 threads per warp)
    // There are 1024 / 32 = 32 warps total
    std::cout << "Analysis of Memory Bank Access for n=1024 (10 bits)" << std::endl;
    std::cout << "Each Bank = index % 32" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    for (int warpId = 0; warpId < 4; warpId++) { // Analyzing first 4 warps as sample
        // Using a vector of 32 vectors to see which threads hit which bank
        std::vector<int> bank_map[32];
        
        int startThread = warpId * 32;
        int endThread = startThread + 32;

        for (int tid = startThread; tid < endThread; tid++) {
            unsigned int reversedIdx = bit_revese[tid];
            int bank = reversedIdx % 32;
            bank_map[bank].push_back(tid % 32);
        }

        std::cout << "\n>>> WARP " << warpId << " (Threads " << startThread << "-" << endThread-1 << "):" << std::endl;
        
        int conflict_count = 0;
        for (int b = 0; b < 32; b++) {
            if (!bank_map[b].empty()) {
                std::cout << "Bank [" << std::setw(2) << b << "]: accessed by " 
                          << bank_map[b].size() << " threads. TIDs: ";
                for (int t : bank_map[b]) std::cout << t << " ";
                std::cout << std::endl;
                
                if (bank_map[b].size() > 1) conflict_count = bank_map[b].size();
            }
        }
        std::cout << "Warp " << warpId << " has a " << conflict_count << "-way Bank Conflict!" << std::endl;
    }

    std::cout << "\n..." << std::endl;
    std::cout << "[Rest of the 32 warps follow the same pattern]" << std::endl;

    return 0;
}

int main() {
    const int N = 4096; 
    std::vector<float> h_input(N);
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(1); 
    }

    std::vector<float> cpu_buffer = h_input;
    std::vector<float> gpu_buffer = h_input;

    std::cout << "Starting Prefix Sum comparison (N = " << N << ")..." << std::endl;
    printArray(h_input.data(), N);

    cpuPrefixSum(cpu_buffer.data(), N);
    // printArray(cpu_buffer.data(), N, -1);

    gpuBitReversePrefixSumShuffleTwice(gpu_buffer.data(), N);
    printArray(gpu_buffer.data(), N, -1);


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