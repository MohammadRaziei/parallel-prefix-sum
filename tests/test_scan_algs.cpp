#include "utest/utest.h"
#include "prefix_sum.h"
#include <vector>
#include <numeric> // for std::iota and std::fill

/**
 * Test 1: Compare Consistency (CPU vs Vanilla GPU vs Bit-Reverse GPU)
 * Using float data with random-like values.
 */
UTEST(PrefixSum, ComparisonTest) {
    const int N = 1024; // Single block limit
    std::vector<float> h_input(N);
    
    // Fill with some data
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i % 10);
    }

    std::vector<float> cpu_res = h_input;
    std::vector<float> vanilla_res = h_input;
    std::vector<float> bitrev_res = h_input;

    // 1. Run CPU
    cpuPrefixSum(cpu_res.data(), N);

    // 2. Run Vanilla GPU
    gpuVanillaPrefixSum(vanilla_res.data(), N);

    // 3. Run Bit-Reverse GPU
    gpuBitReversePrefixSum(bitrev_res.data(), N);

    // Compare bit-reverse result with CPU and Vanilla
    for (int i = 0; i < N; ++i) {
        // Use EXPECT_NEAR for floats due to precision
        EXPECT_NEAR(cpu_res[i], vanilla_res[i], 1e-4f);
        EXPECT_NEAR(cpu_res[i], bitrev_res[i], 1e-4f);
    }
}

/**
 * Test 2: The "All Ones" Test for Integer Indices
 * If input is [1, 1, 1, ...], the Exclusive Scan output should be [0, 1, 2, ...]
 */
UTEST(PrefixSum, AllOnesIndexTest) {
    const int N = 512; 
    std::vector<int> h_data(N, 1); // Fill with all 1s

    // Run the GPU scan
    gpuVanillaPrefixSum(h_data.data(), N);

    // Verify indices
    for (int i = 0; i < N; ++i) {
        // For Exclusive Scan: result[i] should be i
        // (i.e. 0, 1, 2, 3, ...)
        EXPECT_EQ(i+1, h_data[i]);
    }
}

/**
 * Test 3: Bit-Reverse "All Ones" with Unsigned Int
 */
UTEST(PrefixSum, BitReverseAllOnesTest) {
    const int N = 256;
    std::vector<unsigned int> h_data(N, 1u);

    gpuBitReversePrefixSum(h_data.data(), N);

    for (unsigned int i = 0; i < N; ++i) {
        EXPECT_EQ(i+1, h_data[i]);
    }
}