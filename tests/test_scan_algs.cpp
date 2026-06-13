#include "utest/utest.h"
#include "prefix_sum.h"
#include <vector>
#include <cstdio>


/**
 * Structure to hold algorithm metadata
 */
template <typename T>
struct ScanAlgorithm {
    const char* name;
    void (*func)(T*, int, float*);
};

/**
 * Fixture to manage the list of algorithms
 */
struct PrefixSumFixture {
    static inline ScanAlgorithm<int> int_algos[] = {
        {"Vanilla-Blelloch-Int", gpuVanillaPrefixSum<int>},
        {"Swizzled-Blelloch-Int", gpuSwizzledPrefixSum<int>},
        {"Padded-Blelloch-Int", gpuPaddedVanillaPrefixSum<int>},
        {"BitReverseSimple-Blelloch-Int", gpuBitReversePrefixSumSimple<int>},
        {"BitReverseWarp-Blelloch-Int", gpuBitReversePrefixSumWarp<int>},
        {"BitReverseShuffle-Blelloch-Int", gpuBitReversePrefixSumShuffle<int>},
        {"BitReverseShuffleTwice-Blelloch-Int", gpuBitReversePrefixSumShuffleTwice<int>},
        {"BitReverseSwizzled-Blelloch-Int", gpuBitReverseSwizzledPrefixSum<int>},
        // {"BitReverseVectorize-Blelloch-Int", gpuBitReversePrefixSumVectorize<int>} 
    };

    static inline ScanAlgorithm<float> float_algos[] = {
        {"Vanilla-Blelloch-Float", gpuVanillaPrefixSum<float>},
        {"Swizzled-Blelloch-Float", gpuSwizzledPrefixSum<float>},
        {"Padded-Blelloch-Float", gpuPaddedVanillaPrefixSum<float>},
        {"BitReverseSimple-Blelloch-Float", gpuBitReversePrefixSumSimple<float>}, 
        {"BitReverseWarp-Blelloch-Float", gpuBitReversePrefixSumWarp<float>},
        {"BitReverseShuffle-Blelloch-Float", gpuBitReversePrefixSumShuffle<float>},
        {"BitReverseShuffleTwice-Blelloch-Float", gpuBitReversePrefixSumShuffleTwice<float>}, 
        {"BitReverseSwizzled-Blelloch-Float", gpuBitReverseSwizzledPrefixSum<float>},
        // {"BitReverseVectorize-Blelloch-Float", gpuBitReversePrefixSumVectorize<float>} 
    };
    int current_index;
};

#define NUM_INT_ALGOS (sizeof(PrefixSumFixture::int_algos) / sizeof(ScanAlgorithm<int>))
#define NUM_FLOAT_ALGOS (sizeof(PrefixSumFixture::float_algos) / sizeof(ScanAlgorithm<float>))

UTEST_I_SETUP(PrefixSumFixture) { 
    utest_fixture->current_index = (int)utest_index;
}
UTEST_I_TEARDOWN(PrefixSumFixture) { (void)utest_fixture; (void)utest_index; }

/**
 * TEST 1: Integer All-Ones & Consistency Test (Inclusive)
 * - Verifies that [1, 1, 1, ...] results in [1, 2, 3, ...] (Index + 1)
 */
UTEST_I(PrefixSumFixture, IntegerConsistencyAndInclusiveIndexTest, NUM_INT_ALGOS) {
    const int N = 1024;
    auto& algo = PrefixSumFixture::int_algos[utest_fixture->current_index];

    UTEST_PRINTF("[   INFO   ] Testing Inclusive Algo: %s\n", algo.name);

    std::vector<int> h_input(N, 1); 
    std::vector<int> cpu_ref = h_input;
    std::vector<int> gpu_res = h_input;

    // Run CPU and GPU versions
    cpuPrefixSum(cpu_ref.data(), N);
    float kernel_time_ms;
    algo.func(gpu_res.data(), N, &kernel_time_ms);

    UTEST_PRINTF("[   INFO   ] %s -> Kernel Time : %.4gms\n", algo.name, kernel_time_ms);

    for (int i = 0; i < N; ++i) {
        // Requirement for All-Ones Inclusive: output[i] == i + 1
        EXPECT_EQ(i + 1, gpu_res[i]);
        
        // Match CPU Reference exactly
        EXPECT_EQ(cpu_ref[i], gpu_res[i]);
    }
}

/**
 * TEST 2: Floating Point Random Data Test (Inclusive)
 */
UTEST_I(PrefixSumFixture, FloatInclusiveConsistencyTest, NUM_FLOAT_ALGOS) {
    const int N = 1024;
    auto& algo = PrefixSumFixture::float_algos[utest_fixture->current_index];

    UTEST_PRINTF("[   INFO   ] Testing Inclusive Algo: %s\n", algo.name);

    std::vector<float> h_input(N);
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i % 11) * 0.1f;
    }

    std::vector<float> cpu_ref = h_input;
    std::vector<float> gpu_res = h_input;

    cpuPrefixSum(cpu_ref.data(), N);
    float kernel_time_ms;
    algo.func(gpu_res.data(), N, &kernel_time_ms);

    UTEST_PRINTF("[   INFO   ] %s -> Kernel Time : %.4gms\n", algo.name, kernel_time_ms);

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(cpu_ref[i], gpu_res[i], 1e-4f);
    }
}