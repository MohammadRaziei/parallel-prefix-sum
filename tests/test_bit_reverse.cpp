#include "utest/utest.h"
#include "gpu_prefix_sum.cuh"
#include <vector>

/**
 * Recursive formula using simple integer math
 */
int calculate_expected_omega(int i, int n_val) {
    if (i == 0) {
        return n_val - 1;
    }
    int prev_omega = calculate_expected_omega(i / 2, n_val);
    
    int term1 = (prev_omega - 1) / 2;
    int term2 = (1 - (i % 2)) * (n_val / 2);
    
    return term1 + term2;
}

/**
 * Fixture to store the iteration index
 */
struct BitReverseRangeTest {
    int current_index;
};

UTEST_I_SETUP(BitReverseRangeTest) {
    utest_fixture->current_index = (int)utest_index;
}

UTEST_I_TEARDOWN(BitReverseRangeTest) {
    (void)utest_fixture;
    (void)utest_index;
}

/**
 * Test loop from n=8 to n=2048 (9 iterations)
 */
UTEST_I(BitReverseRangeTest, RangeFrom8To2048, 9) {
    const int current_nBits = 3 + utest_fixture->current_index;
    const int n_val = 1 << current_nBits;

    // Use int vectors for simplicity
    std::vector<int> h_in(n_val);
    std::vector<int> h_out(n_val);

    for (int i = 0; i < n_val; i++) {
        h_in[i] = i;
    }

    // Call CUDA wrapper (casting vectors to the required pointer type if needed)
    // Note: If your wrapper expects unsigned int, just cast it there.
    gpuBitReverse((unsigned int*)h_out.data(), (unsigned int*)h_in.data(), current_nBits, n_val);

    for (int i = 0; i < n_val; i++) {
        int actual_omega_i = (n_val - 1) - h_out[i];
        int expected_omega_i = calculate_expected_omega(i, n_val);

        // Simple integer equality check
        EXPECT_EQ(expected_omega_i, actual_omega_i);
    }
}