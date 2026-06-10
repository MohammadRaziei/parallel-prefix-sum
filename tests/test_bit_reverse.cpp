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
 * Test loop from n=8 to n=4096 (10 iterations)
 */
UTEST_I(BitReverseRangeTest, RangeFrom8To2048, 10) {
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
        if (i > 0) {
            // Sanity check to ensure outputs are not all the same (which would indicate a potential issue)
            EXPECT_NE(h_out[i], h_out[i - 1]); // Ensure outputs are not all the same (sanity check)
        }

        // Compare the GPU output with the expected value from the recursive formula
        int actual_omega_i = (n_val - 1) - h_out[i];
        int expected_omega_i = calculate_expected_omega(i, n_val);

        // Simple integer equality check
        EXPECT_EQ(expected_omega_i, actual_omega_i);
    }
}





/**
 * Fixture to compare the two different implementations
 */
struct BitReverseComparisonFixture {
    int current_index;
};

UTEST_I_SETUP(BitReverseComparisonFixture) {
    utest_fixture->current_index = (int)utest_index;
}

UTEST_I_TEARDOWN(BitReverseComparisonFixture) {
    (void)utest_fixture;
    (void)utest_index;
}

/**
 * Parameterized Test: Compare gpuBitReverse vs gpuBitReversePow2
 * Iterations: 10 (from 2^3 to 2^12)
 */
UTEST_I(BitReverseComparisonFixture, CompareImplementations, 10) {
    // 1. Calculate parameters based on index
    const int m_bits = 3 + utest_fixture->current_index; // 3, 4, ..., 12
    const int size = 1 << m_bits;                       // 8, 16, ..., 4096
    const int n_pow2 = size;                            // For the Pow2 version, n is the size

    // 2. Prepare data
    std::vector<unsigned int> h_in(size);
    std::vector<unsigned int> h_out_bits(size);
    std::vector<unsigned int> h_out_pow2(size);

    for (int i = 0; i < size; i++) {
        h_in[i] = (unsigned int)i;
    }

    // 3. Run both implementations
    // Implementation A: Uses m_bits (e.g. 3)
    gpuBitReverse(h_out_bits.data(), h_in.data(), m_bits, size);

    // Implementation B: Uses n_pow2 (e.g. 8)
    gpuBitReversePow2(h_out_pow2.data(), h_in.data(), n_pow2, size);

    // 4. Verification: The outputs must be identical
    for (int i = 0; i < size; i++) {
        // We compare the results of the two GPU implementations directly
        // They should match bit-for-bit
        EXPECT_EQ(h_out_bits[i], h_out_pow2[i]);

        if (i > 0) {
            // Additional sanity check to ensure outputs are not all the same (which would indicate a potential issue)
            EXPECT_NE(h_out_bits[i], h_out_bits[i - 1]); // Ensure outputs are not all the same (sanity check)
        }
    }
}
