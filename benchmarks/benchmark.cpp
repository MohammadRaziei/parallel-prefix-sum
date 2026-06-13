#include "gpubench.h"
#include "prefix_sum.h"
#include <vector>

int main() {
    int n = 1 << 20; 
    std::vector<float> h_data(n, 1.f);

    // Create benchmark object (3 warm-ups, 50 measurements)
    GpuBench bench(5, 200);

    bench.run("Vanilla Prefix Sum", [&](float* t) {
        gpuVanillaPrefixSum(h_data.data(), n, t);
    });

    bench.run("Bit-Reverse Simple", [&](float* t) {
        gpuBitReversePrefixSumSimple(h_data.data(), n, t);
    });

    bench.run("Bit-Reverse Warp", [&](float* t) {
        gpuBitReversePrefixSumWarp(h_data.data(), n, t);
    });

    bench.run("Bit-Reverse Shuffle", [&](float* t) {
        gpuBitReversePrefixSumShuffle(h_data.data(), n, t);
    });

    bench.run("Bit-Reverse Shuffle Twice", [&](float* t) {
        gpuBitReversePrefixSumShuffleTwice(h_data.data(), n, t);
    });


    // Output the beautiful table
    bench.print_results();

    return 0;
}