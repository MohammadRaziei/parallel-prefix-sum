#include "gpubench.h"
#include "prefix_sum.h"
#include <vector>

int main() {
    int n = 1 << 20; 
    std::vector<float> h_data(n, 1.f);
    
    // Calculate total bytes processed (Read + Write)
    size_t total_bytes = static_cast<size_t>(n) * sizeof(float) * 2;

    // Create benchmark object (3 warm-ups, 50 measurements)
    GpuBench bench(5, 200);

    bench.run("Vanilla Prefix Sum", total_bytes, [&](float* t) {
        gpuVanillaPrefixSum(h_data.data(), n, t);
    });

    bench.run("Padded Vanilla Scan", total_bytes, [&](float* t) {
        gpuPaddedVanillaPrefixSum(h_data.data(), n, t);
    });

    bench.run("Swizzled Scan", total_bytes, [&](float* t) {
        gpuSwizzledPrefixSum(h_data.data(), n, t);
    });
    
    bench.run("Bit-Reverse Simple", total_bytes, [&](float* t) {
        gpuBitReversePrefixSumSimple(h_data.data(), n, t);
    });

    bench.run("Bit-Reverse Warp", total_bytes, [&](float* t) {
        gpuBitReversePrefixSumWarp(h_data.data(), n, t);
    });

    bench.run("Bit-Reverse Shuffle", total_bytes, [&](float* t) {
        gpuBitReversePrefixSumShuffle(h_data.data(), n, t);
    });

    bench.run("Bit-Reverse Shuffle Twice", total_bytes, [&](float* t) {
        gpuBitReversePrefixSumShuffleTwice(h_data.data(), n, t);
    });

    bench.run("Bit-Reverse Vectorize", total_bytes, [&](float* t) {
        gpuBitReversePrefixSumVectorize(h_data.data(), n, t);
    });

    bench.run("Bit-Reverse Swizzled", total_bytes, [&](float* t) {
        gpuBitReverseSwizzledPrefixSum(h_data.data(), n, t);
    });
    

    // Output the beautiful table
    bench.print_results();

    return 0;
}