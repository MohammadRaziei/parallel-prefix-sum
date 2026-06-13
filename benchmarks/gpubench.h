#ifndef GPU_BENCH_H
#define GPU_BENCH_H

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cmath>

class GpuBench {
private:
    struct Record {
        std::string name;
        double avg;
        double median;
        double min;
        double max;
        double std_dev;
        double throughput; // GB/s
    };

    std::vector<Record> records;
    int warmup_iters;
    int measure_iters;

public:
    GpuBench(int warmup = 3, int measure = 100) 
        : warmup_iters(warmup), measure_iters(measure) {}

    /**
     * @param name Name of the benchmark
     * @param data_size_bytes Total bytes processed (Read + Write) to calculate throughput
     * @param kernel_func Lambda or function pointer wrapping the kernel call
     */
    template <typename Func>
    void run(const std::string& name, size_t data_size_bytes, Func&& kernel_func) {
        float internal_time = 0;

        // 1. Warm-up phase
        for (int i = 0; i < warmup_iters; ++i) {
            kernel_func(&internal_time);
        }

        // 2. Measurement phase
        std::vector<float> timings;
        timings.reserve(measure_iters);

        for (int i = 0; i < measure_iters; ++i) {
            kernel_func(&internal_time);
            timings.push_back(internal_time);
        }

        // 3. Statistical Analysis
        std::sort(timings.begin(), timings.end());
        
        double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
        double avg = sum / measure_iters;
        double median = (measure_iters % 2 == 0) 
            ? (timings[measure_iters/2 - 1] + timings[measure_iters/2]) / 2.0
            : timings[measure_iters/2];
        
        double sq_sum = 0;
        for(float t : timings) sq_sum += (t - avg) * (t - avg);
        double std_dev = std::sqrt(sq_sum / measure_iters);

        // 4. Calculate Throughput (GB/s)
        // Formula: (Bytes / 10^9) / (seconds) 
        // which is: Bytes / (avg_ms * 10^6)
        double throughput = 0;
        if (data_size_bytes > 0 && avg > 0) {
            throughput = static_cast<double>(data_size_bytes) / (avg * 1e6);
        }

        records.push_back({name, avg, median, (double)timings.front(), (double)timings.back(), std_dev, throughput});
    }

    void print_results() {
        const int w_name = 25;
        const int w_val = 12;
        const int w_tp = 15;

        std::cout << "\n" << std::left 
                  << "| " << std::setw(w_name) << "Benchmark Name" 
                  << "| " << std::setw(w_val) << "Avg (ms)" 
                  << "| " << std::setw(w_val) << "Median" 
                  << "| " << std::setw(w_val) << "Min" 
                  << "| " << std::setw(w_val) << "Max" 
                  << "| " << std::setw(w_tp) << "TP (GB/s)" << " |" << std::endl;
        
        std::cout << "|" << std::string(w_name + 1, '-') 
                  << "|" << std::string(w_val + 1, '-') 
                  << "|" << std::string(w_val + 1, '-') 
                  << "|" << std::string(w_val + 1, '-') 
                  << "|" << std::string(w_val + 1, '-') 
                  << "|" << std::string(w_tp + 2, '-') << "|" << std::endl;

        for (const auto& r : records) {
            std::cout << std::left << "| " 
                      << std::setw(w_name) << r.name 
                      << "| " << std::setw(w_val) << std::fixed << std::setprecision(4) << r.avg
                      << "| " << std::setw(w_val) << r.median
                      << "| " << std::setw(w_val) << r.min
                      << "| " << std::setw(w_val) << r.max;
            
            if (r.throughput > 0)
                std::cout << "| " << std::setw(w_tp) << std::fixed << std::setprecision(2) << r.throughput;
            else
                std::cout << "| " << std::setw(w_tp) << "N/A";
            
            std::cout << " |" << std::endl;
        }
    }
};

#endif