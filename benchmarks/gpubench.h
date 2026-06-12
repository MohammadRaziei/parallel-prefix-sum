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
    };

    std::vector<Record> records;
    int warmup_iters;
    int measure_iters;

public:
    GpuBench(int warmup = 3, int measure = 100) 
        : warmup_iters(warmup), measure_iters(measure) {}

    // Flexible run: accepts only name and a callable (lambda)
    template <typename Func>
    void run(const std::string& name, Func&& kernel_func) {
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

        records.push_back({name, avg, median, (double)timings.front(), (double)timings.back(), std_dev});
    }

    void print_results() {
        const int w_name = 25;
        const int w_val = 12;

        std::cout << "\n" << std::left 
                  << "| " << std::setw(w_name) << "Benchmark Name" 
                  << "| " << std::setw(w_val) << "Avg (ms)" 
                  << "| " << std::setw(w_val) << "Median" 
                  << "| " << std::setw(w_val) << "Min" 
                  << "| " << std::setw(w_val) << "Max" 
                  << "| " << std::setw(w_val) << "StdDev" << " |" << std::endl;
        
        std::cout << "|" << std::string(w_name + 1, '-') 
                  << "|" << std::string(w_val + 1, '-') 
                  << "|" << std::string(w_val + 1, '-') 
                  << "|" << std::string(w_val + 1, '-') 
                  << "|" << std::string(w_val + 1, '-') 
                  << "|" << std::string(w_val + 2, '-') << "|" << std::endl;

        for (const auto& r : records) {
            std::cout << std::left << "| " 
                      << std::setw(w_name) << r.name 
                      << "| " << std::setw(w_val) << std::fixed << std::setprecision(4) << r.avg
                      << "| " << std::setw(w_val) << r.median
                      << "| " << std::setw(w_val) << r.min
                      << "| " << std::setw(w_val) << r.max
                      << "| " << std::setw(w_val) << r.std_dev << " |" << std::endl;
        }
    }
};

#endif