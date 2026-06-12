cmake_minimum_required(VERSION 3.18)
project(ParallelPrefixSum LANGUAGES CXX CUDA)

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/src)

# 1. Create a library for CPU code
add_library(cpu_prefix_sum STATIC src/cpu_prefix_sum.cpp src/cpu_prefix_sum.h)

# 2. Create a library for GPU code (Compiled by NVCC)
add_library(gpu_prefix_sum STATIC src/gpu_prefix_sum.cu  src/gpu_prefix_sum.cuh)
set_target_properties(gpu_prefix_sum PROPERTIES CUDA_ARCHITECTURES 75)

add_library(prefix_sum INTERFACE src/prefix_sum.h)
target_link_libraries(prefix_sum INTERFACE cpu_prefix_sum gpu_prefix_sum CUDA::cudart)


# 3. Create the main executable
add_executable(prefix_sum_example examples/main.cpp src/prefix_sum.h)
target_link_libraries(prefix_sum_example PRIVATE prefix_sum)
set_target_properties(prefix_sum_example PROPERTIES LINKER_LANGUAGE CUDA)


add_subdirectory(benchmarks)

enable_testing()
add_subdirectory(tests)