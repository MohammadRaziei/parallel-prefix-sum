cmake_minimum_required(VERSION 3.18)
project(ParallelPrefixSum LANGUAGES CXX CUDA)

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# 1. Create a library for CPU code
add_library(cpu_prefix_sum STATIC cpu_prefix_sum.cpp cpu_prefix_sum.h)

# 2. Create a library for GPU code (Compiled by NVCC)
add_library(gpu_prefix_sum STATIC gpu_prefix_sum.cu gpu_prefix_sum.cuh)
set_target_properties(gpu_prefix_sum PROPERTIES CUDA_ARCHITECTURES 75)

# 3. Create the main executable
add_executable(prefix_sum main.cpp prefix_sum.h)
set_target_properties(prefix_sum PROPERTIES LINKER_LANGUAGE CUDA)


# 4. Link everything together
# We link the libraries + the CUDA Runtime
target_link_libraries(prefix_sum PRIVATE 
    cpu_prefix_sum 
    gpu_prefix_sum 
    CUDA::cudart
)

