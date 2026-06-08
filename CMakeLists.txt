cmake_minimum_required(VERSION 3.18)
project(CudaPrefixSum LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Adjust compute_75 to match your specific GPU architecture
set(CMAKE_CUDA_ARCHITECTURES 75)

add_executable(prefix_sum main.cpp gpu_prefix_sum.cu cpu_prefix_sum.cpp)
set_target_properties(prefix_sum PROPERTIES CUDA_ARCHITECTURES 75)
    
target_include_directories(prefix_sum PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})