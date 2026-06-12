cmake_minimum_required(VERSION 3.18)

add_executable(cpp_benchmark ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cpp)

target_link_libraries(cpp_benchmark PRIVATE prefix_sum)
set_target_properties(cpp_benchmark PROPERTIES LINKER_LANGUAGE CUDA)




