cmake_minimum_required(VERSION 3.18)

file(GLOB TEST_SOURCES "*.cpp")


add_executable(cpp_tests ${TEST_SOURCES} utest/utest_main.cxx)

# set_target_properties(cpp_tests PROPERTIES
#     CXX_STANDARD 17
#     CXX_STANDARD_REQUIRED ON
#     CUDA_STANDARD 17
#     CUDA_STANDARD_REQUIRED ON
#     CUDA_SEPARABLE_COMPILATION ON
# )
target_link_libraries(cpp_tests PRIVATE prefix_sum)
set_target_properties(cpp_tests PROPERTIES LINKER_LANGUAGE CUDA)





add_test(NAME PrefixSumTests COMMAND $<TARGET_FILE:cpp_tests>)