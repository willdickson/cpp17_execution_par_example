cmake_minimum_required (VERSION 3.0)
set(CMAKE_CXX_COMPILER "g++-9")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")
project(example)
add_executable(main main.cpp)
include_directories(main)
target_link_libraries(main tbb armadillo)






