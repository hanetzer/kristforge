cmake_minimum_required(VERSION 3.10)
project(kristforge)

set(CMAKE_CXX_STANDARD 17)

add_executable(kristforge main.cpp kristforge.cpp kristforge.h kristforge_opencl.cpp cl_amd.h)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIR})
target_link_libraries(kristforge ${OpenCL_LIBRARIES})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(kristforge Threads::Threads)