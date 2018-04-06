cmake_minimum_required(VERSION 3.10)
project(kristforge)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(kristforge main.cpp kristforge.cpp kristforge.h kristforge_opencl.cpp cl_amd.h)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIR})
target_link_libraries(kristforge ${OpenCL_LIBRARIES})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(kristforge Threads::Threads)

find_package(TCLAP REQUIRED)
include_directories(${TCLAP_INCLUDE_DIR})

pkg_check_modules(CURLPP REQUIRED curlpp)
target_link_libraries(kristforge ${CURLPP_LDFLAGS})

pkg_check_modules(JSONCPP jsoncpp)
target_link_libraries(kristforge ${JSONCPP_LIBRARIES})