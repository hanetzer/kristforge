cmake_minimum_required(VERSION 3.10)
project(kristforge)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(kristforge main.cpp kristforge.cpp kristforge.h kristforge_opencl.cpp cl_amd.h krist.cpp krist.h)

find_package(OpenCL REQUIRED)
target_include_directories(kristforge PUBLIC ${OpenCL_INCLUDE_DIR})
target_link_libraries(kristforge PUBLIC ${OpenCL_LIBRARIES})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(kristforge PRIVATE Threads::Threads)

find_package(TCLAP REQUIRED)
target_include_directories(kristforge PRIVATE ${TCLAP_INCLUDE_DIR})

pkg_check_modules(CURLPP REQUIRED curlpp)
target_link_libraries(kristforge PRIVATE ${CURLPP_LDFLAGS})

pkg_check_modules(JSONCPP jsoncpp)
target_link_libraries(kristforge PRIVATE ${JSONCPP_LIBRARIES})

find_path(UWEBSOCKETS_INCLUDE_DIRS uWS)
find_library(UWEBSOCKETS_LIBRARIES uWS)
target_include_directories(kristforge PRIVATE ${UWEBSOCKETS_INCLUDE_DIRS})
target_link_libraries(kristforge PRIVATE ${UWEBSOCKETS_LIBRARIES})

# dependency of uwebsockets
find_package(ZLIB REQUIRED)
target_link_libraries(kristforge PRIVATE ${ZLIB_LIBRARIES})