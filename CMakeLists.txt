cmake_minimum_required(VERSION 3.27)
project(tarea02blas)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

include(FetchContent)

FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 10.1.1
)

FetchContent_MakeAvailable(fmt)

add_executable(tarea02blas main.cpp)

target_link_libraries(tarea02blas fmt::fmt)
