cmake_minimum_required(VERSION 3.15)
project(bmmpy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(bmmpy_core STATIC 
    src/stub.cpp
)
target_include_directories(bmmpy_core PUBLIC include)

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

add_subdirectory(extern/nanobind)

nanobind_add_module(
    bmmpy
    NB_DOMAIN bmmpy
    python/bindings/main.cpp
)

target_link_libraries(bmmpy PRIVATE bmmpy_core)

add_executable(cli src/cli.cpp) 
target_link_libraries(cli PRIVATE bmmpy_core)