cmake_minimum_required(VERSION 3.15)

project(bmmpy LANGUAGES CXX)


# Global project settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# bmmpy_core is linked into the Python extension, so it must be PIC-safe.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)
option(BMMPY_BUILD_BENCHMARKS "Build benchmark targets" OFF)

# Project version
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" PYPROJECT_CONTENT)
string(REGEX MATCH "version = \"([^\"]+)\"" _ "${PYPROJECT_CONTENT}")

if(NOT CMAKE_MATCH_1)
    message(FATAL_ERROR "Failed to read version from pyproject.toml")
endif()

set(BMMPY_VERSION "${CMAKE_MATCH_1}")


# Core C++ library
add_library(bmmpy_core STATIC)

target_sources(
    bmmpy_core
    PRIVATE
        src/stub.cpp

        src/core/row_window.cpp
        src/core/bit_matrix.cpp
        src/core/bit_ops_scalar.cpp
        src/core/bit_ops_dispatch.cpp

        src/math/comb.cpp
        src/math/fwht.cpp
        src/math/fwht_scalar.cpp
        src/math/fwht_dispatch.cpp

        src/apply/greedy_selection.cpp

        src/search/fwht_search.cpp
        src/search/mitm_fwht_search.cpp

        # fwht16
        src/fwht16/engine.cpp
        src/fwht16/gpu_executor.cpp
        src/fwht16/cpu_executor.cpp
        src/fwht16/cpu_kernel_scalar.cpp
        # src/fwht16/cpu_kernel_avx2.cpp
        src/fwht16/cpu_dispatch.cpp
)

target_include_directories(
    bmmpy_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_definitions(
    bmmpy_core
    PUBLIC
        BMMPY_VERSION="${BMMPY_VERSION}"
)


# Optional AVX2 backend
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" BMMPY_ARCH)

if(BMMPY_ARCH MATCHES "^(x86_64|amd64|i[3-6]86)$")
    target_sources(
        bmmpy_core
        PRIVATE
            src/core/bit_ops_avx2.cpp
            src/math/fwht_avx2.cpp
            src/fwht16/cpu_kernel_avx2.cpp
    )

    target_compile_definitions(
        bmmpy_core
        PRIVATE
            BMMPY_HAS_AVX2_BACKEND=1
    )

    if(MSVC)
        set_source_files_properties(
            src/core/bit_ops_avx2.cpp
            src/math/fwht_avx2.cpp
            src/fwht16/cpu_kernel_avx2.cpp

            PROPERTIES
                COMPILE_OPTIONS "/arch:AVX2"
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_source_files_properties(
            src/core/bit_ops_avx2.cpp
            src/math/fwht_avx2.cpp
            src/fwht16/cpu_kernel_avx2.cpp
            PROPERTIES
                COMPILE_OPTIONS "-mavx2"
        )
    endif()
endif()


# Third-party dependencies
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    target_link_libraries(bmmpy_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(bmmpy_core PRIVATE BMMPY_HAS_OPENMP=1)
else()
    message(STATUS "OpenMP not found; building without parallel FWHT backend")
endif()

find_package(
    Python 3.12 REQUIRED
    COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule
)

add_subdirectory(extern/nanobind)


# Python extension module
nanobind_add_module(
    _bmmpy
    NB_DOMAIN bmmpy
    STABLE_ABI
    python/bindings/main.cpp
    python/bindings/bit_matrix.cpp
    python/bindings/candidate.cpp
    python/bindings/search.cpp
    python/bindings/apply.cpp
    python/bindings/runtime.cpp
    python/bindings/math.cpp
    python/bindings/window.cpp
)

nanobind_add_stub(
    bmmpy_stub
    MODULE _bmmpy
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/bmmpy/_bmmpy.pyi"
    PYTHON_PATH $<TARGET_FILE_DIR:_bmmpy>
    DEPENDS _bmmpy
)

target_link_libraries(_bmmpy PRIVATE bmmpy_core)

# Install the native module inside the Python package directory.
install(
    TARGETS _bmmpy
    LIBRARY DESTINATION bmmpy
    RUNTIME DESTINATION bmmpy
)


# Native tools
add_executable(cli src/cli.cpp)
target_link_libraries(cli PRIVATE bmmpy_core)

# Benchmarks
if(BMMPY_BUILD_BENCHMARKS)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "" FORCE)

    add_subdirectory(extern/benchmark EXCLUDE_FROM_ALL)

    add_executable(fwht16_bench bench/fwht16_bench.cpp)
    target_link_libraries(
        fwht16_bench
        PRIVATE
            bmmpy_core
            benchmark::benchmark
            benchmark::benchmark_main
    )
endif()

# Tests
if(BUILD_TESTING)
    add_executable(bit_matrix_tests tests/bit_matrix_tests.cpp)
    target_link_libraries(bit_matrix_tests PRIVATE bmmpy_core)
    add_test(NAME bit_matrix_tests COMMAND bit_matrix_tests)

    add_executable(math_tests tests/math_tests.cpp)
    target_link_libraries(math_tests PRIVATE bmmpy_core)
    add_test(NAME math_tests COMMAND math_tests)

    add_executable(fwht16_tests tests/fwht16_tests.cpp)
    target_link_libraries(fwht16_tests PRIVATE bmmpy_core)
    add_test(NAME fwht16_tests COMMAND fwht16_tests)
endif()