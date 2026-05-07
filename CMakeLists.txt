cmake_minimum_required(VERSION 3.15)

project(bmmpy LANGUAGES CXX)

option(BMMPY_ENABLE_CUDA "Build the optional CUDA backend" OFF)
option(BMMPY_CUDA_STATIC_RUNTIME "Link CUDA runtime statically for CUDA builds" OFF)

# Global project settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# bmmpy_core is linked into the Python extension, so it must be PIC-safe.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)
include(CheckLanguage)

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
        src/core/bit_matrix_algebra.cpp
        src/core/bit_matrix_hash.cpp
        src/core/bit_matrix_rows.cpp
        src/core/bit_matrix_io.cpp
        src/core/bit_ops_scalar.cpp
        src/core/bit_ops_dispatch.cpp

        src/math/comb.cpp
        src/math/fwht.cpp
        src/math/fwht_scalar.cpp
        src/math/fwht_dispatch.cpp

        src/apply/greedy_applier.cpp
        src/apply/global_greedy_applier.cpp

        src/search/fwht_search.cpp
        src/search/mitm_fwht_search.cpp
        src/search/split_window_prep.cpp
        src/search/cuda_mitm_fwht_search.cpp
        src/search/bruteforce_search.cpp
        src/search/cuda_mitm_fwht_search.cpp
        src/search/bruteforce_search.cpp
        src/search/cuda_bruteforce_search.cpp
        src/search/sa_selector.cpp
        src/search/sa_selector_pairwise.cpp
        src/search/sa_selector_higher_order.cpp

        src/solver/exact_basis_solver.cpp
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
            PROPERTIES
                COMPILE_OPTIONS "/arch:AVX2"
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_source_files_properties(
            src/core/bit_ops_avx2.cpp
            src/math/fwht_avx2.cpp
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

if(BMMPY_ENABLE_CUDA)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "BMMPY_ENABLE_CUDA=ON but no CUDA compiler was found")
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    target_sources(
        bmmpy_core
        PRIVATE
            src/search/cuda_mitm_fwht_launch.cu
            src/search/cuda_bruteforce_launch.cu
    )

    if(BMMPY_CUDA_STATIC_RUNTIME)
        target_link_libraries(bmmpy_core PRIVATE CUDA::cudart_static)
    else()
        target_link_libraries(bmmpy_core PRIVATE CUDA::cudart)
    endif()
    target_compile_definitions(bmmpy_core PRIVATE BMMPY_HAS_CUDA_BACKEND=1)
else()
    message(STATUS "CUDA backend disabled")
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
    python/bindings/solver.cpp
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


# Tests
if(BUILD_TESTING)
    function(bmmpy_add_cpp_test target)
        add_executable(${target} ${ARGN})
        target_link_libraries(${target} PRIVATE bmmpy_core)
        target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests/support")
        add_test(NAME ${target} COMMAND ${target})
    endfunction()

    bmmpy_add_cpp_test(
        bit_matrix_core_tests
        tests/cpp/bit_matrix_storage_tests.cpp
        tests/cpp/bit_matrix_rows_tests.cpp
        tests/cpp/bit_matrix_io_tests.cpp
        tests/cpp/bit_matrix_hash_tests.cpp
    )
    set_tests_properties(bit_matrix_core_tests PROPERTIES LABELS "cpp;bit_matrix")

    bmmpy_add_cpp_test(
        math_core_tests
        tests/cpp/fwht_tests.cpp
    )
    set_tests_properties(math_core_tests PROPERTIES LABELS "cpp;math")

    bmmpy_add_cpp_test(
        search_cpu_tests
        tests/cpp/search_cpu_tests.cpp
        tests/cpp/sa_selector_tests.cpp
    )
    set_tests_properties(search_cpu_tests PROPERTIES LABELS "cpp;search")

    bmmpy_add_cpp_test(
        search_cuda_tests
        tests/cpp/search_cuda_tests.cpp
    )
    set_tests_properties(search_cuda_tests PROPERTIES LABELS "cpp;search;cuda" RESOURCE_LOCK "gpu")

    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pytest"
        RESULT_VARIABLE BMMPY_HAS_PYTEST
        OUTPUT_QUIET
        ERROR_QUIET
    )

    if(BMMPY_HAS_PYTEST EQUAL 0)
        set(BMMPY_TEST_PYTHON_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-python")

        add_custom_command(
            TARGET _bmmpy POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E rm -rf "${BMMPY_TEST_PYTHON_DIR}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${BMMPY_TEST_PYTHON_DIR}"
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${CMAKE_CURRENT_SOURCE_DIR}/bmmpy"
                    "${BMMPY_TEST_PYTHON_DIR}/bmmpy"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:_bmmpy>"
                    "${BMMPY_TEST_PYTHON_DIR}/bmmpy/"
            COMMENT "Staging build-tree bmmpy package for pytest"
        )

        add_test(
            NAME python_api_tests
            COMMAND ${CMAKE_COMMAND} -E env
                    "PYTHONPATH=${BMMPY_TEST_PYTHON_DIR}"
                    "BMMPY_EXPECT_CUDA_COMPILED=$<IF:$<BOOL:${BMMPY_ENABLE_CUDA}>,ON,OFF>"
                    ${Python_EXECUTABLE} -m pytest -q "${CMAKE_CURRENT_SOURCE_DIR}/tests/python"
        )
        set_tests_properties(
            python_api_tests
            PROPERTIES
                LABELS "python;api"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )
    else()
        message(STATUS "pytest not available in ${Python_EXECUTABLE}; skipping Python tests")
    endif()
endif()