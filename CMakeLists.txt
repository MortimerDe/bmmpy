cmake_minimum_required(VERSION 3.15)

project(bmmpy LANGUAGES CXX)


# Global project settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# bmmpy_core is linked into the Python extension, so it must be PIC-safe.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)


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
find_package(OpenMP REQUIRED)
target_link_libraries(bmmpy_core PUBLIC OpenMP::OpenMP_CXX)

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

add_subdirectory(extern/nanobind)


# Python extension module
nanobind_add_module(
    _bmmpy
    NB_DOMAIN bmmpy
    python/bindings/main.cpp
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
    add_executable(bit_matrix_tests tests/bit_matrix_tests.cpp)
    target_link_libraries(bit_matrix_tests PRIVATE bmmpy_core)
    add_test(NAME bit_matrix_tests COMMAND bit_matrix_tests)

    add_executable(math_tests tests/math_tests.cpp)
    target_link_libraries(math_tests PRIVATE bmmpy_core)
    add_test(NAME math_tests COMMAND math_tests)
endif()