cmake_minimum_required(VERSION 3.15)

project(bmmpy LANGUAGES CXX)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(BMMPY_IS_TOP_LEVEL ON)
else()
    set(BMMPY_IS_TOP_LEVEL OFF)
endif()

option(BMMPY_ENABLE_CUDA "Build the optional CUDA backend" ${BMMPY_IS_TOP_LEVEL})
option(BMMPY_CUDA_STATIC_RUNTIME "Link CUDA runtime statically for CUDA builds" OFF)
option(BMMPY_BUILD_PYTHON "Build the optional Python bindings" ${BMMPY_IS_TOP_LEVEL})
option(BMMPY_INSTALL "Create install rules for bmmpy" ${BMMPY_IS_TOP_LEVEL})

if(BMMPY_IS_TOP_LEVEL)
    include(CTest)
    set(BMMPY_BUILD_TESTS_DEFAULT ${BUILD_TESTING})
else()
    set(BMMPY_BUILD_TESTS_DEFAULT OFF)
endif()

option(BMMPY_BUILD_TESTS "Build the bmmpy test suite" ${BMMPY_BUILD_TESTS_DEFAULT})

# Global project settings
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the core library PIC-safe so it can be reused from optional shared modules.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CheckLanguage)
include(GNUInstallDirs)

if(BMMPY_BUILD_TESTS)
    if(BMMPY_IS_TOP_LEVEL AND NOT BUILD_TESTING)
        enable_testing()
    elseif(NOT BMMPY_IS_TOP_LEVEL AND NOT BUILD_TESTING)
        message(FATAL_ERROR
            "BMMPY_BUILD_TESTS=ON requires the parent project to enable testing before adding bmmpy"
        )
    endif()
endif()

# Project version
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" PYPROJECT_CONTENT)
string(REGEX MATCH "version = \"([^\"]+)\"" _ "${PYPROJECT_CONTENT}")

if(NOT CMAKE_MATCH_1)
    message(FATAL_ERROR "Failed to read version from pyproject.toml")
endif()

set(BMMPY_VERSION "${CMAKE_MATCH_1}")

# Core C++ library
add_library(bmmpy_core STATIC)
add_library(bmmpy::core ALIAS bmmpy_core)
set_target_properties(bmmpy_core PROPERTIES EXPORT_NAME core)

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
        src/search/cuda_bruteforce_search.cpp
        src/search/sa_selector.cpp
        src/search/sa_selector_pairwise.cpp
        src/search/sa_selector_higher_order.cpp

        src/solver/exact_basis_solver.cpp

        src/algebra/transforms.cpp
        src/algebra/mastrovito.cpp
        src/algebra/transform_utils.cpp

        src/ga/island.cpp
        src/ga/bounded_channel.cpp
        src/ga/island_model.cpp
        src/ga/genetic_algorithm_core.cpp
        src/ga/genetic_algorithm_population.cpp
        src/ga/genetic_algorithm_operators.cpp
        src/ga/genetic_algorithm_lifecycle.cpp
)

target_include_directories(
    bmmpy_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
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
set(BMMPY_PACKAGE_NEEDS_OPENMP OFF)
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    target_link_libraries(bmmpy_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(bmmpy_core PRIVATE BMMPY_HAS_OPENMP=1)
    set(BMMPY_PACKAGE_NEEDS_OPENMP ON)
else()
    message(STATUS "OpenMP not found; building without parallel FWHT backend")
endif()

set(BMMPY_PACKAGE_NEEDS_CUDA OFF)
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
        target_link_libraries(bmmpy_core PUBLIC CUDA::cudart_static)
    else()
        target_link_libraries(bmmpy_core PUBLIC CUDA::cudart)
    endif()

    target_compile_definitions(bmmpy_core PRIVATE BMMPY_HAS_CUDA_BACKEND=1)
    set(BMMPY_PACKAGE_NEEDS_CUDA ON)
else()
    message(STATUS "CUDA backend disabled")
endif()

if(BMMPY_BUILD_PYTHON)
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
        python/bindings/algebra.cpp
        python/bindings/candidate.cpp
        python/bindings/search.cpp
        python/bindings/apply.cpp
        python/bindings/runtime.cpp
        python/bindings/math.cpp
        python/bindings/window.cpp
        python/bindings/solver.cpp
        python/bindings/ga.cpp
    )

    nanobind_add_stub(
        bmmpy_stub
        MODULE _bmmpy
        OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/bmmpy/_bmmpy.pyi"
        PYTHON_PATH $<TARGET_FILE_DIR:_bmmpy>
        DEPENDS _bmmpy
    )

    target_link_libraries(_bmmpy PRIVATE bmmpy_core)

    if(BMMPY_INSTALL)
        # Install the native module inside the Python package directory.
        install(
            TARGETS _bmmpy
            LIBRARY DESTINATION bmmpy
            RUNTIME DESTINATION bmmpy
        )
    endif()
endif()

if(BMMPY_INSTALL)
    include(CMakePackageConfigHelpers)

    install(
        TARGETS bmmpy_core
        EXPORT bmmpyTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )

    install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/bmmpyConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/bmmpyConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/bmmpy"
    )

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/bmmpyConfigVersion.cmake"
        VERSION "${BMMPY_VERSION}"
        COMPATIBILITY SameMajorVersion
    )

    install(
        EXPORT bmmpyTargets
        FILE bmmpyTargets.cmake
        NAMESPACE bmmpy::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/bmmpy"
    )

    install(
        FILES
            "${CMAKE_CURRENT_BINARY_DIR}/bmmpyConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/bmmpyConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/bmmpy"
    )
endif()

# Tests
if(BMMPY_BUILD_TESTS)
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

    bmmpy_add_cpp_test(
        algebra_core_tests
        tests/cpp/algebra_core_tests.cpp
    )
    set_tests_properties(algebra_core_tests PROPERTIES LABELS "cpp;algebra")

    bmmpy_add_cpp_test(
        umbrella_header_tests
        tests/cpp/umbrella_header_tests.cpp
    )
    set_tests_properties(umbrella_header_tests PROPERTIES LABELS "cpp;api")

    if(BMMPY_BUILD_PYTHON)
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
endif()