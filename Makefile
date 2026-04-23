BUILD_DIR := build
BUILD_TYPE ?= Debug
PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
CIBW_BUILD ?= cp312-*
CIBW_OUTPUT ?= dist
VERSION ?=
BENCH_BUILD_DIR ?= build-bench
ENABLE_CUDA ?= ON
CUDA_MANYLINUX_IMAGE ?= bmmpy-manylinux_2_28-cuda13.0
CUDA_WHEEL_CMAKE_ARGS ?= -DBMMPY_ENABLE_CUDA=ON -DBMMPY_CUDA_STATIC_RUNTIME=ON -DCMAKE_CUDA_ARCHITECTURES=all-major

.PHONY: all configure build cli clean clean-build dev bump_ver stubs test test-py wheel wheel-manylinux wheel-manylinux-cuda wheel-tools release configure-bench build-bench bench

all: build

configure:
	cmake -S . -B $(BUILD_DIR) -G Ninja -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBMMPY_ENABLE_CUDA=$(ENABLE_CUDA)

build: configure
	cmake --build $(BUILD_DIR)

cli: build
	./$(BUILD_DIR)/cli

clean:
	cmake --build $(BUILD_DIR) --target clean

clean-build:
	rm -rf $(BUILD_DIR) dist wheelhouse

dev:
	CMAKE_ARGS="-DBMMPY_ENABLE_CUDA=$(ENABLE_CUDA)" $(PIP) install -e .

wheel:
	rm -rf dist
	$(PIP) wheel . -w dist

wheel-tools:
	$(PIP) install cibuildwheel

wheel-manylinux:
	rm -rf $(CIBW_OUTPUT)
	CIBW_BUILD="$(CIBW_BUILD)" \
	CIBW_MANYLINUX_X86_64_IMAGE=manylinux2014 \
	$(PYTHON) -m cibuildwheel --platform linux --output-dir $(CIBW_OUTPUT) \


wheel-manylinux-cuda:
	rm -rf $(CIBW_OUTPUT)
	CIBW_BUILD="$(CIBW_BUILD)" \
	CIBW_MANYLINUX_X86_64_IMAGE=$(CUDA_MANYLINUX_IMAGE) \
	CIBW_ENVIRONMENT='CMAKE_ARGS="$(CUDA_WHEEL_CMAKE_ARGS)" CUDACXX=/usr/local/cuda/bin/nvcc' \
	$(PYTHON) -m cibuildwheel --platform linux --output-dir $(CIBW_OUTPUT)

bump_ver:
	$(PYTHON) scripts/bump_ver.py

release:
	test -n "$(VERSION)"
	$(PYTHON) scripts/release.py $(VERSION) --push

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

stubs: configure
	cmake --build $(BUILD_DIR) --target bmmpy_stub

test-py: dev
	$(PYTHON) -m unittest discover -s tests -p 'python_api_tests.py'

configure-bench:
	cmake -S . -B $(BENCH_BUILD_DIR) -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DBMMPY_BUILD_BENCHMARKS=ON

build-bench: configure-bench
	cmake --build $(BENCH_BUILD_DIR) --target fwht16_bench

bench: build-bench
	mkdir -p bench_results 
	./$(BENCH_BUILD_DIR)/fwht16_bench --benchmark_out=bench_results/results_$(shell date +%Y%m%d_%H%M%S).json --benchmark_out_format=json